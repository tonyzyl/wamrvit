from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.activations import SwiGLU
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb, get_1d_rotary_pos_embed
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import (
    logging,
)
from torch.nn.attention.flex_attention import flex_attention

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class RopeAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0. "
                "Please upgrade your pytorch version."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,  # (B, N, C)
        encoder_hidden_states: torch.Tensor = None,  # (B, N_enc, C)
        image_rotary_emb: torch.Tensor | None = None,
        cond_image_rotary_emb: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        level_idx: torch.Tensor | None = None,  # (N,) long, or None
    ) -> torch.Tensor:

        # QKV projections → (B, N, inner_dim)
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # dispatch_attention_fn expects (B, N, H, D_head).
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        encoder_query = encoder_key = encoder_value = None
        if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)  # (B, N_enc, inner_dim)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

        query = attn.norm_q(query)  # (B, N, H, D_head)
        key = attn.norm_k(key)

        # Apply Rotary Embeddings if provided (Note seq dim=1 for B,N,H,D).
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        _use_scale_bias = (
            hasattr(attn, "use_scale_bias") and attn.use_scale_bias and level_idx is not None
        )

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))  # (B, N_enc, H, D_head)
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            if cond_image_rotary_emb is not None:
                encoder_query = apply_rotary_emb(
                    encoder_query, cond_image_rotary_emb, sequence_dim=1
                )
                encoder_key = apply_rotary_emb(encoder_key, cond_image_rotary_emb, sequence_dim=1)

            # Concat along seq axis → (B, N+N_enc, H, D_head).
            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)

        if _use_scale_bias:
            # flex_attention with inline score_mod — bias computation fused
            # into the kernel (no N×N materialization).
            _scale_bias_table = attn.scale_bias_table  # (2*max_level_idx+1, H)
            _max_level_idx = attn.max_level_idx
            _level_idx = level_idx  # (N,) long

            def _scale_bias_score_mod(score, b, h, q_idx, kv_idx):
                delta = _level_idx[kv_idx] - _level_idx[q_idx]
                delta = delta.clamp(-_max_level_idx, _max_level_idx)
                return score + _scale_bias_table[delta + _max_level_idx, h]

            # flex_attention expects (B, H, N, D_head).
            hidden_states = flex_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                score_mod=_scale_bias_score_mod,
            ).transpose(1, 2)  # back to (B, N, H, D_head)
        else:
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )  # (B, N, H, D_head)

        # Merge heads → (B, N, inner_dim).
        hidden_states = hidden_states.flatten(2)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is None:
            return hidden_states
        else:
            return hidden_states, encoder_hidden_states


class RopeAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = RopeAttnProcessor
    _available_processors = [RopeAttnProcessor]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        eps: float = 1e-6,
        out_dim: int = None,
        elementwise_affine: bool = True,
        processor=None,
        use_scale_bias: bool = False,
        max_level_idx: int = 2,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads

        self.use_bias = bias
        self.dropout = dropout

        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        # Q/K/V projections
        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)

        # QK Norm (head_dim unchanged — applies per-head)
        self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)

        self.use_scale_bias = use_scale_bias
        if use_scale_bias:
            self.max_level_idx = max_level_idx
            self.scale_bias_table = nn.Parameter(torch.zeros(2 * max_level_idx + 1, self.heads))

        # to_out: inner_dim → out_dim
        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(torch.nn.Dropout(dropout))

        if added_kv_proj_dim is not None:
            self.norm_added_q = torch.nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = torch.nn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = torch.nn.Linear(
                added_kv_proj_dim, self.inner_dim, bias=added_proj_bias
            )
            self.add_k_proj = torch.nn.Linear(
                added_kv_proj_dim, self.inner_dim, bias=added_proj_bias
            )
            self.add_v_proj = torch.nn.Linear(
                added_kv_proj_dim, self.inner_dim, bias=added_proj_bias
            )
            self.to_add_out = torch.nn.Linear(self.inner_dim, query_dim, bias=out_bias)

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        image_rotary_emb: torch.Tensor | None = None,
        cond_image_rotary_emb: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        level_idx: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        kwargs["level_idx"] = level_idx
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        unused_kwargs = [k for k, _ in kwargs.items() if k not in attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"joint_attention_kwargs {unused_kwargs} are not expected by "
                f"{self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states,
            image_rotary_emb,
            cond_image_rotary_emb,
            attention_mask=attention_mask,
            **kwargs,
        )


class RotaryPositionalEmbeddingFromCenters(nn.Module):
    """
    Generate rotary positional embeddings (cos, sin) for queries/keys using
    continuous centers. The head dimension is split according to `rope_dim_list`.

    This generic class handles any number of center coordinates (2D, 3D, etc.)
    provided the centers input shape matches the length of `rope_dim_list`.

    Requirements:
      - sum(rope_dim_list) must equal head_dim.
      - centers shape: [B, N, D] or [N, D] where D == len(rope_dim_list).

    Notes:
      - get_1d_rotary_pos_embed must be available in the scope.
    """

    def __init__(
        self,
        rope_dim_list: list[int],
        theta: float | list[float] = 10000.0,
        ntk_factor: float = 1.0,
        scale: float | list[float] | None = None,
    ) -> None:
        super().__init__()

        self.rope_dim_list = rope_dim_list
        self.ntk_factor = ntk_factor
        self.num_segments = len(rope_dim_list)

        if isinstance(theta, float) or isinstance(theta, int):
            self.theta = [float(theta)] * self.num_segments
        else:
            assert len(theta) == self.num_segments, (
                f"Length of theta ({len(theta)}) must match "
                f"len(rope_dim_list) ({self.num_segments})"
            )
            self.theta = theta

        if scale is None:
            scale_vals = [1.0] * self.num_segments
        elif isinstance(scale, (float, int)):
            scale_vals = [float(scale)] * self.num_segments
        else:
            assert len(scale) == self.num_segments, (
                f"Length of scale ({len(scale)}) must match "
                f"len(rope_dim_list) ({self.num_segments})"
            )
            scale_vals = scale

        self.register_buffer(
            "scale_tensor",
            torch.tensor(scale_vals, dtype=torch.float32),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        centers: [B, N, D] or [N, D] where D is the number of segments.
        Returns (cos, sin) each [N, D].
        """
        # Flatten batch dimension: [B, N, D] -> [S, D]
        if centers.dim() == 3:
            centers_flat = centers.reshape(-1, centers.shape[-1])
        elif centers.dim() == 2:
            centers_flat = centers
        else:
            raise ValueError(f"centers must be [N, D] or [B, N, D], got {centers.shape}")

        if centers_flat.shape[-1] != self.num_segments:
            raise ValueError(
                f"Last dimension of centers ({centers_flat.shape[-1]}) must match "
                f"len(rope_dim_list) ({self.num_segments})"
            )

        device = centers_flat.device

        scales = self.scale_tensor.to(device)  # [D]

        freqs = []
        for i, dim in enumerate(self.rope_dim_list):
            coord = centers_flat[:, i] * scales[i]
            freq = get_1d_rotary_pos_embed(
                dim,
                coord,
                theta=self.theta[i],
                use_real=True,
                ntk_factor=self.ntk_factor,
            )
            freqs.append(freq)

        cos = torch.cat([f[0] for f in freqs], dim=-1).to(device)
        sin = torch.cat([f[1] for f in freqs], dim=-1).to(device)

        return cos, sin


class LevelConditionedNorm(nn.Module):
    """LayerNorm with per-level learnable scale/shift. Identity-init (zero scale,
    zero shift) so the norm behaves exactly like ``nn.LayerNorm(..., elementwise_affine=False)``
    at initialization and only diverges as training moves the per-level embedding.

    Applied as ``LN(x) * (1 + scale[level_idx]) + shift[level_idx]``.

    When ``level_idx is None`` (e.g., regular/non-adaptive path), falls back
    to plain LN output — so the module is safe to instantiate regardless of
    whether the caller actually supplies level labels at forward time.
    """

    def __init__(self, hidden_size: int, num_levels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=False)
        # (num_levels, 2 * hidden_size) — scale|shift concatenated per level.
        self.embed = nn.Embedding(num_levels, 2 * hidden_size)
        nn.init.zeros_(self.embed.weight)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, level_idx: torch.Tensor | None = None) -> torch.Tensor:
        x_norm = self.norm(x)
        if level_idx is None:
            return x_norm
        emb = self.embed(level_idx)  # (N, 2 * hidden_size)
        scale, shift = emb.chunk(2, dim=-1)  # (N, hidden_size) each
        # Broadcast leading batch dim: (B, N, hidden_size) = (B, N, D) * (1, N, D) + (1, N, D).
        return x_norm * (1.0 + scale.unsqueeze(0)) + shift.unsqueeze(0)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        bias: bool = True,
        eps: float = 1e-6,
        use_scale_bias: bool = False,
        max_level_idx: int = 2,
        concat: bool = True,
        level_conditioned_norm: bool = False,  # when True, swap self.norm for LevelConditionedNorm
    ):
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim
        mlp_dim = int(hidden_size * mlp_ratio)

        self.attn = RopeAttention(
            query_dim=hidden_size,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=bias,
            out_bias=bias,
            eps=eps,
            processor=RopeAttnProcessor(),
            use_scale_bias=use_scale_bias,
            max_level_idx=max_level_idx,
        )

        if level_conditioned_norm:
            num_levels = int(max_level_idx) + 1
            self.norm = LevelConditionedNorm(hidden_size, num_levels, eps=eps)
        else:
            self.norm = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=False)
        self.proj_mlp = nn.Linear(hidden_size, mlp_dim)
        self.act_mlp = SwiGLU(dim_in=mlp_dim, dim_out=mlp_dim)

        self.concat = concat

        # --- Routing Logic for Projections ---
        if self.concat:
            # proj_out takes the concatenated dimensions
            self.proj_out = nn.Linear(hidden_size + mlp_dim, hidden_size)
        else:
            # proj_out solely handles the Attention output projection
            self.proj_mlp_down = nn.Linear(mlp_dim, hidden_size)
            self.proj_out = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, N, hidden_size)
        temb_mod_hs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        level_idx: torch.Tensor | None = None,  # (N,) long, or None
    ) -> torch.Tensor:

        if isinstance(self.norm, LevelConditionedNorm):
            norm_hidden_states = self.norm(hidden_states, level_idx)  # (B, N, hidden_size)
        else:
            norm_hidden_states = self.norm(hidden_states)  # (B, N, hidden_size)
        if temb_mod_hs is not None:
            mod_shift, mod_scale, mod_gate = temb_mod_hs
            norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

        mlp_hidden_states = self.proj_mlp(norm_hidden_states)  # (B, N, mlp_dim)
        mlp_hidden_states = self.act_mlp(mlp_hidden_states)  # (B, N, mlp_dim) after SwiGLU

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            level_idx=level_idx,
        )  # (B, N, hidden_size)

        if self.concat:
            # Concat-then-project path: (B, N, hidden+mlp) → (B, N, hidden_size).
            attn_output = torch.cat([attn_output, mlp_hidden_states], dim=2)
            attn_output = self.proj_out(attn_output)
        else:
            # Parallel-sum path: project each branch to hidden_size and add.
            mlp_hidden_states = self.proj_mlp_down(mlp_hidden_states)  # (B, N, hidden_size)
            attn_output = self.proj_out(attn_output)  # (B, N, hidden_size)
            attn_output = attn_output + mlp_hidden_states

        if temb_mod_hs is not None:
            hidden_states = hidden_states + mod_gate * attn_output
        else:
            hidden_states = hidden_states + attn_output

        return hidden_states  # (B, N, hidden_size)


class QuadTreeTransformer(
    ModelMixin, AttentionMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin
):
    """
    Minimal transformer that consumes quadtree tokens and applies RoPE built from
    (cx, cy, h) centers. Batch size 1 expected; no padding logic.

    Recommendation on rope config:
    - Set last segment (cell scale) to have dim = 4 *(num_levels+1) to capture
      level differences well.
    - Scale the x and y axes to (H/tile_height)*(2^(num_levels-1)) and
      (W/tile_width)*(2^(num_levels-1)) respectively, equal to the number of
      finest-level cells along each dimension.
    tile_height: the largest cell height at level 0 (coarsest).
    - Set level_idx scale to 1. See docs/rope_scale_config.md for details.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        return_seq_len: int = 1,
        patch_size: int | tuple[int, int] = 32,
        patch_size_t: int | None = 1,
        num_attention_heads: int = 8,
        attention_head_dim: int = 96,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        rope_theta: float = 10000.0,
        rope_axes_dim: tuple[int, int, int] = (32, 32, 32),
        rope_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        eps: float = 1e-6,
        adaptive: bool = True,
        cell_scale_mode: str = "area",  # for config logging only
        use_scale_bias: bool = False,
        max_level_idx: int = 2,  # only utilized if use_scale_bias = True
        concat_mlp: bool = True,  # whether to concatenate attention and mlp outputs before the final projection in BasicTransformerBlock  # noqa: E501
        multi_scale_patch: bool = False,  # when True, use MultiScalePatchEmbed + MultiScaleOutputHead (native value_storage)  # noqa: E501
        multi_scale_channel_mult: int = 2,  # channel growth factor per shrink stage (1 = channel-constant)  # noqa: E501
        multi_scale_shared_finest: bool = False,  # Design B: share finest-level patch proj + linear head across all levels with per-level 1×1×1 reducer/expander. Requires multi_scale_patch.  # noqa: E501
        level_conditioned_norm: bool = False,  # when True, per-block + output norms become level-conditioned (AdaLN-on-level, identity-init)  # noqa: E501
    ) -> None:
        super().__init__()

        assert attention_head_dim % len(rope_axes_dim) == 0, (
            f"attention_head_dim must be divisible by the number of rope axes "
            f"({len(rope_axes_dim)})"
        )
        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels
        self.out_channels = out_channels

        self.in_channels = in_channels
        self.return_seq_len = return_seq_len

        # Expose key dims
        self.inner_dim = inner_dim  # alias often used in diffusers-style configs
        self.attention_head_dim = attention_head_dim

        self.adaptive = adaptive
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t

        self.multi_scale_patch = multi_scale_patch
        self.multi_scale_shared_finest = multi_scale_shared_finest
        if multi_scale_shared_finest and not multi_scale_patch:
            raise ValueError("multi_scale_shared_finest requires multi_scale_patch=True.")
        if multi_scale_patch:
            if not adaptive:
                raise ValueError("multi_scale_patch requires adaptive=True.")
            self.x_embedder = MultiScalePatchEmbed(
                max_level_idx=max_level_idx,
                patch_size_t=patch_size_t,
                base_patch_h=patch_size[0],
                base_patch_w=patch_size[1],
                in_chans=in_channels,
                embed_dim=inner_dim,
                channel_mult=multi_scale_channel_mult,
                shared_finest=multi_scale_shared_finest,
            )
            self.output_head = MultiScaleOutputHead(
                max_level_idx=max_level_idx,
                patch_size_t=patch_size_t,
                base_patch_h=patch_size[0],
                base_patch_w=patch_size[1],
                embed_dim=inner_dim,
                out_channels=out_channels,
                return_seq_len=return_seq_len,
                channel_mult=multi_scale_channel_mult,
                shared_finest=multi_scale_shared_finest,
            )
            self.output_proj = None  # unused in multi-scale mode
        else:
            self.x_embedder = PatchEmbed(
                (patch_size_t, patch_size[0], patch_size[1]), in_channels, inner_dim
            )
            self.output_proj = nn.Linear(
                inner_dim, out_channels * patch_size[0] * patch_size[1] * self.return_seq_len
            )
            self.output_head = None

        self.rope = RotaryPositionalEmbeddingFromCenters(
            rope_dim_list=list(rope_axes_dim),
            theta=rope_theta,
            scale=rope_scale,
        )

        self.level_conditioned_norm = level_conditioned_norm

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    num_attention_heads,
                    attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                    use_scale_bias=use_scale_bias,
                    max_level_idx=max_level_idx,
                    concat=concat_mlp,
                    level_conditioned_norm=level_conditioned_norm,
                )
                for _ in range(num_layers)
            ]
        )

        # self.norm_out = AdaLayerNormContinuous(
        # inner_dim, inner_dim, elementwise_affine=False, eps=eps, bias=False
        # )
        if level_conditioned_norm:
            self.norm_out = LevelConditionedNorm(inner_dim, int(max_level_idx) + 1, eps=eps)
        else:
            self.norm_out = nn.LayerNorm(inner_dim, eps=eps, elementwise_affine=False, bias=False)

        self.gradient_checkpointing = False

    # fmt: off
    def forward_multi_scale(
        self,
        values_by_level: dict[int, torch.Tensor],  # {lvl: (N_l, C, T, Ph*s_l, Pw*s_l)}, s_l = 2^(max_level_idx - lvl)  # noqa: E501
        leaf_to_bucket: torch.Tensor,              # (N, 2) long — (level, position_within_bucket)  # noqa: E501
        centers: torch.Tensor,                     # (N, 3) — (cx, cy, cell_scale_val) in canonical leaf order  # noqa: E501
    ) -> dict[int, torch.Tensor]:
        # fmt: on
        """
        values_by_level: {level: (N_l, C, T, H_l, W_l)} with H_l=Ph*s_l, W_l=Pw*s_l.
        leaf_to_bucket:  (N, 2) long tensor, rows match canonical leaf (Morton) order.
        centers:         (N, 3) in canonical leaf order (same order as leaf_to_bucket rows).
        Returns: {level: (N_l, C_out, T_out, H_l, W_l)} with N_l matching input buckets.
        """
        assert self.multi_scale_patch, "forward_multi_scale requires multi_scale_patch=True."
        # bucket_sizes[lvl] = N_l for each level (possibly zero).
        bucket_sizes = [
            int(values_by_level[lvl].shape[0]) for lvl in range(self.config.max_level_idx + 1)
        ]

        tokens = self.x_embedder(values_by_level, leaf_to_bucket)  # (1, N, D)

        with torch.autocast(tokens.device.type, torch.float32):
            rotary_emb = self.rope(centers)  # (cos, sin), each (N, head_dim)

        level_idx = centers[..., 2].long()  # (N,)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.transformer_blocks:
                tokens = self._gradient_checkpointing_func(
                    block,
                    tokens,
                    None,
                    rotary_emb,
                    None,
                    level_idx,
                )  # (1, N, D)
        else:
            for block in self.transformer_blocks:
                tokens = block(
                    tokens,
                    None,
                    rotary_emb,
                    level_idx=level_idx,
                )  # (1, N, D)

        if isinstance(self.norm_out, LevelConditionedNorm):
            tokens = self.norm_out(tokens, level_idx)  # (1, N, D)
        else:
            tokens = self.norm_out(tokens)  # (1, N, D)
        return self.output_head(
            tokens, leaf_to_bucket, bucket_sizes
        )  # {lvl: (N_l, C_out, R, H_l, W_l)}

    def forward(self, x_tokens: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """
        x_tokens: (N, C, T, Ph, Pw) adaptive — one leaf per token, uniform-patch mode.
                  (B, C, T, H,  W)  regular  — full-grid mode.
        centers:  (N, 3), storing (cx, cy, cell_scale_val).
        Returns:  (N, C_out, R, Ph, Pw) adaptive, or (B, C_out, R, H, W) regular.
        """
        if self.adaptive:
            N, C, T, H, W = x_tokens.shape  # H == Ph, W == Pw
            assert H == self.patch_size[0] and W == self.patch_size[1], (
                f"Adaptive cell spatial dims must exactly match patch_size, "
                f"but got H={H}, W={W} and patch_size={self.patch_size}"
            )
            x_tokens = self.x_embedder(x_tokens)  # (N, 1, D)
            x_tokens = x_tokens.flatten(0, 1).unsqueeze(0)  # (1, N, D)
        else:
            B, C, T, H, W = x_tokens.shape
            x_tokens = self.x_embedder(x_tokens)  # (B, N_tokens, D), N_tokens = (H/p_h)*(W/p_w)

        assert T == self.patch_size_t, (
            f"For now, we assume and T_in==p_t, but got T={T} and p_t={self.patch_size_t}"
        )

        with torch.autocast(x_tokens.device.type, torch.float32):
            rotary_emb = self.rope(centers)  # (cos, sin), each (N, head_dim)

        level_idx = centers[..., 2].long() if self.config.adaptive else None  # (N,) or None

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.transformer_blocks:
                x_tokens = self._gradient_checkpointing_func(
                    block,
                    x_tokens,
                    None,
                    rotary_emb,
                    None,  # attention_mask
                    level_idx,
                )  # (B, N, D)
        else:
            for block in self.transformer_blocks:
                x_tokens = block(
                    x_tokens,
                    None,
                    rotary_emb,
                    level_idx=level_idx,
                )  # (B, N, D)

        if isinstance(self.norm_out, LevelConditionedNorm):
            x_tokens = self.norm_out(x_tokens, level_idx)  # (B, N, D)
        else:
            x_tokens = self.norm_out(x_tokens)  # (B, N, D)
        x_tokens = self.output_proj(x_tokens)  # (B, N, C_out * R * p_h * p_w)

        if self.adaptive:
            x_tokens = x_tokens.squeeze(0)  # (N, C_out*R*p_h*p_w)
            return x_tokens.reshape(
                N, self.out_channels, self.return_seq_len, H, W
            )  # (N, C_out, R, Ph, Pw)
        else:
            p_h, p_w = self.patch_size
            H_p, W_p = H // p_h, W // p_w
            # (B, H_p*W_p, C_out*R*p_h*p_w) → (B, H_p, W_p, C_out, R, p_h, p_w)
            x_tokens = x_tokens.reshape(
                B, H_p, W_p, self.out_channels, self.return_seq_len, p_h, p_w
            )
            # (B, C_out, R, H_p, p_h, W_p, p_w)
            x_tokens = x_tokens.permute(0, 3, 4, 1, 5, 2, 6)
            # → (B, C_out, R, H_p*p_h, W_p*p_w) = (B, C_out, R, H, W)
            x_tokens = x_tokens.flatten(5, 6).flatten(3, 4)
            return x_tokens


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int | tuple[int, int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()

        patch_size = (
            (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)  # -> (B, C, T/p_t, H/p_h, W/p_w)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # BCTHW -> BNC
        return hidden_states


class ShrinkBlock(nn.Module):
    """Input-side 2x spatial downsample with optional channel widening.

    (N, in_c, T, H, W) -> (N, out_c, T, H/2, W/2). Time dim untouched.
    SwiGLU activation matches the backbone MLP. No LayerNorm — per-leaf LN
    would destroy cross-channel physical-magnitude relationships.
    """

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.gate = SwiGLU(dim_in=out_c, dim_out=out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.movedim(1, -1)  # (N, T, H, W, out_c) for channel-last SwiGLU
        x = self.gate(x)
        x = x.movedim(-1, 1)
        return x


class ExpandBlock(nn.Module):
    """Output-side 2x spatial upsample with optional channel narrowing.

    (N, in_c, H, W) -> (N, out_c, 2H, 2W) via Conv2d(in_c, 4*out_c, 3x3) + PixelShuffle(2).

    Internal stages (is_last=False): Conv -> SwiGLU -> PixelShuffle.
    Final stage (is_last=True): Conv -> PixelShuffle only (no activation so the cascade
    can emit arbitrary-magnitude values including negatives).

    Caller folds time R into batch before calling.
    """

    def __init__(self, in_c: int, out_c: int, is_last: bool = False) -> None:
        super().__init__()
        self.is_last = is_last
        self.conv = nn.Conv2d(in_c, 4 * out_c, kernel_size=3, padding=1)
        self.gate = None if is_last else SwiGLU(dim_in=4 * out_c, dim_out=4 * out_c)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.gate is not None:
            x = x.movedim(1, -1)  # (N, H, W, 4*out_c) for channel-last SwiGLU
            x = self.gate(x)
            x = x.movedim(-1, 1)
        return self.shuffle(x)


class MultiScalePatchEmbed(nn.Module):
    """Multi-scale patchify with per-level channel-widening cascade.

    Level ℓ (depth d = max_level_idx − ℓ):
      (C, p_t, Ph·s, Pw·s)
        --> d ShrinkBlocks, stage k: (C·m^k -> C·m^(k+1)), halving spatial
      (C·m^d, p_t, Ph, Pw)
        --> per-level path or shared-finest path (see shared_finest below)
      token (D,)

    m = channel_mult. Finest (d=0) has no shrink cascade.

    Two final-projection paths:

    - ``shared_finest=False`` (default): per-level ``projs[ℓ]: Conv3d(C·m^d,
      D, (p_t, Ph, Pw))``.

    - ``shared_finest=True`` (Design B): a single ``proj_shared: Conv3d(C, D,
      (p_t, Ph, Pw))`` is used across all levels. Coarse levels first pass
      through ``reducers[ℓ]: Conv3d(C·m^d, C, 1×1×1)`` which compresses the
      cascade's widened channels back to ``in_chans`` before the shared
      projection. The finest level (d=0) has no cascade and no reducer (the
      reducer is ``nn.Identity``), so its path is architecturally identical
      to uniform mode's ``PatchEmbed``.
    """

    def __init__(
        self,
        max_level_idx: int,
        patch_size_t: int,
        base_patch_h: int,
        base_patch_w: int,
        in_chans: int,
        embed_dim: int,
        channel_mult: int = 2,
        shared_finest: bool = False,
    ) -> None:
        super().__init__()
        self.max_level_idx = max_level_idx
        self.embed_dim = embed_dim
        self.patch_size_t = patch_size_t
        self.base_patch_h = base_patch_h
        self.base_patch_w = base_patch_w
        self.channel_mult = channel_mult
        self.shared_finest = shared_finest

        kernel = (patch_size_t, base_patch_h, base_patch_w)

        self.shrink_cascades = nn.ModuleList()
        for lvl in range(max_level_idx + 1):
            depth = max_level_idx - lvl
            stages = []
            for k in range(depth):
                in_c = in_chans * (channel_mult**k)
                out_c = in_chans * (channel_mult ** (k + 1))
                stages.append(ShrinkBlock(in_c, out_c))
            self.shrink_cascades.append(nn.Sequential(*stages))

        if shared_finest:
            self.proj_shared = nn.Conv3d(in_chans, embed_dim, kernel_size=kernel, stride=kernel)
            self.reducers = nn.ModuleList()
            for lvl in range(max_level_idx + 1):
                depth = max_level_idx - lvl
                if depth == 0:
                    self.reducers.append(nn.Identity())
                else:
                    end_c = in_chans * (channel_mult**depth)
                    self.reducers.append(nn.Conv3d(end_c, in_chans, kernel_size=1))
        else:
            self.projs = nn.ModuleList()
            for lvl in range(max_level_idx + 1):
                depth = max_level_idx - lvl
                end_c = in_chans * (channel_mult**depth)
                self.projs.append(nn.Conv3d(end_c, embed_dim, kernel_size=kernel, stride=kernel))

    # fmt: off
    def forward(
        self,
        values_by_level: dict[int, torch.Tensor],  # {lvl: (N_l, C, p_t, Ph*s, Pw*s)}, s = 2^(max_level_idx - lvl)  # noqa: E501
        leaf_to_bucket: torch.Tensor,              # (N, 2) long — (level, position_within_bucket)  # noqa: E501
    ) -> torch.Tensor:
        # fmt: on
        """
        values_by_level: {level: (N_l, C, T, H_l, W_l)}
        leaf_to_bucket:  (N, 2) long tensor (level, position_within_bucket).
        Returns (1, N, D) tokens in canonical leaf order.
        """
        N = int(leaf_to_bucket.shape[0])
        device = leaf_to_bucket.device
        ref_dtype = None
        for lvl in values_by_level:
            if values_by_level[lvl].shape[0] > 0:
                ref_dtype = values_by_level[lvl].dtype
                break
        if ref_dtype is None:
            if self.shared_finest:
                ref_dtype = next(self.proj_shared.parameters()).dtype
            else:
                ref_dtype = next(self.projs[self.max_level_idx].parameters()).dtype

        tokens = torch.zeros(N, self.embed_dim, device=device, dtype=ref_dtype)  # (N, D)
        for lvl, bucket in values_by_level.items():
            if bucket.shape[0] == 0:
                continue
            # bucket: (N_l, C, p_t, Ph*s, Pw*s); shrink cascade halves spatial
            # and widens channels each stage.
            x = self.shrink_cascades[lvl](bucket)  # (N_l, C*m^d, p_t, Ph, Pw)
            if self.shared_finest:
                x = self.reducers[lvl](x)  # (N_l, C, p_t, Ph, Pw); Identity at d=0
                out = self.proj_shared(x)  # (N_l, D, 1, 1, 1)
            else:
                out = self.projs[lvl](x)  # (N_l, D, 1, 1, 1)
            out = out.flatten(1)  # (N_l, D)
            # Scatter N_l bucket tokens into their positions in the flat (N, D) sequence.
            mask = leaf_to_bucket[:, 0] == lvl  # (N,) bool
            positions = leaf_to_bucket[mask, 1].long()  # (N_l,)
            tokens[mask] = out[positions].to(tokens.dtype)
        return tokens.unsqueeze(0)  # (1, N, D)


class MultiScaleOutputHead(nn.Module):
    """Multi-scale output head with per-level channel-narrowing cascade.

    Level ℓ (depth d = max_level_idx − ℓ):
      token (D,)
        --> per-level path or shared-finest path (see shared_finest below)
      (C_out·m^d, R, Ph, Pw), fold R into batch
        --> d ExpandBlocks, stage k: (C_out·m^(d-k) -> C_out·m^(d-k-1)), doubling spatial
      (C_out, R, Ph·s, Pw·s)

    m = channel_mult. Finest (d=0) has no expand cascade.
    Final expand stage emits raw magnitudes (no activation).

    Two initial-projection paths (symmetric to MultiScalePatchEmbed):

    - ``shared_finest=False`` (default): per-level ``linears[ℓ]: Linear(D,
      C_out·m^d · R · Ph · Pw)``.

    - ``shared_finest=True`` (Design B): a single ``linear_shared: Linear(D,
      C_out · R · Ph · Pw)``. Coarse levels then pass the reshaped
      ``(C_out, R, Ph, Pw)`` tensor through ``expanders[ℓ]: Conv3d(C_out,
      C_out·m^d, 1×1×1)`` to widen channels for the expand cascade. The
      finest level (d=0) has no expander and no cascade — the linear's
      output reshapes directly to ``(C_out, R, Ph, Pw)`` which is already
      the final output shape, identical to uniform mode's reshape-from-
      ``output_proj``.
    """

    def __init__(
        self,
        max_level_idx: int,
        patch_size_t: int,
        base_patch_h: int,
        base_patch_w: int,
        embed_dim: int,
        out_channels: int,
        return_seq_len: int,
        channel_mult: int = 2,
        shared_finest: bool = False,
    ) -> None:
        super().__init__()
        self.max_level_idx = max_level_idx
        self.patch_size_t = patch_size_t
        self.base_patch_h = base_patch_h
        self.base_patch_w = base_patch_w
        self.out_channels = out_channels
        self.return_seq_len = return_seq_len
        self.channel_mult = channel_mult
        self.shared_finest = shared_finest

        if shared_finest:
            self.linear_shared = nn.Linear(
                embed_dim, out_channels * return_seq_len * base_patch_h * base_patch_w
            )
            self.expanders = nn.ModuleList()
            for lvl in range(max_level_idx + 1):
                depth = max_level_idx - lvl
                if depth == 0:
                    self.expanders.append(nn.Identity())
                else:
                    start_c = out_channels * (channel_mult**depth)
                    self.expanders.append(nn.Conv3d(out_channels, start_c, kernel_size=1))
        else:
            self.linears = nn.ModuleList()
            for lvl in range(max_level_idx + 1):
                depth = max_level_idx - lvl
                start_c = out_channels * (channel_mult**depth)
                out_dim = start_c * return_seq_len * base_patch_h * base_patch_w
                self.linears.append(nn.Linear(embed_dim, out_dim))

        self.expand_cascades = nn.ModuleList()
        for lvl in range(max_level_idx + 1):
            depth = max_level_idx - lvl
            stages = []
            for k in range(depth):
                in_c = out_channels * (channel_mult ** (depth - k))
                out_c = out_channels * (channel_mult ** (depth - k - 1))
                is_last = k == depth - 1
                stages.append(ExpandBlock(in_c, out_c, is_last=is_last))
            self.expand_cascades.append(nn.Sequential(*stages))

    def forward(
        self,
        tokens: torch.Tensor,  # (1, N, D)
        leaf_to_bucket: torch.Tensor,  # (N, 2) long — (level, position_within_bucket)
        bucket_sizes: list[int],  # [N_0, N_1, ..., N_L]
    ) -> dict[int, torch.Tensor]:
        """
        tokens: (1, N, D). leaf_to_bucket: (N, 2). bucket_sizes: [N_0, ..., N_L].
        Returns {level: (N_l, C_out, R, H_l, W_l)} with H_l=Ph*s, W_l=Pw*s, s=2^(L-lvl).
        """
        tokens = tokens.squeeze(0)  # (N, D)
        out_by_level: dict[int, torch.Tensor] = {}
        R = self.return_seq_len
        Ph = self.base_patch_h
        Pw = self.base_patch_w
        m = self.channel_mult
        for lvl in range(self.max_level_idx + 1):
            N_l = int(bucket_sizes[lvl])
            depth = self.max_level_idx - lvl  # d
            s = 1 << depth  # 2^d
            h_l = Ph * s
            w_l = Pw * s
            if N_l == 0:
                out_by_level[lvl] = tokens.new_zeros(0, self.out_channels, R, h_l, w_l)
                continue
            # Gather this level's tokens from the flat sequence in bucket-position order.
            mask = leaf_to_bucket[:, 0] == lvl  # (N,) bool
            positions = leaf_to_bucket[mask, 1].long()  # (N_l,) bucket indices
            idx_masked = torch.nonzero(mask, as_tuple=False).squeeze(1)  # (N_l,) flat indices
            reorder = torch.empty(N_l, dtype=torch.long, device=tokens.device)
            reorder[positions] = idx_masked  # maps bucket slot → flat index
            lvl_tokens = tokens[reorder]  # (N_l, D) in bucket order

            start_c = self.out_channels * (m**depth)  # C_out * m^d
            if self.shared_finest:
                projected = self.linear_shared(lvl_tokens)  # (N_l, C_out*R*Ph*Pw)
                x = projected.reshape(N_l, self.out_channels, R, Ph, Pw)
                if depth > 0:
                    x = self.expanders[lvl](x)  # (N_l, start_c, R, Ph, Pw)
            else:
                projected = self.linears[lvl](lvl_tokens)  # (N_l, start_c*R*Ph*Pw)
                x = projected.reshape(N_l, start_c, R, Ph, Pw)

            if depth == 0:
                # Finest level: x is (N_l, C_out, R, Ph, Pw) = final output shape.
                out_by_level[lvl] = x
            else:
                # Fold R into batch so PixelShuffle(2) operates per-frame: (N_l*R, start_c, Ph, Pw).
                x = x.permute(0, 2, 1, 3, 4).reshape(N_l * R, start_c, Ph, Pw)
                # Expand cascade halves channels and doubles spatial each stage
                # → (N_l*R, C_out, h_l, w_l).
                x = self.expand_cascades[lvl](x)
                # Unfold R back out of batch → (N_l, C_out, R, h_l, w_l).
                x = (
                    x.reshape(N_l, R, self.out_channels, h_l, w_l)
                    .permute(0, 2, 1, 3, 4)
                    .contiguous()
                )
                out_by_level[lvl] = x
        return out_by_level  # {lvl: (N_l, C_out, R, H_l, W_l)}
