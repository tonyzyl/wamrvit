from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.activations import SwiGLU
from diffusers.models.attention import AttentionMixin
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.modeling_utils import ModelMixin

from wamrvit.attention import RopeAttention, RopeAttnProcessor
from wamrvit.encoder import MultiScalePatchEmbed, PatchEmbed
from wamrvit.multi_scale import MultiScaleBucketLayout


class RotaryPositionalEmbeddingFromCenters(nn.Module):
    """
    Generate rotary positional embeddings (cos, sin) for queries/keys using
    continuous centers. The head dimension is split according to `rope_dim_list`.

    This generic class handles any number of center coordinates (2D, 3D, etc.)
    provided the centers input shape matches the length of `rope_dim_list`.

    Requirements:
      - sum(rope_dim_list) must equal the attention head dimension.
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
            if len(theta) != self.num_segments:
                raise ValueError(
                    f"Length of theta ({len(theta)}) must match "
                    f"len(rope_dim_list) ({self.num_segments})"
                )
            self.theta = theta

        if scale is None:
            scale_vals = [1.0] * self.num_segments
        elif isinstance(scale, (float, int)):
            scale_vals = [float(scale)] * self.num_segments
        else:
            if len(scale) != self.num_segments:
                raise ValueError(
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
    - Set level_idx scale to 1.
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
        rope_axes_dim: tuple[int, ...] = (32, 32, 32),
        rope_scale: tuple[float, ...] = (1.0, 1.0, 1.0),
        eps: float = 1e-6,
        adaptive: bool = True,
        cell_scale_mode: str = "area",  # for config logging only
        use_scale_bias: bool = False,
        max_level_idx: int = 2,  # only utilized if use_scale_bias = True
        concat_mlp: bool = True,  # whether to concatenate attention and mlp outputs before the final projection in BasicTransformerBlock  # noqa: E501
        multi_scale_patch: bool = False,  # when True, use MultiScalePatchEmbed + MultiScaleOutputHead (native value_storage)  # noqa: E501
        multi_scale_channel_mult: int = 2,  # channel growth factor per shrink stage (1 = channel-constant)  # noqa: E501
        multi_scale_shared_finest: bool = False,  # serialized-config compatibility only; projections are always per-level  # noqa: E501
        level_conditioned_norm: bool = False,  # when True, per-block + output norms become level-conditioned (AdaLN-on-level, identity-init)  # noqa: E501
        temporal_mixer_type: str = "conv3d",
        temporal_mixer_dim: int = 256,
        temporal_mixer_heads: int = 8,
        temporal_mixer_time_frequencies: int = 8,
        temporal_mixer_num_slots: int = 2,
        temporal_mixer_value_dim: int | None = None,
        temporal_mixer_routing: str = "content_time",
    ) -> None:
        super().__init__()

        rope_axes_dim = tuple(int(dim) for dim in rope_axes_dim)
        if not rope_axes_dim or any(dim <= 0 or dim % 2 != 0 for dim in rope_axes_dim):
            raise ValueError(
                f"rope_axes_dim must contain positive even dimensions, got {rope_axes_dim}."
            )
        if sum(rope_axes_dim) != attention_head_dim:
            raise ValueError(
                "sum(rope_axes_dim) must equal attention_head_dim; "
                f"got {sum(rope_axes_dim)} != {attention_head_dim}."
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
                temporal_mixer_type=temporal_mixer_type,
                temporal_mixer_dim=temporal_mixer_dim,
                temporal_mixer_heads=temporal_mixer_heads,
                temporal_mixer_time_frequencies=temporal_mixer_time_frequencies,
                temporal_mixer_num_slots=temporal_mixer_num_slots,
                temporal_mixer_value_dim=temporal_mixer_value_dim,
                temporal_mixer_routing=temporal_mixer_routing,
                eps=eps,
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
            )
            self.output_proj = None  # unused in multi-scale mode
        else:
            self.x_embedder = PatchEmbed(
                (patch_size_t, patch_size[0], patch_size[1]),
                in_channels,
                inner_dim,
                temporal_mixer_type=temporal_mixer_type,
                temporal_mixer_dim=temporal_mixer_dim,
                temporal_mixer_heads=temporal_mixer_heads,
                temporal_mixer_time_frequencies=temporal_mixer_time_frequencies,
                temporal_mixer_num_slots=temporal_mixer_num_slots,
                temporal_mixer_value_dim=temporal_mixer_value_dim,
                temporal_mixer_routing=temporal_mixer_routing,
                eps=eps,
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

    @property
    def uses_observed_time(self) -> bool:
        """Whether rollout should supply physical observation gaps to this encoder."""
        return self.x_embedder.capabilities.uses_observed_time

    # fmt: off
    def forward_multi_scale(
        self,
        values_by_level: dict[int, torch.Tensor],  # {lvl: (N_l, C, T, Ph*s_l, Pw*s_l)}, s_l = 2^(max_level_idx - lvl)  # noqa: E501
        leaf_to_bucket: torch.Tensor,              # (N, 2) long — (level, position_within_bucket)  # noqa: E501
        centers: torch.Tensor,                     # (N, 3) — (cx, cy, cell_scale_val) in canonical leaf order  # noqa: E501
        bucket_layout: MultiScaleBucketLayout | None = None,
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

        if bucket_layout is None:
            bucket_layout = MultiScaleBucketLayout.from_mapping(leaf_to_bucket, bucket_sizes)
        bucket_layout.validate(
            leaf_to_bucket, bucket_sizes,
            centers=centers if bucket_layout.centers is not None else None,
        )

        tokens = self.x_embedder(
            values_by_level,
            leaf_to_bucket,
            bucket_layout=bucket_layout,
        )  # (1, N, D)
        level_idx = centers[..., 2].long()  # (N,)

        with torch.autocast(tokens.device.type, torch.float32):
            rotary_emb = self.rope(centers)  # (cos, sin), each (N, head_dim)

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
            tokens, leaf_to_bucket, bucket_sizes, bucket_layout=bucket_layout
        )  # {lvl: (N_l, C_out, R, H_l, W_l)}

    def forward(
        self,
        x_tokens: torch.Tensor | dict[int, torch.Tensor],
        centers: torch.Tensor,
        leaf_to_bucket: torch.Tensor | None = None,
        bucket_layout: MultiScaleBucketLayout | None = None,
    ) -> torch.Tensor | dict[int, torch.Tensor]:
        """
        x_tokens:  (N, C, T, Ph, Pw) adaptive — one leaf per token, uniform-patch mode.
                   (B, C, T, H,  W)  regular  — full-grid mode.
                   {level: (N_l, C, T, H_l, W_l)} native — requires leaf_to_bucket.
        Native training must enter through this forward so DDP can synchronize gradients.
        centers:   (N, 3), storing (cx, cy, cell_scale_val).
        Returns:   (N, C_out, R, Ph, Pw) adaptive, (B, C_out, R, H, W) regular,
                   or {level: (N_l, C_out, R, H_l, W_l)} native.
        """
        if isinstance(x_tokens, dict):
            if leaf_to_bucket is None:
                raise ValueError("Native forward requires leaf_to_bucket.")
            return self.forward_multi_scale(
                x_tokens,
                leaf_to_bucket,
                centers,
                bucket_layout=bucket_layout,
            )
        if bucket_layout is not None:
            raise ValueError("bucket_layout is only supported for native inputs.")
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
            f"For now, we assume T_in==p_t, but got T={T} and p_t={self.patch_size_t}"
        )

        level_idx = centers[..., 2].long() if self.adaptive else None  # (N,) or None
        with torch.autocast(x_tokens.device.type, torch.float32):
            rotary_emb = self.rope(centers)  # (cos, sin), each (N, head_dim)

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


class MultiScaleOutputHead(nn.Module):
    """Multi-scale output head with per-level channel-narrowing cascade.

    Level ℓ (depth d = max_level_idx − ℓ):
      token (D,)
        --> per-level Linear(D, C_out·m^d · R · Ph · Pw)
      (C_out·m^d, R, Ph, Pw), fold R into batch
        --> d ExpandBlocks, stage k: (C_out·m^(d-k) -> C_out·m^(d-k-1)), doubling spatial
      (C_out, R, Ph·s, Pw·s)

    m = channel_mult. Finest (d=0) has no expand cascade.
    Final expand stage emits raw magnitudes (no activation).

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
    ) -> None:
        super().__init__()
        self.max_level_idx = max_level_idx
        self.patch_size_t = patch_size_t
        self.base_patch_h = base_patch_h
        self.base_patch_w = base_patch_w
        self.out_channels = out_channels
        self.return_seq_len = return_seq_len
        self.channel_mult = channel_mult

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
        *,
        bucket_layout: MultiScaleBucketLayout | None = None,
    ) -> dict[int, torch.Tensor]:
        """
        tokens: (1, N, D). leaf_to_bucket: (N, 2). bucket_sizes: [N_0, ..., N_L].
        Returns {level: (N_l, C_out, R, H_l, W_l)} with H_l=Ph*s, W_l=Pw*s, s=2^(L-lvl).
        """
        if bucket_layout is None:
            bucket_layout = MultiScaleBucketLayout.from_mapping(leaf_to_bucket, bucket_sizes)
        bucket_layout.validate(leaf_to_bucket, bucket_sizes)
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
            lvl_tokens = tokens.index_select(0, bucket_layout.indices[lvl])  # bucket order

            start_c = self.out_channels * (m**depth)  # C_out * m^d
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
