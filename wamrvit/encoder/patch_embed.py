"""Upstream-style patch adapters with fixed-cadence encoder assembly."""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.models.activations import SwiGLU

from wamrvit.encoder.conv3d import Conv3dTubeletProjection
from wamrvit.encoder.encoder_builder import build_temporal_mixer, encoder_capabilities
from wamrvit.multi_scale import MultiScaleBucketLayout


class ShrinkBlock(nn.Module):
    """Input-side 2x spatial downsample with optional channel widening.

    Used by: MultiScalePatchEmbed.shrink_cascades.
    (N, in_c, T, H, W) -> (N, out_c, T, H/2, W/2). Time dim untouched.
    SwiGLU activation matches the backbone MLP. No LayerNorm — per-leaf LN
    would destroy cross-channel physical-magnitude relationships. Reuse the
    tubelet projection's CUDA layout handling; a time kernel of one preserves T.
    """

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = Conv3dTubeletProjection(in_c, out_c, (1, 2, 2))
        self.gate = SwiGLU(dim_in=out_c, dim_out=out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.movedim(1, -1)  # (N, T, H, W, out_c) for channel-last SwiGLU
        x = self.gate(x)
        x = x.movedim(-1, 1)
        return x


class PatchEmbed(nn.Module):
    """Assemble fused Conv3D or per-frame spatial + temporal encoders.

    Used by: QuadTreeTransformer.x_embedder for uniform patches and regular grids.
    Owns the historical projection/mixer attributes directly to preserve state
    keys. Only Conv3D is registered in the public release; the factorized branch
    is the upstream construction hook for fixed-history extensions.
    """

    def __init__(
        self,
        patch_size: int | tuple[int, int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        temporal_mixer_type: str = "conv3d",
        temporal_mixer_dim: int = 256,
        temporal_mixer_heads: int = 8,
        temporal_mixer_time_frequencies: int = 8,
        eps: float = 1e-6,
        temporal_mixer_num_slots: int = 2,
        temporal_mixer_value_dim: int | None = None,
        temporal_mixer_routing: str = "content_time",
    ) -> None:
        super().__init__()
        patch_size = (
            (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )
        if len(patch_size) != 3:
            raise ValueError(f"patch_size must contain (T, H, W), got {patch_size}.")
        self.patch_size = tuple(int(size) for size in patch_size)
        self.temporal_mixer_type = temporal_mixer_type
        self.capabilities = encoder_capabilities(temporal_mixer_type)
        if not self.capabilities.factorized:
            self.proj = Conv3dTubeletProjection(in_chans, embed_dim, self.patch_size)
        else:
            patch_t, patch_h, patch_w = self.patch_size
            self.spatial_proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=(patch_h, patch_w),
                stride=(patch_h, patch_w),
            )
            self.temporal_mixer = build_temporal_mixer(
                mixer_type=temporal_mixer_type,
                embed_dim=embed_dim,
                num_frames=patch_t,
                temporal_dim=temporal_mixer_dim,
                num_heads=temporal_mixer_heads,
                num_time_frequencies=temporal_mixer_time_frequencies,
                eps=eps,
                num_slots=temporal_mixer_num_slots,
                value_dim=temporal_mixer_value_dim,
                routing=temporal_mixer_routing,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.capabilities.factorized:
            hidden_states = self.proj(hidden_states)  # -> (B, D, T/p_t, H/p_h, W/p_w)
            return hidden_states.flatten(2).transpose(1, 2)  # BCTHW -> BND

        if hidden_states.dim() != 5:
            raise ValueError(
                f"hidden_states must have shape (B, C, T, H, W), got {hidden_states.shape}."
            )
        batch, channels, frames, height, width = hidden_states.shape
        expected_frames = self.patch_size[0]
        if frames != expected_frames:
            raise ValueError(f"Expected T={expected_frames} input frames, got T={frames}.")
        per_frame = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        per_frame = self.spatial_proj(per_frame)  # (B*T, D, H_p, W_p)
        _, embed_dim, height_p, width_p = per_frame.shape
        frame_tokens = (
            per_frame.reshape(batch, frames, embed_dim, height_p, width_p)
            .permute(0, 3, 4, 1, 2)
            .reshape(batch, height_p * width_p, frames, embed_dim)
        )
        # The public pipeline supplies fixed histories, not temporal metadata.
        return self.temporal_mixer(
            frame_tokens, dt_in=None, observation_mask=None
        )  # (B, H_p*W_p, D)


class MultiScalePatchEmbed(nn.Module):
    """Native patch assembly with per-level spatial shrink cascades.

    Level ``l`` uses ``max_level_idx-l`` shrink blocks to reduce its patch to
    ``(Ph, Pw)`` while preserving time and widening channels. Conv3D consumes
    the complete tubelet. Factorized extensions use per-frame Conv2D projections
    and one shared temporal mixer, as upstream does. Tokens are scattered into
    canonical leaf order. Shared-finest Design B is not enabled.
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
        temporal_mixer_type: str = "conv3d",
        temporal_mixer_dim: int = 256,
        temporal_mixer_heads: int = 8,
        temporal_mixer_time_frequencies: int = 8,
        eps: float = 1e-6,
        temporal_mixer_num_slots: int = 2,
        temporal_mixer_value_dim: int | None = None,
        temporal_mixer_routing: str = "content_time",
    ) -> None:
        super().__init__()
        self.max_level_idx = max_level_idx
        self.embed_dim = embed_dim
        self.patch_size_t = patch_size_t
        self.base_patch_h = base_patch_h
        self.base_patch_w = base_patch_w
        self.channel_mult = channel_mult
        self.temporal_mixer_type = temporal_mixer_type
        self.capabilities = encoder_capabilities(temporal_mixer_type)

        self.shrink_cascades = nn.ModuleList()
        for lvl in range(max_level_idx + 1):
            depth = max_level_idx - lvl
            stages = []
            for k in range(depth):
                in_c = in_chans * (channel_mult**k)
                out_c = in_chans * (channel_mult ** (k + 1))
                stages.append(ShrinkBlock(in_c, out_c))
            self.shrink_cascades.append(nn.Sequential(*stages))

        if self.capabilities.factorized:
            self.temporal_mixer = build_temporal_mixer(
                mixer_type=temporal_mixer_type,
                embed_dim=embed_dim,
                num_frames=patch_size_t,
                temporal_dim=temporal_mixer_dim,
                num_heads=temporal_mixer_heads,
                num_time_frequencies=temporal_mixer_time_frequencies,
                eps=eps,
                num_slots=temporal_mixer_num_slots,
                value_dim=temporal_mixer_value_dim,
                routing=temporal_mixer_routing,
            )
            self.spatial_projs = nn.ModuleList()
            for lvl in range(max_level_idx + 1):
                depth = max_level_idx - lvl
                end_c = in_chans * (channel_mult**depth)
                self.spatial_projs.append(
                    nn.Conv2d(
                        end_c,
                        embed_dim,
                        kernel_size=(base_patch_h, base_patch_w),
                        stride=(base_patch_h, base_patch_w),
                    )
                )
        else:
            kernel = (patch_size_t, base_patch_h, base_patch_w)
            self.projs = nn.ModuleList()
            for lvl in range(max_level_idx + 1):
                depth = max_level_idx - lvl
                end_c = in_chans * (channel_mult**depth)
                self.projs.append(Conv3dTubeletProjection(end_c, embed_dim, kernel))

    def _attention_project(self, x: torch.Tensor, projection: nn.Conv2d) -> torch.Tensor:
        """Upstream per-frame projection hook, restricted to fixed histories."""
        leaves, channels, frames, height, width = x.shape
        if frames != self.patch_size_t:
            raise ValueError(f"Expected T={self.patch_size_t} input frames, got T={frames}.")
        per_frame = x.permute(0, 2, 1, 3, 4).reshape(leaves * frames, channels, height, width)
        per_frame = projection(per_frame)  # (N_l*T, D, 1, 1)
        if per_frame.shape[-2:] != (1, 1):
            raise ValueError(
                "Multi-scale spatial projection must produce one token per frame, "
                f"got shape {tuple(per_frame.shape)}."
            )
        frame_tokens = per_frame.reshape(leaves, frames, self.embed_dim)
        return self.temporal_mixer(
            frame_tokens, dt_in=None, observation_mask=None
        )  # (N_l, D)

    def forward(
        self,
        values_by_level: dict[int, torch.Tensor],  # {lvl: (N_l, C, T, Ph*s, Pw*s)}
        leaf_to_bucket: torch.Tensor,  # (N, 2) long — (level, position_within_bucket)
        bucket_layout: MultiScaleBucketLayout | None = None,
    ) -> torch.Tensor:
        """Return ``(1, N, D)`` tokens in canonical leaf order.

        Each level has spatial scale ``s = 2**(max_level_idx - level)``.
        """
        bucket_sizes = [
            values_by_level[level].shape[0] if level in values_by_level else 0
            for level in range(self.max_level_idx + 1)
        ]
        if bucket_layout is None:
            bucket_layout = MultiScaleBucketLayout.from_mapping(leaf_to_bucket, bucket_sizes)
        bucket_layout.validate(leaf_to_bucket, bucket_sizes)
        N = int(leaf_to_bucket.shape[0])
        device = leaf_to_bucket.device
        ref_dtype = None
        for lvl in values_by_level:
            if values_by_level[lvl].shape[0] > 0:
                ref_dtype = values_by_level[lvl].dtype
                break
        if ref_dtype is None:
            projection = (
                self.projs[self.max_level_idx]
                if not self.capabilities.factorized
                else self.spatial_projs[self.max_level_idx]
            )
            ref_dtype = next(projection.parameters()).dtype

        tokens = torch.zeros(N, self.embed_dim, device=device, dtype=ref_dtype)  # (N, D)
        for lvl, bucket in values_by_level.items():
            if bucket.shape[0] == 0:
                continue
            leaf_indices = bucket_layout.indices[lvl]
            x = self.shrink_cascades[lvl](bucket)  # (N_l, C*m^d, T, Ph, Pw)
            if not self.capabilities.factorized:
                out = self.projs[lvl](x).flatten(1)  # (N_l, D)
            else:
                out = self._attention_project(x, self.spatial_projs[lvl])
            # Scatter bucket tokens into the flat canonical sequence.
            tokens.index_copy_(0, leaf_indices, out.to(tokens.dtype))
        return tokens.unsqueeze(0)  # (1, N, D)
