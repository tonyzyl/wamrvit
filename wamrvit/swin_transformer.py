"""SwinV2 baseline wrapper for regular-grid spatiotemporal forecasting.

Uses a timm SwinV2 backbone with a tublet Conv3d for temporal collapse and
a lightweight FPN decoder for spatial reconstruction. Compatible with the
existing diffusers save_pretrained/from_pretrained checkpointing.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


class FPNDecoder(nn.Module):
    """Simple Feature Pyramid Network decoder.

    Takes multi-scale feature maps from the backbone, projects each to a
    common channel dimension, then progressively upsamples and adds coarser
    features into finer ones.
    """

    def __init__(self, in_channels_list: list[int], out_channels: int):
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list]
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        # Project all stages to decoder dim
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway: progressively upsample coarser and add to finer
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        return laterals[0]  # finest resolution


class SwinV2Transformer(ModelMixin, ConfigMixin):
    """SwinV2 wrapper for spatiotemporal forecasting on regular grids.

    Architecture:
        Input (B, C, T, H, W)
        -> Tublet Conv3d collapses temporal dim -> (B, tublet_dim, H, W)
        -> SwinV2 backbone (features_only) -> multi-scale features
        -> FPN decoder -> (B, decoder_channels, H0, W0) at stage-0 resolution
        -> Output head reconstructs full spatial resolution -> (B, C_out, 1, H, W)

    Note: timm's SwinV2 creates buffers during __init__ that are incompatible
    with diffusers' meta-device initialization. Use low_cpu_mem_usage=False
    when calling from_pretrained().
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # timm's SwinV2 creates buffers in __init__ that are incompatible
        # with diffusers' meta-device (accelerate) initialization.
        kwargs["low_cpu_mem_usage"] = False
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    @register_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        return_seq_len: int = 1,
        patch_size: int | list[int] = (10, 5),
        patch_size_t: int = 2,
        # SwinV2 backbone config
        swin_embed_dim: int = 192,
        swin_depths: list[int] | tuple[int, ...] = (2, 2, 18),
        swin_num_heads: list[int] | tuple[int, ...] = (6, 12, 24),
        swin_window_size: int = 4,
        swin_pretrained: bool = False,
        img_size: list[int] | tuple[int, int] = (1120, 400),
        # Decoder
        decoder_channels: int = 256,
        # Kept for config compat with training script but unused
        adaptive: bool = False,
    ):
        super().__init__()

        out_channels = out_channels or in_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.return_seq_len = return_seq_len

        if isinstance(patch_size, int):
            patch_size = [patch_size, patch_size]
        self.patch_size = list(patch_size)
        self.patch_size_t = patch_size_t

        # Tublet embedding: collapse temporal dim, keep spatial dims
        tublet_dim = swin_embed_dim
        self.tublet_embed = nn.Conv3d(
            in_channels,
            tublet_dim,
            kernel_size=(patch_size_t, 1, 1),
            stride=(patch_size_t, 1, 1),
        )

        # SwinV2 backbone via timm (features_only for multi-scale outputs)
        self.backbone = timm.create_model(
            "swinv2_base_window8_256",
            pretrained=swin_pretrained,
            features_only=True,
            in_chans=tublet_dim,
            img_size=list(img_size),
            patch_size=list(patch_size),
            window_size=swin_window_size,
            embed_dim=swin_embed_dim,
            depths=list(swin_depths),
            num_heads=list(swin_num_heads),
        )

        # Determine backbone output channels from a dummy forward pass
        # timm's feature_info gives us the channel counts
        stage_channels = self.backbone.feature_info.channels()
        # Only keep stages matching the number of depths (swin_depths)
        num_stages = len(swin_depths)
        stage_channels = stage_channels[:num_stages]

        # FPN decoder
        self.decoder = FPNDecoder(stage_channels, decoder_channels)

        # Output head: upsample from stage-0 resolution back to full spatial
        # Stage 0 resolution = H / patch_size[0], W / patch_size[1] (swin's
        # first patch embedding divides by patch_size, then each stage halves).
        # For simplicity we use the same linear-fold pattern as QuadTreeTransformer:
        # project decoder_channels -> out_channels * patch_h * patch_w * return_seq_len
        # then fold back to spatial dims.
        p_h, p_w = self.patch_size
        self.output_proj = nn.Linear(decoder_channels, out_channels * p_h * p_w * return_seq_len)

    def forward(
        self,
        x_tokens: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_tokens: (B, C, T, H, W) regular grid input
        centers:  accepted but ignored (compatibility with training loop)
        Returns:  (B, C_out, return_seq_len, H, W)
        """
        B, C, T, H, W = x_tokens.shape

        # 1. Tublet embed: collapse temporal dim
        x = self.tublet_embed(x_tokens)  # (B, tublet_dim, T', H, W)
        x = x.squeeze(2)  # (B, tublet_dim, H, W) assuming T'=1

        # 2. SwinV2 backbone: multi-scale features
        features = self.backbone(x)
        # Only keep the stages we have decoder laterals for
        num_stages = len(self.decoder.lateral_convs)
        features = features[:num_stages]
        # timm SwinV2 outputs channels-last (B, H, W, C); convert to (B, C, H, W)
        features = [f.permute(0, 3, 1, 2) for f in features]

        # 3. FPN decoder -> finest stage resolution
        x = self.decoder(features)  # (B, decoder_channels, H0, W0)

        # 4. Output head: reshape to patches then fold back
        # x is (B, decoder_channels, H0, W0) where H0 = H/p_h, W0 = W/p_w
        # (accounting for swin patch embedding and possible stage downsampling)
        H0, W0 = x.shape[2], x.shape[3]
        p_h, p_w = self.patch_size

        # Permute to (B, H0, W0, decoder_channels) for the linear proj
        x = x.permute(0, 2, 3, 1)  # (B, H0, W0, decoder_channels)
        x = self.output_proj(x)  # (B, H0, W0, C_out * p_h * p_w * T_out)

        # Reshape: (B, H0, W0, C_out, T_out, p_h, p_w)
        x = x.reshape(B, H0, W0, self.out_channels, self.return_seq_len, p_h, p_w)

        # Permute to (B, C_out, T_out, H0, p_h, W0, p_w) and flatten spatial
        x = x.permute(0, 3, 4, 1, 5, 2, 6)  # (B, C_out, T_out, H0, p_h, W0, p_w)
        x = x.reshape(B, self.out_channels, self.return_seq_len, H0 * p_h, W0 * p_w)

        return x
