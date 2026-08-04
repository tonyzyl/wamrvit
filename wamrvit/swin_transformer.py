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
        Input (B, C, T_in, H, W)
        -> composed temporal-spatial Conv3d -> (B, swin_embed_dim, 1, H0, W0)
        -> SwinV2 backbone stages (features_only) -> multi-scale features
        -> FPN decoder -> (B, decoder_channels, H0, W0) at stage-0 resolution
        -> Output head reconstructs -> (B, C_out, T_out, H, W)

    ``patch_size_t`` configures T_in and ``return_seq_len`` configures T_out;
    both are arbitrary positive construction-time sequence lengths.

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

        if patch_size_t < 1 or return_seq_len < 1:
            raise ValueError("patch_size_t (T_in) and return_seq_len (T_out) must be positive.")

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

        # The temporal tublet and spatial patch projections are adjacent linear
        # maps, so execute them as one composed Conv3d. During training the
        # composed tensors remain in the autograd graph; evaluation reuses a
        # non-persistent cache derived from checkpoint parameters.
        self.register_buffer("_composed_patch_weight", None, persistent=False)
        self.register_buffer("_composed_patch_bias", None, persistent=False)
        self._validate_composed_patch_embed()

    def _validate_composed_patch_embed(self) -> None:
        patch_proj = self.backbone.patch_embed.proj
        if not isinstance(patch_proj, nn.Conv2d):
            raise TypeError("Expected the Swin patch projection to be nn.Conv2d.")
        if self.tublet_embed.groups != 1 or patch_proj.groups != 1:
            raise ValueError("Composed Swin patch embedding requires ungrouped convolutions.")
        if self.tublet_embed.kernel_size != (self.patch_size_t, 1, 1):
            raise ValueError("Unexpected temporal tublet kernel for composed embedding.")
        if self.tublet_embed.stride != (self.patch_size_t, 1, 1):
            raise ValueError("Unexpected temporal tublet stride for composed embedding.")
        if patch_proj.kernel_size != tuple(self.patch_size):
            raise ValueError("Unexpected spatial patch kernel for composed embedding.")
        if patch_proj.stride != tuple(self.patch_size):
            raise ValueError("Unexpected spatial patch stride for composed embedding.")
        if patch_proj.in_channels != self.tublet_embed.out_channels:
            raise ValueError("Temporal and spatial embedding channels do not compose.")
        if self.tublet_embed.padding != (0, 0, 0) or patch_proj.padding != (0, 0):
            raise ValueError("Composed Swin patch embedding requires zero padding.")
        if self.tublet_embed.dilation != (1, 1, 1) or patch_proj.dilation != (1, 1):
            raise ValueError("Composed Swin patch embedding requires unit dilation.")

    def _clear_composed_patch_cache(self) -> None:
        self._composed_patch_weight = None
        self._composed_patch_bias = None

    def train(self, mode: bool = True):
        if mode:
            self._clear_composed_patch_cache()
        return super().train(mode)

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        self._clear_composed_patch_cache()
        return result

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self._clear_composed_patch_cache()
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._clear_composed_patch_cache()
        return result

    def _compose_patch_parameters(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compose checkpoint-compatible factors into one Conv3d kernel."""
        temporal_weight = self.tublet_embed.weight[:, :, :, 0, 0]
        temporal_bias = self.tublet_embed.bias
        patch_proj = self.backbone.patch_embed.proj
        patch_weight = patch_proj.weight
        patch_bias = patch_proj.bias

        # Do not let a surrounding BF16 autocast permanently round the factors
        # while constructing the reusable evaluation cache. Source parameter
        # dtypes are preserved, and training remains fully differentiable.
        with torch.autocast(device_type=temporal_weight.device.type, enabled=False):
            composed_weight = torch.einsum(
                "oeuv,ect->octuv", patch_weight, temporal_weight
            ).contiguous()
            if temporal_bias is None:
                composed_bias = None if patch_bias is None else patch_bias.clone()
            else:
                temporal_contribution = torch.einsum(
                    "oeuv,e->o", patch_weight, temporal_bias
                )
                composed_bias = (
                    temporal_contribution
                    if patch_bias is None
                    else patch_bias + temporal_contribution
                )
                composed_bias = composed_bias.contiguous()
        return composed_weight, composed_bias

    def _get_composed_patch_parameters(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.training:
            # Recompute every training forward: caching would retain an obsolete
            # autograd graph after the optimizer updates the factor parameters.
            return self._compose_patch_parameters()
        if self._composed_patch_weight is None:
            with torch.no_grad():
                weight, bias = self._compose_patch_parameters()
            self._composed_patch_weight = weight
            self._composed_patch_bias = bias
        return self._composed_patch_weight, self._composed_patch_bias

    def _forward_backbone_composed(self, x_tokens: torch.Tensor) -> list[torch.Tensor]:
        if x_tokens.ndim != 5:
            raise ValueError(
                "SwinV2 expects input shape (B,C,T_in,H,W); "
                f"got {tuple(x_tokens.shape)}."
            )
        if x_tokens.shape[2] != self.patch_size_t:
            raise ValueError(
                "SwinV2 was configured for "
                f"T_in={self.patch_size_t}, but received T_in={x_tokens.shape[2]}. "
                "Set patch_size_t to the desired input sequence length."
            )

        patch_embed = self.backbone.patch_embed
        height, width = x_tokens.shape[-2:]
        if patch_embed.dynamic_img_pad:
            raise ValueError("Composed Swin patch embedding does not support dynamic padding.")
        if patch_embed.img_size is not None and patch_embed.strict_img_size:
            if (height, width) != tuple(patch_embed.img_size):
                raise ValueError(
                    f"Expected input size {tuple(patch_embed.img_size)}, got {(height, width)}."
                )
        p_h, p_w = self.patch_size
        if height % p_h != 0 or width % p_w != 0:
            raise ValueError(
                f"Input size {(height, width)} is not divisible by patch size {(p_h, p_w)}."
            )

        weight, bias = self._get_composed_patch_parameters()
        x = F.conv3d(
            x_tokens,
            weight,
            bias,
            stride=(self.patch_size_t, p_h, p_w),
        )
        if x.shape[2] != 1:
            raise RuntimeError(f"Expected one temporal output, got shape {tuple(x.shape)}.")

        # timm Swin stages consume channels-last tensors after PatchEmbed norm.
        x = x.squeeze(2).permute(0, 2, 3, 1)
        x = patch_embed.norm(x)
        features = []
        for name, module in self.backbone.items():
            if name == "patch_embed":
                continue
            x = module(x)
            if name in self.backbone.return_layers:
                features.append(x)
        return features

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

        # 1--2. Collapse the configured input sequence and spatially patchify
        # with one composed Conv3d. T_in and T_out are independent constructor
        # parameters; neither is hard-coded to the paper's 2-to-1 setting.
        features = self._forward_backbone_composed(x_tokens)
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
