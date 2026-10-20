"""Legacy fused spatial-temporal tubelet projection."""

from __future__ import annotations

import torch
import torch.nn as nn


class Conv3dTubeletProjection(nn.Conv3d):
    """Independent spatial kernels for every frame in a full tubelet.

    ``(B, C, T, H, W) -> (B, D, T/p_t, H/p_h, W/p_w)``.
    Uniform and native adapters determine how the resulting tokens are assembled.
    This is exactly nn.Conv3d with kernel=stride=patch_size, with no additional
    normalization or temporal factorization. Subclassing rather than wrapping
    preserves the historical ``proj.weight``/``proj.bias`` checkpoint layout,
    initialization, and convolution parameter classification by optimizers.
    CUDA activations use channels-last-3d to retain fast cuDNN dispatch (notably
    for BF16 on PyTorch 2.9/cuDNN 9.10). Parameter objects and layouts are untouched;
    cuDNN handles its own kernel-layout conversion, including after checkpoint load.
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        patch_size: tuple[int, int, int],
    ) -> None:
        super().__init__(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.is_cuda and input.ndim in (4, 5):
            # Also preserve nn.Conv3d's unbatched (C,T,H,W) input contract.
            unbatched = input.ndim == 4
            if unbatched:
                input = input.unsqueeze(0)
            # Unlike contiguous(), to(memory_format=...) assigns unambiguous
            # channels-last strides even when C == 1. Already-correct layouts
            # are a no-op. Do not mutate or cache copies of trainable weights.
            input = input.to(memory_format=torch.channels_last_3d)
            output = super().forward(input)
            return output.squeeze(0) if unbatched else output
        return super().forward(input)
