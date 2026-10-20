"""Upstream-compatible encoder capabilities and lazily imported patch adapters."""

from importlib import import_module
from typing import TYPE_CHECKING

from wamrvit.encoder.encoder_builder import (
    EncoderCapabilities,
    build_temporal_mixer,
    encoder_capabilities,
)

if TYPE_CHECKING:
    from wamrvit.encoder.conv3d import Conv3dTubeletProjection
    from wamrvit.encoder.patch_embed import MultiScalePatchEmbed, PatchEmbed, ShrinkBlock

_LAZY_EXPORTS = {
    "Conv3dTubeletProjection": ".conv3d",
    "MultiScalePatchEmbed": ".patch_embed",
    "PatchEmbed": ".patch_embed",
    "ShrinkBlock": ".patch_embed",
}

__all__ = [
    "Conv3dTubeletProjection",
    "EncoderCapabilities",
    "MultiScalePatchEmbed",
    "PatchEmbed",
    "ShrinkBlock",
    "build_temporal_mixer",
    "encoder_capabilities",
]


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
