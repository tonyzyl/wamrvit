"""Single selection point for encoder designs and their input capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn


@dataclass(frozen=True)
class EncoderCapabilities:
    factorized: bool
    uses_observed_time: bool
    variable_history: bool = False
    observation_mask: bool = False
    latest_bypass: bool = False


@dataclass(frozen=True)
class _EncoderDesign:
    capabilities: EncoderCapabilities
    temporal_mixer: str | None = None  # Dotted constructor path; resolved only when building.
    mixer_options: tuple[str, ...] = ()


# Fused Conv3D is deliberately not forced through a per-frame spatial encoder.
# Keep upstream's registry contract, but ship only the retained paper design.
_ENCODER_DESIGNS = {
    "conv3d": _EncoderDesign(EncoderCapabilities(factorized=False, uses_observed_time=False)),
}


def _encoder_design(mixer_type: str) -> _EncoderDesign:
    try:
        return _ENCODER_DESIGNS[mixer_type]
    except KeyError:
        raise ValueError(
            f"Unknown temporal_mixer_type {mixer_type!r}; "
            f"expected one of {tuple(_ENCODER_DESIGNS)}."
        ) from None


def encoder_capabilities(mixer_type: str) -> EncoderCapabilities:
    """Return the input contract shared by uniform/native adapters and callers."""
    return _encoder_design(mixer_type).capabilities


def build_temporal_mixer(
    mixer_type: str,
    embed_dim: int,
    num_frames: int,
    temporal_dim: int,
    num_heads: int,
    num_time_frequencies: int,
    eps: float,
    num_slots: int,
    value_dim: int | None,
    routing: str,
    latest_bypass: bool = False,
) -> nn.Module:
    """Build a local ``(..., T, D) -> (..., D)`` temporal mixer.

    Every mixer accepts ``dt_in=None, observation_mask=None``. Capabilities
    describe whether non-None inputs and variable history lengths are supported.
    Encoder wrappers retain their existing attribute names for checkpoint parity.
    """
    design = _encoder_design(mixer_type)
    if design.temporal_mixer is None:
        raise ValueError(f"{mixer_type!r} is a fused encoder, not a per-frame temporal mixer.")
    if latest_bypass and not design.capabilities.latest_bypass:
        raise ValueError(f"{mixer_type!r} does not support the opt-in latest bypass.")
    kwargs = dict(
        embed_dim=embed_dim,
        temporal_dim=temporal_dim,
        num_heads=num_heads,
        num_time_frequencies=num_time_frequencies,
        eps=eps,
    )
    optional = dict(
        num_frames=num_frames,
        num_slots=num_slots,
        value_dim=value_dim,
        routing=routing,
        latest_bypass=latest_bypass,
    )
    kwargs.update({name: optional[name] for name in design.mixer_options})
    module_name, class_name = design.temporal_mixer.rsplit(".", 1)
    mixer_cls = getattr(import_module(module_name), class_name)
    return mixer_cls(**kwargs)
