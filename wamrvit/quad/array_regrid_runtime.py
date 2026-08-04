from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numba import get_num_threads, set_num_threads

from wamrvit.quad.array_regrid_types import ArrayFallback, Channel, Tol


_COARSEN_PARALLEL_FILL_THRESHOLD = 64
_COARSEN_PARALLEL_ACCEPT_THRESHOLD = 64
_REFINE_PARALLEL_FILL_THRESHOLD = 64
_SOURCE_REFINE_PARALLEL_FILL_THRESHOLD = 16
_ACTIVE_PROTECTED_MASK_PARALLEL_THRESHOLD = 512
_WARM_ARRAY_REGRID_ACTIVE_CAPACITY = 8


def configure_array_regrid_num_threads(num_threads: Optional[int]) -> Optional[int]:
    """Set Numba's process-wide thread mask for array regrid kernels."""
    if num_threads is None:
        return None
    parsed = int(num_threads)
    if parsed <= 0:
        raise ValueError(
            f"array regrid num_threads must be positive, got {num_threads!r}."
        )
    set_num_threads(parsed)
    return parsed


def _array_regrid_config_status(capacity: int) -> dict[str, Any]:
    return {
        "capacity": int(capacity),
        "numba_threads": int(get_num_threads()),
        "thresholds": {
            "coarsen_fill": int(_COARSEN_PARALLEL_FILL_THRESHOLD),
            "coarsen_accept": int(_COARSEN_PARALLEL_ACCEPT_THRESHOLD),
            "refine_fill": int(_REFINE_PARALLEL_FILL_THRESHOLD),
            "source_refine_fill": int(_SOURCE_REFINE_PARALLEL_FILL_THRESHOLD),
            "protected_mask": int(_ACTIVE_PROTECTED_MASK_PARALLEL_THRESHOLD),
        },
    }


def _format_status_scalar(value: Any) -> str:
    if isinstance(value, (np.floating, float)):
        return f"{float(value):g}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)


def _format_status_sequence(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    return ",".join(_format_status_scalar(item) for item in values)


def _array_regrid_param_status(
    *,
    tol_frac: Tol,
    channel: Channel,
    max_passes: int,
    coarsen_ratio: float,
    adapt_nearby: int,
) -> dict[str, Any]:
    return {
        "tol_frac": _format_status_sequence(tol_frac),
        "channel": _format_status_sequence(channel),
        "max_passes": int(max_passes),
        "coarsen_ratio": float(coarsen_ratio),
        "adapt_nearby": int(adapt_nearby),
    }


def _normalize_channels(channel: Channel, channels: int) -> list[int]:
    if channel is None:
        selected = list(range(channels))
    elif isinstance(channel, (int, np.integer)):
        selected = [channel]
    else:
        selected = list(channel)
    return [int(ch) for ch in selected if 0 <= int(ch) < channels]


def _normalize_array_regrid_mode(array_regrid_mode: str) -> str:
    mode = str(array_regrid_mode)
    if mode == "approximate":
        raise ValueError(
            "array_regrid_mode='approximate' was removed; use the production "
            "array_regrid_mode='parity'."
        )
    if mode != "parity":
        raise ValueError(f"Unknown array_regrid_mode={mode!r}; expected 'parity'.")
    return mode


def _normalize_tolerances(tol_frac: Tol, num_channels: int) -> np.ndarray:
    if isinstance(tol_frac, (float, int)):
        return np.full(num_channels, float(tol_frac), dtype=np.float64)
    out = np.asarray(list(tol_frac), dtype=np.float64)
    if out.shape[0] != num_channels:
        raise ValueError(
            "The length of tol_frac must match the number of specified channels."
        )
    return out


def _ensure_capacity(num_leaves: int, capacity: int) -> None:
    if int(num_leaves) > int(capacity):
        raise ArrayFallback("capacity_exceeded")
