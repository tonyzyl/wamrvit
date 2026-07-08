from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

from wamrvit.quad.adapt_wavelet import regrid
from wamrvit.quad.quad_utils import quadtree_to_tensor, tensor_to_quadtree
from wamrvit.quad.quadtree import _ceil_log2
from wamrvit.quad.quadtree_kernels import morton2D_jit, resize_patch_bilinear_jit


Tol = Union[float, Sequence[float], np.ndarray]
Channel = Optional[Union[int, Sequence[int]]]
FineBounds = Tuple[np.ndarray, np.ndarray, np.ndarray]
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
        raise ValueError(f"array regrid num_threads must be positive, got {num_threads!r}.")
    set_num_threads(parsed)
    return parsed


@dataclass
class FlatTopology:
    values: np.ndarray
    tile_ix: np.ndarray
    tile_iy: np.ndarray
    level_idx: np.ndarray
    x_idx: np.ndarray
    y_idx: np.ndarray
    domain: Dict[str, Any]
    cell_scale_mode: Optional[str]


@dataclass
class ActiveTopology:
    values: np.ndarray
    tile_ix: np.ndarray
    tile_iy: np.ndarray
    level_idx: np.ndarray
    x_idx: np.ndarray
    y_idx: np.ndarray
    active_slots: np.ndarray
    active_count: int
    next_slot: int
    domain: Dict[str, Any]
    cell_scale_mode: Optional[str]


@dataclass
class SequenceSourceActiveTopology(ActiveTopology):
    source_sequence: np.ndarray
    initial_slots: int
    sequence_channels: int
    sequence_timesteps: int
    source_sequence_copied: bool


class _ArrayFallback(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _array_regrid_config_status(capacity: int) -> Dict[str, int]:
    return {
        "capacity": int(capacity),
        "numba_threads": int(get_num_threads()),
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
) -> Dict[str, Any]:
    return {
        "tol_frac": _format_status_sequence(tol_frac),
        "channel": _format_status_sequence(channel),
        "max_passes": int(max_passes),
        "coarsen_ratio": float(coarsen_ratio),
        "adapt_nearby": int(adapt_nearby),
    }


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _fine_size(domain: Dict[str, Any]) -> int:
    return 1 << int(domain["max_level_idx"])


def _domain_grid_shape(domain: Dict[str, Any]) -> Tuple[int, int]:
    nx_tiles = int(round((float(domain["xmax"]) - float(domain["xmin"])) / float(domain["tile_width"])))
    ny_tiles = int(round((float(domain["ymax"]) - float(domain["ymin"])) / float(domain["tile_height"])))
    scale = _fine_size(domain)
    return nx_tiles * scale, ny_tiles * scale


def _import_topology_with_values(
    values: np.ndarray,
    meta: Dict[str, Any],
    cell_scale_mode: Optional[str],
) -> FlatTopology:
    centers = _as_numpy(meta["centers"])
    levels = _as_numpy(meta["levels"]).astype(np.int16, copy=False)
    domain = dict(meta["domain"])

    if "tiles" in meta and "xy_idx" in meta:
        tiles = _as_numpy(meta["tiles"]).astype(np.int64, copy=False)
        xy_idx = _as_numpy(meta["xy_idx"]).astype(np.int64, copy=False)
        tile_ix = tiles[:, 0].astype(np.int32, copy=True)
        tile_iy = tiles[:, 1].astype(np.int32, copy=True)
        x_idx = xy_idx[:, 1].astype(np.int32, copy=True)
        y_idx = xy_idx[:, 2].astype(np.int32, copy=True)
        return FlatTopology(values, tile_ix, tile_iy, levels.astype(np.int16, copy=True), x_idx, y_idx, domain, cell_scale_mode)

    xmin = float(domain["xmin"])
    xmax = float(domain["xmax"])
    ymin = float(domain["ymin"])
    ymax = float(domain["ymax"])
    tile_width = float(domain["tile_width"])
    tile_height = float(domain["tile_height"])
    lx = xmax - xmin
    ly = ymax - ymin

    if cell_scale_mode is not None:
        abs_cx = centers[:, 0].astype(np.float64) * lx + xmin
        abs_cy = centers[:, 1].astype(np.float64) * ly + ymin
    else:
        abs_cx = centers[:, 0].astype(np.float64)
        abs_cy = centers[:, 1].astype(np.float64)

    widths = tile_width / np.left_shift(1, levels.astype(np.int64))
    heights = tile_height / np.left_shift(1, levels.astype(np.int64))
    x_left = abs_cx - 0.5 * widths
    y_bottom = abs_cy - 0.5 * heights

    tile_ix = np.floor((x_left - xmin) / tile_width + 1.0e-9).astype(np.int32)
    tile_iy = np.floor((y_bottom - ymin) / tile_height + 1.0e-9).astype(np.int32)

    tile_x0 = xmin + tile_ix.astype(np.float64) * tile_width
    tile_y0 = ymin + tile_iy.astype(np.float64) * tile_height
    x_idx = np.rint((x_left - tile_x0) / widths).astype(np.int32)
    y_idx = np.rint((y_bottom - tile_y0) / heights).astype(np.int32)

    return FlatTopology(
        values=values,
        tile_ix=tile_ix,
        tile_iy=tile_iy,
        level_idx=levels.astype(np.int16, copy=True),
        x_idx=x_idx,
        y_idx=y_idx,
        domain=domain,
        cell_scale_mode=cell_scale_mode,
    )


def _import_topology(
    data: np.ndarray,
    meta: Dict[str, Any],
    cell_scale_mode: Optional[str],
) -> FlatTopology:
    values = np.ascontiguousarray(_as_numpy(data), dtype=np.float32)
    return _import_topology_with_values(values, meta, cell_scale_mode)


def _import_topology_metadata(
    num_leaves: int,
    meta: Dict[str, Any],
    cell_scale_mode: Optional[str],
) -> FlatTopology:
    placeholder = np.empty((int(num_leaves), 0, 0, 0), dtype=np.float32)
    return _import_topology_with_values(placeholder, meta, cell_scale_mode)


def _leaf_fine_bounds(top: FlatTopology) -> FineBounds:
    max_level = int(top.domain["max_level_idx"])
    scale = 1 << (max_level - top.level_idx.astype(np.int64))
    x0 = (top.tile_ix.astype(np.int64) << max_level) + top.x_idx.astype(np.int64) * scale
    y0 = (top.tile_iy.astype(np.int64) << max_level) + top.y_idx.astype(np.int64) * scale
    return x0, y0, scale


@njit(boundscheck=False, cache=True)
def _build_owner_grid_jit(nx: int, ny: int, x0: np.ndarray, y0: np.ndarray, scale: np.ndarray):
    owner = np.full((ny, nx), -1, dtype=np.int32)
    for i in range(x0.shape[0]):
        xi = int(x0[i])
        yi = int(y0[i])
        si = int(scale[i])
        if xi < 0 or yi < 0 or xi + si > nx or yi + si > ny:
            return owner, 1
        for yy in range(yi, yi + si):
            for xx in range(xi, xi + si):
                if owner[yy, xx] != -1:
                    return owner, 2
                owner[yy, xx] = i
    for yy in range(ny):
        for xx in range(nx):
            if owner[yy, xx] < 0:
                return owner, 3
    return owner, 0


@njit(boundscheck=False)
def _append_unique_neighbor(tmp: np.ndarray, count: int, candidate: int, leaf_idx: int) -> int:
    if candidate < 0 or candidate == leaf_idx:
        return count
    for i in range(count):
        if tmp[i] == candidate:
            return count
    tmp[count] = candidate
    return count + 1


@njit(boundscheck=False)
def _face_neighbors_for_leaf_jit(
    owner: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    scale: np.ndarray,
    leaf_idx: int,
) -> np.ndarray:
    xi = int(x0[leaf_idx])
    yi = int(y0[leaf_idx])
    si = int(scale[leaf_idx])
    ny, nx = owner.shape
    tmp = np.empty(max(1, 4 * si), dtype=np.int32)
    count = 0

    if xi > 0:
        xx = xi - 1
        for yy in range(yi, yi + si):
            count = _append_unique_neighbor(tmp, count, int(owner[yy, xx]), leaf_idx)
    if xi + si < nx:
        xx = xi + si
        for yy in range(yi, yi + si):
            count = _append_unique_neighbor(tmp, count, int(owner[yy, xx]), leaf_idx)
    if yi > 0:
        yy = yi - 1
        for xx in range(xi, xi + si):
            count = _append_unique_neighbor(tmp, count, int(owner[yy, xx]), leaf_idx)
    if yi + si < ny:
        yy = yi + si
        for xx in range(xi, xi + si):
            count = _append_unique_neighbor(tmp, count, int(owner[yy, xx]), leaf_idx)

    return tmp[:count].copy()


def _build_owner_grid(top: FlatTopology) -> np.ndarray:
    nx, ny = _domain_grid_shape(top.domain)
    x0, y0, scale = _leaf_fine_bounds(top)
    owner, err = _build_owner_grid_jit(nx, ny, x0, y0, scale)
    if err == 1:
        raise ValueError("leaf_outside_domain")
    if err == 2:
        raise ValueError("overlapping_leaves")
    if err == 3:
        raise ValueError("incomplete_domain_coverage")
    return owner


@njit(boundscheck=False, cache=True)
def _build_active_owner_grid_jit(
    nx: int,
    ny: int,
    active_slots: np.ndarray,
    active_count: int,
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    level_idx: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    max_level: int,
):
    owner = np.full((ny, nx), -1, dtype=np.int32)
    for pos in range(active_count):
        slot = int(active_slots[pos])
        scale = 1 << (max_level - int(level_idx[slot]))
        xi = (int(tile_ix[slot]) << max_level) + int(x_idx[slot]) * scale
        yi = (int(tile_iy[slot]) << max_level) + int(y_idx[slot]) * scale
        if xi < 0 or yi < 0 or xi + scale > nx or yi + scale > ny:
            return owner, 1
        for yy in range(yi, yi + scale):
            for xx in range(xi, xi + scale):
                if owner[yy, xx] != -1:
                    return owner, 2
                owner[yy, xx] = slot
    for yy in range(ny):
        for xx in range(nx):
            if owner[yy, xx] < 0:
                return owner, 3
    return owner, 0


def _build_active_owner_grid(active: ActiveTopology) -> np.ndarray:
    nx, ny = _domain_grid_shape(active.domain)
    owner, err = _build_active_owner_grid_jit(
        nx,
        ny,
        active.active_slots,
        int(active.active_count),
        active.tile_ix,
        active.tile_iy,
        active.level_idx,
        active.x_idx,
        active.y_idx,
        int(active.domain["max_level_idx"]),
    )
    if err == 1:
        raise ValueError("leaf_outside_domain")
    if err == 2:
        raise ValueError("overlapping_leaves")
    if err == 3:
        raise ValueError("incomplete_domain_coverage")
    return owner


def _face_neighbors_for_leaf(
    top: FlatTopology,
    owner: np.ndarray,
    leaf_idx: int,
    bounds: Optional[FineBounds] = None,
) -> np.ndarray:
    x0, y0, scale = bounds if bounds is not None else _leaf_fine_bounds(top)
    return _face_neighbors_for_leaf_jit(owner, x0, y0, scale, int(leaf_idx))


@njit(boundscheck=False)
def _infer_max_level_from_bounds_jit(level_idx: np.ndarray, scale: np.ndarray) -> int:
    max_level = 0
    for i in range(level_idx.shape[0]):
        level = int(level_idx[i])
        size = int(scale[i])
        while size > 1:
            level += 1
            size //= 2
        if level > max_level:
            max_level = level
    return max_level


@njit(boundscheck=False, inline="always")
def _balance_level_can_need_refinement_jit(leaf_level: int, max_level: int) -> bool:
    return leaf_level < max_level - 1


@njit(boundscheck=False)
def _find_balance_refinement_mask_jit(
    level_idx: np.ndarray,
    owner: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    scale: np.ndarray,
    max_level: int = -1,
) -> np.ndarray:
    if max_level < 0:
        max_level = _infer_max_level_from_bounds_jit(level_idx, scale)

    mask = np.zeros(level_idx.shape[0], dtype=np.bool_)
    ny, nx = owner.shape
    for leaf_idx in range(level_idx.shape[0]):
        leaf_level = int(level_idx[leaf_idx])
        if not _balance_level_can_need_refinement_jit(leaf_level, max_level):
            continue

        xi = int(x0[leaf_idx])
        yi = int(y0[leaf_idx])
        si = int(scale[leaf_idx])
        too_fine = leaf_level + 1

        if xi > 0:
            xx = xi - 1
            for yy in range(yi, yi + si):
                neighbor = int(owner[yy, xx])
                if neighbor >= 0 and int(level_idx[neighbor]) > too_fine:
                    mask[leaf_idx] = True
                    break
        if mask[leaf_idx]:
            continue
        if xi + si < nx:
            xx = xi + si
            for yy in range(yi, yi + si):
                neighbor = int(owner[yy, xx])
                if neighbor >= 0 and int(level_idx[neighbor]) > too_fine:
                    mask[leaf_idx] = True
                    break
        if mask[leaf_idx]:
            continue
        if yi > 0:
            yy = yi - 1
            for xx in range(xi, xi + si):
                neighbor = int(owner[yy, xx])
                if neighbor >= 0 and int(level_idx[neighbor]) > too_fine:
                    mask[leaf_idx] = True
                    break
        if mask[leaf_idx]:
            continue
        if yi + si < ny:
            yy = yi + si
            for xx in range(xi, xi + si):
                neighbor = int(owner[yy, xx])
                if neighbor >= 0 and int(level_idx[neighbor]) > too_fine:
                    mask[leaf_idx] = True
                    break

    return mask


def _side_neighbor_indices(
    owner: np.ndarray,
    x0: int,
    y0: int,
    size: int,
    exclude: set[int],
) -> np.ndarray:
    ny, nx = owner.shape
    parts = []
    if x0 > 0:
        parts.append(owner[y0:y0 + size, x0 - 1])
    if x0 + size < nx:
        parts.append(owner[y0:y0 + size, x0 + size])
    if y0 > 0:
        parts.append(owner[y0 - 1, x0:x0 + size])
    if y0 + size < ny:
        parts.append(owner[y0 + size, x0:x0 + size])
    if not parts:
        return np.empty(0, dtype=np.int32)
    neighbors = np.unique(np.concatenate([p.ravel() for p in parts]))
    return np.array([int(n) for n in neighbors if int(n) >= 0 and int(n) not in exclude], dtype=np.int32)


def _find_balance_refinements(top: FlatTopology, owner: np.ndarray) -> np.ndarray:
    max_level = int(top.domain["max_level_idx"])
    bounds = _leaf_fine_bounds(top)
    mask = _find_balance_refinement_mask_jit(top.level_idx, owner, bounds[0], bounds[1], bounds[2], max_level)
    return np.flatnonzero(mask).astype(np.int32, copy=False)


@njit(boundscheck=False)
def _active_side_has_too_fine_neighbor_jit(
    level_idx_arr: np.ndarray,
    owner: np.ndarray,
    tile_ix: int,
    tile_iy: int,
    level: int,
    x_idx: int,
    y_idx: int,
    max_level: int,
    too_fine_level: int,
    side: int,
) -> bool:
    scale = 1 << (max_level - level)
    x0 = (tile_ix << max_level) + x_idx * scale
    y0 = (tile_iy << max_level) + y_idx * scale

    if side == 0:
        xx = x0
        for yy in range(y0, y0 + scale):
            neighbor = int(owner[yy, xx])
            if neighbor >= 0 and int(level_idx_arr[neighbor]) > too_fine_level:
                return True
    elif side == 1:
        xx = x0 + scale - 1
        for yy in range(y0, y0 + scale):
            neighbor = int(owner[yy, xx])
            if neighbor >= 0 and int(level_idx_arr[neighbor]) > too_fine_level:
                return True
    elif side == 2:
        yy = y0
        for xx in range(x0, x0 + scale):
            neighbor = int(owner[yy, xx])
            if neighbor >= 0 and int(level_idx_arr[neighbor]) > too_fine_level:
                return True
    else:
        yy = y0 + scale - 1
        for xx in range(x0, x0 + scale):
            neighbor = int(owner[yy, xx])
            if neighbor >= 0 and int(level_idx_arr[neighbor]) > too_fine_level:
                return True
    return False


@njit(boundscheck=False)
def _active_object_style_face_has_too_fine_neighbor_jit(
    level_idx_arr: np.ndarray,
    owner: np.ndarray,
    tile_ix_arr: np.ndarray,
    tile_iy_arr: np.ndarray,
    x_idx_arr: np.ndarray,
    y_idx_arr: np.ndarray,
    max_level: int,
    slot: int,
    direction: int,
) -> bool:
    fine = 1 << max_level
    nx_tiles = owner.shape[1] // fine
    ny_tiles = owner.shape[0] // fine
    tile_ix = int(tile_ix_arr[slot])
    tile_iy = int(tile_iy_arr[slot])
    leaf_level = int(level_idx_arr[slot])
    too_fine_level = leaf_level + 1
    cur_level = leaf_level
    cur_x = int(x_idx_arr[slot])
    cur_y = int(y_idx_arr[slot])

    while cur_level > 0:
        if direction == 0:
            if cur_x & 1:
                return _active_side_has_too_fine_neighbor_jit(
                    level_idx_arr, owner, tile_ix, tile_iy, cur_level, cur_x - 1, cur_y, max_level, too_fine_level, 1
                )
        elif direction == 1:
            if (cur_x & 1) == 0:
                return _active_side_has_too_fine_neighbor_jit(
                    level_idx_arr, owner, tile_ix, tile_iy, cur_level, cur_x + 1, cur_y, max_level, too_fine_level, 0
                )
        elif direction == 2:
            if cur_y & 1:
                return _active_side_has_too_fine_neighbor_jit(
                    level_idx_arr, owner, tile_ix, tile_iy, cur_level, cur_x, cur_y - 1, max_level, too_fine_level, 3
                )
        else:
            if (cur_y & 1) == 0:
                return _active_side_has_too_fine_neighbor_jit(
                    level_idx_arr, owner, tile_ix, tile_iy, cur_level, cur_x, cur_y + 1, max_level, too_fine_level, 2
                )

        cur_x //= 2
        cur_y //= 2
        cur_level -= 1

    if direction == 0 and tile_ix > 0:
        return _active_side_has_too_fine_neighbor_jit(
            level_idx_arr, owner, tile_ix - 1, tile_iy, 0, 0, 0, max_level, too_fine_level, 1
        )
    if direction == 1 and tile_ix + 1 < nx_tiles:
        return _active_side_has_too_fine_neighbor_jit(
            level_idx_arr, owner, tile_ix + 1, tile_iy, 0, 0, 0, max_level, too_fine_level, 0
        )
    if direction == 2 and tile_iy > 0:
        return _active_side_has_too_fine_neighbor_jit(
            level_idx_arr, owner, tile_ix, tile_iy - 1, 0, 0, 0, max_level, too_fine_level, 3
        )
    if direction == 3 and tile_iy + 1 < ny_tiles:
        return _active_side_has_too_fine_neighbor_jit(
            level_idx_arr, owner, tile_ix, tile_iy + 1, 0, 0, 0, max_level, too_fine_level, 2
        )
    return False


@njit(boundscheck=False, cache=True)
def _find_active_balance_refinement_slots_jit(
    active_slots: np.ndarray,
    active_count: int,
    level_idx: np.ndarray,
    owner: np.ndarray,
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    max_level: int,
) -> np.ndarray:
    mask = np.zeros(level_idx.shape[0], dtype=np.bool_)
    for pos in range(active_count):
        slot = int(active_slots[pos])
        leaf_level = int(level_idx[slot])
        if not _balance_level_can_need_refinement_jit(leaf_level, max_level):
            continue
        for direction in range(4):
            if _active_object_style_face_has_too_fine_neighbor_jit(
                level_idx,
                owner,
                tile_ix,
                tile_iy,
                x_idx,
                y_idx,
                max_level,
                slot,
                direction,
            ):
                mask[slot] = True
                break
    return mask


def _find_active_balance_refinements(active: ActiveTopology, owner: np.ndarray) -> np.ndarray:
    mask = _find_active_balance_refinement_slots_jit(
        active.active_slots,
        int(active.active_count),
        active.level_idx,
        owner,
        active.tile_ix,
        active.tile_iy,
        active.x_idx,
        active.y_idx,
        int(active.domain["max_level_idx"]),
    )
    return np.flatnonzero(mask).astype(np.int32, copy=False)


def _normalize_channels(channel: Channel, channels: int) -> list[int]:
    if channel is None:
        selected = list(range(channels))
    elif isinstance(channel, int):
        selected = [channel]
    else:
        selected = list(channel)
    return [int(ch) for ch in selected if 0 <= int(ch) < channels]


def _normalize_array_regrid_mode(array_regrid_mode: str) -> str:
    mode = str(array_regrid_mode)
    if mode not in {"parity", "approximate"}:
        raise ValueError(
            f"Unknown array_regrid_mode={array_regrid_mode!r}; expected 'parity' or 'approximate'."
        )
    return mode


def _normalize_tolerances(tol_frac: Tol, num_channels: int) -> np.ndarray:
    if isinstance(tol_frac, (float, int)):
        return np.full(num_channels, float(tol_frac), dtype=np.float64)
    out = np.asarray(list(tol_frac), dtype=np.float64)
    if out.shape[0] != num_channels:
        raise ValueError("The length of tol_frac must match the number of specified channels.")
    return out


def _channel_ranges(values: np.ndarray, channels: list[int]) -> np.ndarray:
    ranges = np.zeros(len(channels), dtype=np.float64)
    for i, ch in enumerate(channels):
        patch = values[:, ch, :, :]
        ranges[i] = float(np.max(patch) - np.min(patch))
    return ranges


def _initial_channel_ranges(top: FlatTopology, channels: list[int]) -> np.ndarray:
    return _channel_ranges(top.values, channels)


def _initial_sequence_channel_ranges(sequence: np.ndarray, channels: list[int]) -> np.ndarray:
    if sequence.ndim != 5:
        raise ValueError(f"Expected sequence shape (N,C,T,H,W), got {sequence.shape}.")
    _, num_channels, timesteps, _, _ = sequence.shape
    total_channels = int(num_channels) * int(timesteps)
    ranges = np.zeros(len(channels), dtype=np.float64)
    for i, flat_channel in enumerate(channels):
        ch = int(flat_channel)
        if ch < 0 or ch >= total_channels:
            continue
        timestep = ch // int(num_channels)
        channel = ch - timestep * int(num_channels)
        patch = sequence[:, channel, timestep, :, :].astype(np.float32, copy=False)
        ranges[i] = float(np.max(patch) - np.min(patch))
    return ranges


@njit(boundscheck=False)
def _active_channel_ranges_jit(
    values: np.ndarray,
    active_slots: np.ndarray,
    active_count: int,
    channels: np.ndarray,
) -> np.ndarray:
    ranges = np.zeros(channels.shape[0], dtype=np.float64)
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    for channel_pos in range(channels.shape[0]):
        ch = int(channels[channel_pos])
        if active_count == 0:
            ranges[channel_pos] = 0.0
            continue
        first_slot = int(active_slots[0])
        v_min = values[first_slot, ch, 0, 0]
        v_max = v_min
        for pos in range(active_count):
            slot = int(active_slots[pos])
            for yy in range(patch_h):
                for xx in range(patch_w):
                    val = values[slot, ch, yy, xx]
                    if val < v_min:
                        v_min = val
                    elif val > v_max:
                        v_max = val
        ranges[channel_pos] = float(v_max - v_min)
    return ranges


def _active_channel_ranges(active: ActiveTopology, channels: list[int]) -> np.ndarray:
    return _active_channel_ranges_jit(
        active.values,
        active.active_slots,
        int(active.active_count),
        np.asarray(channels, dtype=np.int64),
    )


@njit(boundscheck=False)
def _compute_refine_mask_jit(
    values: np.ndarray,
    level_idx: np.ndarray,
    channels: np.ndarray,
    tolerances: np.ndarray,
    owner: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    scale: np.ndarray,
    max_level: int,
) -> np.ndarray:
    num_leaves = values.shape[0]
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    center_y = patch_h // 2
    center_x = patch_w // 2
    ny, nx = owner.shape
    mask = np.zeros(num_leaves, dtype=np.bool_)

    for tol_idx in range(channels.shape[0]):
        ch = int(channels[tol_idx])
        tol = float(tolerances[tol_idx])

        for leaf_idx in range(num_leaves):
            if int(level_idx[leaf_idx]) >= max_level:
                continue
            v_min = values[leaf_idx, ch, 0, 0]
            v_max = v_min
            for yy in range(patch_h):
                for xx in range(patch_w):
                    val = values[leaf_idx, ch, yy, xx]
                    if val < v_min:
                        v_min = val
                    elif val > v_max:
                        v_max = val
            if float(v_max - v_min) > tol:
                mask[leaf_idx] = True

        for leaf_idx in range(num_leaves):
            if mask[leaf_idx] or int(level_idx[leaf_idx]) >= max_level:
                continue
            xi = int(x0[leaf_idx])
            yi = int(y0[leaf_idx])
            si = int(scale[leaf_idx])
            center = values[leaf_idx, ch, center_y, center_x]

            if xi > 0:
                xx = xi - 1
                for yy in range(yi, yi + si):
                    neighbor = int(owner[yy, xx])
                    if neighbor >= 0 and neighbor != leaf_idx and abs(float(center - values[neighbor, ch, center_y, center_x])) > tol:
                        mask[leaf_idx] = True
                        break
            if mask[leaf_idx]:
                continue
            if xi + si < nx:
                xx = xi + si
                for yy in range(yi, yi + si):
                    neighbor = int(owner[yy, xx])
                    if neighbor >= 0 and neighbor != leaf_idx and abs(float(center - values[neighbor, ch, center_y, center_x])) > tol:
                        mask[leaf_idx] = True
                        break
            if mask[leaf_idx]:
                continue
            if yi > 0:
                yy = yi - 1
                for xx in range(xi, xi + si):
                    neighbor = int(owner[yy, xx])
                    if neighbor >= 0 and neighbor != leaf_idx and abs(float(center - values[neighbor, ch, center_y, center_x])) > tol:
                        mask[leaf_idx] = True
                        break
            if mask[leaf_idx]:
                continue
            if yi + si < ny:
                yy = yi + si
                for xx in range(xi, xi + si):
                    neighbor = int(owner[yy, xx])
                    if neighbor >= 0 and neighbor != leaf_idx and abs(float(center - values[neighbor, ch, center_y, center_x])) > tol:
                        mask[leaf_idx] = True
                        break

    return mask


def _compute_refine_mask(
    top: FlatTopology,
    channels: list[int],
    tolerances: np.ndarray,
    owner: np.ndarray,
) -> np.ndarray:
    max_level = int(top.domain["max_level_idx"])
    if not channels:
        return np.zeros(top.values.shape[0], dtype=bool)
    x0, y0, scale = _leaf_fine_bounds(top)
    return _compute_refine_mask_jit(
        top.values,
        top.level_idx,
        np.asarray(channels, dtype=np.int64),
        tolerances.astype(np.float64, copy=False),
        owner,
        x0,
        y0,
        scale,
        max_level,
    )


@njit(boundscheck=False, cache=True)
def _compute_active_refine_slots_jit(
    values: np.ndarray,
    active_slots: np.ndarray,
    active_count: int,
    level_idx: np.ndarray,
    channels: np.ndarray,
    tolerances: np.ndarray,
    owner: np.ndarray,
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    max_level: int,
) -> np.ndarray:
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    center_y = patch_h // 2
    center_x = patch_w // 2
    ny, nx = owner.shape
    mask = np.zeros(values.shape[0], dtype=np.bool_)

    for tol_idx in range(channels.shape[0]):
        ch = int(channels[tol_idx])
        tol = float(tolerances[tol_idx])

        for pos in range(active_count):
            slot = int(active_slots[pos])
            if int(level_idx[slot]) >= max_level:
                continue
            v_min = values[slot, ch, 0, 0]
            v_max = v_min
            for yy in range(patch_h):
                for xx in range(patch_w):
                    val = values[slot, ch, yy, xx]
                    if val < v_min:
                        v_min = val
                    elif val > v_max:
                        v_max = val
            if float(v_max - v_min) > tol:
                mask[slot] = True

        for pos in range(active_count):
            slot = int(active_slots[pos])
            leaf_level = int(level_idx[slot])
            if mask[slot] or leaf_level >= max_level:
                continue
            scale = 1 << (max_level - leaf_level)
            xi = (int(tile_ix[slot]) << max_level) + int(x_idx[slot]) * scale
            yi = (int(tile_iy[slot]) << max_level) + int(y_idx[slot]) * scale
            center = values[slot, ch, center_y, center_x]

            if xi > 0:
                xx = xi - 1
                for yy in range(yi, yi + scale):
                    neighbor = int(owner[yy, xx])
                    if neighbor >= 0 and neighbor != slot and abs(float(center - values[neighbor, ch, center_y, center_x])) > tol:
                        mask[slot] = True
                        break
            if mask[slot]:
                continue
            if xi + scale < nx:
                xx = xi + scale
                for yy in range(yi, yi + scale):
                    neighbor = int(owner[yy, xx])
                    if neighbor >= 0 and neighbor != slot and abs(float(center - values[neighbor, ch, center_y, center_x])) > tol:
                        mask[slot] = True
                        break
            if mask[slot]:
                continue
            if yi > 0:
                yy = yi - 1
                for xx in range(xi, xi + scale):
                    neighbor = int(owner[yy, xx])
                    if neighbor >= 0 and neighbor != slot and abs(float(center - values[neighbor, ch, center_y, center_x])) > tol:
                        mask[slot] = True
                        break
            if mask[slot]:
                continue
            if yi + scale < ny:
                yy = yi + scale
                for xx in range(xi, xi + scale):
                    neighbor = int(owner[yy, xx])
                    if neighbor >= 0 and neighbor != slot and abs(float(center - values[neighbor, ch, center_y, center_x])) > tol:
                        mask[slot] = True
                        break
    return mask


def _compute_active_refine_slots(
    active: ActiveTopology,
    channels: list[int],
    tolerances: np.ndarray,
    owner: np.ndarray,
) -> np.ndarray:
    if not channels:
        return np.empty(0, dtype=np.int32)
    mask = _compute_active_refine_slots_jit(
        active.values,
        active.active_slots,
        int(active.active_count),
        active.level_idx,
        np.asarray(channels, dtype=np.int64),
        tolerances.astype(np.float64, copy=False),
        owner,
        active.tile_ix,
        active.tile_iy,
        active.x_idx,
        active.y_idx,
        int(active.domain["max_level_idx"]),
    )
    return np.flatnonzero(mask).astype(np.int32, copy=False)


@njit(boundscheck=False, cache=True)
def _compute_active_refine_slots_object_style_jit(
    values: np.ndarray,
    active_slots: np.ndarray,
    active_count: int,
    level_idx: np.ndarray,
    channels: np.ndarray,
    tolerances: np.ndarray,
    owner: np.ndarray,
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    xmin: float,
    ymin: float,
    tile_width: float,
    tile_height: float,
    max_level: int,
) -> np.ndarray:
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    mask = np.zeros(level_idx.shape[0], dtype=np.bool_)
    for tol_idx in range(channels.shape[0]):
        ch = int(channels[tol_idx])
        tol = float(tolerances[tol_idx])
        for pos in range(active_count):
            slot = int(active_slots[pos])
            leaf_level = int(level_idx[slot])
            if leaf_level >= max_level:
                continue
            v_min = values[slot, ch, 0, 0]
            v_max = v_min
            for yy in range(patch_h):
                for xx in range(patch_w):
                    val = values[slot, ch, yy, xx]
                    if val < v_min:
                        v_min = val
                    elif val > v_max:
                        v_max = val
            internal_err = float(v_max - v_min)
            cell_w = tile_width / float(1 << leaf_level)
            cell_h = tile_height / float(1 << leaf_level)
            cx = xmin + float(tile_ix[slot]) * tile_width + (float(x_idx[slot]) + 0.5) * cell_w
            cy = ymin + float(tile_iy[slot]) * tile_height + (float(y_idx[slot]) + 0.5) * cell_h
            center = _bilinear_sample_patch_jit(
                values[slot, ch],
                0.5 * float(patch_w) - 0.5,
                0.5 * float(patch_h) - 0.5,
            )
            boundary_err = 0.0
            sample_x = np.empty(4, dtype=np.float64)
            sample_y = np.empty(4, dtype=np.float64)
            sample_x[0] = cx - cell_w
            sample_y[0] = cy
            sample_x[1] = cx + cell_w
            sample_y[1] = cy
            sample_x[2] = cx
            sample_y[2] = cy - cell_h
            sample_x[3] = cx
            sample_y[3] = cy + cell_h
            for direction in range(4):
                neighbor = _owner_slot_for_physical_point_jit(
                    owner,
                    xmin,
                    ymin,
                    tile_width,
                    tile_height,
                    max_level,
                    sample_x[direction],
                    sample_y[direction],
                )
                neighbor_value = 0.0
                if neighbor >= 0:
                    neighbor_value = _active_sample_value_object_style_jit(
                        values,
                        neighbor,
                        ch,
                        sample_x[direction],
                        sample_y[direction],
                        tile_ix,
                        tile_iy,
                        level_idx,
                        x_idx,
                        y_idx,
                        xmin,
                        ymin,
                        tile_width,
                        tile_height,
                        max_level,
                    )
                diff = abs(float(center - neighbor_value))
                if diff > boundary_err:
                    boundary_err = diff
            if max(internal_err, boundary_err) > tol:
                mask[slot] = True
    return mask


def _compute_active_refine_slots_object_style(
    active: ActiveTopology,
    channels: list[int],
    tolerances: np.ndarray,
    owner: np.ndarray,
) -> np.ndarray:
    if not channels:
        return np.empty(0, dtype=np.int32)
    domain = active.domain
    mask = _compute_active_refine_slots_object_style_jit(
        active.values,
        active.active_slots,
        int(active.active_count),
        active.level_idx,
        np.asarray(channels, dtype=np.int64),
        tolerances.astype(np.float64, copy=False),
        owner,
        active.tile_ix,
        active.tile_iy,
        active.x_idx,
        active.y_idx,
        float(domain["xmin"]),
        float(domain["ymin"]),
        float(domain["tile_width"]),
        float(domain["tile_height"]),
        int(domain["max_level_idx"]),
    )
    return np.flatnonzero(mask).astype(np.int32, copy=False)


@njit(boundscheck=False, inline="always")
def _source_or_workspace_value(
    source_sequence: np.ndarray,
    values: np.ndarray,
    initial_slots: int,
    sequence_channels: int,
    slot: int,
    flat_ch: int,
    yy: int,
    xx: int,
) -> np.float32:
    if slot < initial_slots:
        t_idx = flat_ch // sequence_channels
        ch = flat_ch - t_idx * sequence_channels
        return source_sequence[slot, ch, t_idx, yy, xx]
    return values[slot - initial_slots, flat_ch, yy, xx]


@njit(boundscheck=False, inline="always")
def _bilinear_sample_patch_jit(patch: np.ndarray, px: float, py: float) -> float:
    patch_h, patch_w = patch.shape
    x_floor_float = np.floor(px)
    y_floor_float = np.floor(py)
    x_f = int(x_floor_float)
    y_f = int(y_floor_float)
    x_c = x_f + 1
    y_c = y_f + 1
    if x_f < 0:
        x_f = 0
    elif x_f >= patch_w:
        x_f = patch_w - 1
    if x_c < 0:
        x_c = 0
    elif x_c >= patch_w:
        x_c = patch_w - 1
    if y_f < 0:
        y_f = 0
    elif y_f >= patch_h:
        y_f = patch_h - 1
    if y_c < 0:
        y_c = 0
    elif y_c >= patch_h:
        y_c = patch_h - 1
    wx = px - x_floor_float
    wy = py - y_floor_float
    val_ff = patch[y_f, x_f]
    val_fc = patch[y_f, x_c]
    val_cf = patch[y_c, x_f]
    val_cc = patch[y_c, x_c]
    top = val_ff * (1.0 - wx) + val_fc * wx
    bottom = val_cf * (1.0 - wx) + val_cc * wx
    return float(top * (1.0 - wy) + bottom * wy)


@njit(boundscheck=False, inline="always")
def _owner_slot_for_physical_point_jit(
    owner: np.ndarray,
    xmin: float,
    ymin: float,
    tile_width: float,
    tile_height: float,
    max_level: int,
    x: float,
    y: float,
) -> int:
    fine = 1 << max_level
    fine_dx = tile_width / float(fine)
    fine_dy = tile_height / float(fine)
    nx = owner.shape[1]
    ny = owner.shape[0]
    xmax = xmin + fine_dx * float(nx)
    ymax = ymin + fine_dy * float(ny)
    if x < xmin or y < ymin or x > xmax or y > ymax:
        return -1
    ix = int(np.floor((x - xmin) / fine_dx))
    iy = int(np.floor((y - ymin) / fine_dy))
    if ix < 0:
        ix = 0
    elif ix >= nx:
        ix = nx - 1
    if iy < 0:
        iy = 0
    elif iy >= ny:
        iy = ny - 1
    return int(owner[iy, ix])


@njit(boundscheck=False, inline="always")
def _active_sample_value_object_style_jit(
    values: np.ndarray,
    slot: int,
    flat_ch: int,
    x: float,
    y: float,
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    level_idx: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    xmin: float,
    ymin: float,
    tile_width: float,
    tile_height: float,
    max_level: int,
) -> float:
    level = int(level_idx[slot])
    scale = 1 << level
    cell_w = tile_width / float(scale)
    cell_h = tile_height / float(scale)
    left = xmin + float(tile_ix[slot]) * tile_width + float(x_idx[slot]) * cell_w
    bottom = ymin + float(tile_iy[slot]) * tile_height + float(y_idx[slot]) * cell_h
    u = (x - left) / cell_w
    v = (y - bottom) / cell_h
    if u < 0.0:
        u = 0.0
    elif u > 1.0:
        u = 1.0
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    px = u * float(patch_w) - 0.5
    py = v * float(patch_h) - 0.5
    return _bilinear_sample_patch_jit(values[slot, flat_ch], px, py)


@njit(boundscheck=False, inline="always")
def _active_source_sample_value_object_style_jit(
    source_sequence: np.ndarray,
    values: np.ndarray,
    initial_slots: int,
    sequence_channels: int,
    slot: int,
    flat_ch: int,
    x: float,
    y: float,
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    level_idx: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    xmin: float,
    ymin: float,
    tile_width: float,
    tile_height: float,
    max_level: int,
) -> float:
    level = int(level_idx[slot])
    scale = 1 << level
    cell_w = tile_width / float(scale)
    cell_h = tile_height / float(scale)
    left = xmin + float(tile_ix[slot]) * tile_width + float(x_idx[slot]) * cell_w
    bottom = ymin + float(tile_iy[slot]) * tile_height + float(y_idx[slot]) * cell_h
    u = (x - left) / cell_w
    v = (y - bottom) / cell_h
    if u < 0.0:
        u = 0.0
    elif u > 1.0:
        u = 1.0
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    px = u * float(patch_w) - 0.5
    py = v * float(patch_h) - 0.5
    if slot < initial_slots:
        t_idx = flat_ch // sequence_channels
        ch = flat_ch - t_idx * sequence_channels
        return _bilinear_sample_patch_jit(source_sequence[slot, ch, t_idx], px, py)
    return _bilinear_sample_patch_jit(values[slot - initial_slots, flat_ch], px, py)


@njit(boundscheck=False, cache=True)
def _compute_active_refine_slots_from_source_jit(
    source_sequence: np.ndarray,
    values: np.ndarray,
    initial_slots: int,
    sequence_channels: int,
    active_slots: np.ndarray,
    active_count: int,
    level_idx: np.ndarray,
    channels: np.ndarray,
    tolerances: np.ndarray,
    owner: np.ndarray,
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    max_level: int,
) -> np.ndarray:
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    center_y = patch_h // 2
    center_x = patch_w // 2
    ny, nx = owner.shape
    mask = np.zeros(level_idx.shape[0], dtype=np.bool_)

    for tol_idx in range(channels.shape[0]):
        ch = int(channels[tol_idx])
        tol = float(tolerances[tol_idx])

        for pos in range(active_count):
            slot = int(active_slots[pos])
            if int(level_idx[slot]) >= max_level:
                continue
            v_min = _source_or_workspace_value(
                source_sequence, values, initial_slots, sequence_channels, slot, ch, 0, 0
            )
            v_max = v_min
            for yy in range(patch_h):
                for xx in range(patch_w):
                    val = _source_or_workspace_value(
                        source_sequence, values, initial_slots, sequence_channels, slot, ch, yy, xx
                    )
                    if val < v_min:
                        v_min = val
                    elif val > v_max:
                        v_max = val
            if float(v_max - v_min) > tol:
                mask[slot] = True

        for pos in range(active_count):
            slot = int(active_slots[pos])
            leaf_level = int(level_idx[slot])
            if mask[slot] or leaf_level >= max_level:
                continue
            scale = 1 << (max_level - leaf_level)
            xi = (int(tile_ix[slot]) << max_level) + int(x_idx[slot]) * scale
            yi = (int(tile_iy[slot]) << max_level) + int(y_idx[slot]) * scale
            center = _source_or_workspace_value(
                source_sequence, values, initial_slots, sequence_channels, slot, ch, center_y, center_x
            )

            if xi > 0:
                xx = xi - 1
                for yy in range(yi, yi + scale):
                    neighbor = int(owner[yy, xx])
                    if (
                        neighbor >= 0
                        and neighbor != slot
                        and abs(
                            float(
                                center
                                - _source_or_workspace_value(
                                    source_sequence,
                                    values,
                                    initial_slots,
                                    sequence_channels,
                                    neighbor,
                                    ch,
                                    center_y,
                                    center_x,
                                )
                            )
                        )
                        > tol
                    ):
                        mask[slot] = True
                        break
            if mask[slot]:
                continue
            if xi + scale < nx:
                xx = xi + scale
                for yy in range(yi, yi + scale):
                    neighbor = int(owner[yy, xx])
                    if (
                        neighbor >= 0
                        and neighbor != slot
                        and abs(
                            float(
                                center
                                - _source_or_workspace_value(
                                    source_sequence,
                                    values,
                                    initial_slots,
                                    sequence_channels,
                                    neighbor,
                                    ch,
                                    center_y,
                                    center_x,
                                )
                            )
                        )
                        > tol
                    ):
                        mask[slot] = True
                        break
            if mask[slot]:
                continue
            if yi > 0:
                yy = yi - 1
                for xx in range(xi, xi + scale):
                    neighbor = int(owner[yy, xx])
                    if (
                        neighbor >= 0
                        and neighbor != slot
                        and abs(
                            float(
                                center
                                - _source_or_workspace_value(
                                    source_sequence,
                                    values,
                                    initial_slots,
                                    sequence_channels,
                                    neighbor,
                                    ch,
                                    center_y,
                                    center_x,
                                )
                            )
                        )
                        > tol
                    ):
                        mask[slot] = True
                        break
            if mask[slot]:
                continue
            if yi + scale < ny:
                yy = yi + scale
                for xx in range(xi, xi + scale):
                    neighbor = int(owner[yy, xx])
                    if (
                        neighbor >= 0
                        and neighbor != slot
                        and abs(
                            float(
                                center
                                - _source_or_workspace_value(
                                    source_sequence,
                                    values,
                                    initial_slots,
                                    sequence_channels,
                                    neighbor,
                                    ch,
                                    center_y,
                                    center_x,
                                )
                            )
                        )
                        > tol
                    ):
                        mask[slot] = True
                        break
    return mask


def _compute_active_refine_slots_from_source(
    active: SequenceSourceActiveTopology,
    channels: list[int],
    tolerances: np.ndarray,
    owner: np.ndarray,
) -> np.ndarray:
    if not channels:
        return np.empty(0, dtype=np.int32)
    mask = _compute_active_refine_slots_from_source_jit(
        active.source_sequence,
        active.values,
        int(active.initial_slots),
        int(active.sequence_channels),
        active.active_slots,
        int(active.active_count),
        active.level_idx,
        np.asarray(channels, dtype=np.int64),
        tolerances.astype(np.float64, copy=False),
        owner,
        active.tile_ix,
        active.tile_iy,
        active.x_idx,
        active.y_idx,
        int(active.domain["max_level_idx"]),
    )
    return np.flatnonzero(mask).astype(np.int32, copy=False)


@njit(boundscheck=False, cache=True)
def _compute_active_refine_slots_from_source_object_style_jit(
    source_sequence: np.ndarray,
    values: np.ndarray,
    initial_slots: int,
    sequence_channels: int,
    active_slots: np.ndarray,
    active_count: int,
    level_idx: np.ndarray,
    channels: np.ndarray,
    tolerances: np.ndarray,
    owner: np.ndarray,
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    max_level: int,
    xmin: float,
    ymin: float,
    tile_width: float,
    tile_height: float,
) -> np.ndarray:
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    mask = np.zeros(level_idx.shape[0], dtype=np.bool_)

    for tol_idx in range(channels.shape[0]):
        ch = int(channels[tol_idx])
        tol = float(tolerances[tol_idx])

        for pos in range(active_count):
            slot = int(active_slots[pos])
            leaf_level = int(level_idx[slot])
            if leaf_level >= max_level:
                continue
            v_min = _source_or_workspace_value(
                source_sequence,
                values,
                initial_slots,
                sequence_channels,
                slot,
                ch,
                0,
                0,
            )
            v_max = v_min
            for yy in range(patch_h):
                for xx in range(patch_w):
                    val = _source_or_workspace_value(
                        source_sequence,
                        values,
                        initial_slots,
                        sequence_channels,
                        slot,
                        ch,
                        yy,
                        xx,
                    )
                    if val < v_min:
                        v_min = val
                    elif val > v_max:
                        v_max = val
            internal_err = float(v_max - v_min)
            cell_w = tile_width / float(1 << leaf_level)
            cell_h = tile_height / float(1 << leaf_level)
            cx = xmin + float(tile_ix[slot]) * tile_width + (float(x_idx[slot]) + 0.5) * cell_w
            cy = ymin + float(tile_iy[slot]) * tile_height + (float(y_idx[slot]) + 0.5) * cell_h
            center = _active_source_sample_value_object_style_jit(
                source_sequence,
                values,
                initial_slots,
                sequence_channels,
                slot,
                ch,
                cx,
                cy,
                tile_ix,
                tile_iy,
                level_idx,
                x_idx,
                y_idx,
                xmin,
                ymin,
                tile_width,
                tile_height,
                max_level,
            )
            boundary_err = 0.0
            sample_x = np.empty(4, dtype=np.float64)
            sample_y = np.empty(4, dtype=np.float64)
            sample_x[0] = cx - cell_w
            sample_y[0] = cy
            sample_x[1] = cx + cell_w
            sample_y[1] = cy
            sample_x[2] = cx
            sample_y[2] = cy - cell_h
            sample_x[3] = cx
            sample_y[3] = cy + cell_h
            for direction in range(4):
                neighbor = _owner_slot_for_physical_point_jit(
                    owner,
                    xmin,
                    ymin,
                    tile_width,
                    tile_height,
                    max_level,
                    sample_x[direction],
                    sample_y[direction],
                )
                neighbor_value = 0.0
                if neighbor >= 0:
                    neighbor_value = _active_source_sample_value_object_style_jit(
                        source_sequence,
                        values,
                        initial_slots,
                        sequence_channels,
                        neighbor,
                        ch,
                        sample_x[direction],
                        sample_y[direction],
                        tile_ix,
                        tile_iy,
                        level_idx,
                        x_idx,
                        y_idx,
                        xmin,
                        ymin,
                        tile_width,
                        tile_height,
                        max_level,
                    )
                diff = abs(float(center - neighbor_value))
                if diff > boundary_err:
                    boundary_err = diff
            if max(internal_err, boundary_err) > tol:
                mask[slot] = True
    return mask


def _compute_active_refine_slots_from_source_object_style(
    active: SequenceSourceActiveTopology,
    channels: list[int],
    tolerances: np.ndarray,
    owner: np.ndarray,
) -> np.ndarray:
    if not channels:
        return np.empty(0, dtype=np.int32)
    domain = active.domain
    mask = _compute_active_refine_slots_from_source_object_style_jit(
        active.source_sequence,
        active.values,
        int(active.initial_slots),
        int(active.sequence_channels),
        active.active_slots,
        int(active.active_count),
        active.level_idx,
        np.asarray(channels, dtype=np.int64),
        tolerances.astype(np.float64, copy=False),
        owner,
        active.tile_ix,
        active.tile_iy,
        active.x_idx,
        active.y_idx,
        int(domain["max_level_idx"]),
        float(domain["xmin"]),
        float(domain["ymin"]),
        float(domain["tile_width"]),
        float(domain["tile_height"]),
    )
    return np.flatnonzero(mask).astype(np.int32, copy=False)


def _write_split_patch_bilinear(out: np.ndarray, value: np.ndarray) -> None:
    # Bilinear-upsample each quadrant back to the full patch, matching object
    # QuadCell.refine (quadtree.py) and the array active-path fill. Replaces the old
    # nearest/repeat upsampler, which silently diverged from object on every refined cell.
    _, patch_h, patch_w = value.shape
    h_mid = patch_h // 2
    w_mid = patch_w // 2
    quads = (
        value[:, 0:h_mid, 0:w_mid],
        value[:, 0:h_mid, w_mid:patch_w],
        value[:, h_mid:patch_h, 0:w_mid],
        value[:, h_mid:patch_h, w_mid:patch_w],
    )
    for child, quad in enumerate(quads):
        up = resize_patch_bilinear_jit(np.ascontiguousarray(quad), patch_h, patch_w)
        out[child] = up.astype(out.dtype, copy=False)


def _assemble_refined_topology(
    top: FlatTopology,
    refine_indices: np.ndarray,
    protected: Optional[np.ndarray] = None,
) -> Tuple[FlatTopology, Optional[np.ndarray]]:
    refine_indices = np.sort(np.asarray(refine_indices, dtype=np.intp))
    if refine_indices.size == 0:
        return top, protected

    new_count = top.values.shape[0] + 3 * refine_indices.size
    values = np.empty((new_count,) + top.values.shape[1:], dtype=top.values.dtype)
    tile_ix = np.empty(new_count, dtype=np.int32)
    tile_iy = np.empty(new_count, dtype=np.int32)
    level_idx = np.empty(new_count, dtype=np.int16)
    x_idx = np.empty(new_count, dtype=np.int32)
    y_idx = np.empty(new_count, dtype=np.int32)
    new_protected = np.empty(new_count, dtype=bool) if protected is not None else None

    child_dx = (0, 1, 0, 1)
    child_dy = (0, 0, 1, 1)

    out_i = 0
    src_start = 0
    for parent_idx in refine_indices:
        parent_i = int(parent_idx)
        if parent_i > src_start:
            span = parent_i - src_start
            dst_end = out_i + span
            values[out_i:dst_end] = top.values[src_start:parent_i]
            tile_ix[out_i:dst_end] = top.tile_ix[src_start:parent_i]
            tile_iy[out_i:dst_end] = top.tile_iy[src_start:parent_i]
            level_idx[out_i:dst_end] = top.level_idx[src_start:parent_i]
            x_idx[out_i:dst_end] = top.x_idx[src_start:parent_i]
            y_idx[out_i:dst_end] = top.y_idx[src_start:parent_i]
            if new_protected is not None:
                new_protected[out_i:dst_end] = protected[src_start:parent_i]
            out_i = dst_end

        _write_split_patch_bilinear(values[out_i:out_i + 4], top.values[parent_i])
        child_level = np.int16(int(top.level_idx[parent_i]) + 1)
        parent_tile_ix = top.tile_ix[parent_i]
        parent_tile_iy = top.tile_iy[parent_i]
        parent_x_idx = int(top.x_idx[parent_i]) * 2
        parent_y_idx = int(top.y_idx[parent_i]) * 2
        parent_protected = bool(protected[parent_i]) if new_protected is not None else False
        for child in range(4):
            tile_ix[out_i] = parent_tile_ix
            tile_iy[out_i] = parent_tile_iy
            level_idx[out_i] = child_level
            x_idx[out_i] = parent_x_idx + child_dx[child]
            y_idx[out_i] = parent_y_idx + child_dy[child]
            if new_protected is not None:
                new_protected[out_i] = parent_protected
            out_i += 1
        src_start = parent_i + 1

    if src_start < top.values.shape[0]:
        span = top.values.shape[0] - src_start
        dst_end = out_i + span
        values[out_i:dst_end] = top.values[src_start:]
        tile_ix[out_i:dst_end] = top.tile_ix[src_start:]
        tile_iy[out_i:dst_end] = top.tile_iy[src_start:]
        level_idx[out_i:dst_end] = top.level_idx[src_start:]
        x_idx[out_i:dst_end] = top.x_idx[src_start:]
        y_idx[out_i:dst_end] = top.y_idx[src_start:]
        if new_protected is not None:
            new_protected[out_i:dst_end] = protected[src_start:]

    refined = FlatTopology(values, tile_ix, tile_iy, level_idx, x_idx, y_idx, dict(top.domain), top.cell_scale_mode)
    return refined, new_protected


def _refine_masked(
    top: FlatTopology,
    mask: np.ndarray,
    protected: Optional[np.ndarray] = None,
) -> Tuple[FlatTopology, Optional[np.ndarray]]:
    refine_indices = np.flatnonzero(mask)
    if refine_indices.size == 0:
        return top, protected

    return _assemble_refined_topology(top, refine_indices, protected=protected)


def _ensure_capacity(num_leaves: int, capacity: int) -> None:
    if num_leaves > capacity:
        raise _ArrayFallback("capacity_exceeded")


def _active_from_flat(
    top: FlatTopology,
    capacity: int,
    protected: Optional[np.ndarray] = None,
) -> Tuple[ActiveTopology, Optional[np.ndarray]]:
    n_leaves = int(top.values.shape[0])
    _ensure_capacity(n_leaves, capacity)

    values = np.empty((int(capacity),) + top.values.shape[1:], dtype=top.values.dtype)
    tile_ix = np.empty(int(capacity), dtype=np.int32)
    tile_iy = np.empty(int(capacity), dtype=np.int32)
    level_idx = np.empty(int(capacity), dtype=np.int16)
    x_idx = np.empty(int(capacity), dtype=np.int32)
    y_idx = np.empty(int(capacity), dtype=np.int32)
    active_slots = np.empty(int(capacity), dtype=np.int32)

    values[:n_leaves] = top.values
    tile_ix[:n_leaves] = top.tile_ix
    tile_iy[:n_leaves] = top.tile_iy
    level_idx[:n_leaves] = top.level_idx
    x_idx[:n_leaves] = top.x_idx
    y_idx[:n_leaves] = top.y_idx
    active_slots[:n_leaves] = np.arange(n_leaves, dtype=np.int32)

    active_protected = None
    if protected is not None:
        active_protected = np.zeros(int(capacity), dtype=bool)
        active_protected[:n_leaves] = protected

    active = ActiveTopology(
        values=values,
        tile_ix=tile_ix,
        tile_iy=tile_iy,
        level_idx=level_idx,
        x_idx=x_idx,
        y_idx=y_idx,
        active_slots=active_slots,
        active_count=n_leaves,
        next_slot=n_leaves,
        domain=dict(top.domain),
        cell_scale_mode=top.cell_scale_mode,
    )
    return active, active_protected


@njit(boundscheck=False, parallel=True)
def _copy_sequence_to_flat_values_jit(sequence: np.ndarray, out: np.ndarray) -> None:
    n_leaves, channels, timesteps, patch_h, patch_w = sequence.shape
    for leaf in prange(n_leaves):
        for t_idx in range(timesteps):
            flat_ch_offset = t_idx * channels
            for ch in range(channels):
                flat_ch = flat_ch_offset + ch
                for yy in range(patch_h):
                    for xx in range(patch_w):
                        out[leaf, flat_ch, yy, xx] = sequence[leaf, ch, t_idx, yy, xx]


def _flat_channel_view_from_sequence(sequence: np.ndarray) -> Optional[np.ndarray]:
    n_leaves, channels, timesteps, patch_h, patch_w = sequence.shape
    itemsize = int(sequence.dtype.itemsize)
    expected_strides = (
        timesteps * channels * patch_h * patch_w * itemsize,
        patch_h * patch_w * itemsize,
        channels * patch_h * patch_w * itemsize,
        patch_w * itemsize,
        itemsize,
    )
    if tuple(int(stride) for stride in sequence.strides) != expected_strides:
        return None
    flat = sequence.transpose(0, 2, 1, 3, 4).reshape(n_leaves, timesteps * channels, patch_h, patch_w)
    if not np.shares_memory(flat, sequence) or not flat.flags.c_contiguous:
        return None
    return flat


def _copy_sequence_to_flat_values(sequence: np.ndarray, out: np.ndarray) -> None:
    if sequence.ndim != 5:
        raise ValueError(f"Expected sequence shape (N,C,T,H,W), got {sequence.shape}.")
    n_leaves, channels, timesteps, patch_h, patch_w = sequence.shape
    if out.shape[0] < n_leaves:
        raise _ArrayFallback("capacity_exceeded")
    if out.shape[1:] != (timesteps * channels, patch_h, patch_w):
        raise ValueError(
            f"Flat output shape {out.shape} is incompatible with sequence shape {sequence.shape}."
        )
    flat_view = _flat_channel_view_from_sequence(sequence)
    if flat_view is not None:
        out[:n_leaves] = flat_view
        return
    _copy_sequence_to_flat_values_jit(sequence, out)


def _flatten_sequence_array(sequence: np.ndarray) -> np.ndarray:
    if sequence.ndim != 5:
        raise ValueError(f"Expected sequence shape (N,C,T,H,W), got {sequence.shape}.")
    n_leaves, channels, timesteps, patch_h, patch_w = sequence.shape
    flat = np.empty((n_leaves, timesteps * channels, patch_h, patch_w), dtype=np.float32)
    _copy_sequence_to_flat_values(sequence, flat)
    return flat


def _active_from_sequence(
    sequence: np.ndarray,
    top: FlatTopology,
    capacity: int,
    protected: Optional[np.ndarray] = None,
) -> Tuple[ActiveTopology, Optional[np.ndarray]]:
    if sequence.ndim != 5:
        raise ValueError(f"Expected sequence shape (N,C,T,H,W), got {sequence.shape}.")
    n_leaves, channels, timesteps, patch_h, patch_w = sequence.shape
    if n_leaves != int(top.level_idx.shape[0]):
        raise ValueError("sequence leaf count does not match topology metadata")
    _ensure_capacity(n_leaves, capacity)

    values = np.empty((int(capacity), timesteps * channels, patch_h, patch_w), dtype=np.float32)
    tile_ix = np.empty(int(capacity), dtype=np.int32)
    tile_iy = np.empty(int(capacity), dtype=np.int32)
    level_idx = np.empty(int(capacity), dtype=np.int16)
    x_idx = np.empty(int(capacity), dtype=np.int32)
    y_idx = np.empty(int(capacity), dtype=np.int32)
    active_slots = np.empty(int(capacity), dtype=np.int32)

    _copy_sequence_to_flat_values(sequence, values)
    tile_ix[:n_leaves] = top.tile_ix
    tile_iy[:n_leaves] = top.tile_iy
    level_idx[:n_leaves] = top.level_idx
    x_idx[:n_leaves] = top.x_idx
    y_idx[:n_leaves] = top.y_idx
    active_slots[:n_leaves] = np.arange(n_leaves, dtype=np.int32)

    active_protected = None
    if protected is not None:
        active_protected = np.zeros(int(capacity), dtype=bool)
        active_protected[:n_leaves] = protected

    active = ActiveTopology(
        values=values,
        tile_ix=tile_ix,
        tile_iy=tile_iy,
        level_idx=level_idx,
        x_idx=x_idx,
        y_idx=y_idx,
        active_slots=active_slots,
        active_count=int(n_leaves),
        next_slot=int(n_leaves),
        domain=dict(top.domain),
        cell_scale_mode=top.cell_scale_mode,
    )
    return active, active_protected


def _active_from_sequence_source(
    sequence: np.ndarray,
    top: FlatTopology,
    capacity: int,
    protected: Optional[np.ndarray] = None,
) -> Tuple[SequenceSourceActiveTopology, Optional[np.ndarray]]:
    if sequence.ndim != 5:
        raise ValueError(f"Expected sequence shape (N,C,T,H,W), got {sequence.shape}.")
    n_leaves, channels, timesteps, patch_h, patch_w = sequence.shape
    if n_leaves != int(top.level_idx.shape[0]):
        raise ValueError("sequence leaf count does not match topology metadata")
    _ensure_capacity(n_leaves, capacity)

    source_sequence_copied = not (sequence.dtype == np.float32 and sequence.flags.c_contiguous)
    source_sequence = np.ascontiguousarray(sequence, dtype=np.float32)
    values = np.empty((int(capacity) - int(n_leaves), timesteps * channels, patch_h, patch_w), dtype=np.float32)
    tile_ix = np.empty(int(capacity), dtype=np.int32)
    tile_iy = np.empty(int(capacity), dtype=np.int32)
    level_idx = np.empty(int(capacity), dtype=np.int16)
    x_idx = np.empty(int(capacity), dtype=np.int32)
    y_idx = np.empty(int(capacity), dtype=np.int32)
    active_slots = np.empty(int(capacity), dtype=np.int32)

    tile_ix[:n_leaves] = top.tile_ix
    tile_iy[:n_leaves] = top.tile_iy
    level_idx[:n_leaves] = top.level_idx
    x_idx[:n_leaves] = top.x_idx
    y_idx[:n_leaves] = top.y_idx
    active_slots[:n_leaves] = np.arange(n_leaves, dtype=np.int32)

    active_protected = None
    if protected is not None:
        active_protected = np.zeros(int(capacity), dtype=bool)
        active_protected[:n_leaves] = protected

    active = SequenceSourceActiveTopology(
        values=values,
        tile_ix=tile_ix,
        tile_iy=tile_iy,
        level_idx=level_idx,
        x_idx=x_idx,
        y_idx=y_idx,
        active_slots=active_slots,
        active_count=int(n_leaves),
        next_slot=int(n_leaves),
        domain=dict(top.domain),
        cell_scale_mode=top.cell_scale_mode,
        source_sequence=source_sequence,
        initial_slots=int(n_leaves),
        sequence_channels=int(channels),
        sequence_timesteps=int(timesteps),
        source_sequence_copied=bool(source_sequence_copied),
    )
    return active, active_protected


def _active_to_flat(
    active: ActiveTopology,
    protected: Optional[np.ndarray] = None,
) -> Tuple[FlatTopology, Optional[np.ndarray]]:
    slots = active.active_slots[:active.active_count]
    top = FlatTopology(
        values=np.ascontiguousarray(active.values[slots], dtype=active.values.dtype),
        tile_ix=active.tile_ix[slots].astype(np.int32, copy=True),
        tile_iy=active.tile_iy[slots].astype(np.int32, copy=True),
        level_idx=active.level_idx[slots].astype(np.int16, copy=True),
        x_idx=active.x_idx[slots].astype(np.int32, copy=True),
        y_idx=active.y_idx[slots].astype(np.int32, copy=True),
        domain=dict(active.domain),
        cell_scale_mode=active.cell_scale_mode,
    )
    if protected is None:
        return top, None
    return top, protected[slots].astype(bool, copy=True)


def _ensure_active_append_capacity(active: ActiveTopology, append_count: int, new_active_count: int) -> None:
    capacity = int(active.active_slots.shape[0])
    if new_active_count > capacity or int(active.next_slot) + int(append_count) > capacity:
        raise _ArrayFallback("capacity_exceeded")


@njit(boundscheck=False, cache=True)
def _bilinear_fill_one_parent(values, slot, child_start):
    # Fill the 4 child slots of a refined parent by bilinear-upsampling each quadrant
    # back to the full patch -- byte-identical to object QuadCell.refine (quadtree.py:131):
    # slice into quadrants (uneven sizes for odd patch dims) then resize_patch_bilinear_jit
    # each to (patch_h, patch_w). resize returns float64; the per-element write casts to
    # the values dtype, exactly as object does on export.
    channels = values.shape[1]
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    h_mid = patch_h // 2
    w_mid = patch_w // 2
    for child in range(4):
        child_slot = child_start + child
        y0 = 0 if child < 2 else h_mid
        y1 = h_mid if child < 2 else patch_h
        x0 = 0 if (child == 0 or child == 2) else w_mid
        x1 = w_mid if (child == 0 or child == 2) else patch_w
        quad = np.ascontiguousarray(values[slot, :, y0:y1, x0:x1])
        up = resize_patch_bilinear_jit(quad, patch_h, patch_w)
        for ch in range(channels):
            for yy in range(patch_h):
                for xx in range(patch_w):
                    values[child_slot, ch, yy, xx] = up[ch, yy, xx]


@njit(boundscheck=False, cache=True)
def _fill_refined_child_values_sequential_jit(
    values: np.ndarray,
    parent_slots: np.ndarray,
    child_starts: np.ndarray,
    refine_count: int,
) -> None:
    for refine_pos in range(refine_count):
        _bilinear_fill_one_parent(
            values, int(parent_slots[refine_pos]), int(child_starts[refine_pos]))


@njit(boundscheck=False, parallel=True, cache=True)
def _fill_refined_child_values_parallel_jit(
    values: np.ndarray,
    parent_slots: np.ndarray,
    child_starts: np.ndarray,
    refine_count: int,
) -> None:
    for refine_pos in prange(refine_count):
        _bilinear_fill_one_parent(
            values, int(parent_slots[refine_pos]), int(child_starts[refine_pos]))


@njit(boundscheck=False, parallel=True, cache=True)
def _fill_refined_child_values_block_parallel_jit(
    values: np.ndarray,
    parent_slots: np.ndarray,
    child_starts: np.ndarray,
    refine_count: int,
) -> None:
    for refine_pos in prange(refine_count):
        _bilinear_fill_one_parent(
            values, int(parent_slots[refine_pos]), int(child_starts[refine_pos]))


@njit(boundscheck=False, cache=True)
def _fill_refined_child_values_jit(
    values: np.ndarray,
    parent_slots: np.ndarray,
    child_starts: np.ndarray,
    refine_count: int,
) -> None:
    if refine_count >= _REFINE_PARALLEL_FILL_THRESHOLD:
        _fill_refined_child_values_block_parallel_jit(values, parent_slots, child_starts, refine_count)
    else:
        _fill_refined_child_values_sequential_jit(values, parent_slots, child_starts, refine_count)


@njit(boundscheck=False, cache=True)
def _bilinear_fill_from_source(source_sequence, values, src_slot, child_ws_start, sequence_channels):
    # Source-mode bilinear upsample, matching object/copy: per timestep, resize the source
    # quadrant (sequence_channels, h_q, w_q) -> (patch_h, patch_w). Quadrant sizes are
    # uneven for odd patch dims, mirroring object QuadCell.refine (quadtree.py:122-129).
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    h_mid = patch_h // 2
    w_mid = patch_w // 2
    timesteps = source_sequence.shape[2]
    for child in range(4):
        child_ws = child_ws_start + child
        y0 = 0 if child < 2 else h_mid
        y1 = h_mid if child < 2 else patch_h
        x0 = 0 if (child == 0 or child == 2) else w_mid
        x1 = w_mid if (child == 0 or child == 2) else patch_w
        for t_idx in range(timesteps):
            quad = np.ascontiguousarray(source_sequence[src_slot, :, t_idx, y0:y1, x0:x1])
            up = resize_patch_bilinear_jit(quad, patch_h, patch_w)
            for ch in range(sequence_channels):
                flat_ch = t_idx * sequence_channels + ch
                for yy in range(patch_h):
                    for xx in range(patch_w):
                        values[child_ws, flat_ch, yy, xx] = up[ch, yy, xx]


@njit(boundscheck=False, cache=True)
def _fill_refined_child_values_from_source_sequential_jit(
    source_sequence: np.ndarray,
    values: np.ndarray,
    initial_slots: int,
    sequence_channels: int,
    parent_slots: np.ndarray,
    child_starts: np.ndarray,
    refine_count: int,
) -> None:
    for refine_pos in range(refine_count):
        slot = int(parent_slots[refine_pos])
        child_ws_start = int(child_starts[refine_pos]) - initial_slots
        if slot < initial_slots:
            _bilinear_fill_from_source(
                source_sequence, values, slot, child_ws_start, sequence_channels)
        else:
            _bilinear_fill_one_parent(values, slot - initial_slots, child_ws_start)


@njit(boundscheck=False, parallel=True, cache=True)
def _fill_refined_child_values_from_source_parallel_jit(
    source_sequence: np.ndarray,
    values: np.ndarray,
    initial_slots: int,
    sequence_channels: int,
    parent_slots: np.ndarray,
    child_starts: np.ndarray,
    refine_count: int,
) -> None:
    for refine_pos in prange(refine_count):
        slot = int(parent_slots[refine_pos])
        child_ws_start = int(child_starts[refine_pos]) - initial_slots
        if slot < initial_slots:
            _bilinear_fill_from_source(
                source_sequence, values, slot, child_ws_start, sequence_channels)
        else:
            _bilinear_fill_one_parent(values, slot - initial_slots, child_ws_start)


@njit(boundscheck=False, cache=True)
def _fill_refined_child_values_from_source_jit(
    source_sequence: np.ndarray,
    values: np.ndarray,
    initial_slots: int,
    sequence_channels: int,
    parent_slots: np.ndarray,
    child_starts: np.ndarray,
    refine_count: int,
) -> None:
    if refine_count >= _SOURCE_REFINE_PARALLEL_FILL_THRESHOLD:
        _fill_refined_child_values_from_source_parallel_jit(
            source_sequence,
            values,
            initial_slots,
            sequence_channels,
            parent_slots,
            child_starts,
            refine_count,
        )
    else:
        _fill_refined_child_values_from_source_sequential_jit(
            source_sequence,
            values,
            initial_slots,
            sequence_channels,
            parent_slots,
            child_starts,
            refine_count,
        )


@njit(boundscheck=False, cache=True)
def _active_refine_slots_jit(
    values: np.ndarray,
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    level_idx: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    active_slots: np.ndarray,
    active_count: int,
    next_slot: int,
    refine_mask: np.ndarray,
    protected: np.ndarray,
    has_protected: bool,
) -> Tuple[np.ndarray, int, int]:
    new_active_slots = np.empty_like(active_slots)
    refined_parent_slots = np.empty(active_count, dtype=np.int32)
    refined_child_starts = np.empty(active_count, dtype=np.int32)
    out_count = 0
    refine_count = 0
    child_dx = (0, 1, 0, 1)
    child_dy = (0, 0, 1, 1)

    for active_pos in range(active_count):
        slot = int(active_slots[active_pos])
        if not refine_mask[slot]:
            new_active_slots[out_count] = slot
            out_count += 1
            continue

        child_start = next_slot
        child_stop = child_start + 4
        next_slot = child_stop
        refined_parent_slots[refine_count] = slot
        refined_child_starts[refine_count] = child_start
        refine_count += 1
        child_level = np.int16(int(level_idx[slot]) + 1)
        child_x0 = int(x_idx[slot]) * 2
        child_y0 = int(y_idx[slot]) * 2
        parent_tile_ix = tile_ix[slot]
        parent_tile_iy = tile_iy[slot]
        parent_protected = False
        if has_protected:
            parent_protected = bool(protected[slot])
            protected[slot] = False

        for child in range(4):
            child_slot = child_start + child
            new_active_slots[out_count + child] = child_slot
            tile_ix[child_slot] = parent_tile_ix
            tile_iy[child_slot] = parent_tile_iy
            level_idx[child_slot] = child_level
            x_idx[child_slot] = child_x0 + child_dx[child]
            y_idx[child_slot] = child_y0 + child_dy[child]
            if has_protected:
                protected[child_slot] = parent_protected

        out_count += 4

    if refine_count > 0:
        _fill_refined_child_values_jit(values, refined_parent_slots, refined_child_starts, refine_count)

    return new_active_slots, out_count, next_slot


@njit(boundscheck=False, cache=True)
def _active_refine_slots_from_source_jit(
    source_sequence: np.ndarray,
    values: np.ndarray,
    initial_slots: int,
    sequence_channels: int,
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    level_idx: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    active_slots: np.ndarray,
    active_count: int,
    next_slot: int,
    refine_mask: np.ndarray,
    protected: np.ndarray,
    has_protected: bool,
) -> Tuple[np.ndarray, int, int]:
    new_active_slots = np.empty_like(active_slots)
    refined_parent_slots = np.empty(active_count, dtype=np.int32)
    refined_child_starts = np.empty(active_count, dtype=np.int32)
    out_count = 0
    refine_count = 0
    child_dx = (0, 1, 0, 1)
    child_dy = (0, 0, 1, 1)

    for active_pos in range(active_count):
        slot = int(active_slots[active_pos])
        if not refine_mask[slot]:
            new_active_slots[out_count] = slot
            out_count += 1
            continue

        child_start = next_slot
        child_stop = child_start + 4
        next_slot = child_stop
        refined_parent_slots[refine_count] = slot
        refined_child_starts[refine_count] = child_start
        refine_count += 1
        child_level = np.int16(int(level_idx[slot]) + 1)
        child_x0 = int(x_idx[slot]) * 2
        child_y0 = int(y_idx[slot]) * 2
        parent_tile_ix = tile_ix[slot]
        parent_tile_iy = tile_iy[slot]
        parent_protected = False
        if has_protected:
            parent_protected = bool(protected[slot])
            protected[slot] = False

        for child in range(4):
            child_slot = child_start + child
            new_active_slots[out_count + child] = child_slot
            tile_ix[child_slot] = parent_tile_ix
            tile_iy[child_slot] = parent_tile_iy
            level_idx[child_slot] = child_level
            x_idx[child_slot] = child_x0 + child_dx[child]
            y_idx[child_slot] = child_y0 + child_dy[child]
            if has_protected:
                protected[child_slot] = parent_protected

        out_count += 4

    if refine_count > 0:
        _fill_refined_child_values_from_source_jit(
            source_sequence,
            values,
            initial_slots,
            sequence_channels,
            refined_parent_slots,
            refined_child_starts,
            refine_count,
        )

    return new_active_slots, out_count, next_slot


def warm_array_regrid_kernels(
    *,
    flat_channels: int,
    patch_size: Union[int, Sequence[int]],
    capacity: int = 8,
    warm_source: bool = False,
    warm_copy: bool = True,
    output_layout: str = "both",
    array_regrid_mode: str = "parity",
) -> None:
    array_regrid_mode = _normalize_array_regrid_mode(array_regrid_mode)
    flat_channels = int(flat_channels)
    if isinstance(patch_size, (tuple, list)):
        if len(patch_size) != 2:
            raise ValueError("patch_size sequence must contain height and width.")
        patch_h = int(patch_size[0])
        patch_w = int(patch_size[1])
    else:
        patch_h = int(patch_size)
        patch_w = patch_h
    if flat_channels <= 0 or patch_h <= 0 or patch_w <= 0:
        raise ValueError("flat_channels and patch dimensions must be positive.")
    if output_layout not in {"flat", "sequence", "both"}:
        raise ValueError("output_layout must be one of 'flat', 'sequence', or 'both'.")
    warm_flat_export = output_layout in {"flat", "both"}
    warm_sequence_export = output_layout in {"sequence", "both"}

    # Parity mode currently reuses most active topology kernels; direct public-call warmup covers parity-only kernels.
    _ = array_regrid_mode

    # Numba specializes these kernels on dtype/layout, not concrete shapes.
    # Keep warmup buffers small so rollout setup does not allocate or walk the
    # full production capacity just to compile representative kernels.
    capacity = min(max(5, int(capacity)), _WARM_ARRAY_REGRID_ACTIVE_CAPACITY)
    warm_patch_h = max(1, min(2, patch_h))
    warm_patch_w = max(1, min(2, patch_w))
    protected_count = int(_ACTIVE_PROTECTED_MASK_PARALLEL_THRESHOLD)
    protected_slots = np.arange(protected_count, dtype=np.int32)
    protected_channels = np.array([0], dtype=np.int64)
    protected_tolerances = np.array([1.0], dtype=np.float64)

    coarsen_values = np.zeros((8, flat_channels, 2, 2), dtype=np.float32)
    coarsen_tile_ix = np.zeros(8, dtype=np.int32)
    coarsen_tile_iy = np.zeros(8, dtype=np.int32)
    coarsen_level_idx = np.zeros(8, dtype=np.int16)
    coarsen_level_idx[:4] = 1
    coarsen_x_idx = np.zeros(8, dtype=np.int32)
    coarsen_y_idx = np.zeros(8, dtype=np.int32)
    coarsen_x_idx[:4] = np.array([0, 1, 0, 1], dtype=np.int32)
    coarsen_y_idx[:4] = np.array([0, 0, 1, 1], dtype=np.int32)
    coarsen_active_slots = np.arange(8, dtype=np.int32)
    coarsen_protected = np.zeros(8, dtype=bool)
    coarsen_group_keys = np.array([[0, 0, 0, 0, 0]], dtype=np.int64)
    coarsen_sibling_groups = np.array([[0, 1, 2, 3]], dtype=np.int32)
    coarsen_channels = np.array([0], dtype=np.int64)
    coarsen_tolerances = np.array([1.0], dtype=np.float64)
    coarsen_owner = np.array([[0, 1], [2, 3]], dtype=np.int32)

    coarsen_count = int(_COARSEN_PARALLEL_FILL_THRESHOLD)
    coarsen_fill_groups = np.arange(4 * coarsen_count, dtype=np.int32).reshape(coarsen_count, 4)
    coarsen_accepted_groups = np.arange(coarsen_count, dtype=np.int32)
    coarsen_accept_count = int(_COARSEN_PARALLEL_ACCEPT_THRESHOLD)
    coarsen_accept_level_idx = np.ones(4 * coarsen_accept_count, dtype=np.int16)
    coarsen_accept_protected = np.zeros(4 * coarsen_accept_count, dtype=bool)
    coarsen_accept_group_keys = np.zeros((coarsen_accept_count, 5), dtype=np.int64)
    coarsen_accept_groups = np.arange(4 * coarsen_accept_count, dtype=np.int32).reshape(coarsen_accept_count, 4)
    coarsen_accept_channels = np.array([0], dtype=np.int64)
    coarsen_accept_tolerances = np.array([1.0], dtype=np.float64)
    coarsen_accept_owner = np.full((2, 2), -1, dtype=np.int32)

    export_values = np.zeros((8, flat_channels, warm_patch_h, warm_patch_w), dtype=np.float32)
    export_slots = np.array([3, 1, 4, 0], dtype=np.int32)
    export_flat = np.empty((4, flat_channels, warm_patch_h, warm_patch_w), dtype=np.float32)
    export_sequence = np.empty((4, flat_channels, 1, warm_patch_h, warm_patch_w), dtype=np.float32)
    if warm_copy:
        values = np.zeros((capacity, flat_channels, warm_patch_h, warm_patch_w), dtype=np.float32)
        tile_ix = np.zeros(capacity, dtype=np.int32)
        tile_iy = np.zeros(capacity, dtype=np.int32)
        level_idx = np.zeros(capacity, dtype=np.int16)
        x_idx = np.zeros(capacity, dtype=np.int32)
        y_idx = np.zeros(capacity, dtype=np.int32)
        active_slots = np.zeros(capacity, dtype=np.int32)
        refine_mask = np.zeros(capacity, dtype=bool)
        refine_mask[0] = True
        protected = np.zeros(capacity, dtype=bool)

        _active_refine_slots_jit(
            values,
            tile_ix,
            tile_iy,
            level_idx,
            x_idx,
            y_idx,
            active_slots,
            1,
            1,
            refine_mask,
            protected,
            False,
        )

        refine_count = int(_REFINE_PARALLEL_FILL_THRESHOLD)
        warm_values = np.zeros((5 * refine_count, flat_channels, warm_patch_h, warm_patch_w), dtype=np.float32)
        parent_slots = np.arange(refine_count, dtype=np.int32)
        child_starts = refine_count + 4 * np.arange(refine_count, dtype=np.int32)
        _fill_refined_child_values_jit(warm_values, parent_slots, child_starts, np.int64(refine_count))

        protected_values = np.zeros((protected_count, flat_channels, warm_patch_h, warm_patch_w), dtype=np.float32)
        _active_protected_mask_parallel_jit(
            protected_values,
            protected_slots,
            protected_count,
            protected_channels,
            protected_tolerances,
        )

        _active_coarsen_once_jit(
            coarsen_values,
            coarsen_tile_ix,
            coarsen_tile_iy,
            coarsen_level_idx,
            coarsen_x_idx,
            coarsen_y_idx,
            coarsen_active_slots,
            4,
            4,
            coarsen_protected,
            coarsen_group_keys,
            coarsen_sibling_groups,
            coarsen_channels,
            coarsen_tolerances,
            coarsen_owner,
            1,
        )

        coarsen_fill_values = np.zeros(
            (5 * coarsen_count, flat_channels, warm_patch_h, warm_patch_w),
            dtype=np.float32,
        )
        _fill_accepted_coarsened_parent_values_jit(
            coarsen_fill_values,
            coarsen_fill_groups,
            coarsen_accepted_groups,
            np.int64(coarsen_count),
            4 * coarsen_count,
        )

        coarsen_accept_values = np.zeros(
            (4 * coarsen_accept_count, flat_channels, warm_patch_h, warm_patch_w),
            dtype=np.float32,
        )
        _active_coarsen_acceptance_mask_parallel_jit(
            coarsen_accept_values,
            coarsen_accept_level_idx,
            coarsen_accept_protected,
            coarsen_accept_group_keys,
            coarsen_accept_groups,
            coarsen_accept_channels,
            coarsen_accept_tolerances,
            coarsen_accept_owner,
            1,
        )

        if warm_flat_export:
            _copy_ordered_active_values_jit(export_values, export_slots, export_flat)
        if warm_sequence_export:
            _copy_ordered_active_values_to_sequence_jit(
                export_values,
                export_slots,
                export_sequence,
                flat_channels,
                1,
            )
    if warm_source:
        source_sequence = np.zeros((capacity, flat_channels, 1, warm_patch_h, warm_patch_w), dtype=np.float32)
        source_values = np.zeros((capacity, flat_channels, warm_patch_h, warm_patch_w), dtype=np.float32)
        source_tile_ix = np.zeros(capacity, dtype=np.int32)
        source_tile_iy = np.zeros(capacity, dtype=np.int32)
        source_level_idx = np.zeros(capacity, dtype=np.int16)
        source_x_idx = np.zeros(capacity, dtype=np.int32)
        source_y_idx = np.zeros(capacity, dtype=np.int32)
        source_active_slots = np.zeros(capacity, dtype=np.int32)
        source_refine_mask = np.zeros(capacity, dtype=bool)
        source_refine_mask[0] = True
        source_protected = np.zeros(capacity, dtype=bool)
        _active_refine_slots_from_source_jit(
            source_sequence,
            source_values,
            1,
            flat_channels,
            source_tile_ix,
            source_tile_iy,
            source_level_idx,
            source_x_idx,
            source_y_idx,
            source_active_slots,
            1,
            1,
            source_refine_mask,
            source_protected,
            False,
        )

        source_refine_count = int(_SOURCE_REFINE_PARALLEL_FILL_THRESHOLD)
        source_fill_sequence = np.zeros(
            (source_refine_count, flat_channels, 1, warm_patch_h, warm_patch_w),
            dtype=np.float32,
        )
        source_fill_values = np.zeros(
            (5 * source_refine_count, flat_channels, warm_patch_h, warm_patch_w),
            dtype=np.float32,
        )
        source_parent_slots = np.arange(source_refine_count, dtype=np.int32)
        source_child_starts = source_refine_count + 4 * np.arange(source_refine_count, dtype=np.int32)
        _fill_refined_child_values_from_source_jit(
            source_fill_sequence,
            source_fill_values,
            source_refine_count,
            flat_channels,
            source_parent_slots,
            source_child_starts,
            np.int64(source_refine_count),
        )

        source_protected_values = np.zeros(
            (protected_count, flat_channels, warm_patch_h, warm_patch_w),
            dtype=np.float32,
        )
        source_protected_sequence = np.zeros(
            (protected_count, flat_channels, 1, warm_patch_h, warm_patch_w),
            dtype=np.float32,
        )
        _active_protected_mask_from_source_jit(
            source_protected_sequence,
            source_protected_values,
            protected_count,
            flat_channels,
            protected_count,
            protected_slots,
            protected_count,
            protected_channels,
            protected_tolerances,
        )

        source_coarsen_count = int(_COARSEN_PARALLEL_FILL_THRESHOLD)
        source_coarsen_sequence = np.zeros(
            (4 * source_coarsen_count, flat_channels, 1, warm_patch_h, warm_patch_w),
            dtype=np.float32,
        )
        source_coarsen_values = np.zeros(
            (5 * source_coarsen_count, flat_channels, warm_patch_h, warm_patch_w),
            dtype=np.float32,
        )
        source_coarsen_groups = np.arange(4 * source_coarsen_count, dtype=np.int32).reshape(
            source_coarsen_count,
            4,
        )
        source_accepted_groups = np.arange(source_coarsen_count, dtype=np.int32)
        _fill_accepted_coarsened_parent_values_from_source_jit(
            source_coarsen_sequence,
            source_coarsen_values,
            4 * source_coarsen_count,
            flat_channels,
            source_coarsen_groups,
            source_accepted_groups,
            np.int64(source_coarsen_count),
            4 * source_coarsen_count,
        )

        source_accept_values = np.zeros(
            (4 * coarsen_accept_count, flat_channels, warm_patch_h, warm_patch_w),
            dtype=np.float32,
        )
        source_accept_sequence = np.zeros(
            (4 * coarsen_accept_count, flat_channels, 1, warm_patch_h, warm_patch_w),
            dtype=np.float32,
        )
        _active_coarsen_acceptance_mask_from_source_jit(
            source_accept_sequence,
            source_accept_values,
            4 * coarsen_accept_count,
            flat_channels,
            coarsen_accept_level_idx,
            coarsen_accept_protected,
            coarsen_accept_group_keys,
            coarsen_accept_groups,
            coarsen_accept_channels,
            coarsen_accept_tolerances,
            coarsen_accept_owner,
            1,
        )

        _active_coarsen_once_from_source_jit(
            source_sequence,
            coarsen_values.copy(),
            4,
            flat_channels,
            coarsen_tile_ix.copy(),
            coarsen_tile_iy.copy(),
            coarsen_level_idx.copy(),
            coarsen_x_idx.copy(),
            coarsen_y_idx.copy(),
            coarsen_active_slots.copy(),
            4,
            4,
            coarsen_protected.copy(),
            coarsen_group_keys,
            coarsen_sibling_groups,
            coarsen_channels,
            coarsen_tolerances,
            coarsen_owner,
            1,
        )

        if warm_flat_export:
            _copy_ordered_source_active_values_jit(
                source_sequence,
                export_values,
                4,
                flat_channels,
                export_slots,
                export_flat,
            )
        if warm_sequence_export:
            _copy_ordered_source_active_values_to_sequence_jit(
                source_sequence,
                export_values,
                4,
                export_slots,
                export_sequence,
            )


def _active_refine_slots(
    active: ActiveTopology,
    refine_slots: np.ndarray,
    protected: Optional[np.ndarray] = None,
) -> Tuple[ActiveTopology, Optional[np.ndarray]]:
    refine_slots = np.unique(np.asarray(refine_slots, dtype=np.int32))
    if refine_slots.size == 0:
        return active, protected

    capacity = int(active.active_slots.shape[0])
    refine_mask = np.zeros(capacity, dtype=bool)
    refine_mask[refine_slots] = True
    active_slots = active.active_slots[:active.active_count]
    refine_count = int(np.count_nonzero(refine_mask[active_slots]))
    if refine_count == 0:
        return active, protected

    _ensure_active_append_capacity(
        active,
        append_count=4 * refine_count,
        new_active_count=active.active_count + 3 * refine_count,
    )

    protected_array = protected if protected is not None else np.empty(0, dtype=bool)
    new_active_slots, out_count, next_slot = _active_refine_slots_jit(
        active.values,
        active.tile_ix,
        active.tile_iy,
        active.level_idx,
        active.x_idx,
        active.y_idx,
        active.active_slots,
        int(active.active_count),
        int(active.next_slot),
        refine_mask,
        protected_array,
        protected is not None,
    )
    active.active_slots = new_active_slots
    active.active_count = out_count
    active.next_slot = next_slot
    return active, protected


def _active_refine_slots_from_source(
    active: SequenceSourceActiveTopology,
    refine_slots: np.ndarray,
    protected: Optional[np.ndarray] = None,
) -> Tuple[SequenceSourceActiveTopology, Optional[np.ndarray]]:
    refine_slots = np.unique(np.asarray(refine_slots, dtype=np.int32))
    if refine_slots.size == 0:
        return active, protected

    capacity = int(active.active_slots.shape[0])
    refine_mask = np.zeros(capacity, dtype=bool)
    refine_mask[refine_slots] = True
    active_slots = active.active_slots[:active.active_count]
    refine_count = int(np.count_nonzero(refine_mask[active_slots]))
    if refine_count == 0:
        return active, protected

    _ensure_active_append_capacity(
        active,
        append_count=4 * refine_count,
        new_active_count=active.active_count + 3 * refine_count,
    )

    protected_array = protected if protected is not None else np.empty(0, dtype=bool)
    new_active_slots, out_count, next_slot = _active_refine_slots_from_source_jit(
        active.source_sequence,
        active.values,
        int(active.initial_slots),
        int(active.sequence_channels),
        active.tile_ix,
        active.tile_iy,
        active.level_idx,
        active.x_idx,
        active.y_idx,
        active.active_slots,
        int(active.active_count),
        int(active.next_slot),
        refine_mask,
        protected_array,
        protected is not None,
    )
    active.active_slots = new_active_slots
    active.active_count = out_count
    active.next_slot = next_slot
    return active, protected


def _active_assemble_coarsened_topology(
    active: ActiveTopology,
    protected: np.ndarray,
    accepted: Dict[Tuple[int, int, int, int, int], Dict[str, Any]],
) -> Tuple[ActiveTopology, np.ndarray]:
    if not accepted:
        return active, protected

    accepted_items = list(accepted.items())
    skipped: set[int] = set()
    for _, item in accepted_items:
        skipped.update(int(child) for child in item["children"])

    parent_count = len(accepted_items)
    keep_count = sum(1 for slot in active.active_slots[:active.active_count] if int(slot) not in skipped)
    new_active_count = keep_count + parent_count
    _ensure_active_append_capacity(active, append_count=parent_count, new_active_count=new_active_count)

    new_active_slots = np.empty_like(active.active_slots)
    out_i = 0
    for slot_value in active.active_slots[:active.active_count]:
        slot = int(slot_value)
        if slot in skipped:
            protected[slot] = False
            continue
        new_active_slots[out_i] = slot
        out_i += 1

    next_slot = int(active.next_slot)
    for key, item in accepted_items:
        parent_slot = next_slot
        next_slot += 1
        parent_tile_ix, parent_tile_iy, parent_level, parent_x_idx, parent_y_idx = key
        active.values[parent_slot] = item["value"]
        active.tile_ix[parent_slot] = parent_tile_ix
        active.tile_iy[parent_slot] = parent_tile_iy
        active.level_idx[parent_slot] = parent_level
        active.x_idx[parent_slot] = parent_x_idx
        active.y_idx[parent_slot] = parent_y_idx
        protected[parent_slot] = False
        new_active_slots[out_i] = parent_slot
        out_i += 1

    active.active_slots = new_active_slots
    active.active_count = out_i
    active.next_slot = next_slot
    return active, protected


def _ensure_2to1_balance(
    top: FlatTopology,
    capacity: int,
    protected: Optional[np.ndarray] = None,
) -> Tuple[FlatTopology, Optional[np.ndarray]]:
    while True:
        owner = _build_owner_grid(top)
        refine_indices = _find_balance_refinements(top, owner)
        if refine_indices.size == 0:
            return top, protected
        _ensure_capacity(top.values.shape[0] + 3 * refine_indices.size, capacity)
        mask = np.zeros(top.values.shape[0], dtype=bool)
        mask[refine_indices] = True
        top, protected = _refine_masked(top, mask, protected=protected)


def _active_ensure_2to1_balance(
    active: ActiveTopology,
    protected: Optional[np.ndarray] = None,
) -> Tuple[ActiveTopology, Optional[np.ndarray]]:
    active, protected, _, _ = _active_ensure_2to1_balance_counted(active, protected=protected)
    return active, protected


def _active_ensure_2to1_balance_counted(
    active: ActiveTopology,
    protected: Optional[np.ndarray] = None,
) -> Tuple[ActiveTopology, Optional[np.ndarray], int, int]:
    refine_calls = 0
    refined_parents = 0
    while True:
        owner = _build_active_owner_grid(active)
        refine_slots = _find_active_balance_refinements(active, owner)
        if refine_slots.size == 0:
            return active, protected, refine_calls, refined_parents
        refine_calls += 1
        refined_parents += int(refine_slots.size)
        active, protected = _active_refine_slots(active, refine_slots, protected=protected)


def _active_ensure_2to1_balance_counted_from_source(
    active: SequenceSourceActiveTopology,
    protected: Optional[np.ndarray] = None,
) -> Tuple[SequenceSourceActiveTopology, Optional[np.ndarray], int, int]:
    refine_calls = 0
    refined_parents = 0
    while True:
        owner = _build_active_owner_grid(active)
        refine_slots = _find_active_balance_refinements(active, owner)
        if refine_slots.size == 0:
            return active, protected, refine_calls, refined_parents
        refine_calls += 1
        refined_parents += int(refine_slots.size)
        active, protected = _active_refine_slots_from_source(active, refine_slots, protected=protected)


def _active_overlapping_ancestor_slots(active: ActiveTopology) -> np.ndarray:
    slots = active.active_slots[: active.active_count]
    x0, y0, scale = _active_leaf_fine_bounds(active, slots)
    ancestors: list[int] = []
    for i, slot in enumerate(slots):
        si = int(scale[i])
        for j in range(slots.shape[0]):
            if i == j or si <= int(scale[j]):
                continue
            if (
                int(x0[i]) <= int(x0[j])
                and int(y0[i]) <= int(y0[j])
                and int(x0[j]) + int(scale[j]) <= int(x0[i]) + si
                and int(y0[j]) + int(scale[j]) <= int(y0[i]) + si
            ):
                ancestors.append(int(slot))
                break
    return np.unique(np.asarray(ancestors, dtype=np.int32))


def _active_fully_covered_ancestor_slots(active: ActiveTopology) -> np.ndarray:
    slots = active.active_slots[: active.active_count]
    x0, y0, scale = _active_leaf_fine_bounds(active, slots)
    covered: list[int] = []
    for i, slot in enumerate(slots):
        si = int(scale[i])
        covered_cells: set[tuple[int, int]] = set()
        for j in range(slots.shape[0]):
            sj = int(scale[j])
            if i == j or si <= sj:
                continue
            if (
                int(x0[i]) <= int(x0[j])
                and int(y0[i]) <= int(y0[j])
                and int(x0[j]) + sj <= int(x0[i]) + si
                and int(y0[j]) + sj <= int(y0[i]) + si
            ):
                for yy in range(int(y0[j]), int(y0[j]) + sj):
                    for xx in range(int(x0[j]), int(x0[j]) + sj):
                        covered_cells.add((xx, yy))
                if len(covered_cells) == si * si:
                    break
        if len(covered_cells) == si * si:
            covered.append(int(slot))
    return np.unique(np.asarray(covered, dtype=np.int32))


def _active_remove_slots(
    active: ActiveTopology,
    protected: Optional[np.ndarray],
    remove_slots: np.ndarray,
) -> Tuple[ActiveTopology, Optional[np.ndarray]]:
    remove_slots = np.unique(np.asarray(remove_slots, dtype=np.int32))
    if remove_slots.size == 0:
        return active, protected

    remove_mask = np.zeros(int(active.active_slots.shape[0]), dtype=bool)
    remove_mask[remove_slots] = True
    current = active.active_slots[: active.active_count]
    kept = current[~remove_mask[current]]
    active.active_slots[: kept.shape[0]] = kept
    active.active_count = int(kept.shape[0])
    if protected is not None:
        protected[remove_slots] = False
    return active, protected


def _active_refine_slots_for_repair(
    active: ActiveTopology,
    refine_slots: np.ndarray,
    protected: Optional[np.ndarray],
) -> Tuple[ActiveTopology, Optional[np.ndarray]]:
    if isinstance(active, SequenceSourceActiveTopology):
        return _active_refine_slots_from_source(active, refine_slots, protected=protected)
    return _active_refine_slots(active, refine_slots, protected=protected)


def _active_owner_grid_after_overlap_repair(
    active: ActiveTopology,
    protected: Optional[np.ndarray],
) -> Tuple[ActiveTopology, Optional[np.ndarray], np.ndarray, int, int, int]:
    repair_invalid_slots = 0
    repair_calls = 0
    repair_parents = 0
    max_iterations = int(active.domain["max_level_idx"]) + 1

    for _ in range(max_iterations):
        try:
            owner = _build_active_owner_grid(active)
        except ValueError as exc:
            if str(exc) != "overlapping_leaves":
                raise _ArrayFallback("invalid_topology") from exc
            covered_ancestors = _active_fully_covered_ancestor_slots(active)
            if covered_ancestors.size:
                repair_invalid_slots += int(covered_ancestors.size)
                active, protected = _active_remove_slots(active, protected, covered_ancestors)
                continue
            ancestor_slots = _active_overlapping_ancestor_slots(active)
            if ancestor_slots.size == 0:
                raise _ArrayFallback("invalid_topology") from exc
            repair_invalid_slots += int(ancestor_slots.size)
            active, protected = _active_refine_slots_for_repair(active, ancestor_slots, protected)
            repair_calls += 1
            repair_parents += int(ancestor_slots.size)
            continue

        return active, protected, owner, repair_invalid_slots, repair_calls, repair_parents

    raise _ArrayFallback("invalid_topology")


def _repair_active_balance_or_raise(
    active: ActiveTopology,
    protected: Optional[np.ndarray],
) -> Tuple[ActiveTopology, Optional[np.ndarray], Dict[str, int]]:
    active, protected, owner, overlap_invalid, overlap_calls, overlap_parents = _active_owner_grid_after_overlap_repair(
        active,
        protected,
    )
    invalid_slots = _find_active_balance_refinements(active, owner)
    if invalid_slots.size == 0:
        return active, protected, {
            "repair_invalid_slots": int(overlap_invalid),
            "repair_refine_calls": int(overlap_calls),
            "repair_refine_parents": int(overlap_parents),
        }
    active, protected, repair_calls, repair_parents = _active_ensure_2to1_balance_counted(
        active,
        protected=protected,
    )
    owner = _build_active_owner_grid(active)
    remaining = _find_active_balance_refinements(active, owner)
    if remaining.size:
        raise _ArrayFallback("invalid_topology")
    return active, protected, {
        "repair_invalid_slots": int(overlap_invalid) + int(invalid_slots.size),
        "repair_refine_calls": int(overlap_calls) + int(repair_calls),
        "repair_refine_parents": int(overlap_parents) + int(repair_parents),
    }


def _repair_source_active_balance_or_raise(
    active: SequenceSourceActiveTopology,
    protected: Optional[np.ndarray],
) -> Tuple[SequenceSourceActiveTopology, Optional[np.ndarray], Dict[str, int]]:
    active, protected, owner, overlap_invalid, overlap_calls, overlap_parents = _active_owner_grid_after_overlap_repair(
        active,
        protected,
    )
    invalid_slots = _find_active_balance_refinements(active, owner)
    if invalid_slots.size == 0:
        return active, protected, {
            "repair_invalid_slots": int(overlap_invalid),
            "repair_refine_calls": int(overlap_calls),
            "repair_refine_parents": int(overlap_parents),
        }
    active, protected, repair_calls, repair_parents = _active_ensure_2to1_balance_counted_from_source(
        active,
        protected=protected,
    )
    owner = _build_active_owner_grid(active)
    remaining = _find_active_balance_refinements(active, owner)
    if remaining.size:
        raise _ArrayFallback("invalid_topology")
    return active, protected, {
        "repair_invalid_slots": int(overlap_invalid) + int(invalid_slots.size),
        "repair_refine_calls": int(overlap_calls) + int(repair_calls),
        "repair_refine_parents": int(overlap_parents) + int(repair_parents),
    }


def _protected_mask(top: FlatTopology, channels: list[int], tolerances: np.ndarray) -> np.ndarray:
    protected = np.zeros(top.values.shape[0], dtype=bool)
    for tol_idx, ch in enumerate(channels):
        patch = top.values[:, ch, :, :]
        protected |= np.ptp(patch, axis=(1, 2)) >= tolerances[tol_idx]
    return protected


@njit(boundscheck=False, cache=True)
def _active_protected_mask_sequential_jit(
    values: np.ndarray,
    active_slots: np.ndarray,
    active_count: int,
    channels: np.ndarray,
    tolerances: np.ndarray,
) -> np.ndarray:
    protected = np.zeros(values.shape[0], dtype=np.bool_)
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    for tol_idx in range(channels.shape[0]):
        ch = int(channels[tol_idx])
        tol = float(tolerances[tol_idx])
        for pos in range(active_count):
            slot = int(active_slots[pos])
            v_min = values[slot, ch, 0, 0]
            v_max = v_min
            for yy in range(patch_h):
                for xx in range(patch_w):
                    val = values[slot, ch, yy, xx]
                    if val < v_min:
                        v_min = val
                    elif val > v_max:
                        v_max = val
            if float(v_max - v_min) >= tol:
                protected[slot] = True
    return protected


@njit(boundscheck=False, parallel=True, cache=True)
def _active_protected_mask_parallel_jit(
    values: np.ndarray,
    active_slots: np.ndarray,
    active_count: int,
    channels: np.ndarray,
    tolerances: np.ndarray,
) -> np.ndarray:
    protected = np.zeros(values.shape[0], dtype=np.bool_)
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    for pos in prange(active_count):
        slot = int(active_slots[pos])
        is_protected = False
        for tol_idx in range(channels.shape[0]):
            ch = int(channels[tol_idx])
            tol = float(tolerances[tol_idx])
            v_min = values[slot, ch, 0, 0]
            v_max = v_min
            for yy in range(patch_h):
                for xx in range(patch_w):
                    val = values[slot, ch, yy, xx]
                    if val < v_min:
                        v_min = val
                    elif val > v_max:
                        v_max = val
            if float(v_max - v_min) >= tol:
                is_protected = True
                break
        if is_protected:
            protected[slot] = True
    return protected


@njit(boundscheck=False, cache=True)
def _active_protected_mask_jit(
    values: np.ndarray,
    active_slots: np.ndarray,
    active_count: int,
    channels: np.ndarray,
    tolerances: np.ndarray,
) -> np.ndarray:
    if active_count >= _ACTIVE_PROTECTED_MASK_PARALLEL_THRESHOLD:
        return _active_protected_mask_parallel_jit(values, active_slots, active_count, channels, tolerances)
    return _active_protected_mask_sequential_jit(values, active_slots, active_count, channels, tolerances)


def _active_protected_mask(active: ActiveTopology, channels: list[int], tolerances: np.ndarray) -> np.ndarray:
    return _active_protected_mask_jit(
        active.values,
        active.active_slots,
        int(active.active_count),
        np.asarray(channels, dtype=np.int64),
        tolerances.astype(np.float64, copy=False),
    )


@njit(boundscheck=False, cache=True)
def _active_protected_mask_from_source_sequential_jit(
    source_sequence: np.ndarray,
    values: np.ndarray,
    initial_slots: int,
    sequence_channels: int,
    slot_capacity: int,
    active_slots: np.ndarray,
    active_count: int,
    channels: np.ndarray,
    tolerances: np.ndarray,
) -> np.ndarray:
    protected = np.zeros(slot_capacity, dtype=np.bool_)
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    for tol_idx in range(channels.shape[0]):
        ch = int(channels[tol_idx])
        tol = float(tolerances[tol_idx])
        for pos in range(active_count):
            slot = int(active_slots[pos])
            v_min = _source_or_workspace_value(
                source_sequence, values, initial_slots, sequence_channels, slot, ch, 0, 0
            )
            v_max = v_min
            for yy in range(patch_h):
                for xx in range(patch_w):
                    val = _source_or_workspace_value(
                        source_sequence, values, initial_slots, sequence_channels, slot, ch, yy, xx
                    )
                    if val < v_min:
                        v_min = val
                    elif val > v_max:
                        v_max = val
            if float(v_max - v_min) >= tol:
                protected[slot] = True
    return protected


@njit(boundscheck=False, parallel=True, cache=True)
def _active_protected_mask_from_source_parallel_jit(
    source_sequence: np.ndarray,
    values: np.ndarray,
    initial_slots: int,
    sequence_channels: int,
    slot_capacity: int,
    active_slots: np.ndarray,
    active_count: int,
    channels: np.ndarray,
    tolerances: np.ndarray,
) -> np.ndarray:
    protected = np.zeros(slot_capacity, dtype=np.bool_)
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    for pos in prange(active_count):
        slot = int(active_slots[pos])
        is_protected = False
        for tol_idx in range(channels.shape[0]):
            ch = int(channels[tol_idx])
            tol = float(tolerances[tol_idx])
            v_min = _source_or_workspace_value(
                source_sequence, values, initial_slots, sequence_channels, slot, ch, 0, 0
            )
            v_max = v_min
            for yy in range(patch_h):
                for xx in range(patch_w):
                    val = _source_or_workspace_value(
                        source_sequence, values, initial_slots, sequence_channels, slot, ch, yy, xx
                    )
                    if val < v_min:
                        v_min = val
                    elif val > v_max:
                        v_max = val
            if float(v_max - v_min) >= tol:
                is_protected = True
                break
        if is_protected:
            protected[slot] = True
    return protected


@njit(boundscheck=False, cache=True)
def _active_protected_mask_from_source_jit(
    source_sequence: np.ndarray,
    values: np.ndarray,
    initial_slots: int,
    sequence_channels: int,
    slot_capacity: int,
    active_slots: np.ndarray,
    active_count: int,
    channels: np.ndarray,
    tolerances: np.ndarray,
) -> np.ndarray:
    if active_count >= _ACTIVE_PROTECTED_MASK_PARALLEL_THRESHOLD:
        return _active_protected_mask_from_source_parallel_jit(
            source_sequence,
            values,
            initial_slots,
            sequence_channels,
            slot_capacity,
            active_slots,
            active_count,
            channels,
            tolerances,
        )
    return _active_protected_mask_from_source_sequential_jit(
        source_sequence,
        values,
        initial_slots,
        sequence_channels,
        slot_capacity,
        active_slots,
        active_count,
        channels,
        tolerances,
    )


def _active_protected_mask_from_source(
    active: SequenceSourceActiveTopology,
    channels: list[int],
    tolerances: np.ndarray,
) -> np.ndarray:
    return _active_protected_mask_from_source_jit(
        active.source_sequence,
        active.values,
        int(active.initial_slots),
        int(active.sequence_channels),
        int(active.active_slots.shape[0]),
        active.active_slots,
        int(active.active_count),
        np.asarray(channels, dtype=np.int64),
        tolerances.astype(np.float64, copy=False),
    )


@njit(boundscheck=False)
def _dilate_mask_from_owner_jit(
    protected: np.ndarray,
    owner: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    scale: np.ndarray,
    passes: int,
) -> np.ndarray:
    current = protected.copy()
    ny, nx = owner.shape
    for _ in range(max(0, passes)):
        next_mask = current.copy()
        for leaf_idx in range(current.shape[0]):
            if not current[leaf_idx]:
                continue

            xi = int(x0[leaf_idx])
            yi = int(y0[leaf_idx])
            si = int(scale[leaf_idx])

            if xi > 0:
                xx = xi - 1
                for yy in range(yi, yi + si):
                    neighbor = int(owner[yy, xx])
                    if neighbor >= 0:
                        next_mask[neighbor] = True
            if xi + si < nx:
                xx = xi + si
                for yy in range(yi, yi + si):
                    neighbor = int(owner[yy, xx])
                    if neighbor >= 0:
                        next_mask[neighbor] = True
            if yi > 0:
                yy = yi - 1
                for xx in range(xi, xi + si):
                    neighbor = int(owner[yy, xx])
                    if neighbor >= 0:
                        next_mask[neighbor] = True
            if yi + si < ny:
                yy = yi + si
                for xx in range(xi, xi + si):
                    neighbor = int(owner[yy, xx])
                    if neighbor >= 0:
                        next_mask[neighbor] = True
        current = next_mask
    return current


@njit(boundscheck=False)
def _dilate_mask_for_indices_jit(
    protected_indices: np.ndarray,
    protected: np.ndarray,
    owner: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    scale: np.ndarray,
    passes: int,
) -> np.ndarray:
    current = protected.copy()
    ny, nx = owner.shape
    active = np.empty(protected.shape[0], dtype=np.int32)
    active_count = 0
    for i in range(protected_indices.shape[0]):
        active[active_count] = int(protected_indices[i])
        active_count += 1

    for _ in range(max(0, passes)):
        next_active = np.empty(protected.shape[0], dtype=np.int32)
        next_count = 0
        for active_pos in range(active_count):
            leaf_idx = int(active[active_pos])
            xi = int(x0[leaf_idx])
            yi = int(y0[leaf_idx])
            si = int(scale[leaf_idx])

            if xi > 0:
                xx = xi - 1
                for yy in range(yi, yi + si):
                    neighbor = int(owner[yy, xx])
                    if neighbor >= 0 and not current[neighbor]:
                        current[neighbor] = True
                        next_active[next_count] = neighbor
                        next_count += 1
            if xi + si < nx:
                xx = xi + si
                for yy in range(yi, yi + si):
                    neighbor = int(owner[yy, xx])
                    if neighbor >= 0 and not current[neighbor]:
                        current[neighbor] = True
                        next_active[next_count] = neighbor
                        next_count += 1
            if yi > 0:
                yy = yi - 1
                for xx in range(xi, xi + si):
                    neighbor = int(owner[yy, xx])
                    if neighbor >= 0 and not current[neighbor]:
                        current[neighbor] = True
                        next_active[next_count] = neighbor
                        next_count += 1
            if yi + si < ny:
                yy = yi + si
                for xx in range(xi, xi + si):
                    neighbor = int(owner[yy, xx])
                    if neighbor >= 0 and not current[neighbor]:
                        current[neighbor] = True
                        next_active[next_count] = neighbor
                        next_count += 1

        active = next_active
        active_count = next_count
        if active_count == 0:
            break

    return current


@njit(boundscheck=False)
def _dilate_active_mask_for_slots_jit(
    protected_slots: np.ndarray,
    protected: np.ndarray,
    owner: np.ndarray,
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    level_idx: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    max_level: int,
    passes: int,
) -> np.ndarray:
    current = protected.copy()
    ny, nx = owner.shape
    active = np.empty(protected.shape[0], dtype=np.int32)
    active_count = 0
    for i in range(protected_slots.shape[0]):
        active[active_count] = int(protected_slots[i])
        active_count += 1

    for _ in range(max(0, passes)):
        next_active = np.empty(protected.shape[0], dtype=np.int32)
        next_count = 0
        for active_pos in range(active_count):
            slot = int(active[active_pos])
            level = int(level_idx[slot])
            scale = 1 << (max_level - level)
            xi = (int(tile_ix[slot]) << max_level) + int(x_idx[slot]) * scale
            yi = (int(tile_iy[slot]) << max_level) + int(y_idx[slot]) * scale

            if xi > 0:
                xx = xi - 1
                for yy in range(yi, yi + scale):
                    neighbor = int(owner[yy, xx])
                    if neighbor >= 0 and not current[neighbor]:
                        current[neighbor] = True
                        next_active[next_count] = neighbor
                        next_count += 1
            if xi + scale < nx:
                xx = xi + scale
                for yy in range(yi, yi + scale):
                    neighbor = int(owner[yy, xx])
                    if neighbor >= 0 and not current[neighbor]:
                        current[neighbor] = True
                        next_active[next_count] = neighbor
                        next_count += 1
            if yi > 0:
                yy = yi - 1
                for xx in range(xi, xi + scale):
                    neighbor = int(owner[yy, xx])
                    if neighbor >= 0 and not current[neighbor]:
                        current[neighbor] = True
                        next_active[next_count] = neighbor
                        next_count += 1
            if yi + scale < ny:
                yy = yi + scale
                for xx in range(xi, xi + scale):
                    neighbor = int(owner[yy, xx])
                    if neighbor >= 0 and not current[neighbor]:
                        current[neighbor] = True
                        next_active[next_count] = neighbor
                        next_count += 1

        active = next_active
        active_count = next_count
        if active_count == 0:
            break

    return current


@njit(boundscheck=False)
def _mark_active_neighbor_slot_jit(
    current: np.ndarray,
    next_active: np.ndarray,
    next_count: int,
    neighbor: int,
) -> int:
    if neighbor >= 0 and not current[neighbor]:
        current[neighbor] = True
        next_active[next_count] = neighbor
        return next_count + 1
    return next_count


@njit(boundscheck=False)
def _mark_active_sibling_side_jit(
    current: np.ndarray,
    next_active: np.ndarray,
    next_count: int,
    owner: np.ndarray,
    tile_ix: int,
    tile_iy: int,
    level: int,
    x_idx: int,
    y_idx: int,
    max_level: int,
    side: int,
) -> int:
    scale = 1 << (max_level - level)
    x0 = (tile_ix << max_level) + x_idx * scale
    y0 = (tile_iy << max_level) + y_idx * scale

    if side == 0:  # west side
        xx = x0
        for yy in range(y0, y0 + scale):
            next_count = _mark_active_neighbor_slot_jit(current, next_active, next_count, int(owner[yy, xx]))
    elif side == 1:  # east side
        xx = x0 + scale - 1
        for yy in range(y0, y0 + scale):
            next_count = _mark_active_neighbor_slot_jit(current, next_active, next_count, int(owner[yy, xx]))
    elif side == 2:  # south side
        yy = y0
        for xx in range(x0, x0 + scale):
            next_count = _mark_active_neighbor_slot_jit(current, next_active, next_count, int(owner[yy, xx]))
    else:  # north side
        yy = y0 + scale - 1
        for xx in range(x0, x0 + scale):
            next_count = _mark_active_neighbor_slot_jit(current, next_active, next_count, int(owner[yy, xx]))
    return next_count


@njit(boundscheck=False)
def _mark_active_object_style_face_neighbors_jit(
    current: np.ndarray,
    next_active: np.ndarray,
    next_count: int,
    owner: np.ndarray,
    tile_ix_arr: np.ndarray,
    tile_iy_arr: np.ndarray,
    level_idx_arr: np.ndarray,
    x_idx_arr: np.ndarray,
    y_idx_arr: np.ndarray,
    max_level: int,
    slot: int,
    direction: int,
) -> int:
    fine = 1 << max_level
    nx_tiles = owner.shape[1] // fine
    ny_tiles = owner.shape[0] // fine
    tile_ix = int(tile_ix_arr[slot])
    tile_iy = int(tile_iy_arr[slot])
    level = int(level_idx_arr[slot])
    x_idx = int(x_idx_arr[slot])
    y_idx = int(y_idx_arr[slot])

    cur_level = level
    cur_x = x_idx
    cur_y = y_idx
    while cur_level > 0:
        if direction == 0:  # west
            if cur_x & 1:
                return _mark_active_sibling_side_jit(
                    current, next_active, next_count, owner, tile_ix, tile_iy,
                    cur_level, cur_x - 1, cur_y, max_level, 1,
                )
        elif direction == 1:  # east
            if (cur_x & 1) == 0:
                return _mark_active_sibling_side_jit(
                    current, next_active, next_count, owner, tile_ix, tile_iy,
                    cur_level, cur_x + 1, cur_y, max_level, 0,
                )
        elif direction == 2:  # south
            if cur_y & 1:
                return _mark_active_sibling_side_jit(
                    current, next_active, next_count, owner, tile_ix, tile_iy,
                    cur_level, cur_x, cur_y - 1, max_level, 3,
                )
        else:  # north
            if (cur_y & 1) == 0:
                return _mark_active_sibling_side_jit(
                    current, next_active, next_count, owner, tile_ix, tile_iy,
                    cur_level, cur_x, cur_y + 1, max_level, 2,
                )

        cur_x //= 2
        cur_y //= 2
        cur_level -= 1

    if direction == 0 and tile_ix > 0:
        return _mark_active_sibling_side_jit(
            current, next_active, next_count, owner, tile_ix - 1, tile_iy, 0, 0, 0, max_level, 1
        )
    if direction == 1 and tile_ix + 1 < nx_tiles:
        return _mark_active_sibling_side_jit(
            current, next_active, next_count, owner, tile_ix + 1, tile_iy, 0, 0, 0, max_level, 0
        )
    if direction == 2 and tile_iy > 0:
        return _mark_active_sibling_side_jit(
            current, next_active, next_count, owner, tile_ix, tile_iy - 1, 0, 0, 0, max_level, 3
        )
    if direction == 3 and tile_iy + 1 < ny_tiles:
        return _mark_active_sibling_side_jit(
            current, next_active, next_count, owner, tile_ix, tile_iy + 1, 0, 0, 0, max_level, 2
        )
    return next_count


@njit(boundscheck=False, cache=True)
def _dilate_active_mask_object_neighbors_jit(
    protected_slots: np.ndarray,
    protected: np.ndarray,
    owner: np.ndarray,
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    level_idx: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    max_level: int,
    passes: int,
) -> np.ndarray:
    current = protected.copy()
    active = np.empty(protected.shape[0], dtype=np.int32)
    active_count = 0
    for i in range(protected_slots.shape[0]):
        active[active_count] = int(protected_slots[i])
        active_count += 1

    for _ in range(max(0, passes)):
        next_active = np.empty(protected.shape[0], dtype=np.int32)
        next_count = 0
        for active_pos in range(active_count):
            slot = int(active[active_pos])
            for direction in range(4):
                next_count = _mark_active_object_style_face_neighbors_jit(
                    current,
                    next_active,
                    next_count,
                    owner,
                    tile_ix,
                    tile_iy,
                    level_idx,
                    x_idx,
                    y_idx,
                    max_level,
                    slot,
                    direction,
                )

        active = next_active
        active_count = next_count
        if active_count == 0:
            break

    return current


def _dilate_protected(top: FlatTopology, protected: np.ndarray, passes: int) -> np.ndarray:
    if passes <= 0 or not np.any(protected):
        return protected
    owner = _build_owner_grid(top)
    x0, y0, scale = _leaf_fine_bounds(top)
    protected_indices = np.flatnonzero(protected).astype(np.int32)
    return _dilate_mask_for_indices_jit(protected_indices, protected, owner, x0, y0, scale, int(passes))


def _active_dilate_protected(active: ActiveTopology, protected: np.ndarray, passes: int) -> np.ndarray:
    if passes <= 0 or not np.any(protected):
        return protected
    owner = _build_active_owner_grid(active)
    protected_slots = np.flatnonzero(protected).astype(np.int32)
    return _dilate_active_mask_object_neighbors_jit(
        protected_slots,
        protected,
        owner,
        active.tile_ix,
        active.tile_iy,
        active.level_idx,
        active.x_idx,
        active.y_idx,
        int(active.domain["max_level_idx"]),
        int(passes),
    )



@njit(boundscheck=False)
def _protected_region_refine_mask_for_indices_jit(
    protected_indices: np.ndarray,
    protected: np.ndarray,
    level_idx: np.ndarray,
    owner: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    scale: np.ndarray,
    max_level: int,
) -> np.ndarray:
    mask = np.zeros(protected.shape[0], dtype=np.bool_)
    ny, nx = owner.shape

    for protected_pos in range(protected_indices.shape[0]):
        leaf_idx = int(protected_indices[protected_pos])
        leaf_level = int(level_idx[leaf_idx])
        if leaf_level <= 0:
            continue

        xi = int(x0[leaf_idx])
        yi = int(y0[leaf_idx])
        si = int(scale[leaf_idx])

        if xi > 0:
            xx = xi - 1
            for yy in range(yi, yi + si):
                neighbor = int(owner[yy, xx])
                if (
                    neighbor >= 0
                    and protected[neighbor]
                    and int(level_idx[neighbor]) < leaf_level
                    and int(level_idx[neighbor]) < max_level
                ):
                    mask[neighbor] = True
        if xi + si < nx:
            xx = xi + si
            for yy in range(yi, yi + si):
                neighbor = int(owner[yy, xx])
                if (
                    neighbor >= 0
                    and protected[neighbor]
                    and int(level_idx[neighbor]) < leaf_level
                    and int(level_idx[neighbor]) < max_level
                ):
                    mask[neighbor] = True
        if yi > 0:
            yy = yi - 1
            for xx in range(xi, xi + si):
                neighbor = int(owner[yy, xx])
                if (
                    neighbor >= 0
                    and protected[neighbor]
                    and int(level_idx[neighbor]) < leaf_level
                    and int(level_idx[neighbor]) < max_level
                ):
                    mask[neighbor] = True
        if yi + si < ny:
            yy = yi + si
            for xx in range(xi, xi + si):
                neighbor = int(owner[yy, xx])
                if (
                    neighbor >= 0
                    and protected[neighbor]
                    and int(level_idx[neighbor]) < leaf_level
                    and int(level_idx[neighbor]) < max_level
                ):
                    mask[neighbor] = True

    return mask


@njit(boundscheck=False)
def _active_protected_region_refine_slots_jit(
    protected_slots: np.ndarray,
    protected: np.ndarray,
    level_idx: np.ndarray,
    owner: np.ndarray,
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    max_level: int,
) -> np.ndarray:
    mask = np.zeros(protected.shape[0], dtype=np.bool_)
    ny, nx = owner.shape

    for protected_pos in range(protected_slots.shape[0]):
        slot = int(protected_slots[protected_pos])
        leaf_level = int(level_idx[slot])
        if leaf_level <= 0:
            continue

        scale = 1 << (max_level - leaf_level)
        xi = (int(tile_ix[slot]) << max_level) + int(x_idx[slot]) * scale
        yi = (int(tile_iy[slot]) << max_level) + int(y_idx[slot]) * scale

        if xi > 0:
            xx = xi - 1
            for yy in range(yi, yi + scale):
                neighbor = int(owner[yy, xx])
                if (
                    neighbor >= 0
                    and protected[neighbor]
                    and int(level_idx[neighbor]) < leaf_level
                    and int(level_idx[neighbor]) < max_level
                ):
                    mask[neighbor] = True
        if xi + scale < nx:
            xx = xi + scale
            for yy in range(yi, yi + scale):
                neighbor = int(owner[yy, xx])
                if (
                    neighbor >= 0
                    and protected[neighbor]
                    and int(level_idx[neighbor]) < leaf_level
                    and int(level_idx[neighbor]) < max_level
                ):
                    mask[neighbor] = True
        if yi > 0:
            yy = yi - 1
            for xx in range(xi, xi + scale):
                neighbor = int(owner[yy, xx])
                if (
                    neighbor >= 0
                    and protected[neighbor]
                    and int(level_idx[neighbor]) < leaf_level
                    and int(level_idx[neighbor]) < max_level
                ):
                    mask[neighbor] = True
        if yi + scale < ny:
            yy = yi + scale
            for xx in range(xi, xi + scale):
                neighbor = int(owner[yy, xx])
                if (
                    neighbor >= 0
                    and protected[neighbor]
                    and int(level_idx[neighbor]) < leaf_level
                    and int(level_idx[neighbor]) < max_level
                ):
                    mask[neighbor] = True

    return mask


@njit(boundscheck=False)
def _mark_active_lower_protected_side_jit(
    mask: np.ndarray,
    protected: np.ndarray,
    level_idx_arr: np.ndarray,
    owner: np.ndarray,
    tile_ix: int,
    tile_iy: int,
    level: int,
    x_idx: int,
    y_idx: int,
    max_level: int,
    leaf_level: int,
    side: int,
) -> None:
    scale = 1 << (max_level - level)
    x0 = (tile_ix << max_level) + x_idx * scale
    y0 = (tile_iy << max_level) + y_idx * scale

    if side == 0:
        xx = x0
        for yy in range(y0, y0 + scale):
            neighbor = int(owner[yy, xx])
            if neighbor >= 0 and protected[neighbor] and int(level_idx_arr[neighbor]) < leaf_level and int(level_idx_arr[neighbor]) < max_level:
                mask[neighbor] = True
    elif side == 1:
        xx = x0 + scale - 1
        for yy in range(y0, y0 + scale):
            neighbor = int(owner[yy, xx])
            if neighbor >= 0 and protected[neighbor] and int(level_idx_arr[neighbor]) < leaf_level and int(level_idx_arr[neighbor]) < max_level:
                mask[neighbor] = True
    elif side == 2:
        yy = y0
        for xx in range(x0, x0 + scale):
            neighbor = int(owner[yy, xx])
            if neighbor >= 0 and protected[neighbor] and int(level_idx_arr[neighbor]) < leaf_level and int(level_idx_arr[neighbor]) < max_level:
                mask[neighbor] = True
    else:
        yy = y0 + scale - 1
        for xx in range(x0, x0 + scale):
            neighbor = int(owner[yy, xx])
            if neighbor >= 0 and protected[neighbor] and int(level_idx_arr[neighbor]) < leaf_level and int(level_idx_arr[neighbor]) < max_level:
                mask[neighbor] = True


@njit(boundscheck=False)
def _mark_active_object_style_lower_protected_neighbors_jit(
    mask: np.ndarray,
    protected: np.ndarray,
    level_idx_arr: np.ndarray,
    owner: np.ndarray,
    tile_ix_arr: np.ndarray,
    tile_iy_arr: np.ndarray,
    x_idx_arr: np.ndarray,
    y_idx_arr: np.ndarray,
    max_level: int,
    slot: int,
    direction: int,
) -> None:
    fine = 1 << max_level
    nx_tiles = owner.shape[1] // fine
    ny_tiles = owner.shape[0] // fine
    tile_ix = int(tile_ix_arr[slot])
    tile_iy = int(tile_iy_arr[slot])
    leaf_level = int(level_idx_arr[slot])
    cur_level = leaf_level
    cur_x = int(x_idx_arr[slot])
    cur_y = int(y_idx_arr[slot])

    while cur_level > 0:
        if direction == 0:
            if cur_x & 1:
                _mark_active_lower_protected_side_jit(
                    mask, protected, level_idx_arr, owner, tile_ix, tile_iy,
                    cur_level, cur_x - 1, cur_y, max_level, leaf_level, 1,
                )
                return
        elif direction == 1:
            if (cur_x & 1) == 0:
                _mark_active_lower_protected_side_jit(
                    mask, protected, level_idx_arr, owner, tile_ix, tile_iy,
                    cur_level, cur_x + 1, cur_y, max_level, leaf_level, 0,
                )
                return
        elif direction == 2:
            if cur_y & 1:
                _mark_active_lower_protected_side_jit(
                    mask, protected, level_idx_arr, owner, tile_ix, tile_iy,
                    cur_level, cur_x, cur_y - 1, max_level, leaf_level, 3,
                )
                return
        else:
            if (cur_y & 1) == 0:
                _mark_active_lower_protected_side_jit(
                    mask, protected, level_idx_arr, owner, tile_ix, tile_iy,
                    cur_level, cur_x, cur_y + 1, max_level, leaf_level, 2,
                )
                return

        cur_x //= 2
        cur_y //= 2
        cur_level -= 1

    if direction == 0 and tile_ix > 0:
        _mark_active_lower_protected_side_jit(
            mask, protected, level_idx_arr, owner, tile_ix - 1, tile_iy, 0, 0, 0, max_level, leaf_level, 1
        )
    elif direction == 1 and tile_ix + 1 < nx_tiles:
        _mark_active_lower_protected_side_jit(
            mask, protected, level_idx_arr, owner, tile_ix + 1, tile_iy, 0, 0, 0, max_level, leaf_level, 0
        )
    elif direction == 2 and tile_iy > 0:
        _mark_active_lower_protected_side_jit(
            mask, protected, level_idx_arr, owner, tile_ix, tile_iy - 1, 0, 0, 0, max_level, leaf_level, 3
        )
    elif direction == 3 and tile_iy + 1 < ny_tiles:
        _mark_active_lower_protected_side_jit(
            mask, protected, level_idx_arr, owner, tile_ix, tile_iy + 1, 0, 0, 0, max_level, leaf_level, 2
        )


@njit(boundscheck=False, cache=True)
def _active_protected_region_refine_slots_object_neighbors_jit(
    protected_slots: np.ndarray,
    protected: np.ndarray,
    level_idx: np.ndarray,
    owner: np.ndarray,
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    max_level: int,
) -> np.ndarray:
    mask = np.zeros(protected.shape[0], dtype=np.bool_)
    for protected_pos in range(protected_slots.shape[0]):
        slot = int(protected_slots[protected_pos])
        if int(level_idx[slot]) <= 0:
            continue
        for direction in range(4):
            _mark_active_object_style_lower_protected_neighbors_jit(
                mask,
                protected,
                level_idx,
                owner,
                tile_ix,
                tile_iy,
                x_idx,
                y_idx,
                max_level,
                slot,
                direction,
            )
    return mask


def _expand_protected_region(
    top: FlatTopology,
    protected: np.ndarray,
    passes: int,
    capacity: int,
) -> Tuple[FlatTopology, np.ndarray]:
    max_level = int(top.domain["max_level_idx"])
    for _ in range(max(0, passes)):
        owner = _build_owner_grid(top)
        x0, y0, scale = _leaf_fine_bounds(top)
        protected_indices = np.flatnonzero(protected).astype(np.int32)
        mask = _protected_region_refine_mask_for_indices_jit(
            protected_indices,
            protected,
            top.level_idx,
            owner,
            x0,
            y0,
            scale,
            max_level,
        )
        refine_count = int(np.count_nonzero(mask))
        if refine_count == 0:
            break
        _ensure_capacity(top.values.shape[0] + 3 * refine_count, capacity)
        top, protected = _refine_masked(top, mask, protected=protected)
        top, protected = _ensure_2to1_balance(top, capacity, protected=protected)
    return top, protected


def _active_expand_protected_region(
    active: ActiveTopology,
    protected: np.ndarray,
    passes: int,
) -> Tuple[ActiveTopology, np.ndarray]:
    active, protected, _, _, _, _ = _active_expand_protected_region_counted(active, protected, passes)
    return active, protected


def _active_expand_protected_region_counted(
    active: ActiveTopology,
    protected: np.ndarray,
    passes: int,
) -> Tuple[ActiveTopology, np.ndarray, int, int, int, int]:
    max_level = int(active.domain["max_level_idx"])
    protected_refine_calls = 0
    protected_refine_parents = 0
    balance_refine_calls = 0
    balance_refine_parents = 0
    for _ in range(max(0, passes)):
        owner = _build_active_owner_grid(active)
        protected_slots = np.flatnonzero(protected).astype(np.int32)
        refine_mask = _active_protected_region_refine_slots_object_neighbors_jit(
            protected_slots,
            protected,
            active.level_idx,
            owner,
            active.tile_ix,
            active.tile_iy,
            active.x_idx,
            active.y_idx,
            max_level,
        )
        refine_slots = np.flatnonzero(refine_mask).astype(np.int32, copy=False)
        if refine_slots.size == 0:
            break
        protected_refine_calls += 1
        protected_refine_parents += int(refine_slots.size)
        active, protected = _active_refine_slots(active, refine_slots, protected=protected)
        active, protected, balance_calls, balance_parents = _active_ensure_2to1_balance_counted(
            active,
            protected=protected,
        )
        balance_refine_calls += int(balance_calls)
        balance_refine_parents += int(balance_parents)
    return (
        active,
        protected,
        protected_refine_calls,
        protected_refine_parents,
        balance_refine_calls,
        balance_refine_parents,
    )


def _active_expand_protected_region_counted_from_source(
    active: SequenceSourceActiveTopology,
    protected: np.ndarray,
    passes: int,
) -> Tuple[SequenceSourceActiveTopology, np.ndarray, int, int, int, int]:
    max_level = int(active.domain["max_level_idx"])
    protected_refine_calls = 0
    protected_refine_parents = 0
    balance_refine_calls = 0
    balance_refine_parents = 0
    for _ in range(max(0, passes)):
        owner = _build_active_owner_grid(active)
        protected_slots = np.flatnonzero(protected).astype(np.int32)
        refine_mask = _active_protected_region_refine_slots_object_neighbors_jit(
            protected_slots,
            protected,
            active.level_idx,
            owner,
            active.tile_ix,
            active.tile_iy,
            active.x_idx,
            active.y_idx,
            max_level,
        )
        refine_slots = np.flatnonzero(refine_mask).astype(np.int32, copy=False)
        if refine_slots.size == 0:
            break
        protected_refine_calls += 1
        protected_refine_parents += int(refine_slots.size)
        active, protected = _active_refine_slots_from_source(active, refine_slots, protected=protected)
        active, protected, balance_calls, balance_parents = _active_ensure_2to1_balance_counted_from_source(
            active,
            protected=protected,
        )
        balance_refine_calls += int(balance_calls)
        balance_refine_parents += int(balance_parents)
    return (
        active,
        protected,
        protected_refine_calls,
        protected_refine_parents,
        balance_refine_calls,
        balance_refine_parents,
    )


def _coarsen_values(children: Dict[int, np.ndarray]) -> np.ndarray:
    return _coarsen_values_direct(children)


@njit(boundscheck=False)
def _coarsen_quadrants_mean_jit(
    v_sw: np.ndarray,
    v_se: np.ndarray,
    v_nw: np.ndarray,
    v_ne: np.ndarray,
) -> np.ndarray:
    channels, patch_h, patch_w = v_sw.shape
    out = np.empty_like(v_sw)
    half_h = patch_h // 2
    half_w = patch_w // 2
    for ch in range(channels):
        for yy in range(patch_h):
            use_north = yy >= half_h
            child_y = 2 * yy
            if use_north:
                child_y = 2 * (yy - half_h)
            for xx in range(patch_w):
                use_east = xx >= half_w
                child_x = 2 * xx
                if use_east:
                    child_x = 2 * (xx - half_w)
                if use_north:
                    child = v_ne if use_east else v_nw
                else:
                    child = v_se if use_east else v_sw
                out[ch, yy, xx] = 0.25 * (
                    child[ch, child_y, child_x]
                    + child[ch, child_y + 1, child_x]
                    + child[ch, child_y, child_x + 1]
                    + child[ch, child_y + 1, child_x + 1]
                )
    return out


def _coarsen_values_direct(children: Dict[int, np.ndarray]) -> np.ndarray:
    v_sw = children[0]
    return _coarsen_quadrants_mean_jit(v_sw, children[1], children[2], children[3]).astype(v_sw.dtype, copy=False)


def _parent_side_has_too_fine_neighbor(
    top: FlatTopology,
    owner: np.ndarray,
    child_indices: list[int],
    parent_level: int,
    parent_x_idx: int,
    parent_y_idx: int,
    tile_ix: int,
    tile_iy: int,
) -> bool:
    max_level = int(top.domain["max_level_idx"])
    parent_size = 1 << (max_level - parent_level)
    parent_x0 = (int(tile_ix) << max_level) + int(parent_x_idx) * parent_size
    parent_y0 = (int(tile_iy) << max_level) + int(parent_y_idx) * parent_size
    exclude = set(int(i) for i in child_indices)
    neighbors = _side_neighbor_indices(owner, parent_x0, parent_y0, parent_size, exclude)
    return bool(neighbors.size and np.any(top.level_idx[neighbors].astype(np.int64) >= parent_level + 2))


def _active_parent_side_has_too_fine_neighbor(
    active: ActiveTopology,
    owner: np.ndarray,
    child_slots: list[int],
    parent_level: int,
    parent_x_idx: int,
    parent_y_idx: int,
    tile_ix: int,
    tile_iy: int,
) -> bool:
    return bool(
        _active_parent_side_has_too_fine_neighbor_jit(
            active.level_idx,
            owner,
            np.asarray(child_slots, dtype=np.int32),
            int(parent_level),
            int(parent_x_idx),
            int(parent_y_idx),
            int(tile_ix),
            int(tile_iy),
            int(active.domain["max_level_idx"]),
        )
    )


@njit(boundscheck=False)
def _contains_child_slot_jit(child_slots: np.ndarray, slot: int) -> bool:
    for i in range(child_slots.shape[0]):
        if int(child_slots[i]) == slot:
            return True
    return False


@njit(boundscheck=False)
def _active_side_has_min_level_neighbor_excluding_jit(
    level_idx_arr: np.ndarray,
    owner: np.ndarray,
    child_slots: np.ndarray,
    tile_ix: int,
    tile_iy: int,
    level: int,
    x_idx: int,
    y_idx: int,
    max_level: int,
    min_level: int,
    side: int,
) -> bool:
    scale = 1 << (max_level - level)
    x0 = (tile_ix << max_level) + x_idx * scale
    y0 = (tile_iy << max_level) + y_idx * scale

    if side == 0:
        xx = x0
        for yy in range(y0, y0 + scale):
            neighbor = int(owner[yy, xx])
            if (
                neighbor >= 0
                and not _contains_child_slot_jit(child_slots, neighbor)
                and int(level_idx_arr[neighbor]) >= min_level
            ):
                return True
    elif side == 1:
        xx = x0 + scale - 1
        for yy in range(y0, y0 + scale):
            neighbor = int(owner[yy, xx])
            if (
                neighbor >= 0
                and not _contains_child_slot_jit(child_slots, neighbor)
                and int(level_idx_arr[neighbor]) >= min_level
            ):
                return True
    elif side == 2:
        yy = y0
        for xx in range(x0, x0 + scale):
            neighbor = int(owner[yy, xx])
            if (
                neighbor >= 0
                and not _contains_child_slot_jit(child_slots, neighbor)
                and int(level_idx_arr[neighbor]) >= min_level
            ):
                return True
    else:
        yy = y0 + scale - 1
        for xx in range(x0, x0 + scale):
            neighbor = int(owner[yy, xx])
            if (
                neighbor >= 0
                and not _contains_child_slot_jit(child_slots, neighbor)
                and int(level_idx_arr[neighbor]) >= min_level
            ):
                return True
    return False


@njit(boundscheck=False)
def _active_object_style_cell_face_has_min_level_neighbor_excluding_jit(
    level_idx_arr: np.ndarray,
    owner: np.ndarray,
    child_slots: np.ndarray,
    tile_ix: int,
    tile_iy: int,
    level: int,
    x_idx: int,
    y_idx: int,
    max_level: int,
    min_level: int,
    direction: int,
) -> bool:
    fine = 1 << max_level
    nx_tiles = owner.shape[1] // fine
    ny_tiles = owner.shape[0] // fine
    cur_level = level
    cur_x = x_idx
    cur_y = y_idx

    while cur_level > 0:
        if direction == 0:
            if cur_x & 1:
                return _active_side_has_min_level_neighbor_excluding_jit(
                    level_idx_arr, owner, child_slots, tile_ix, tile_iy, cur_level, cur_x - 1, cur_y, max_level, min_level, 1
                )
        elif direction == 1:
            if (cur_x & 1) == 0:
                return _active_side_has_min_level_neighbor_excluding_jit(
                    level_idx_arr, owner, child_slots, tile_ix, tile_iy, cur_level, cur_x + 1, cur_y, max_level, min_level, 0
                )
        elif direction == 2:
            if cur_y & 1:
                return _active_side_has_min_level_neighbor_excluding_jit(
                    level_idx_arr, owner, child_slots, tile_ix, tile_iy, cur_level, cur_x, cur_y - 1, max_level, min_level, 3
                )
        else:
            if (cur_y & 1) == 0:
                return _active_side_has_min_level_neighbor_excluding_jit(
                    level_idx_arr, owner, child_slots, tile_ix, tile_iy, cur_level, cur_x, cur_y + 1, max_level, min_level, 2
                )

        cur_x //= 2
        cur_y //= 2
        cur_level -= 1

    if direction == 0 and tile_ix > 0:
        return _active_side_has_min_level_neighbor_excluding_jit(
            level_idx_arr, owner, child_slots, tile_ix - 1, tile_iy, 0, 0, 0, max_level, min_level, 1
        )
    if direction == 1 and tile_ix + 1 < nx_tiles:
        return _active_side_has_min_level_neighbor_excluding_jit(
            level_idx_arr, owner, child_slots, tile_ix + 1, tile_iy, 0, 0, 0, max_level, min_level, 0
        )
    if direction == 2 and tile_iy > 0:
        return _active_side_has_min_level_neighbor_excluding_jit(
            level_idx_arr, owner, child_slots, tile_ix, tile_iy - 1, 0, 0, 0, max_level, min_level, 3
        )
    if direction == 3 and tile_iy + 1 < ny_tiles:
        return _active_side_has_min_level_neighbor_excluding_jit(
            level_idx_arr, owner, child_slots, tile_ix, tile_iy + 1, 0, 0, 0, max_level, min_level, 2
        )
    return False


@njit(boundscheck=False)
def _active_parent_side_has_too_fine_neighbor_jit(
    level_idx: np.ndarray,
    owner: np.ndarray,
    child_slots: np.ndarray,
    parent_level: int,
    parent_x_idx: int,
    parent_y_idx: int,
    tile_ix: int,
    tile_iy: int,
    max_level: int,
) -> bool:
    too_fine_level = parent_level + 2
    child_level = parent_level + 1
    for quad in range(child_slots.shape[0]):
        child_x = parent_x_idx * 2 + (quad & 1)
        child_y = parent_y_idx * 2 + (quad >> 1)
        for direction in range(4):
            if _active_object_style_cell_face_has_min_level_neighbor_excluding_jit(
                level_idx,
                owner,
                child_slots,
                tile_ix,
                tile_iy,
                child_level,
                child_x,
                child_y,
                max_level,
                too_fine_level,
                direction,
            ):
                return True
    return False


@njit(boundscheck=False)
def _active_complete_sibling_groups_linear_jit(
    active_slots: np.ndarray,
    active_count: int,
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    level_idx: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    keys = np.empty((active_count, 5), dtype=np.int64)
    children = np.full((active_count, 4), -1, dtype=np.int32)
    masks = np.zeros(active_count, dtype=np.int8)
    valid = np.ones(active_count, dtype=np.bool_)
    group_count = 0

    for pos in range(active_count):
        slot = int(active_slots[pos])
        level = int(level_idx[slot])
        if level <= 0:
            continue
        key0 = int(tile_ix[slot])
        key1 = int(tile_iy[slot])
        key2 = level - 1
        key3 = int(x_idx[slot]) // 2
        key4 = int(y_idx[slot]) // 2

        group = -1
        for group_idx in range(group_count):
            if (
                keys[group_idx, 0] == key0
                and keys[group_idx, 1] == key1
                and keys[group_idx, 2] == key2
                and keys[group_idx, 3] == key3
                and keys[group_idx, 4] == key4
            ):
                group = group_idx
                break

        if group < 0:
            group = group_count
            group_count += 1
            keys[group, 0] = key0
            keys[group, 1] = key1
            keys[group, 2] = key2
            keys[group, 3] = key3
            keys[group, 4] = key4

        quad = (int(x_idx[slot]) & 1) + 2 * (int(y_idx[slot]) & 1)
        quad_bit = np.int8(1 << quad)
        if masks[group] & quad_bit:
            valid[group] = False
        children[group, quad] = slot
        masks[group] = np.int8(masks[group] | quad_bit)

    complete_count = 0
    for group_idx in range(group_count):
        if valid[group_idx] and masks[group_idx] == 15:
            keys[complete_count] = keys[group_idx]
            children[complete_count] = children[group_idx]
            complete_count += 1

    return keys, children, complete_count


@njit(boundscheck=False, inline="always")
def _sibling_group_hash_jit(key0: int, key1: int, key2: int, key3: int, key4: int) -> int:
    h = key0 * 73856093
    h = h ^ (key1 * 19349663)
    h = h ^ (key2 * 83492791)
    h = h ^ (key3 * 2654435761)
    h = h ^ (key4 * 97531)
    return h


@njit(boundscheck=False, cache=True)
def _active_complete_sibling_groups_hashed_jit(
    active_slots: np.ndarray,
    active_count: int,
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    level_idx: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    keys = np.empty((active_count, 5), dtype=np.int64)
    children = np.full((active_count, 4), -1, dtype=np.int32)
    masks = np.zeros(active_count, dtype=np.int8)
    valid = np.ones(active_count, dtype=np.bool_)
    table_size = 1
    while table_size < active_count * 4:
        table_size *= 2
    table = np.full(table_size, -1, dtype=np.int32)
    table_mask = table_size - 1
    group_count = 0

    for pos in range(active_count):
        slot = int(active_slots[pos])
        level = int(level_idx[slot])
        if level <= 0:
            continue
        key0 = int(tile_ix[slot])
        key1 = int(tile_iy[slot])
        key2 = level - 1
        key3 = int(x_idx[slot]) // 2
        key4 = int(y_idx[slot]) // 2

        table_idx = _sibling_group_hash_jit(key0, key1, key2, key3, key4) & table_mask
        group = -1
        while True:
            group_idx = int(table[table_idx])
            if group_idx < 0:
                group = group_count
                group_count += 1
                keys[group, 0] = key0
                keys[group, 1] = key1
                keys[group, 2] = key2
                keys[group, 3] = key3
                keys[group, 4] = key4
                table[table_idx] = group
                break
            if (
                keys[group_idx, 0] == key0
                and keys[group_idx, 1] == key1
                and keys[group_idx, 2] == key2
                and keys[group_idx, 3] == key3
                and keys[group_idx, 4] == key4
            ):
                group = group_idx
                break
            table_idx = (table_idx + 1) & table_mask

        quad = (int(x_idx[slot]) & 1) + 2 * (int(y_idx[slot]) & 1)
        quad_bit = np.int8(1 << quad)
        if masks[group] & quad_bit:
            valid[group] = False
        children[group, quad] = slot
        masks[group] = np.int8(masks[group] | quad_bit)

    complete_count = 0
    for group_idx in range(group_count):
        if valid[group_idx] and masks[group_idx] == 15:
            keys[complete_count] = keys[group_idx]
            children[complete_count] = children[group_idx]
            complete_count += 1

    return keys, children, complete_count


@njit(boundscheck=False, cache=True)
def _active_complete_sibling_groups_jit(
    active_slots: np.ndarray,
    active_count: int,
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    level_idx: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    return _active_complete_sibling_groups_hashed_jit(
        active_slots,
        active_count,
        tile_ix,
        tile_iy,
        level_idx,
        x_idx,
        y_idx,
    )


def _active_complete_sibling_groups(active: ActiveTopology) -> Tuple[np.ndarray, np.ndarray]:
    keys, children, count = _active_complete_sibling_groups_jit(
        active.active_slots,
        int(active.active_count),
        active.tile_ix,
        active.tile_iy,
        active.level_idx,
        active.x_idx,
        active.y_idx,
    )
    return keys[:count].copy(), children[:count].copy()


def _assemble_coarsened_topology(
    top: FlatTopology,
    protected: np.ndarray,
    accepted: Dict[Tuple[int, int, int, int, int], Dict[str, Any]],
) -> Tuple[FlatTopology, np.ndarray]:
    accepted_items = list(accepted.items())
    total_children = sum(len(item["children"]) for _, item in accepted_items)
    skip_indices = np.empty(total_children, dtype=np.intp)
    offset = 0
    for _, item in accepted_items:
        children = item["children"]
        child_count = len(children)
        skip_indices[offset:offset + child_count] = children
        offset += child_count
    skip_indices.sort()

    parent_count = len(accepted_items)
    keep_count = top.values.shape[0] - skip_indices.shape[0]
    new_count = keep_count + parent_count

    values = np.empty((new_count,) + top.values.shape[1:], dtype=top.values.dtype)
    tile_ix = np.empty(new_count, dtype=np.int32)
    tile_iy = np.empty(new_count, dtype=np.int32)
    level_idx = np.empty(new_count, dtype=np.int16)
    x_idx = np.empty(new_count, dtype=np.int32)
    y_idx = np.empty(new_count, dtype=np.int32)
    new_protected = np.zeros(new_count, dtype=bool)

    out_i = 0
    src_start = 0
    for skip_idx in skip_indices:
        src_end = int(skip_idx)
        if src_end > src_start:
            span = src_end - src_start
            dst_end = out_i + span
            values[out_i:dst_end] = top.values[src_start:src_end]
            tile_ix[out_i:dst_end] = top.tile_ix[src_start:src_end]
            tile_iy[out_i:dst_end] = top.tile_iy[src_start:src_end]
            level_idx[out_i:dst_end] = top.level_idx[src_start:src_end]
            x_idx[out_i:dst_end] = top.x_idx[src_start:src_end]
            y_idx[out_i:dst_end] = top.y_idx[src_start:src_end]
            new_protected[out_i:dst_end] = protected[src_start:src_end]
            out_i = dst_end
        src_start = src_end + 1
    if src_start < top.values.shape[0]:
        span = top.values.shape[0] - src_start
        dst_end = out_i + span
        values[out_i:dst_end] = top.values[src_start:]
        tile_ix[out_i:dst_end] = top.tile_ix[src_start:]
        tile_iy[out_i:dst_end] = top.tile_iy[src_start:]
        level_idx[out_i:dst_end] = top.level_idx[src_start:]
        x_idx[out_i:dst_end] = top.x_idx[src_start:]
        y_idx[out_i:dst_end] = top.y_idx[src_start:]
        new_protected[out_i:dst_end] = protected[src_start:]
        out_i = dst_end

    for parent_offset, (key, item) in enumerate(accepted_items):
        out_i = keep_count + parent_offset
        parent_tile_ix, parent_tile_iy, parent_level, parent_x_idx, parent_y_idx = key
        values[out_i] = item["value"]
        tile_ix[out_i] = parent_tile_ix
        tile_iy[out_i] = parent_tile_iy
        level_idx[out_i] = parent_level
        x_idx[out_i] = parent_x_idx
        y_idx[out_i] = parent_y_idx

    coarsened = FlatTopology(values, tile_ix, tile_iy, level_idx, x_idx, y_idx, dict(top.domain), top.cell_scale_mode)
    return coarsened, new_protected


def _coarsen_once(
    top: FlatTopology,
    protected: np.ndarray,
    channels: list[int],
    tolerances: np.ndarray,
) -> Tuple[FlatTopology, np.ndarray, bool]:
    groups: Dict[Tuple[int, int, int, int, int], list[int]] = {}
    for i in range(top.values.shape[0]):
        level = int(top.level_idx[i])
        if level <= 0:
            continue
        key = (
            int(top.tile_ix[i]),
            int(top.tile_iy[i]),
            level - 1,
            int(top.x_idx[i]) // 2,
            int(top.y_idx[i]) // 2,
        )
        groups.setdefault(key, []).append(i)

    owner = _build_owner_grid(top)
    accepted: Dict[Tuple[int, int, int, int, int], Dict[str, Any]] = {}
    for key, child_indices in groups.items():
        if len(child_indices) != 4:
            continue
        if np.any(protected[child_indices]):
            continue

        quadrants = {}
        valid_quadrants = True
        for idx in child_indices:
            dx = int(top.x_idx[idx]) & 1
            dy = int(top.y_idx[idx]) & 1
            quad = dx + 2 * dy
            if quad in quadrants:
                valid_quadrants = False
                break
            quadrants[quad] = idx
        if not valid_quadrants or set(quadrants) != {0, 1, 2, 3}:
            continue

        reject = False
        for tol_idx, ch in enumerate(channels):
            patch_stack = top.values[list(quadrants.values()), ch, :, :]
            if float(np.max(patch_stack) - np.min(patch_stack)) >= float(tolerances[tol_idx]):
                reject = True
                break
        if reject:
            continue

        tile_ix, tile_iy, parent_level, parent_x_idx, parent_y_idx = key
        if _parent_side_has_too_fine_neighbor(
            top,
            owner,
            child_indices,
            parent_level,
            parent_x_idx,
            parent_y_idx,
            tile_ix,
            tile_iy,
        ):
            continue

        accepted[key] = {
            "children": child_indices,
            "value": _coarsen_values({quad: top.values[idx] for quad, idx in quadrants.items()}),
        }

    if not accepted:
        return top, protected, False

    coarsened, new_protected = _assemble_coarsened_topology(top, protected, accepted)
    return coarsened, new_protected, True


@njit(boundscheck=False, cache=True)
def _fill_accepted_coarsened_parent_values_sequential_jit(
    values: np.ndarray,
    sibling_groups: np.ndarray,
    accepted_groups: np.ndarray,
    accepted_count: int,
    parent_start: int,
) -> None:
    channels_total = values.shape[1]
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    half_h = patch_h // 2
    half_w = patch_w // 2
    for accepted_pos in range(accepted_count):
        group_idx = int(accepted_groups[accepted_pos])
        child_slots = sibling_groups[group_idx]
        parent_slot = parent_start + accepted_pos

        for ch in range(channels_total):
            for yy in range(patch_h):
                use_north = yy >= half_h
                child_y = 2 * yy
                if use_north:
                    child_y = 2 * (yy - half_h)
                for xx in range(patch_w):
                    use_east = xx >= half_w
                    child_x = 2 * xx
                    if use_east:
                        child_x = 2 * (xx - half_w)
                    quad = 0
                    if use_east:
                        quad += 1
                    if use_north:
                        quad += 2
                    child_slot = int(child_slots[quad])
                    values[parent_slot, ch, yy, xx] = 0.25 * (
                        values[child_slot, ch, child_y, child_x]
                        + values[child_slot, ch, child_y + 1, child_x]
                        + values[child_slot, ch, child_y, child_x + 1]
                        + values[child_slot, ch, child_y + 1, child_x + 1]
                    )


@njit(boundscheck=False, parallel=True, cache=True)
def _fill_accepted_coarsened_parent_values_parallel_jit(
    values: np.ndarray,
    sibling_groups: np.ndarray,
    accepted_groups: np.ndarray,
    accepted_count: int,
    parent_start: int,
) -> None:
    channels_total = values.shape[1]
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    half_h = patch_h // 2
    half_w = patch_w // 2
    for accepted_pos in prange(accepted_count):
        group_idx = int(accepted_groups[accepted_pos])
        child_slots = sibling_groups[group_idx]
        parent_slot = parent_start + accepted_pos

        for ch in range(channels_total):
            for yy in range(patch_h):
                use_north = yy >= half_h
                child_y = 2 * yy
                if use_north:
                    child_y = 2 * (yy - half_h)
                for xx in range(patch_w):
                    use_east = xx >= half_w
                    child_x = 2 * xx
                    if use_east:
                        child_x = 2 * (xx - half_w)
                    quad = 0
                    if use_east:
                        quad += 1
                    if use_north:
                        quad += 2
                    child_slot = int(child_slots[quad])
                    values[parent_slot, ch, yy, xx] = 0.25 * (
                        values[child_slot, ch, child_y, child_x]
                        + values[child_slot, ch, child_y + 1, child_x]
                        + values[child_slot, ch, child_y, child_x + 1]
                        + values[child_slot, ch, child_y + 1, child_x + 1]
                    )


@njit(boundscheck=False, cache=True)
def _fill_accepted_coarsened_parent_values_jit(
    values: np.ndarray,
    sibling_groups: np.ndarray,
    accepted_groups: np.ndarray,
    accepted_count: int,
    parent_start: int,
) -> None:
    if accepted_count >= _COARSEN_PARALLEL_FILL_THRESHOLD:
        _fill_accepted_coarsened_parent_values_parallel_jit(
            values,
            sibling_groups,
            accepted_groups,
            accepted_count,
            parent_start,
        )
    else:
        _fill_accepted_coarsened_parent_values_sequential_jit(
            values,
            sibling_groups,
            accepted_groups,
            accepted_count,
            parent_start,
        )


@njit(boundscheck=False, cache=True)
def _active_coarsen_acceptance_mask_sequential_jit(
    values: np.ndarray,
    level_idx: np.ndarray,
    protected: np.ndarray,
    group_keys: np.ndarray,
    sibling_groups: np.ndarray,
    channels: np.ndarray,
    tolerances: np.ndarray,
    owner: np.ndarray,
    max_level: int,
) -> np.ndarray:
    accepted = np.zeros(sibling_groups.shape[0], dtype=np.bool_)
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    for group_idx in range(sibling_groups.shape[0]):
        child_slots = sibling_groups[group_idx]
        if (
            protected[int(child_slots[0])]
            or protected[int(child_slots[1])]
            or protected[int(child_slots[2])]
            or protected[int(child_slots[3])]
        ):
            continue

        reject = False
        for tol_idx in range(channels.shape[0]):
            ch = int(channels[tol_idx])
            tol = float(tolerances[tol_idx])
            first_slot = int(child_slots[0])
            v_min = values[first_slot, ch, 0, 0]
            v_max = v_min
            for quad in range(4):
                child_slot = int(child_slots[quad])
                for yy in range(patch_h):
                    for xx in range(patch_w):
                        val = values[child_slot, ch, yy, xx]
                        if val < v_min:
                            v_min = val
                        elif val > v_max:
                            v_max = val
            if float(v_max - v_min) >= tol:
                reject = True
                break
        if reject:
            continue

        key = group_keys[group_idx]
        if _active_parent_side_has_too_fine_neighbor_jit(
            level_idx,
            owner,
            child_slots,
            int(key[2]),
            int(key[3]),
            int(key[4]),
            int(key[0]),
            int(key[1]),
            max_level,
        ):
            continue

        accepted[group_idx] = True
    return accepted


@njit(boundscheck=False, parallel=True, cache=True)
def _active_coarsen_acceptance_mask_parallel_jit(
    values: np.ndarray,
    level_idx: np.ndarray,
    protected: np.ndarray,
    group_keys: np.ndarray,
    sibling_groups: np.ndarray,
    channels: np.ndarray,
    tolerances: np.ndarray,
    owner: np.ndarray,
    max_level: int,
) -> np.ndarray:
    accepted = np.zeros(sibling_groups.shape[0], dtype=np.bool_)
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    for group_idx in prange(sibling_groups.shape[0]):
        child_slots = sibling_groups[group_idx]
        if (
            protected[int(child_slots[0])]
            or protected[int(child_slots[1])]
            or protected[int(child_slots[2])]
            or protected[int(child_slots[3])]
        ):
            continue

        reject = False
        for tol_idx in range(channels.shape[0]):
            ch = int(channels[tol_idx])
            tol = float(tolerances[tol_idx])
            first_slot = int(child_slots[0])
            v_min = values[first_slot, ch, 0, 0]
            v_max = v_min
            for quad in range(4):
                child_slot = int(child_slots[quad])
                for yy in range(patch_h):
                    for xx in range(patch_w):
                        val = values[child_slot, ch, yy, xx]
                        if val < v_min:
                            v_min = val
                        elif val > v_max:
                            v_max = val
            if float(v_max - v_min) >= tol:
                reject = True
                break
        if reject:
            continue

        key = group_keys[group_idx]
        if _active_parent_side_has_too_fine_neighbor_jit(
            level_idx,
            owner,
            child_slots,
            int(key[2]),
            int(key[3]),
            int(key[4]),
            int(key[0]),
            int(key[1]),
            max_level,
        ):
            continue

        accepted[group_idx] = True
    return accepted


@njit(boundscheck=False, cache=True)
def _active_coarsen_acceptance_mask_jit(
    values: np.ndarray,
    level_idx: np.ndarray,
    protected: np.ndarray,
    group_keys: np.ndarray,
    sibling_groups: np.ndarray,
    channels: np.ndarray,
    tolerances: np.ndarray,
    owner: np.ndarray,
    max_level: int,
) -> np.ndarray:
    if sibling_groups.shape[0] >= _COARSEN_PARALLEL_ACCEPT_THRESHOLD:
        return _active_coarsen_acceptance_mask_parallel_jit(
            values,
            level_idx,
            protected,
            group_keys,
            sibling_groups,
            channels,
            tolerances,
            owner,
            max_level,
        )
    return _active_coarsen_acceptance_mask_sequential_jit(
        values,
        level_idx,
        protected,
        group_keys,
        sibling_groups,
        channels,
        tolerances,
        owner,
        max_level,
    )


@njit(boundscheck=False, cache=True)
def _source_original_sibling_group_exceeds_tolerance_jit(
    source_sequence: np.ndarray,
    sequence_channels: int,
    child_slots: np.ndarray,
    flat_channel: int,
    tolerance: float,
) -> bool:
    t_idx = flat_channel // sequence_channels
    ch = flat_channel - t_idx * sequence_channels
    patch_h = source_sequence.shape[3]
    patch_w = source_sequence.shape[4]
    first_slot = int(child_slots[0])
    v_min = source_sequence[first_slot, ch, t_idx, 0, 0]
    v_max = v_min
    for quad in range(4):
        child_slot = int(child_slots[quad])
        for yy in range(patch_h):
            for xx in range(patch_w):
                val = source_sequence[child_slot, ch, t_idx, yy, xx]
                if val < v_min:
                    v_min = val
                elif val > v_max:
                    v_max = val
    return float(v_max - v_min) >= tolerance


@njit(boundscheck=False, cache=True)
def _active_coarsen_acceptance_mask_from_source_sequential_jit(
    source_sequence: np.ndarray,
    values: np.ndarray,
    initial_slots: int,
    sequence_channels: int,
    level_idx: np.ndarray,
    protected: np.ndarray,
    group_keys: np.ndarray,
    sibling_groups: np.ndarray,
    channels: np.ndarray,
    tolerances: np.ndarray,
    owner: np.ndarray,
    max_level: int,
) -> np.ndarray:
    accepted = np.zeros(sibling_groups.shape[0], dtype=np.bool_)
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    for group_idx in range(sibling_groups.shape[0]):
        child_slots = sibling_groups[group_idx]
        if (
            protected[int(child_slots[0])]
            or protected[int(child_slots[1])]
            or protected[int(child_slots[2])]
            or protected[int(child_slots[3])]
        ):
            continue

        all_original = (
            int(child_slots[0]) < initial_slots
            and int(child_slots[1]) < initial_slots
            and int(child_slots[2]) < initial_slots
            and int(child_slots[3]) < initial_slots
        )
        reject = False
        for tol_idx in range(channels.shape[0]):
            ch = int(channels[tol_idx])
            tol = float(tolerances[tol_idx])
            if all_original:
                if _source_original_sibling_group_exceeds_tolerance_jit(
                    source_sequence,
                    sequence_channels,
                    child_slots,
                    ch,
                    tol,
                ):
                    reject = True
                    break
                continue
            first_slot = int(child_slots[0])
            v_min = _source_or_workspace_value(
                source_sequence, values, initial_slots, sequence_channels, first_slot, ch, 0, 0
            )
            v_max = v_min
            for quad in range(4):
                child_slot = int(child_slots[quad])
                for yy in range(patch_h):
                    for xx in range(patch_w):
                        val = _source_or_workspace_value(
                            source_sequence, values, initial_slots, sequence_channels, child_slot, ch, yy, xx
                        )
                        if val < v_min:
                            v_min = val
                        elif val > v_max:
                            v_max = val
            if float(v_max - v_min) >= tol:
                reject = True
                break
        if reject:
            continue

        key = group_keys[group_idx]
        if _active_parent_side_has_too_fine_neighbor_jit(
            level_idx,
            owner,
            child_slots,
            int(key[2]),
            int(key[3]),
            int(key[4]),
            int(key[0]),
            int(key[1]),
            max_level,
        ):
            continue

        accepted[group_idx] = True
    return accepted


@njit(boundscheck=False, parallel=True, cache=True)
def _active_coarsen_acceptance_mask_from_source_parallel_jit(
    source_sequence: np.ndarray,
    values: np.ndarray,
    initial_slots: int,
    sequence_channels: int,
    level_idx: np.ndarray,
    protected: np.ndarray,
    group_keys: np.ndarray,
    sibling_groups: np.ndarray,
    channels: np.ndarray,
    tolerances: np.ndarray,
    owner: np.ndarray,
    max_level: int,
) -> np.ndarray:
    accepted = np.zeros(sibling_groups.shape[0], dtype=np.bool_)
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    for group_idx in prange(sibling_groups.shape[0]):
        child_slots = sibling_groups[group_idx]
        if (
            protected[int(child_slots[0])]
            or protected[int(child_slots[1])]
            or protected[int(child_slots[2])]
            or protected[int(child_slots[3])]
        ):
            continue

        all_original = (
            int(child_slots[0]) < initial_slots
            and int(child_slots[1]) < initial_slots
            and int(child_slots[2]) < initial_slots
            and int(child_slots[3]) < initial_slots
        )
        reject = False
        for tol_idx in range(channels.shape[0]):
            ch = int(channels[tol_idx])
            tol = float(tolerances[tol_idx])
            if all_original:
                if _source_original_sibling_group_exceeds_tolerance_jit(
                    source_sequence,
                    sequence_channels,
                    child_slots,
                    ch,
                    tol,
                ):
                    reject = True
                    break
                continue
            first_slot = int(child_slots[0])
            v_min = _source_or_workspace_value(
                source_sequence, values, initial_slots, sequence_channels, first_slot, ch, 0, 0
            )
            v_max = v_min
            for quad in range(4):
                child_slot = int(child_slots[quad])
                for yy in range(patch_h):
                    for xx in range(patch_w):
                        val = _source_or_workspace_value(
                            source_sequence, values, initial_slots, sequence_channels, child_slot, ch, yy, xx
                        )
                        if val < v_min:
                            v_min = val
                        elif val > v_max:
                            v_max = val
            if float(v_max - v_min) >= tol:
                reject = True
                break
        if reject:
            continue

        key = group_keys[group_idx]
        if _active_parent_side_has_too_fine_neighbor_jit(
            level_idx,
            owner,
            child_slots,
            int(key[2]),
            int(key[3]),
            int(key[4]),
            int(key[0]),
            int(key[1]),
            max_level,
        ):
            continue

        accepted[group_idx] = True
    return accepted


@njit(boundscheck=False, cache=True)
def _active_coarsen_acceptance_mask_from_source_jit(
    source_sequence: np.ndarray,
    values: np.ndarray,
    initial_slots: int,
    sequence_channels: int,
    level_idx: np.ndarray,
    protected: np.ndarray,
    group_keys: np.ndarray,
    sibling_groups: np.ndarray,
    channels: np.ndarray,
    tolerances: np.ndarray,
    owner: np.ndarray,
    max_level: int,
) -> np.ndarray:
    if sibling_groups.shape[0] >= _COARSEN_PARALLEL_ACCEPT_THRESHOLD:
        return _active_coarsen_acceptance_mask_from_source_parallel_jit(
            source_sequence,
            values,
            initial_slots,
            sequence_channels,
            level_idx,
            protected,
            group_keys,
            sibling_groups,
            channels,
            tolerances,
            owner,
            max_level,
        )
    return _active_coarsen_acceptance_mask_from_source_sequential_jit(
        source_sequence,
        values,
        initial_slots,
        sequence_channels,
        level_idx,
        protected,
        group_keys,
        sibling_groups,
        channels,
        tolerances,
        owner,
        max_level,
    )


@njit(boundscheck=False, inline="always")
def _fill_one_coarsened_parent_from_source_original_jit(
    source_sequence: np.ndarray,
    values: np.ndarray,
    initial_slots: int,
    sequence_channels: int,
    child_slots: np.ndarray,
    parent_slot: int,
) -> None:
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    half_h = patch_h // 2
    half_w = patch_w // 2
    timesteps = source_sequence.shape[2]
    parent_workspace_slot = parent_slot - initial_slots

    for quad in range(4):
        child_slot = int(child_slots[quad])
        y_start = 0
        y_stop = half_h
        if quad >= 2:
            y_start = half_h
            y_stop = patch_h
        x_start = 0
        x_stop = half_w
        if quad == 1 or quad == 3:
            x_start = half_w
            x_stop = patch_w

        for t_idx in range(timesteps):
            flat_offset = t_idx * sequence_channels
            for ch in range(sequence_channels):
                flat_ch = flat_offset + ch
                for yy in range(y_start, y_stop):
                    child_y = 2 * (yy - y_start)
                    for xx in range(x_start, x_stop):
                        child_x = 2 * (xx - x_start)
                        values[parent_workspace_slot, flat_ch, yy, xx] = 0.25 * (
                            source_sequence[child_slot, ch, t_idx, child_y, child_x]
                            + source_sequence[child_slot, ch, t_idx, child_y + 1, child_x]
                            + source_sequence[child_slot, ch, t_idx, child_y, child_x + 1]
                            + source_sequence[child_slot, ch, t_idx, child_y + 1, child_x + 1]
                        )


@njit(boundscheck=False, inline="always")
def _fill_one_coarsened_parent_from_source_jit(
    source_sequence: np.ndarray,
    values: np.ndarray,
    initial_slots: int,
    sequence_channels: int,
    child_slots: np.ndarray,
    parent_slot: int,
) -> None:
    channels_total = values.shape[1]
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    half_h = patch_h // 2
    half_w = patch_w // 2

    if (
        int(child_slots[0]) < initial_slots
        and int(child_slots[1]) < initial_slots
        and int(child_slots[2]) < initial_slots
        and int(child_slots[3]) < initial_slots
    ):
        _fill_one_coarsened_parent_from_source_original_jit(
            source_sequence,
            values,
            initial_slots,
            sequence_channels,
            child_slots,
            parent_slot,
        )
        return

    parent_workspace_slot = parent_slot - initial_slots
    for quad in range(4):
        child_slot = int(child_slots[quad])
        y_start = 0
        y_stop = half_h
        if quad >= 2:
            y_start = half_h
            y_stop = patch_h
        x_start = 0
        x_stop = half_w
        if quad == 1 or quad == 3:
            x_start = half_w
            x_stop = patch_w

        if child_slot < initial_slots:
            for flat_ch in range(channels_total):
                t_idx = flat_ch // sequence_channels
                ch = flat_ch - t_idx * sequence_channels
                for yy in range(y_start, y_stop):
                    child_y = 2 * (yy - y_start)
                    for xx in range(x_start, x_stop):
                        child_x = 2 * (xx - x_start)
                        values[parent_workspace_slot, flat_ch, yy, xx] = 0.25 * (
                            source_sequence[child_slot, ch, t_idx, child_y, child_x]
                            + source_sequence[child_slot, ch, t_idx, child_y + 1, child_x]
                            + source_sequence[child_slot, ch, t_idx, child_y, child_x + 1]
                            + source_sequence[child_slot, ch, t_idx, child_y + 1, child_x + 1]
                        )
        else:
            child_workspace_slot = child_slot - initial_slots
            for flat_ch in range(channels_total):
                for yy in range(y_start, y_stop):
                    child_y = 2 * (yy - y_start)
                    for xx in range(x_start, x_stop):
                        child_x = 2 * (xx - x_start)
                        values[parent_workspace_slot, flat_ch, yy, xx] = 0.25 * (
                            values[child_workspace_slot, flat_ch, child_y, child_x]
                            + values[child_workspace_slot, flat_ch, child_y + 1, child_x]
                            + values[child_workspace_slot, flat_ch, child_y, child_x + 1]
                            + values[child_workspace_slot, flat_ch, child_y + 1, child_x + 1]
                        )


@njit(boundscheck=False, cache=True)
def _fill_accepted_coarsened_parent_values_from_source_sequential_jit(
    source_sequence: np.ndarray,
    values: np.ndarray,
    initial_slots: int,
    sequence_channels: int,
    sibling_groups: np.ndarray,
    accepted_groups: np.ndarray,
    accepted_count: int,
    parent_start: int,
) -> None:
    for accepted_pos in range(accepted_count):
        group_idx = int(accepted_groups[accepted_pos])
        child_slots = sibling_groups[group_idx]
        parent_slot = parent_start + accepted_pos
        _fill_one_coarsened_parent_from_source_jit(
            source_sequence,
            values,
            initial_slots,
            sequence_channels,
            child_slots,
            parent_slot,
        )


@njit(boundscheck=False, parallel=True, cache=True)
def _fill_accepted_coarsened_parent_values_from_source_parallel_jit(
    source_sequence: np.ndarray,
    values: np.ndarray,
    initial_slots: int,
    sequence_channels: int,
    sibling_groups: np.ndarray,
    accepted_groups: np.ndarray,
    accepted_count: int,
    parent_start: int,
) -> None:
    for accepted_pos in prange(accepted_count):
        group_idx = int(accepted_groups[accepted_pos])
        child_slots = sibling_groups[group_idx]
        parent_slot = parent_start + accepted_pos
        _fill_one_coarsened_parent_from_source_jit(
            source_sequence,
            values,
            initial_slots,
            sequence_channels,
            child_slots,
            parent_slot,
        )


@njit(boundscheck=False, cache=True)
def _fill_accepted_coarsened_parent_values_from_source_jit(
    source_sequence: np.ndarray,
    values: np.ndarray,
    initial_slots: int,
    sequence_channels: int,
    sibling_groups: np.ndarray,
    accepted_groups: np.ndarray,
    accepted_count: int,
    parent_start: int,
) -> None:
    if accepted_count >= _COARSEN_PARALLEL_FILL_THRESHOLD:
        _fill_accepted_coarsened_parent_values_from_source_parallel_jit(
            source_sequence,
            values,
            initial_slots,
            sequence_channels,
            sibling_groups,
            accepted_groups,
            accepted_count,
            parent_start,
        )
    else:
        _fill_accepted_coarsened_parent_values_from_source_sequential_jit(
            source_sequence,
            values,
            initial_slots,
            sequence_channels,
            sibling_groups,
            accepted_groups,
            accepted_count,
            parent_start,
        )


@njit(boundscheck=False, cache=True)
def _active_coarsen_once_from_source_jit(
    source_sequence: np.ndarray,
    values: np.ndarray,
    initial_slots: int,
    sequence_channels: int,
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    level_idx: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    active_slots: np.ndarray,
    active_count: int,
    next_slot: int,
    protected: np.ndarray,
    group_keys: np.ndarray,
    sibling_groups: np.ndarray,
    channels: np.ndarray,
    tolerances: np.ndarray,
    owner: np.ndarray,
    max_level: int,
) -> Tuple[np.ndarray, int, int, bool, int, int]:
    capacity = level_idx.shape[0]
    accepted_groups = np.empty(sibling_groups.shape[0], dtype=np.int32)
    accepted_count = 0
    accepted_mask = _active_coarsen_acceptance_mask_from_source_jit(
        source_sequence,
        values,
        initial_slots,
        sequence_channels,
        level_idx,
        protected,
        group_keys,
        sibling_groups,
        channels,
        tolerances,
        owner,
        max_level,
    )

    for group_idx in range(sibling_groups.shape[0]):
        if not accepted_mask[group_idx]:
            continue

        if next_slot + accepted_count >= capacity:
            return active_slots, active_count, next_slot, False, 1, accepted_count
        accepted_groups[accepted_count] = group_idx
        accepted_count += 1

    if accepted_count == 0:
        return active_slots, active_count, next_slot, False, 0, 0

    skipped = np.zeros(capacity, dtype=np.bool_)
    for accepted_pos in range(accepted_count):
        child_slots = sibling_groups[int(accepted_groups[accepted_pos])]
        for quad in range(4):
            skipped[int(child_slots[quad])] = True

    new_active_slots = np.empty_like(active_slots)
    out_count = 0
    for active_pos in range(active_count):
        slot = int(active_slots[active_pos])
        if skipped[slot]:
            protected[slot] = False
            continue
        new_active_slots[out_count] = slot
        out_count += 1

    parent_start = next_slot
    for accepted_pos in range(accepted_count):
        group_idx = int(accepted_groups[accepted_pos])
        key = group_keys[group_idx]
        parent_slot = parent_start + accepted_pos

        tile_ix[parent_slot] = int(key[0])
        tile_iy[parent_slot] = int(key[1])
        level_idx[parent_slot] = np.int16(int(key[2]))
        x_idx[parent_slot] = int(key[3])
        y_idx[parent_slot] = int(key[4])
        protected[parent_slot] = False
        new_active_slots[out_count] = parent_slot
        out_count += 1

    _fill_accepted_coarsened_parent_values_from_source_jit(
        source_sequence,
        values,
        initial_slots,
        sequence_channels,
        sibling_groups,
        accepted_groups,
        accepted_count,
        parent_start,
    )
    next_slot = parent_start + accepted_count

    return new_active_slots, out_count, next_slot, True, 0, accepted_count


@njit(boundscheck=False, cache=True)
def _active_coarsen_once_jit(
    values: np.ndarray,
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    level_idx: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    active_slots: np.ndarray,
    active_count: int,
    next_slot: int,
    protected: np.ndarray,
    group_keys: np.ndarray,
    sibling_groups: np.ndarray,
    channels: np.ndarray,
    tolerances: np.ndarray,
    owner: np.ndarray,
    max_level: int,
) -> Tuple[np.ndarray, int, int, bool, int, int]:
    capacity = values.shape[0]
    accepted_groups = np.empty(sibling_groups.shape[0], dtype=np.int32)
    accepted_count = 0
    accepted_mask = _active_coarsen_acceptance_mask_jit(
        values,
        level_idx,
        protected,
        group_keys,
        sibling_groups,
        channels,
        tolerances,
        owner,
        max_level,
    )

    for group_idx in range(sibling_groups.shape[0]):
        if not accepted_mask[group_idx]:
            continue

        if next_slot + accepted_count >= capacity:
            return active_slots, active_count, next_slot, False, 1, accepted_count
        accepted_groups[accepted_count] = group_idx
        accepted_count += 1

    if accepted_count == 0:
        return active_slots, active_count, next_slot, False, 0, 0

    skipped = np.zeros(capacity, dtype=np.bool_)
    for accepted_pos in range(accepted_count):
        child_slots = sibling_groups[int(accepted_groups[accepted_pos])]
        for quad in range(4):
            skipped[int(child_slots[quad])] = True

    new_active_slots = np.empty_like(active_slots)
    out_count = 0
    for active_pos in range(active_count):
        slot = int(active_slots[active_pos])
        if skipped[slot]:
            protected[slot] = False
            continue
        new_active_slots[out_count] = slot
        out_count += 1

    parent_start = next_slot
    for accepted_pos in range(accepted_count):
        group_idx = int(accepted_groups[accepted_pos])
        key = group_keys[group_idx]
        parent_slot = parent_start + accepted_pos

        tile_ix[parent_slot] = int(key[0])
        tile_iy[parent_slot] = int(key[1])
        level_idx[parent_slot] = np.int16(int(key[2]))
        x_idx[parent_slot] = int(key[3])
        y_idx[parent_slot] = int(key[4])
        protected[parent_slot] = False
        new_active_slots[out_count] = parent_slot
        out_count += 1

    _fill_accepted_coarsened_parent_values_jit(
        values,
        sibling_groups,
        accepted_groups,
        accepted_count,
        parent_start,
    )
    next_slot = parent_start + accepted_count

    return new_active_slots, out_count, next_slot, True, 0, accepted_count


def _active_coarsen_once(
    active: ActiveTopology,
    protected: np.ndarray,
    channels: list[int],
    tolerances: np.ndarray,
) -> Tuple[ActiveTopology, np.ndarray, bool, int]:
    group_keys, sibling_groups = _active_complete_sibling_groups(active)
    owner = _build_active_owner_grid(active)
    new_active_slots, active_count, next_slot, changed, err, accepted_count = _active_coarsen_once_jit(
        active.values,
        active.tile_ix,
        active.tile_iy,
        active.level_idx,
        active.x_idx,
        active.y_idx,
        active.active_slots,
        int(active.active_count),
        int(active.next_slot),
        protected,
        group_keys,
        sibling_groups,
        np.asarray(channels, dtype=np.int64),
        tolerances.astype(np.float64, copy=False),
        owner,
        int(active.domain["max_level_idx"]),
    )
    if err:
        raise _ArrayFallback("capacity_exceeded")
    if not changed:
        return active, protected, False, int(accepted_count)

    active.active_slots = new_active_slots
    active.active_count = int(active_count)
    active.next_slot = int(next_slot)
    return active, protected, True, int(accepted_count)


def _active_coarsen_once_conservative(
    active: ActiveTopology,
    protected: np.ndarray,
    channels: list[int],
    tolerances: np.ndarray,
) -> Tuple[ActiveTopology, np.ndarray, bool, int]:
    accepted_total = 0
    changed_any = False
    evaluated: set[tuple[int, int, int, int, int]] = set()
    original_active_slots = active.active_slots.copy()
    original_active_count = int(active.active_count)
    original_next_slot = int(active.next_slot)
    original_protected = protected.copy()

    def restore_original_state() -> None:
        active.active_slots = original_active_slots
        active.active_count = original_active_count
        active.next_slot = original_next_slot
        protected[:] = original_protected

    while True:
        group_keys, sibling_groups = _active_complete_sibling_groups(active)
        owner = _build_active_owner_grid(active)
        accepted_this_scan = False
        pending_found = False
        for group_idx in range(sibling_groups.shape[0]):
            key = group_keys[group_idx]
            key_tuple = (int(key[0]), int(key[1]), int(key[2]), int(key[3]), int(key[4]))
            if key_tuple in evaluated:
                continue
            evaluated.add(key_tuple)
            pending_found = True
            single_keys = group_keys[group_idx : group_idx + 1]
            single_groups = sibling_groups[group_idx : group_idx + 1]
            new_active_slots, active_count, next_slot, changed, err, accepted_count = _active_coarsen_once_jit(
                active.values,
                active.tile_ix,
                active.tile_iy,
                active.level_idx,
                active.x_idx,
                active.y_idx,
                active.active_slots,
                int(active.active_count),
                int(active.next_slot),
                protected,
                single_keys,
                single_groups,
                np.asarray(channels, dtype=np.int64),
                tolerances.astype(np.float64, copy=False),
                owner,
                int(active.domain["max_level_idx"]),
            )
            if err:
                restore_original_state()
                raise _ArrayFallback("capacity_exceeded")
            if changed:
                active.active_slots = new_active_slots
                active.active_count = int(active_count)
                active.next_slot = int(next_slot)
                accepted_total += int(accepted_count)
                changed_any = True
                accepted_this_scan = True
                break
        if not accepted_this_scan:
            if pending_found:
                return active, protected, changed_any, accepted_total
            return active, protected, changed_any, accepted_total


def _active_coarsen_once_from_source(
    active: SequenceSourceActiveTopology,
    protected: np.ndarray,
    channels: list[int],
    tolerances: np.ndarray,
) -> Tuple[SequenceSourceActiveTopology, np.ndarray, bool, int]:
    group_keys, sibling_groups = _active_complete_sibling_groups(active)
    owner = _build_active_owner_grid(active)
    new_active_slots, active_count, next_slot, changed, err, accepted_count = _active_coarsen_once_from_source_jit(
        active.source_sequence,
        active.values,
        int(active.initial_slots),
        int(active.sequence_channels),
        active.tile_ix,
        active.tile_iy,
        active.level_idx,
        active.x_idx,
        active.y_idx,
        active.active_slots,
        int(active.active_count),
        int(active.next_slot),
        protected,
        group_keys,
        sibling_groups,
        np.asarray(channels, dtype=np.int64),
        tolerances.astype(np.float64, copy=False),
        owner,
        int(active.domain["max_level_idx"]),
    )
    if err:
        raise _ArrayFallback("capacity_exceeded")
    if not changed:
        return active, protected, False, int(accepted_count)

    active.active_slots = new_active_slots
    active.active_count = int(active_count)
    active.next_slot = int(next_slot)
    return active, protected, True, int(accepted_count)


def _active_coarsen_once_from_source_conservative(
    active: SequenceSourceActiveTopology,
    protected: np.ndarray,
    channels: list[int],
    tolerances: np.ndarray,
) -> Tuple[SequenceSourceActiveTopology, np.ndarray, bool, int]:
    accepted_total = 0
    changed_any = False
    evaluated: set[tuple[int, int, int, int, int]] = set()
    original_active_slots = active.active_slots.copy()
    original_active_count = int(active.active_count)
    original_next_slot = int(active.next_slot)
    original_protected = protected.copy()

    def restore_original_state() -> None:
        active.active_slots = original_active_slots
        active.active_count = original_active_count
        active.next_slot = original_next_slot
        protected[:] = original_protected

    while True:
        group_keys, sibling_groups = _active_complete_sibling_groups(active)
        owner = _build_active_owner_grid(active)
        accepted_this_scan = False
        pending_found = False
        for group_idx in range(sibling_groups.shape[0]):
            key = group_keys[group_idx]
            key_tuple = (int(key[0]), int(key[1]), int(key[2]), int(key[3]), int(key[4]))
            if key_tuple in evaluated:
                continue
            evaluated.add(key_tuple)
            pending_found = True
            single_keys = group_keys[group_idx : group_idx + 1]
            single_groups = sibling_groups[group_idx : group_idx + 1]
            new_active_slots, active_count, next_slot, changed, err, accepted_count = _active_coarsen_once_from_source_jit(
                active.source_sequence,
                active.values,
                int(active.initial_slots),
                int(active.sequence_channels),
                active.tile_ix,
                active.tile_iy,
                active.level_idx,
                active.x_idx,
                active.y_idx,
                active.active_slots,
                int(active.active_count),
                int(active.next_slot),
                protected,
                single_keys,
                single_groups,
                np.asarray(channels, dtype=np.int64),
                tolerances.astype(np.float64, copy=False),
                owner,
                int(active.domain["max_level_idx"]),
            )
            if err:
                restore_original_state()
                raise _ArrayFallback("capacity_exceeded")
            if changed:
                active.active_slots = new_active_slots
                active.active_count = int(active_count)
                active.next_slot = int(next_slot)
                accepted_total += int(accepted_count)
                changed_any = True
                accepted_this_scan = True
                break
        if not accepted_this_scan:
            if pending_found:
                return active, protected, changed_any, accepted_total
            return active, protected, changed_any, accepted_total


def _nx_ny_tiles(domain: Dict[str, Any]) -> Tuple[int, int]:
    nx_tiles = int(round((float(domain["xmax"]) - float(domain["xmin"])) / float(domain["tile_width"])))
    ny_tiles = int(round((float(domain["ymax"]) - float(domain["ymin"])) / float(domain["tile_height"])))
    return nx_tiles, ny_tiles


def _cell_ids(top: FlatTopology) -> np.ndarray:
    nx_tiles, ny_tiles = _nx_ny_tiles(top.domain)
    max_level = int(top.domain["max_level_idx"])
    xw = max_level
    yw = max_level
    lw = 6
    txw = _ceil_log2(nx_tiles)
    tyw = _ceil_log2(ny_tiles)
    return _cell_ids_jit(
        top.tile_ix,
        top.tile_iy,
        top.level_idx,
        top.x_idx,
        top.y_idx,
        xw,
        yw,
        lw,
        txw,
        tyw,
    )


@njit(boundscheck=False)
def _cell_ids_jit(
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    level_idx: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    xw: int,
    yw: int,
    lw: int,
    txw: int,
    tyw: int,
) -> np.ndarray:
    ids = np.empty(level_idx.shape[0], dtype=np.uint64)
    for i in range(level_idx.shape[0]):
        shift = 0
        uid = 0
        if xw:
            uid |= (int(x_idx[i]) & ((1 << xw) - 1)) << shift
        shift += xw
        if yw:
            uid |= (int(y_idx[i]) & ((1 << yw) - 1)) << shift
        shift += yw
        uid |= (int(level_idx[i]) & ((1 << lw) - 1)) << shift
        shift += lw
        if txw:
            uid |= (int(tile_ix[i]) & ((1 << txw) - 1)) << shift
            shift += txw
        if tyw:
            uid |= (int(tile_iy[i]) & ((1 << tyw) - 1)) << shift
        ids[i] = np.uint64(uid)
    return ids


@njit(boundscheck=False, cache=True)
def _morton_keys_jit(x0: np.ndarray, y0: np.ndarray, bits: int) -> np.ndarray:
    keys = np.empty(x0.shape[0], dtype=np.uint64)
    for i in range(x0.shape[0]):
        keys[i] = np.uint64(morton2D_jit(int(x0[i]), int(y0[i]), bits))
    return keys


def _scale_values_for_centers(top: FlatTopology, cx: np.ndarray, cy: np.ndarray, hx: np.ndarray, hy: np.ndarray) -> np.ndarray:
    mode = top.cell_scale_mode
    if mode == "level_idx":
        return top.level_idx.astype(np.float32)
    grid_w = 2.0 * hx
    grid_h = 2.0 * hy
    if mode == "x":
        return grid_w.astype(np.float32)
    if mode == "y":
        return grid_h.astype(np.float32)
    if mode == "area":
        return (grid_w * grid_h).astype(np.float32)
    if mode == "sqrt_area":
        return np.sqrt(grid_w * grid_h).astype(np.float32)
    if mode == "log_area":
        return np.log(grid_w * grid_h + 1.0e-12).astype(np.float32)
    if mode == "diag":
        return np.sqrt(grid_w**2 + grid_h**2).astype(np.float32)
    raise ValueError(f"Invalid cell_scale_mode '{mode}'")


def _scale_norm_constants(top: FlatTopology) -> Tuple[float, float]:
    mode = top.cell_scale_mode
    max_level = int(top.domain["max_level_idx"])
    tile_width = float(top.domain["tile_width"])
    tile_height = float(top.domain["tile_height"])
    min_grid_w = tile_width / (1 << max_level)
    min_grid_h = tile_height / (1 << max_level)
    max_grid_w = tile_width
    max_grid_h = tile_height
    if mode == "level_idx":
        return 0.0, 1.0
    if mode == "x":
        return min_grid_w, max_grid_w - min_grid_w
    if mode == "y":
        return min_grid_h, max_grid_h - min_grid_h
    if mode == "area":
        return min_grid_w * min_grid_h, max_grid_w * max_grid_h - min_grid_w * min_grid_h
    if mode == "sqrt_area":
        return np.sqrt(min_grid_w * min_grid_h), np.sqrt(max_grid_w * max_grid_h) - np.sqrt(min_grid_w * min_grid_h)
    if mode == "log_area":
        return np.log(min_grid_w * min_grid_h + 1.0e-12), np.log(max_grid_w * max_grid_h + 1.0e-12) - np.log(min_grid_w * min_grid_h + 1.0e-12)
    if mode == "diag":
        return np.sqrt(min_grid_w**2 + min_grid_h**2), np.sqrt(max_grid_w**2 + max_grid_h**2) - np.sqrt(min_grid_w**2 + min_grid_h**2)
    raise ValueError(f"Invalid cell_scale_mode '{mode}'")


def _export_ordered_topology(
    ordered: FlatTopology,
    x0: np.ndarray,
    y0: np.ndarray,
    scale: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    max_level = int(ordered.domain["max_level_idx"])
    xmin = float(ordered.domain["xmin"])
    ymin = float(ordered.domain["ymin"])
    xmax = float(ordered.domain["xmax"])
    ymax = float(ordered.domain["ymax"])
    tile_width = float(ordered.domain["tile_width"])
    tile_height = float(ordered.domain["tile_height"])
    fine = float(1 << max_level)
    dx = tile_width / fine
    dy = tile_height / fine
    cx = xmin + (x0.astype(np.float64) + scale.astype(np.float64) * 0.5) * dx
    cy = ymin + (y0.astype(np.float64) + scale.astype(np.float64) * 0.5) * dy
    hx = scale.astype(np.float64) * dx * 0.5
    hy = scale.astype(np.float64) * dy * 0.5

    if ordered.cell_scale_mode is None:
        centers = np.stack([cx, cy, hx, hy], axis=1).astype(np.float32)
    else:
        s_val = _scale_values_for_centers(ordered, cx, cy, hx, hy).astype(np.float64)
        cell_min, lcell = _scale_norm_constants(ordered)
        centers = np.empty((ordered.values.shape[0], 3), dtype=np.float32)
        centers[:, 0] = ((cx - xmin) / (xmax - xmin)).astype(np.float32)
        centers[:, 1] = ((cy - ymin) / (ymax - ymin)).astype(np.float32)
        centers[:, 2] = ((s_val - cell_min) / lcell).astype(np.float32) if lcell != 0 else 0.0

    meta = {
        "centers": centers,
        "levels": ordered.level_idx.astype(np.int16, copy=True),
        "tiles": np.stack([ordered.tile_ix, ordered.tile_iy], axis=1).astype(np.int16),
        "xy_idx": np.stack([ordered.level_idx.astype(np.int32), ordered.x_idx, ordered.y_idx], axis=1).astype(np.int32),
        "cell_ids": _cell_ids(ordered),
        "domain": dict(ordered.domain),
        "cell_scale_mode": ordered.cell_scale_mode,
    }
    return ordered.values, meta


def _export_topology(top: FlatTopology) -> Tuple[np.ndarray, Dict[str, Any]]:
    x0, y0, scale = _leaf_fine_bounds(top)
    nx_tiles, ny_tiles = _nx_ny_tiles(top.domain)
    max_level = int(top.domain["max_level_idx"])
    bits = max_level + _ceil_log2(max(nx_tiles, ny_tiles))
    keys = _morton_keys_jit(x0, y0, bits)
    order = np.argsort(keys, kind="stable")

    ordered = FlatTopology(
        values=np.ascontiguousarray(top.values[order], dtype=np.float32),
        tile_ix=top.tile_ix[order].astype(np.int32, copy=True),
        tile_iy=top.tile_iy[order].astype(np.int32, copy=True),
        level_idx=top.level_idx[order].astype(np.int16, copy=True),
        x_idx=top.x_idx[order].astype(np.int32, copy=True),
        y_idx=top.y_idx[order].astype(np.int32, copy=True),
        domain=dict(top.domain),
        cell_scale_mode=top.cell_scale_mode,
    )
    return _export_ordered_topology(ordered, x0[order], y0[order], scale[order])


def _active_leaf_fine_bounds(active: ActiveTopology, slots: np.ndarray) -> FineBounds:
    max_level = int(active.domain["max_level_idx"])
    levels = active.level_idx[slots].astype(np.int64)
    scale = 1 << (max_level - levels)
    x0 = (active.tile_ix[slots].astype(np.int64) << max_level) + active.x_idx[slots].astype(np.int64) * scale
    y0 = (active.tile_iy[slots].astype(np.int64) << max_level) + active.y_idx[slots].astype(np.int64) * scale
    return x0, y0, scale


@njit(boundscheck=False, parallel=True, cache=True)
def _copy_ordered_active_values_jit(values: np.ndarray, ordered_slots: np.ndarray, out: np.ndarray) -> None:
    n = ordered_slots.shape[0]
    channels = values.shape[1]
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    for out_i in prange(n):
        slot = int(ordered_slots[out_i])
        for ch in range(channels):
            for yy in range(patch_h):
                for xx in range(patch_w):
                    out[out_i, ch, yy, xx] = values[slot, ch, yy, xx]


@njit(boundscheck=False, parallel=True, cache=True)
def _copy_ordered_active_values_to_sequence_jit(
    values: np.ndarray,
    ordered_slots: np.ndarray,
    out: np.ndarray,
    channels: int,
    timesteps: int,
) -> None:
    n = ordered_slots.shape[0]
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    for out_i in prange(n):
        slot = int(ordered_slots[out_i])
        for ch in range(channels):
            for t in range(timesteps):
                flat_ch = t * channels + ch
                for yy in range(patch_h):
                    for xx in range(patch_w):
                        out[out_i, ch, t, yy, xx] = values[slot, flat_ch, yy, xx]


@njit(boundscheck=False, parallel=True, cache=True)
def _copy_ordered_source_active_values_jit(
    source_sequence: np.ndarray,
    values: np.ndarray,
    initial_slots: int,
    sequence_channels: int,
    ordered_slots: np.ndarray,
    out: np.ndarray,
) -> None:
    n = ordered_slots.shape[0]
    flat_channels = values.shape[1]
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    for out_i in prange(n):
        slot = int(ordered_slots[out_i])
        if slot < initial_slots:
            for flat_ch in range(flat_channels):
                t_idx = flat_ch // sequence_channels
                ch = flat_ch - t_idx * sequence_channels
                for yy in range(patch_h):
                    for xx in range(patch_w):
                        out[out_i, flat_ch, yy, xx] = source_sequence[slot, ch, t_idx, yy, xx]
        else:
            workspace_slot = slot - initial_slots
            for flat_ch in range(flat_channels):
                for yy in range(patch_h):
                    for xx in range(patch_w):
                        out[out_i, flat_ch, yy, xx] = values[workspace_slot, flat_ch, yy, xx]


@njit(boundscheck=False, parallel=True, cache=True)
def _copy_ordered_source_active_values_to_sequence_jit(
    source_sequence: np.ndarray,
    values: np.ndarray,
    initial_slots: int,
    ordered_slots: np.ndarray,
    out: np.ndarray,
) -> None:
    n = ordered_slots.shape[0]
    channels = source_sequence.shape[1]
    timesteps = source_sequence.shape[2]
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    for out_i in prange(n):
        slot = int(ordered_slots[out_i])
        if slot < initial_slots:
            for ch in range(channels):
                for t in range(timesteps):
                    for yy in range(patch_h):
                        for xx in range(patch_w):
                            out[out_i, ch, t, yy, xx] = source_sequence[slot, ch, t, yy, xx]
        else:
            workspace_slot = slot - initial_slots
            for ch in range(channels):
                for t in range(timesteps):
                    flat_ch = t * channels + ch
                    for yy in range(patch_h):
                        for xx in range(patch_w):
                            out[out_i, ch, t, yy, xx] = values[workspace_slot, flat_ch, yy, xx]


def _flat_output_to_sequence_array(flat: np.ndarray, channels: int, timesteps: int) -> np.ndarray:
    if flat.ndim != 4:
        raise ValueError(f"Expected flat regrid output shape (N,T*C,H,W), got {flat.shape}.")
    n, flat_channels, h, w = flat.shape
    channels = int(channels)
    timesteps = int(timesteps)
    if channels * timesteps != flat_channels:
        raise ValueError(f"Expected {channels * timesteps} flat channels, got {flat_channels}.")
    return np.ascontiguousarray(flat.reshape(n, timesteps, channels, h, w).transpose(0, 2, 1, 3, 4))


@njit(boundscheck=False, cache=True)
def _export_active_level_idx_metadata_jit(
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    level_idx: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    ordered_slots: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    scale: np.ndarray,
    max_level: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    tile_width: float,
    tile_height: float,
    xw: int,
    yw: int,
    lw: int,
    txw: int,
    tyw: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = ordered_slots.shape[0]
    centers = np.empty((n, 3), dtype=np.float32)
    levels = np.empty(n, dtype=np.int16)
    tiles = np.empty((n, 2), dtype=np.int16)
    xy_idx = np.empty((n, 3), dtype=np.int32)
    cell_ids = np.empty(n, dtype=np.uint64)

    fine = float(1 << max_level)
    dx = tile_width / fine
    dy = tile_height / fine
    inv_x = 1.0 / (xmax - xmin)
    inv_y = 1.0 / (ymax - ymin)

    for i in range(n):
        slot = int(ordered_slots[i])
        level = int(level_idx[slot])
        x_index = int(x_idx[slot])
        y_index = int(y_idx[slot])
        tx = int(tile_ix[slot])
        ty = int(tile_iy[slot])
        cx = xmin + (float(x0[i]) + float(scale[i]) * 0.5) * dx
        cy = ymin + (float(y0[i]) + float(scale[i]) * 0.5) * dy

        centers[i, 0] = np.float32((cx - xmin) * inv_x)
        centers[i, 1] = np.float32((cy - ymin) * inv_y)
        centers[i, 2] = np.float32(level)
        levels[i] = np.int16(level)
        tiles[i, 0] = np.int16(tx)
        tiles[i, 1] = np.int16(ty)
        xy_idx[i, 0] = level
        xy_idx[i, 1] = x_index
        xy_idx[i, 2] = y_index

        shift = 0
        uid = 0
        if xw:
            uid |= (x_index & ((1 << xw) - 1)) << shift
        shift += xw
        if yw:
            uid |= (y_index & ((1 << yw) - 1)) << shift
        shift += yw
        uid |= (level & ((1 << lw) - 1)) << shift
        shift += lw
        if txw:
            uid |= (tx & ((1 << txw) - 1)) << shift
            shift += txw
        if tyw:
            uid |= (ty & ((1 << tyw) - 1)) << shift
        cell_ids[i] = np.uint64(uid)

    return centers, levels, tiles, xy_idx, cell_ids


def _export_active_topology(
    active: ActiveTopology,
    *,
    output_layout: str = "flat",
    sequence_channels: Optional[int] = None,
    sequence_timesteps: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if output_layout not in {"flat", "sequence"}:
        raise ValueError(f"Unknown output_layout={output_layout!r}; expected 'flat' or 'sequence'.")
    slots = active.active_slots[:active.active_count]
    x0, y0, scale = _active_leaf_fine_bounds(active, slots)
    nx_tiles, ny_tiles = _nx_ny_tiles(active.domain)
    max_level = int(active.domain["max_level_idx"])
    bits = max_level + _ceil_log2(max(nx_tiles, ny_tiles))
    keys = _morton_keys_jit(x0, y0, bits)
    order = np.argsort(keys, kind="stable")
    ordered_slots = slots[order]
    if output_layout == "sequence":
        if sequence_channels is None or sequence_timesteps is None:
            raise ValueError("sequence output requires sequence_channels and sequence_timesteps.")
        sequence_channels = int(sequence_channels)
        sequence_timesteps = int(sequence_timesteps)
        if sequence_channels * sequence_timesteps != int(active.values.shape[1]):
            raise ValueError(
                f"Expected sequence_channels * sequence_timesteps to equal {active.values.shape[1]}, "
                f"got {sequence_channels} * {sequence_timesteps}."
            )
        ordered_values = np.empty(
            (
                ordered_slots.shape[0],
                sequence_channels,
                sequence_timesteps,
                active.values.shape[2],
                active.values.shape[3],
            ),
            dtype=np.float32,
        )
        _copy_ordered_active_values_to_sequence_jit(
            active.values,
            ordered_slots,
            ordered_values,
            sequence_channels,
            sequence_timesteps,
        )
    else:
        ordered_values = np.empty((ordered_slots.shape[0],) + active.values.shape[1:], dtype=np.float32)
        _copy_ordered_active_values_jit(active.values, ordered_slots, ordered_values)

    if active.cell_scale_mode == "level_idx":
        nx_tiles, ny_tiles = _nx_ny_tiles(active.domain)
        max_level = int(active.domain["max_level_idx"])
        centers, levels, tiles, xy_idx, cell_ids = _export_active_level_idx_metadata_jit(
            active.tile_ix,
            active.tile_iy,
            active.level_idx,
            active.x_idx,
            active.y_idx,
            ordered_slots,
            x0[order],
            y0[order],
            scale[order],
            max_level,
            float(active.domain["xmin"]),
            float(active.domain["xmax"]),
            float(active.domain["ymin"]),
            float(active.domain["ymax"]),
            float(active.domain["tile_width"]),
            float(active.domain["tile_height"]),
            max_level,
            max_level,
            6,
            _ceil_log2(nx_tiles),
            _ceil_log2(ny_tiles),
        )
        return ordered_values, {
            "centers": centers,
            "levels": levels,
            "tiles": tiles,
            "xy_idx": xy_idx,
            "cell_ids": cell_ids,
            "domain": dict(active.domain),
            "cell_scale_mode": active.cell_scale_mode,
        }

    if output_layout == "sequence":
        ordered_values = np.empty((ordered_slots.shape[0],) + active.values.shape[1:], dtype=np.float32)
        _copy_ordered_active_values_jit(active.values, ordered_slots, ordered_values)

    ordered = FlatTopology(
        values=ordered_values,
        tile_ix=active.tile_ix[ordered_slots].astype(np.int32, copy=True),
        tile_iy=active.tile_iy[ordered_slots].astype(np.int32, copy=True),
        level_idx=active.level_idx[ordered_slots].astype(np.int16, copy=True),
        x_idx=active.x_idx[ordered_slots].astype(np.int32, copy=True),
        y_idx=active.y_idx[ordered_slots].astype(np.int32, copy=True),
        domain=dict(active.domain),
        cell_scale_mode=active.cell_scale_mode,
    )
    out, out_meta = _export_ordered_topology(ordered, x0[order], y0[order], scale[order])
    if output_layout == "sequence":
        out = _flat_output_to_sequence_array(out, int(sequence_channels), int(sequence_timesteps))
    return out, out_meta


def _export_source_active_topology(
    active: SequenceSourceActiveTopology,
    *,
    output_layout: str = "flat",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if output_layout not in {"flat", "sequence"}:
        raise ValueError(f"Unknown output_layout={output_layout!r}; expected 'flat' or 'sequence'.")
    slots = active.active_slots[:active.active_count]
    x0, y0, scale = _active_leaf_fine_bounds(active, slots)
    nx_tiles, ny_tiles = _nx_ny_tiles(active.domain)
    max_level = int(active.domain["max_level_idx"])
    bits = max_level + _ceil_log2(max(nx_tiles, ny_tiles))
    keys = _morton_keys_jit(x0, y0, bits)
    order = np.argsort(keys, kind="stable")
    ordered_slots = slots[order]

    if output_layout == "sequence":
        ordered_values = np.empty(
            (
                ordered_slots.shape[0],
                int(active.sequence_channels),
                int(active.sequence_timesteps),
                active.values.shape[2],
                active.values.shape[3],
            ),
            dtype=np.float32,
        )
        _copy_ordered_source_active_values_to_sequence_jit(
            active.source_sequence,
            active.values,
            int(active.initial_slots),
            ordered_slots,
            ordered_values,
        )
    else:
        ordered_values = np.empty((ordered_slots.shape[0],) + active.values.shape[1:], dtype=np.float32)
        _copy_ordered_source_active_values_jit(
            active.source_sequence,
            active.values,
            int(active.initial_slots),
            int(active.sequence_channels),
            ordered_slots,
            ordered_values,
        )

    if active.cell_scale_mode == "level_idx":
        nx_tiles, ny_tiles = _nx_ny_tiles(active.domain)
        max_level = int(active.domain["max_level_idx"])
        centers, levels, tiles, xy_idx, cell_ids = _export_active_level_idx_metadata_jit(
            active.tile_ix,
            active.tile_iy,
            active.level_idx,
            active.x_idx,
            active.y_idx,
            ordered_slots,
            x0[order],
            y0[order],
            scale[order],
            max_level,
            float(active.domain["xmin"]),
            float(active.domain["xmax"]),
            float(active.domain["ymin"]),
            float(active.domain["ymax"]),
            float(active.domain["tile_width"]),
            float(active.domain["tile_height"]),
            max_level,
            max_level,
            6,
            _ceil_log2(nx_tiles),
            _ceil_log2(ny_tiles),
        )
        return ordered_values, {
            "centers": centers,
            "levels": levels,
            "tiles": tiles,
            "xy_idx": xy_idx,
            "cell_ids": cell_ids,
            "domain": dict(active.domain),
            "cell_scale_mode": active.cell_scale_mode,
        }

    if output_layout == "sequence":
        flat_values = np.empty((ordered_slots.shape[0],) + active.values.shape[1:], dtype=np.float32)
        _copy_ordered_source_active_values_jit(
            active.source_sequence,
            active.values,
            int(active.initial_slots),
            int(active.sequence_channels),
            ordered_slots,
            flat_values,
        )
        ordered_values_for_meta = flat_values
    else:
        ordered_values_for_meta = ordered_values

    ordered = FlatTopology(
        values=ordered_values_for_meta,
        tile_ix=active.tile_ix[ordered_slots].astype(np.int32, copy=True),
        tile_iy=active.tile_iy[ordered_slots].astype(np.int32, copy=True),
        level_idx=active.level_idx[ordered_slots].astype(np.int16, copy=True),
        x_idx=active.x_idx[ordered_slots].astype(np.int32, copy=True),
        y_idx=active.y_idx[ordered_slots].astype(np.int32, copy=True),
        domain=dict(active.domain),
        cell_scale_mode=active.cell_scale_mode,
    )
    out, out_meta = _export_ordered_topology(ordered, x0[order], y0[order], scale[order])
    if output_layout == "sequence":
        out = _flat_output_to_sequence_array(out, int(active.sequence_channels), int(active.sequence_timesteps))
    return out, out_meta


def _run_array_regrid(
    top: FlatTopology,
    *,
    tol_frac: Tol,
    channel: Channel,
    max_passes: int,
    coarsen_ratio: float,
    adapt_nearby: int,
    capacity: int,
) -> Tuple[Union[FlatTopology, ActiveTopology], Dict[str, float], Dict[str, int]]:
    _ensure_capacity(top.values.shape[0], capacity)
    channels = _normalize_channels(channel, top.values.shape[1])
    timings = {"active_init": 0.0, "refine": 0.0, "protect": 0.0, "coarsen": 0.0, "validate": 0.0}
    counts = {
        "balance_refine_parents": 0,
        "coarsen_passes": 0,
        "coarsen_accepted": 0,
        "final_appended_slots": 0,
        "final_original_slots": int(top.values.shape[0]),
        "initial_slots": int(top.values.shape[0]),
        "peak_slots": int(top.values.shape[0]),
        "protected_refine_parents": 0,
        "refine_calls": 0,
        "refine_parents": 0,
    }
    if not channels:
        return top, timings, counts

    active_init_start = time.perf_counter()
    ranges = _initial_channel_ranges(top, channels)
    active, _ = _active_from_flat(top, capacity=capacity)
    timings["active_init"] = time.perf_counter() - active_init_start

    active, active_timings, active_counts = _run_active_regrid(
        active,
        ranges=ranges,
        channels=channels,
        tol_frac=tol_frac,
        max_passes=max_passes,
        coarsen_ratio=coarsen_ratio,
        adapt_nearby=adapt_nearby,
    )
    timings.update(active_timings)
    counts.update(active_counts)
    return active, timings, counts


def _run_array_regrid_parity(
    top: FlatTopology,
    *,
    tol_frac: Tol,
    channel: Channel,
    max_passes: int,
    coarsen_ratio: float,
    adapt_nearby: int,
    capacity: int,
) -> Tuple[Union[FlatTopology, ActiveTopology], Dict[str, float], Dict[str, int]]:
    _ensure_capacity(top.values.shape[0], capacity)
    channels = _normalize_channels(channel, top.values.shape[1])
    timings = {"active_init": 0.0, "refine": 0.0, "protect": 0.0, "coarsen": 0.0, "validate": 0.0}
    counts = {
        "balance_refine_parents": 0,
        "coarsen_passes": 0,
        "coarsen_accepted": 0,
        "final_appended_slots": 0,
        "final_original_slots": int(top.values.shape[0]),
        "initial_slots": int(top.values.shape[0]),
        "peak_slots": int(top.values.shape[0]),
        "protected_refine_parents": 0,
        "repair_invalid_slots": 0,
        "repair_refine_calls": 0,
        "repair_refine_parents": 0,
        "refine_calls": 0,
        "refine_parents": 0,
    }
    if not channels:
        return top, timings, counts

    active_init_start = time.perf_counter()
    ranges = _initial_channel_ranges(top, channels)
    active, _ = _active_from_flat(top, capacity=capacity)
    timings["active_init"] = time.perf_counter() - active_init_start

    active, active_timings, active_counts = _run_active_regrid_parity(
        active,
        ranges=ranges,
        channels=channels,
        tol_frac=tol_frac,
        max_passes=max_passes,
        coarsen_ratio=coarsen_ratio,
        adapt_nearby=adapt_nearby,
    )
    timings.update(active_timings)
    counts.update(active_counts)
    return active, timings, counts


def _run_array_regrid_from_sequence(
    sequence: np.ndarray,
    top: FlatTopology,
    *,
    tol_frac: Tol,
    channel: Channel,
    max_passes: int,
    coarsen_ratio: float,
    adapt_nearby: int,
    capacity: int,
) -> Tuple[ActiveTopology, Dict[str, float], Dict[str, int]]:
    if sequence.ndim != 5:
        raise ValueError(f"Expected sequence shape (N,C,T,H,W), got {sequence.shape}.")
    channels = _normalize_channels(channel, int(sequence.shape[1]) * int(sequence.shape[2]))
    timings = {"active_init": 0.0, "refine": 0.0, "protect": 0.0, "coarsen": 0.0, "validate": 0.0}
    counts = {
        "balance_refine_parents": 0,
        "coarsen_passes": 0,
        "coarsen_accepted": 0,
        "final_appended_slots": 0,
        "final_original_slots": int(sequence.shape[0]),
        "initial_slots": int(sequence.shape[0]),
        "peak_slots": int(sequence.shape[0]),
        "protected_refine_parents": 0,
        "refine_calls": 0,
        "refine_parents": 0,
    }

    active_init_start = time.perf_counter()
    ranges = _initial_sequence_channel_ranges(sequence, channels)
    active, _ = _active_from_sequence(sequence, top, capacity=capacity)
    timings["active_init"] = time.perf_counter() - active_init_start

    active, active_timings, active_counts = _run_active_regrid(
        active,
        ranges=ranges,
        channels=channels,
        tol_frac=tol_frac,
        max_passes=max_passes,
        coarsen_ratio=coarsen_ratio,
        adapt_nearby=adapt_nearby,
    )
    timings.update(active_timings)
    counts.update(active_counts)
    return active, timings, counts


def _run_array_regrid_from_sequence_parity(
    sequence: np.ndarray,
    top: FlatTopology,
    *,
    tol_frac: Tol,
    channel: Channel,
    max_passes: int,
    coarsen_ratio: float,
    adapt_nearby: int,
    capacity: int,
) -> Tuple[ActiveTopology, Dict[str, float], Dict[str, int]]:
    if sequence.ndim != 5:
        raise ValueError(f"Expected sequence shape (N,C,T,H,W), got {sequence.shape}.")
    channels = _normalize_channels(channel, int(sequence.shape[1]) * int(sequence.shape[2]))
    timings = {"active_init": 0.0, "refine": 0.0, "protect": 0.0, "coarsen": 0.0, "validate": 0.0}
    counts = {
        "balance_refine_parents": 0,
        "coarsen_passes": 0,
        "coarsen_accepted": 0,
        "final_appended_slots": 0,
        "final_original_slots": int(sequence.shape[0]),
        "initial_slots": int(sequence.shape[0]),
        "peak_slots": int(sequence.shape[0]),
        "protected_refine_parents": 0,
        "refine_calls": 0,
        "refine_parents": 0,
    }

    active_init_start = time.perf_counter()
    ranges = _initial_sequence_channel_ranges(sequence, channels)
    active, _ = _active_from_sequence(sequence, top, capacity=capacity)
    timings["active_init"] = time.perf_counter() - active_init_start

    active, active_timings, active_counts = _run_active_regrid_parity(
        active,
        ranges=ranges,
        channels=channels,
        tol_frac=tol_frac,
        max_passes=max_passes,
        coarsen_ratio=coarsen_ratio,
        adapt_nearby=adapt_nearby,
    )
    timings.update(active_timings)
    counts.update(active_counts)
    return active, timings, counts


def _run_array_regrid_from_sequence_source(
    sequence: np.ndarray,
    top: FlatTopology,
    *,
    tol_frac: Tol,
    channel: Channel,
    max_passes: int,
    coarsen_ratio: float,
    adapt_nearby: int,
    capacity: int,
) -> Tuple[SequenceSourceActiveTopology, Dict[str, float], Dict[str, int]]:
    if sequence.ndim != 5:
        raise ValueError(f"Expected sequence shape (N,C,T,H,W), got {sequence.shape}.")
    channels = _normalize_channels(channel, int(sequence.shape[1]) * int(sequence.shape[2]))
    timings = {"active_init": 0.0, "refine": 0.0, "protect": 0.0, "coarsen": 0.0, "validate": 0.0}
    counts = {
        "balance_refine_parents": 0,
        "coarsen_passes": 0,
        "coarsen_accepted": 0,
        "final_appended_slots": 0,
        "final_original_slots": int(sequence.shape[0]),
        "initial_slots": int(sequence.shape[0]),
        "peak_slots": int(sequence.shape[0]),
        "protected_refine_parents": 0,
        "refine_calls": 0,
        "refine_parents": 0,
    }

    active_init_start = time.perf_counter()
    ranges = _initial_sequence_channel_ranges(sequence, channels)
    active, _ = _active_from_sequence_source(sequence, top, capacity=capacity)
    counts["source_sequence_copied"] = int(active.source_sequence_copied)
    counts["source_workspace_slots"] = int(active.values.shape[0])
    counts["source_workspace_bytes"] = int(active.values.nbytes)
    timings["active_init"] = time.perf_counter() - active_init_start

    active, active_timings, active_counts = _run_source_active_regrid(
        active,
        ranges=ranges,
        channels=channels,
        tol_frac=tol_frac,
        max_passes=max_passes,
        coarsen_ratio=coarsen_ratio,
        adapt_nearby=adapt_nearby,
    )
    timings.update(active_timings)
    counts.update(active_counts)
    return active, timings, counts


def _run_array_regrid_from_sequence_source_parity(
    sequence: np.ndarray,
    top: FlatTopology,
    *,
    tol_frac: Tol,
    channel: Channel,
    max_passes: int,
    coarsen_ratio: float,
    adapt_nearby: int,
    capacity: int,
) -> Tuple[SequenceSourceActiveTopology, Dict[str, float], Dict[str, int]]:
    if sequence.ndim != 5:
        raise ValueError(f"Expected sequence shape (N,C,T,H,W), got {sequence.shape}.")
    channels = _normalize_channels(channel, int(sequence.shape[1]) * int(sequence.shape[2]))
    timings = {"active_init": 0.0, "refine": 0.0, "protect": 0.0, "coarsen": 0.0, "validate": 0.0}
    counts = {
        "balance_refine_parents": 0,
        "coarsen_passes": 0,
        "coarsen_accepted": 0,
        "final_appended_slots": 0,
        "final_original_slots": int(sequence.shape[0]),
        "initial_slots": int(sequence.shape[0]),
        "peak_slots": int(sequence.shape[0]),
        "protected_refine_parents": 0,
        "refine_calls": 0,
        "refine_parents": 0,
    }

    active_init_start = time.perf_counter()
    ranges = _initial_sequence_channel_ranges(sequence, channels)
    active, _ = _active_from_sequence_source(sequence, top, capacity=capacity)
    counts["source_sequence_copied"] = int(active.source_sequence_copied)
    counts["source_workspace_slots"] = int(active.values.shape[0])
    counts["source_workspace_bytes"] = int(active.values.nbytes)
    timings["active_init"] = time.perf_counter() - active_init_start

    active, active_timings, active_counts = _run_source_active_regrid_parity(
        active,
        ranges=ranges,
        channels=channels,
        tol_frac=tol_frac,
        max_passes=max_passes,
        coarsen_ratio=coarsen_ratio,
        adapt_nearby=adapt_nearby,
    )
    timings.update(active_timings)
    counts.update(active_counts)
    return active, timings, counts


def _run_active_regrid(
    active: ActiveTopology,
    *,
    ranges: np.ndarray,
    channels: list[int],
    tol_frac: Tol,
    max_passes: int,
    coarsen_ratio: float,
    adapt_nearby: int,
) -> Tuple[ActiveTopology, Dict[str, float], Dict[str, int]]:
    timings = {"refine": 0.0, "protect": 0.0, "coarsen": 0.0, "validate": 0.0}
    initial_slots = int(active.active_count)
    counts = {
        "balance_refine_parents": 0,
        "coarsen_passes": 0,
        "coarsen_accepted": 0,
        "final_appended_slots": 0,
        "final_original_slots": initial_slots,
        "initial_slots": initial_slots,
        "peak_slots": int(active.next_slot),
        "protected_refine_parents": 0,
        "refine_calls": 0,
        "refine_parents": 0,
    }
    if not channels:
        return active, timings, counts

    refine_start = time.perf_counter()
    tol_fracs = _normalize_tolerances(tol_frac, len(channels))
    tols_refine = tol_fracs * np.abs(ranges)
    tols_coarsen = tols_refine * float(coarsen_ratio)

    for _ in range(max_passes):
        owner = _build_active_owner_grid(active)
        refine_slots = _compute_active_refine_slots(active, channels, tols_refine, owner)
        if refine_slots.size == 0:
            break
        counts["refine_calls"] += 1
        counts["refine_parents"] += int(refine_slots.size)
        active, _ = _active_refine_slots(active, refine_slots)
        active, _, balance_calls, balance_parents = _active_ensure_2to1_balance_counted(active)
        counts["refine_calls"] += int(balance_calls)
        counts["refine_parents"] += int(balance_parents)
        counts["balance_refine_parents"] += int(balance_parents)
    timings["refine"] = time.perf_counter() - refine_start

    protect_start = time.perf_counter()
    protected = np.zeros(int(active.active_slots.shape[0]), dtype=bool)
    if adapt_nearby > 0:
        protected = _active_protected_mask(active, channels, tols_coarsen)
        protected = _active_dilate_protected(active, protected, adapt_nearby)
        (
            active,
            protected,
            protected_calls,
            protected_parents,
            balance_calls,
            balance_parents,
        ) = _active_expand_protected_region_counted(active, protected, adapt_nearby)
        counts["refine_calls"] += int(protected_calls) + int(balance_calls)
        counts["refine_parents"] += int(protected_parents) + int(balance_parents)
        counts["protected_refine_parents"] += int(protected_parents)
        counts["balance_refine_parents"] += int(balance_parents)
    timings["protect"] = time.perf_counter() - protect_start

    coarsen_start = time.perf_counter()
    for _ in range(max_passes):
        active, protected, changed, accepted_count = _active_coarsen_once(active, protected, channels, tols_coarsen)
        counts["coarsen_passes"] += 1
        counts["coarsen_accepted"] += int(accepted_count)
        if not changed:
            break
    timings["coarsen"] = time.perf_counter() - coarsen_start

    validate_start = time.perf_counter()
    owner = _build_active_owner_grid(active)
    if _find_active_balance_refinements(active, owner).size:
        raise _ArrayFallback("invalid_topology")
    timings["validate"] = time.perf_counter() - validate_start
    counts["peak_slots"] = int(active.next_slot)
    final_slots = active.active_slots[: active.active_count]
    final_original = int(np.count_nonzero(final_slots < initial_slots))
    counts["final_original_slots"] = final_original
    counts["final_appended_slots"] = int(active.active_count) - final_original
    return active, timings, counts


def _run_source_active_regrid(
    active: SequenceSourceActiveTopology,
    *,
    ranges: np.ndarray,
    channels: list[int],
    tol_frac: Tol,
    max_passes: int,
    coarsen_ratio: float,
    adapt_nearby: int,
) -> Tuple[SequenceSourceActiveTopology, Dict[str, float], Dict[str, int]]:
    timings = {"refine": 0.0, "protect": 0.0, "coarsen": 0.0, "validate": 0.0}
    initial_slots = int(active.active_count)
    counts = {
        "balance_refine_parents": 0,
        "coarsen_passes": 0,
        "coarsen_accepted": 0,
        "final_appended_slots": 0,
        "final_original_slots": initial_slots,
        "initial_slots": initial_slots,
        "peak_slots": int(active.next_slot),
        "protected_refine_parents": 0,
        "refine_calls": 0,
        "refine_parents": 0,
    }
    if not channels:
        return active, timings, counts

    refine_start = time.perf_counter()
    tol_fracs = _normalize_tolerances(tol_frac, len(channels))
    tols_refine = tol_fracs * np.abs(ranges)
    tols_coarsen = tols_refine * float(coarsen_ratio)

    for _ in range(max_passes):
        owner = _build_active_owner_grid(active)
        refine_slots = _compute_active_refine_slots_from_source(active, channels, tols_refine, owner)
        if refine_slots.size == 0:
            break
        counts["refine_calls"] += 1
        counts["refine_parents"] += int(refine_slots.size)
        active, _ = _active_refine_slots_from_source(active, refine_slots)
        active, _, balance_calls, balance_parents = _active_ensure_2to1_balance_counted_from_source(active)
        counts["refine_calls"] += int(balance_calls)
        counts["refine_parents"] += int(balance_parents)
        counts["balance_refine_parents"] += int(balance_parents)
    timings["refine"] = time.perf_counter() - refine_start

    protect_start = time.perf_counter()
    protected = np.zeros(int(active.active_slots.shape[0]), dtype=bool)
    if adapt_nearby > 0:
        protected = _active_protected_mask_from_source(active, channels, tols_coarsen)
        protected = _active_dilate_protected(active, protected, adapt_nearby)
        (
            active,
            protected,
            protected_calls,
            protected_parents,
            balance_calls,
            balance_parents,
        ) = _active_expand_protected_region_counted_from_source(active, protected, adapt_nearby)
        counts["refine_calls"] += int(protected_calls) + int(balance_calls)
        counts["refine_parents"] += int(protected_parents) + int(balance_parents)
        counts["protected_refine_parents"] += int(protected_parents)
        counts["balance_refine_parents"] += int(balance_parents)
    timings["protect"] = time.perf_counter() - protect_start

    coarsen_start = time.perf_counter()
    for _ in range(max_passes):
        active, protected, changed, accepted_count = _active_coarsen_once_from_source(
            active,
            protected,
            channels,
            tols_coarsen,
        )
        counts["coarsen_passes"] += 1
        counts["coarsen_accepted"] += int(accepted_count)
        if not changed:
            break
    timings["coarsen"] = time.perf_counter() - coarsen_start

    validate_start = time.perf_counter()
    owner = _build_active_owner_grid(active)
    if _find_active_balance_refinements(active, owner).size:
        raise _ArrayFallback("invalid_topology")
    timings["validate"] = time.perf_counter() - validate_start
    counts["peak_slots"] = int(active.next_slot)
    final_slots = active.active_slots[: active.active_count]
    final_original = int(np.count_nonzero(final_slots < initial_slots))
    counts["final_original_slots"] = final_original
    counts["final_appended_slots"] = int(active.active_count) - final_original
    return active, timings, counts


def _run_active_regrid_parity(
    active: ActiveTopology,
    *,
    ranges: np.ndarray,
    channels: list[int],
    tol_frac: Tol,
    max_passes: int,
    coarsen_ratio: float,
    adapt_nearby: int,
) -> Tuple[ActiveTopology, Dict[str, float], Dict[str, int]]:
    """Object-parity active regrid for copy-backed values."""
    timings = {"refine": 0.0, "protect": 0.0, "coarsen": 0.0, "validate": 0.0}
    initial_slots = int(active.active_count)
    counts = {
        "balance_refine_parents": 0,
        "coarsen_passes": 0,
        "coarsen_accepted": 0,
        "final_appended_slots": 0,
        "final_original_slots": initial_slots,
        "initial_slots": initial_slots,
        "peak_slots": int(active.next_slot),
        "protected_refine_parents": 0,
        "refine_calls": 0,
        "refine_parents": 0,
        "repair_invalid_slots": 0,
        "repair_refine_calls": 0,
        "repair_refine_parents": 0,
    }
    if not channels:
        return active, timings, counts

    refine_start = time.perf_counter()
    tol_fracs = _normalize_tolerances(tol_frac, len(channels))
    tols_refine = tol_fracs * np.abs(ranges)
    tols_coarsen = tols_refine * float(coarsen_ratio)

    for _ in range(max_passes):
        owner = _build_active_owner_grid(active)
        refine_slots = _compute_active_refine_slots_object_style(active, channels, tols_refine, owner)
        if refine_slots.size == 0:
            break
        counts["refine_calls"] += 1
        counts["refine_parents"] += int(refine_slots.size)
        active, _ = _active_refine_slots(active, refine_slots)
        active, _, balance_calls, balance_parents = _active_ensure_2to1_balance_counted(active)
        counts["refine_calls"] += int(balance_calls)
        counts["refine_parents"] += int(balance_parents)
        counts["balance_refine_parents"] += int(balance_parents)
    timings["refine"] = time.perf_counter() - refine_start

    protect_start = time.perf_counter()
    protected = np.zeros(int(active.active_slots.shape[0]), dtype=bool)
    if adapt_nearby > 0:
        protected = _active_protected_mask(active, channels, tols_coarsen)
        protected = _active_dilate_protected(active, protected, adapt_nearby)
        (
            active,
            protected,
            protected_calls,
            protected_parents,
            balance_calls,
            balance_parents,
        ) = _active_expand_protected_region_counted(active, protected, adapt_nearby)
        counts["refine_calls"] += int(protected_calls) + int(balance_calls)
        counts["refine_parents"] += int(protected_parents) + int(balance_parents)
        counts["protected_refine_parents"] += int(protected_parents)
        counts["balance_refine_parents"] += int(balance_parents)
    timings["protect"] = time.perf_counter() - protect_start

    coarsen_start = time.perf_counter()
    for _ in range(max_passes):
        active, protected, changed, accepted_count = _active_coarsen_once_conservative(
            active,
            protected,
            channels,
            tols_coarsen,
        )
        counts["coarsen_passes"] += 1
        counts["coarsen_accepted"] += int(accepted_count)
        if not changed:
            break
    timings["coarsen"] = time.perf_counter() - coarsen_start

    validate_start = time.perf_counter()
    active, protected, repair_counts = _repair_active_balance_or_raise(active, protected)
    timings["validate"] = time.perf_counter() - validate_start
    counts.update(repair_counts)
    counts["peak_slots"] = int(active.next_slot)
    final_slots = active.active_slots[: active.active_count]
    final_original = int(np.count_nonzero(final_slots < initial_slots))
    counts["final_original_slots"] = final_original
    counts["final_appended_slots"] = int(active.active_count) - final_original
    return active, timings, counts


def _run_source_active_regrid_parity(
    active: SequenceSourceActiveTopology,
    *,
    ranges: np.ndarray,
    channels: list[int],
    tol_frac: Tol,
    max_passes: int,
    coarsen_ratio: float,
    adapt_nearby: int,
) -> Tuple[SequenceSourceActiveTopology, Dict[str, float], Dict[str, int]]:
    """Object-parity active regrid for source-backed sequence values."""
    timings = {"refine": 0.0, "protect": 0.0, "coarsen": 0.0, "validate": 0.0}
    initial_slots = int(active.active_count)
    counts = {
        "balance_refine_parents": 0,
        "coarsen_passes": 0,
        "coarsen_accepted": 0,
        "final_appended_slots": 0,
        "final_original_slots": initial_slots,
        "initial_slots": initial_slots,
        "peak_slots": int(active.next_slot),
        "protected_refine_parents": 0,
        "refine_calls": 0,
        "refine_parents": 0,
        "repair_invalid_slots": 0,
        "repair_refine_calls": 0,
        "repair_refine_parents": 0,
    }
    if not channels:
        return active, timings, counts

    refine_start = time.perf_counter()
    tol_fracs = _normalize_tolerances(tol_frac, len(channels))
    tols_refine = tol_fracs * np.abs(ranges)
    tols_coarsen = tols_refine * float(coarsen_ratio)

    for _ in range(max_passes):
        owner = _build_active_owner_grid(active)
        refine_slots = _compute_active_refine_slots_from_source_object_style(active, channels, tols_refine, owner)
        if refine_slots.size == 0:
            break
        counts["refine_calls"] += 1
        counts["refine_parents"] += int(refine_slots.size)
        active, _ = _active_refine_slots_from_source(active, refine_slots)
        active, _, balance_calls, balance_parents = _active_ensure_2to1_balance_counted_from_source(active)
        counts["refine_calls"] += int(balance_calls)
        counts["refine_parents"] += int(balance_parents)
        counts["balance_refine_parents"] += int(balance_parents)
    timings["refine"] = time.perf_counter() - refine_start

    protect_start = time.perf_counter()
    protected = np.zeros(int(active.active_slots.shape[0]), dtype=bool)
    if adapt_nearby > 0:
        protected = _active_protected_mask_from_source(active, channels, tols_coarsen)
        protected = _active_dilate_protected(active, protected, adapt_nearby)
        (
            active,
            protected,
            protected_calls,
            protected_parents,
            balance_calls,
            balance_parents,
        ) = _active_expand_protected_region_counted_from_source(active, protected, adapt_nearby)
        counts["refine_calls"] += int(protected_calls) + int(balance_calls)
        counts["refine_parents"] += int(protected_parents) + int(balance_parents)
        counts["protected_refine_parents"] += int(protected_parents)
        counts["balance_refine_parents"] += int(balance_parents)
    timings["protect"] = time.perf_counter() - protect_start

    coarsen_start = time.perf_counter()
    for _ in range(max_passes):
        active, protected, changed, accepted_count = _active_coarsen_once_from_source_conservative(
            active,
            protected,
            channels,
            tols_coarsen,
        )
        counts["coarsen_passes"] += 1
        counts["coarsen_accepted"] += int(accepted_count)
        if not changed:
            break
    timings["coarsen"] = time.perf_counter() - coarsen_start

    validate_start = time.perf_counter()
    active, protected, repair_counts = _repair_source_active_balance_or_raise(active, protected)
    timings["validate"] = time.perf_counter() - validate_start
    counts.update(repair_counts)
    counts["peak_slots"] = int(active.next_slot)
    final_slots = active.active_slots[: active.active_count]
    final_original = int(np.count_nonzero(final_slots < initial_slots))
    counts["final_original_slots"] = final_original
    counts["final_appended_slots"] = int(active.active_count) - final_original
    return active, timings, counts


def object_regrid_from_tensor(
    data: np.ndarray,
    meta: Dict[str, Any],
    *,
    cell_scale_mode: Optional[str],
    tol_frac: Tol,
    channel: Channel,
    max_passes: int,
    coarsen_ratio: float = 0.25,
    adapt_nearby: int = 0,
    disable_warnings: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    start = time.perf_counter()
    qt = tensor_to_quadtree(data, meta, cell_scale_mode=cell_scale_mode)
    after_import = time.perf_counter()
    regrid(
        qt,
        tol_frac=tol_frac,
        channel=channel,
        max_passes=max_passes,
        coarsen_ratio=coarsen_ratio,
        adapt_nearby=adapt_nearby,
        disable_warnings=disable_warnings,
    )
    after_regrid = time.perf_counter()
    out, out_meta = quadtree_to_tensor(qt, return_tensor=False, cell_scale_mode=cell_scale_mode)
    end = time.perf_counter()
    return out, out_meta, {
        "backend": "object",
        "fallback_reason": None,
        **_array_regrid_param_status(
            tol_frac=tol_frac,
            channel=channel,
            max_passes=max_passes,
            coarsen_ratio=coarsen_ratio,
            adapt_nearby=adapt_nearby,
        ),
        "timings": {
            "tensor_to_quadtree": after_import - start,
            "regrid": after_regrid - after_import,
            "quadtree_to_tensor": end - after_regrid,
            "total": end - start,
        },
    }


def topology_is_2to1_balanced(meta: Dict[str, Any]) -> bool:
    centers = _as_numpy(meta["centers"])
    cell_scale_mode = meta.get("cell_scale_mode")
    if cell_scale_mode is None and centers.ndim == 2 and centers.shape[1] == 3:
        cell_scale_mode = "level_idx"
    dummy = np.zeros((len(meta["levels"]), 1, 1, 1), dtype=np.float32)
    try:
        top = _import_topology(dummy, meta, cell_scale_mode)
        owner = _build_owner_grid(top)
    except (KeyError, ValueError):
        return False
    return _find_balance_refinements(top, owner).size == 0


def array_regrid_from_sequence(
    data: np.ndarray,
    meta: Dict[str, Any],
    *,
    cell_scale_mode: Optional[str],
    tol_frac: Tol,
    channel: Channel,
    max_passes: int,
    coarsen_ratio: float = 0.25,
    adapt_nearby: int = 0,
    capacity: int = 8192,
    disable_warnings: bool = True,
    output_layout: str = "flat",
    value_storage: str = "copy",
    array_regrid_mode: str = "parity",
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    if output_layout not in {"flat", "sequence"}:
        raise ValueError(f"Unknown output_layout={output_layout!r}; expected 'flat' or 'sequence'.")
    if value_storage not in {"copy", "source"}:
        raise ValueError(f"Unknown value_storage={value_storage!r}; expected 'copy' or 'source'.")
    array_regrid_mode = _normalize_array_regrid_mode(array_regrid_mode)
    start = time.perf_counter()
    sequence: Optional[np.ndarray] = None
    try:
        sequence = _as_numpy(data)
        if sequence.ndim != 5:
            raise ValueError(f"Expected sequence shape (N,C,T,H,W), got {sequence.shape}.")
        top = _import_topology_metadata(int(sequence.shape[0]), meta, cell_scale_mode)
        after_import = time.perf_counter()
        if value_storage == "source":
            if array_regrid_mode == "parity":
                top, run_timings, run_counts = _run_array_regrid_from_sequence_source_parity(
                    sequence,
                    top,
                    tol_frac=tol_frac,
                    channel=channel,
                    max_passes=max_passes,
                    coarsen_ratio=coarsen_ratio,
                    adapt_nearby=adapt_nearby,
                    capacity=capacity,
                )
            else:
                top, run_timings, run_counts = _run_array_regrid_from_sequence_source(
                    sequence,
                    top,
                    tol_frac=tol_frac,
                    channel=channel,
                    max_passes=max_passes,
                    coarsen_ratio=coarsen_ratio,
                    adapt_nearby=adapt_nearby,
                    capacity=capacity,
                )
        else:
            if array_regrid_mode == "parity":
                top, run_timings, run_counts = _run_array_regrid_from_sequence_parity(
                    sequence,
                    top,
                    tol_frac=tol_frac,
                    channel=channel,
                    max_passes=max_passes,
                    coarsen_ratio=coarsen_ratio,
                    adapt_nearby=adapt_nearby,
                    capacity=capacity,
                )
            else:
                top, run_timings, run_counts = _run_array_regrid_from_sequence(
                    sequence,
                    top,
                    tol_frac=tol_frac,
                    channel=channel,
                    max_passes=max_passes,
                    coarsen_ratio=coarsen_ratio,
                    adapt_nearby=adapt_nearby,
                    capacity=capacity,
                )
        after_run = time.perf_counter()
        if isinstance(top, SequenceSourceActiveTopology):
            out, out_meta = _export_source_active_topology(top, output_layout=output_layout)
        else:
            out, out_meta = _export_active_topology(
                top,
                output_layout=output_layout,
                sequence_channels=int(sequence.shape[1]),
                sequence_timesteps=int(sequence.shape[2]),
            )
        end = time.perf_counter()
        return out, out_meta, {
            "backend": "array",
            "fallback_reason": None,
            **_array_regrid_config_status(capacity),
            **_array_regrid_param_status(
                tol_frac=tol_frac,
                channel=channel,
                max_passes=max_passes,
                coarsen_ratio=coarsen_ratio,
                adapt_nearby=adapt_nearby,
            ),
            "value_storage": value_storage,
            "output_layout": output_layout,
            "array_regrid_mode": array_regrid_mode,
            "timings": {
                "import": after_import - start,
                "run": after_run - after_import,
                "export": end - after_run,
                "total": end - start,
            },
            "run_timings": run_timings,
            "run_counts": run_counts,
        }
    except _ArrayFallback as exc:
        flat = _flatten_sequence_array(sequence if sequence is not None else _as_numpy(data))
        out, out_meta, status = object_regrid_from_tensor(
            flat,
            meta,
            cell_scale_mode=cell_scale_mode,
            tol_frac=tol_frac,
            channel=channel,
            max_passes=max_passes,
            coarsen_ratio=coarsen_ratio,
            adapt_nearby=adapt_nearby,
            disable_warnings=disable_warnings,
        )
        status["fallback_reason"] = exc.reason
        status["output_layout"] = output_layout
        status["value_storage"] = value_storage
        status["array_regrid_mode"] = array_regrid_mode
        status.update(_array_regrid_config_status(capacity))
        status.update(
            _array_regrid_param_status(
                tol_frac=tol_frac,
                channel=channel,
                max_passes=max_passes,
                coarsen_ratio=coarsen_ratio,
                adapt_nearby=adapt_nearby,
            )
        )
        if output_layout == "sequence":
            layout_start = time.perf_counter()
            out = _flat_output_to_sequence_array(out, int(sequence.shape[1]), int(sequence.shape[2]))
            layout_elapsed = time.perf_counter() - layout_start
            status["timings"]["sequence_layout"] = layout_elapsed
            status["timings"]["total"] += layout_elapsed
        return out, out_meta, status
    except ValueError:
        flat = _flatten_sequence_array(sequence if sequence is not None else _as_numpy(data))
        out, out_meta, status = object_regrid_from_tensor(
            flat,
            meta,
            cell_scale_mode=cell_scale_mode,
            tol_frac=tol_frac,
            channel=channel,
            max_passes=max_passes,
            coarsen_ratio=coarsen_ratio,
            adapt_nearby=adapt_nearby,
            disable_warnings=disable_warnings,
        )
        status["fallback_reason"] = "invalid_topology"
        status["output_layout"] = output_layout
        status["value_storage"] = value_storage
        status["array_regrid_mode"] = array_regrid_mode
        status.update(_array_regrid_config_status(capacity))
        status.update(
            _array_regrid_param_status(
                tol_frac=tol_frac,
                channel=channel,
                max_passes=max_passes,
                coarsen_ratio=coarsen_ratio,
                adapt_nearby=adapt_nearby,
            )
        )
        if output_layout == "sequence" and sequence is not None:
            layout_start = time.perf_counter()
            out = _flat_output_to_sequence_array(out, int(sequence.shape[1]), int(sequence.shape[2]))
            layout_elapsed = time.perf_counter() - layout_start
            status["timings"]["sequence_layout"] = layout_elapsed
            status["timings"]["total"] += layout_elapsed
        return out, out_meta, status


# === NATIVE (multi-scale) array regrid ===
@dataclass
class NativeActiveTopology:
    """Active-slot native-resolution topology.

    For an active slot ``s``, its patch is ``values_by_level[level_idx[s]][slot_value_pos[s]]``,
    shape ``(T*C, base_h << (L - level_idx[s]), base_w << (L - level_idx[s]))``, folded T-major.
    ``values_by_level`` is a per-level tuple (len ``L+1``) whose value arrays are sized per-level,
    while ``slot_value_pos`` and the topology arrays (``level_idx``, ``x_idx``, ...) are sized to
    the global ``capacity``.
    """

    values_by_level: Tuple[np.ndarray, ...]
    slot_value_pos: np.ndarray
    tile_ix: np.ndarray
    tile_iy: np.ndarray
    level_idx: np.ndarray
    x_idx: np.ndarray
    y_idx: np.ndarray
    active_slots: np.ndarray
    active_count: int
    next_slot: int
    level_next_pos: np.ndarray
    level_cap: np.ndarray
    domain: Dict[str, Any]
    cell_scale_mode: Optional[str]


def _native_patch_hw(domain: Dict[str, Any], level_idx: int) -> Tuple[int, int]:
    L = int(domain["max_level_idx"])
    base_h = int(domain["base_patch_h"])
    base_w = int(domain["base_patch_w"])
    depth = L - int(level_idx)
    return base_h << depth, base_w << depth


def _native_active_from_buckets(by_level, leaf_to_bucket, meta, *, C, T, capacity, cell_scale_mode):
    n_leaves = int(leaf_to_bucket.shape[0])
    _ensure_capacity(n_leaves, capacity)
    top = _import_topology_metadata(n_leaves, meta, cell_scale_mode)
    domain = dict(top.domain)
    L = int(domain["max_level_idx"])
    tc = T * C

    # Resolve base patch from any non-empty bucket (domain has no base_patch_h/w) and inject into domain.
    # A fully-empty topology has no probe level; route through the documented fallback contract
    # (-> object regrid_native, which fails loud on degenerate input) rather than letting a bare
    # StopIteration escape the entry's `except (_ArrayFallback, ValueError)`.
    probe_lvl = next((l for l in range(L + 1) if l in by_level and by_level[l].shape[0] > 0), None)
    if probe_lvl is None:
        raise _ArrayFallback("empty_topology")
    domain["base_patch_h"] = int(by_level[probe_lvl].shape[-2]) >> (L - probe_lvl)
    domain["base_patch_w"] = int(by_level[probe_lvl].shape[-1]) >> (L - probe_lvl)

    nx_tiles = int(round((float(domain["xmax"]) - float(domain["xmin"])) / float(domain["tile_width"])))
    ny_tiles = int(round((float(domain["ymax"]) - float(domain["ymin"])) / float(domain["tile_height"])))
    # 3x per-level cell budget: covers max cells per level (tiles * 4**l) plus in-flight refine/coarsen churn
    churn_slack = 3
    level_cap = np.empty(L + 1, dtype=np.int32)
    level_next_pos = np.zeros(L + 1, dtype=np.int32)
    vbl = []
    for lvl in range(L + 1):
        h_l, w_l = _native_patch_hw(domain, lvl)
        cap_l = max(int(churn_slack) * nx_tiles * ny_tiles * (4 ** lvl), 1)
        level_cap[lvl] = cap_l
        vbl.append(np.empty((cap_l, tc, h_l, w_l), dtype=np.float32))

    slot_value_pos = np.empty(int(capacity), dtype=np.int32)
    for s in range(n_leaves):
        lvl = int(top.level_idx[s])
        src_pos = int(leaf_to_bucket[s, 1])
        patch = np.ascontiguousarray(by_level[lvl][src_pos], dtype=np.float32)
        if patch.ndim == 4:  # (C, T, H, W) -> (T*C, H, W) folded T-major
            patch = patch.transpose(1, 0, 2, 3).reshape(tc, patch.shape[-2], patch.shape[-1])
        pos = int(level_next_pos[lvl])
        if pos >= int(level_cap[lvl]):
            raise _ArrayFallback("native_level_capacity")
        vbl[lvl][pos] = patch
        slot_value_pos[s] = pos
        level_next_pos[lvl] = pos + 1
    values_by_level = tuple(vbl)

    tile_ix = np.empty(int(capacity), dtype=np.int32); tile_ix[:n_leaves] = top.tile_ix
    tile_iy = np.empty(int(capacity), dtype=np.int32); tile_iy[:n_leaves] = top.tile_iy
    level_idx = np.empty(int(capacity), dtype=np.int16); level_idx[:n_leaves] = top.level_idx
    x_idx = np.empty(int(capacity), dtype=np.int32); x_idx[:n_leaves] = top.x_idx
    y_idx = np.empty(int(capacity), dtype=np.int32); y_idx[:n_leaves] = top.y_idx
    active_slots = np.empty(int(capacity), dtype=np.int32); active_slots[:n_leaves] = np.arange(n_leaves, dtype=np.int32)

    return NativeActiveTopology(
        values_by_level=values_by_level, slot_value_pos=slot_value_pos,
        tile_ix=tile_ix, tile_iy=tile_iy, level_idx=level_idx, x_idx=x_idx, y_idx=y_idx,
        active_slots=active_slots, active_count=n_leaves, next_slot=n_leaves,
        level_next_pos=level_next_pos, level_cap=level_cap, domain=domain, cell_scale_mode=cell_scale_mode)


def _native_active_channel_ranges(active: NativeActiveTopology, channels: list) -> np.ndarray:
    """Per-channel global value range (max - min) over all active levels.

    Parameters
    ----------
    active:
        Native topology struct. Only the first ``level_next_pos[lvl]`` rows of
        ``values_by_level[lvl]`` are valid; empty levels are skipped. MUST be a
        freshly-imported (pre-regrid) topology: after refine/coarsen the counted
        rows include logically-freed (dead) patches, which would corrupt the
        reduction. The assert below enforces this precondition.
    channels:
        Absolute indices into the folded T*C layout. Returned in the same order.

    Returns
    -------
    np.ndarray of shape ``(len(channels),)``, dtype float64 (matches the uniform
    twin ``_active_channel_ranges`` and the object ``channel_range``). Entry ``i``
    is the global ``max - min`` across all valid patches for ``channels[i]``.
    """
    # Valid only pre-regrid: every counted row must be an active slot (no dead rows yet).
    assert int(active.level_next_pos.sum()) == int(active.active_count), \
        "_native_active_channel_ranges must be called on a freshly-imported topology"
    L = len(active.values_by_level) - 1
    ranges = np.zeros(len(channels), dtype=np.float64)
    for i, ch in enumerate(channels):
        v_min = np.inf
        v_max = -np.inf
        for lvl in range(L + 1):
            n_valid = int(active.level_next_pos[lvl])
            if n_valid == 0:
                continue
            patch = active.values_by_level[lvl][:n_valid, ch, :, :]
            lvl_min = float(patch.min())
            lvl_max = float(patch.max())
            if lvl_min < v_min:
                v_min = lvl_min
            if lvl_max > v_max:
                v_max = lvl_max
        if np.isfinite(v_min) and np.isfinite(v_max):
            ranges[i] = float(v_max - v_min)
    return ranges


@njit(boundscheck=False, inline="always")
def _native_sample_value_object_style_jit(
    value_tuple,
    slot_value_pos: np.ndarray,
    level_idx: np.ndarray,
    slot: int,
    ch: int,
    x: float,
    y: float,
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    xmin: float,
    ymin: float,
    tile_width: float,
    tile_height: float,
) -> float:
    """Native twin of ``_active_sample_value_object_style_jit``.

    Identical physical->u,v->px,py->bilinear math, except the neighbor's patch is
    its OWN per-level native patch ``value_tuple[level_idx[slot]][slot_value_pos[slot], ch]``
    whose ``(H_l, W_l)`` is read from that patch's shape (not a fixed shape).

    Signature note: one fewer argument than the uniform twin -- the twin's trailing
    ``max_level`` is unused dead weight there and is dropped here. Do NOT re-add it.
    """
    level = int(level_idx[slot])
    scale = 1 << level
    cell_w = tile_width / float(scale)
    cell_h = tile_height / float(scale)
    left = xmin + float(tile_ix[slot]) * tile_width + float(x_idx[slot]) * cell_w
    bottom = ymin + float(tile_iy[slot]) * tile_height + float(y_idx[slot]) * cell_h
    u = (x - left) / cell_w
    v = (y - bottom) / cell_h
    if u < 0.0:
        u = 0.0
    elif u > 1.0:
        u = 1.0
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    patch = value_tuple[level][int(slot_value_pos[slot]), ch]
    patch_h = patch.shape[0]
    patch_w = patch.shape[1]
    px = u * float(patch_w) - 0.5
    py = v * float(patch_h) - 0.5
    return _bilinear_sample_patch_jit(patch, px, py)


# NUMBA DATA-PASSING (pinned for native refine/coarsen kernels T4-T7):
# value_tuple is a numba UniTuple -- ``active.values_by_level`` as-is, a Python tuple
# of 4-D float32 C-contiguous arrays that all share dtype/ndim/layout, so they map to
# one numba type and the tuple is homogeneous. Numba 0.64 accepts runtime indexing
# ``value_tuple[lvl]`` (lvl a runtime int) on such a UniTuple; verified standalone, so
# no numba.typed.List fallback is needed. Inside: ``arr = value_tuple[lvl]`` ->
# ``patch2d = arr[pos, ch]``.
@njit(boundscheck=False, cache=True)
def _native_active_refine_criterion_jit(
    value_tuple,
    slot_value_pos: np.ndarray,
    level_idx: np.ndarray,
    active_slots: np.ndarray,
    active_count: int,
    owner: np.ndarray,
    channels: np.ndarray,
    tolerances: np.ndarray,
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    xmin: float,
    ymin: float,
    tile_width: float,
    tile_height: float,
    max_level: int,
) -> np.ndarray:
    mask = np.zeros(level_idx.shape[0], dtype=np.bool_)
    for tol_idx in range(channels.shape[0]):
        ch = int(channels[tol_idx])
        tol = float(tolerances[tol_idx])
        for pos in range(active_count):
            slot = int(active_slots[pos])
            leaf_level = int(level_idx[slot])
            if leaf_level >= max_level:
                continue
            patch = value_tuple[leaf_level][int(slot_value_pos[slot]), ch]
            patch_h = patch.shape[0]
            patch_w = patch.shape[1]
            v_min = patch[0, 0]
            v_max = v_min
            for yy in range(patch_h):
                for xx in range(patch_w):
                    val = patch[yy, xx]
                    if val < v_min:
                        v_min = val
                    elif val > v_max:
                        v_max = val
            internal_err = float(v_max - v_min)
            cell_w = tile_width / float(1 << leaf_level)
            cell_h = tile_height / float(1 << leaf_level)
            cx = xmin + float(tile_ix[slot]) * tile_width + (float(x_idx[slot]) + 0.5) * cell_w
            cy = ymin + float(tile_iy[slot]) * tile_height + (float(y_idx[slot]) + 0.5) * cell_h
            center = _bilinear_sample_patch_jit(
                patch,
                0.5 * float(patch_w) - 0.5,
                0.5 * float(patch_h) - 0.5,
            )
            boundary_err = 0.0
            sample_x = np.empty(4, dtype=np.float64)
            sample_y = np.empty(4, dtype=np.float64)
            sample_x[0] = cx - cell_w
            sample_y[0] = cy
            sample_x[1] = cx + cell_w
            sample_y[1] = cy
            sample_x[2] = cx
            sample_y[2] = cy - cell_h
            sample_x[3] = cx
            sample_y[3] = cy + cell_h
            for direction in range(4):
                neighbor = _owner_slot_for_physical_point_jit(
                    owner,
                    xmin,
                    ymin,
                    tile_width,
                    tile_height,
                    max_level,
                    sample_x[direction],
                    sample_y[direction],
                )
                neighbor_value = 0.0
                if neighbor >= 0:
                    neighbor_value = _native_sample_value_object_style_jit(
                        value_tuple,
                        slot_value_pos,
                        level_idx,
                        neighbor,
                        ch,
                        sample_x[direction],
                        sample_y[direction],
                        tile_ix,
                        tile_iy,
                        x_idx,
                        y_idx,
                        xmin,
                        ymin,
                        tile_width,
                        tile_height,
                    )
                diff = abs(float(center - neighbor_value))
                if diff > boundary_err:
                    boundary_err = diff
            if max(internal_err, boundary_err) > tol:
                mask[slot] = True
    return mask


def _compute_native_active_refine_slots(
    active: NativeActiveTopology,
    channels: list,
    tolerances: np.ndarray,
    owner: np.ndarray,
) -> np.ndarray:
    if not channels:
        return np.empty(0, dtype=np.int32)
    domain = active.domain
    mask = _native_active_refine_criterion_jit(
        active.values_by_level,
        active.slot_value_pos,
        active.level_idx,
        active.active_slots,
        int(active.active_count),
        owner,
        np.asarray(channels, dtype=np.int64),
        tolerances.astype(np.float64, copy=False),
        active.tile_ix,
        active.tile_iy,
        active.x_idx,
        active.y_idx,
        float(domain["xmin"]),
        float(domain["ymin"]),
        float(domain["tile_width"]),
        float(domain["tile_height"]),
        int(domain["max_level_idx"]),
    )
    return np.flatnonzero(mask).astype(np.int32, copy=False)


def _native_slot_center(active: NativeActiveTopology, slot: int) -> Tuple[float, float]:
    """Physical center (cx, cy) of an active native slot's cell (pure Python)."""
    domain = active.domain
    xmin = float(domain["xmin"])
    ymin = float(domain["ymin"])
    tile_width = float(domain["tile_width"])
    tile_height = float(domain["tile_height"])
    level = int(active.level_idx[slot])
    cell_w = tile_width / float(1 << level)
    cell_h = tile_height / float(1 << level)
    cx = xmin + float(active.tile_ix[slot]) * tile_width + (float(active.x_idx[slot]) + 0.5) * cell_w
    cy = ymin + float(active.tile_iy[slot]) * tile_height + (float(active.y_idx[slot]) + 0.5) * cell_h
    return cx, cy


# value_tuple WRITE path (T4 native refine fill): the per-level value arrays differ only in
# (H_l, W_l) -- same dtype/ndim/C-layout -- so they share ONE numba type and value_tuple is a
# homogeneous UniTuple (verified: numba.typeof -> UniTuple(array(float32, 4d, C) x (L+1));
# shape is not part of numba's array type). That homogeneity is REQUIRED: runtime indexing
# value_tuple[lvl] for a runtime-int lvl only compiles because every element unifies to the
# same type -- do NOT introduce a non-C-contiguous or different-dtype level array. Reads work
# (the criterion kernel relies on it); WRITES use element loops
# (arr = value_tuple[lc]; arr[r, fc, yy, xx] = ...) -- slice-assignment is avoided; the element
# loop is verified to compile and write correctly for both source (parent, level lp) and
# destination (child, level lp+1) at runtime-varying levels.
@njit(boundscheck=False, cache=True)
def _native_active_refine_slots_jit(
    value_tuple,
    slot_value_pos: np.ndarray,
    level_next_pos: np.ndarray,
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    level_idx: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    active_slots: np.ndarray,
    active_count: int,
    next_slot: int,
    refine_mask: np.ndarray,
    protected: np.ndarray,
    has_protected: bool,
) -> Tuple[np.ndarray, int, int]:
    # Topology bookkeeping replicates the uniform twin _active_refine_slots_jit EXACTLY
    # (it is value-agnostic). The only change is the fill: a native quadrant SLICE (no
    # upsample) into per-level child value rows instead of the uniform upsample.
    new_active_slots = np.empty_like(active_slots)
    out_count = 0
    child_dx = (0, 1, 0, 1)
    child_dy = (0, 0, 1, 1)

    for active_pos in range(active_count):
        slot = int(active_slots[active_pos])
        if not refine_mask[slot]:
            new_active_slots[out_count] = slot
            out_count += 1
            continue

        child_start = next_slot
        next_slot = child_start + 4
        parent_level = int(level_idx[slot])
        child_level = np.int16(parent_level + 1)
        child_x0 = int(x_idx[slot]) * 2
        child_y0 = int(y_idx[slot]) * 2
        parent_tile_ix = tile_ix[slot]
        parent_tile_iy = tile_iy[slot]
        parent_protected = False
        if has_protected:
            parent_protected = bool(protected[slot])
            protected[slot] = False

        # Native quadrant fill source: the parent's own per-level native patch.
        parent_patch = value_tuple[parent_level][int(slot_value_pos[slot])]
        tc = parent_patch.shape[0]
        Hp = parent_patch.shape[1]
        Wp = parent_patch.shape[2]
        hm = Hp // 2
        wm = Wp // 2
        child_arr = value_tuple[parent_level + 1]

        for child in range(4):
            child_slot = child_start + child
            new_active_slots[out_count + child] = child_slot
            tile_ix[child_slot] = parent_tile_ix
            tile_iy[child_slot] = parent_tile_iy
            level_idx[child_slot] = child_level
            x_idx[child_slot] = child_x0 + child_dx[child]
            y_idx[child_slot] = child_y0 + child_dy[child]
            if has_protected:
                protected[child_slot] = parent_protected

            # child 0=SW, 1=SE, 2=NW, 3=NE; patch row 0 = bottom (low y), col 0 = left (low x).
            src_y0 = 0
            src_x0 = 0
            if child >= 2:
                src_y0 = hm
            if child == 1 or child == 3:
                src_x0 = wm
            r = int(level_next_pos[parent_level + 1])
            for fc in range(tc):
                for yy in range(hm):
                    for xx in range(wm):
                        child_arr[r, fc, yy, xx] = parent_patch[fc, src_y0 + yy, src_x0 + xx]
            slot_value_pos[child_slot] = r
            level_next_pos[parent_level + 1] = r + 1

        out_count += 4

    return new_active_slots, out_count, next_slot


def _native_active_refine_slots(
    active: NativeActiveTopology,
    slots: np.ndarray,
    protected: Optional[np.ndarray] = None,
) -> NativeActiveTopology:
    """Native (multi-scale) refine fill: split each refined parent patch into 4 lossless
    quadrants and write them to per-level child value rows.

    Inverse of the Task-7 coarsen-stitch. Topology bookkeeping mirrors the uniform twin
    ``_active_refine_slots`` (value-agnostic); only the fill differs (slice, no resample).
    Mutates ``active``'s pre-allocated arrays in place and returns the re-wrapped struct,
    matching the uniform wrapper's mutate-and-return style.
    """
    slots = np.unique(np.asarray(slots, dtype=np.int32))
    if slots.size == 0:
        return active

    capacity = int(active.active_slots.shape[0])
    refine_mask = np.zeros(capacity, dtype=bool)
    refine_mask[slots] = True
    active_slots = active.active_slots[:active.active_count]
    to_refine = active_slots[refine_mask[active_slots]]
    refine_count = int(to_refine.shape[0])
    if refine_count == 0:
        return active

    # Fail-loud invariant guard (project rule: no silent corruption): a max-level slot has
    # no finer level to refine into. The uniform twin relies on callers never marking such
    # slots; the native fill would otherwise index value_tuple[L+1] (out of range) with a
    # cryptic numba error. T3's criterion already skips max-level; this protects T5/T6 too.
    L = len(active.values_by_level) - 1
    if np.any(active.level_idx[to_refine] >= L):
        raise ValueError(
            f"_native_active_refine_slots: cannot refine max-level slots "
            f"(max_level_idx={L}); caller must not mark them"
        )

    # Slot capacity guard (Python, mirrors _ensure_active_append_capacity): each refined
    # parent appends 4 child slots and grows the active count by 3 (parent leaves, 4 enter).
    if (int(active.next_slot) + 4 * refine_count > capacity
            or active.active_count + 3 * refine_count > capacity):
        raise _ArrayFallback("capacity_exceeded")

    # Per-level value capacity guard: children land at level parent_level+1; the kernel writes
    # 4 child rows PER PARENT, so multiply the per-parent tally by 4 (a parent-count guard
    # would be 4x too loose and permit a silent OOB write with boundscheck=False).
    child_levels = active.level_idx[to_refine].astype(np.int64) + 1
    counts = np.bincount(child_levels, minlength=L + 1) * 4
    for lc in range(L + 1):
        if int(counts[lc]) == 0:
            continue
        if int(active.level_next_pos[lc]) + int(counts[lc]) > int(active.level_cap[lc]):
            raise _ArrayFallback("native_level_capacity")

    protected_array = protected if protected is not None else np.empty(0, dtype=bool)
    new_active_slots, out_count, next_slot = _native_active_refine_slots_jit(
        active.values_by_level,
        active.slot_value_pos,
        active.level_next_pos,
        active.tile_ix,
        active.tile_iy,
        active.level_idx,
        active.x_idx,
        active.y_idx,
        active.active_slots,
        int(active.active_count),
        int(active.next_slot),
        refine_mask,
        protected_array,
        protected is not None,
    )
    active.active_slots = new_active_slots
    active.active_count = out_count
    active.next_slot = next_slot
    return active


def _native_active_ensure_2to1_balance(
    active: NativeActiveTopology,
    protected: Optional[np.ndarray] = None,
) -> Tuple[NativeActiveTopology, Optional[np.ndarray], int, int]:
    """Restore 2:1 balance on a native (multi-scale) active topology.

    Mirrors ``_active_ensure_2to1_balance_counted`` exactly, with one substitution:
    the native ``_native_active_refine_slots`` mutates ``protected`` in place and
    returns only ``active`` (rather than ``(active, protected)``), so the loop body
    assigns only ``active``. The value-agnostic balance decision
    (``_build_active_owner_grid`` + ``_find_active_balance_refinements``) is reused
    unchanged; only the fill differs (exact quadrant slice via the native refine).

    Returns ``(active, protected, refine_calls, refined_parents)`` — same 4-tuple
    shape as the uniform twin so the Task-9 driver can thread ``protected`` and use
    the counts for timing/telemetry.
    """
    refine_calls = 0
    refined_parents = 0
    while True:
        owner = _build_active_owner_grid(active)
        refine_slots = _find_active_balance_refinements(active, owner)
        if refine_slots.size == 0:
            return active, protected, refine_calls, refined_parents
        refine_calls += 1
        refined_parents += int(refine_slots.size)
        active = _native_active_refine_slots(active, refine_slots, protected=protected)


@njit(boundscheck=False, cache=True)
def _native_active_protected_mask_jit(
    value_tuple,
    slot_value_pos: np.ndarray,
    level_idx: np.ndarray,
    active_slots: np.ndarray,
    active_count: int,
    channels: np.ndarray,
    tolerances: np.ndarray,
) -> np.ndarray:
    """Native (multi-scale) protected mask.

    Twin: ``_active_protected_mask_jit`` (the non-source sibling). Same logic --
    for each active slot, an active slot is protected when ANY requested channel's
    patch range ``max - min`` is ``>= tolerances[ch_pos]`` (``>=``, NOT ``>``,
    matching the twin). The ONLY native delta is the per-level value read: instead
    of the twin's fixed-shape ``values[slot, ch, yy, xx]`` we read the slot's OWN
    per-level native patch ``value_tuple[level_idx[slot]][slot_value_pos[slot], ch]``
    whose ``(H_l, W_l)`` comes from that patch's shape -- exactly the read used by
    the ``internal_err`` half of ``_native_active_refine_criterion_jit`` (T3).

    Returns a bool array of length ``capacity`` (slot-indexed; zeros for inactive
    slots), matching the twin's ``np.zeros(values.shape[0])`` length convention.
    """
    capacity = level_idx.shape[0]
    protected = np.zeros(capacity, dtype=np.bool_)
    for tol_idx in range(channels.shape[0]):
        ch = int(channels[tol_idx])
        tol = float(tolerances[tol_idx])
        for pos in range(active_count):
            slot = int(active_slots[pos])
            leaf_level = int(level_idx[slot])
            patch = value_tuple[leaf_level][int(slot_value_pos[slot]), ch]
            patch_h = patch.shape[0]
            patch_w = patch.shape[1]
            v_min = patch[0, 0]
            v_max = v_min
            for yy in range(patch_h):
                for xx in range(patch_w):
                    val = patch[yy, xx]
                    if val < v_min:
                        v_min = val
                    elif val > v_max:
                        v_max = val
            if float(v_max - v_min) >= tol:
                protected[slot] = True
    return protected


def _native_active_protected_mask(
    active: NativeActiveTopology,
    channels: list,
    tolerances: np.ndarray,
) -> np.ndarray:
    """Python wrapper for the native protected mask (Part A of Task 6).

    Mirrors the uniform ``_active_protected_mask`` wrapper, threading the native
    per-level value tuple + ``slot_value_pos``/``level_idx`` into the jit kernel.
    """
    return _native_active_protected_mask_jit(
        active.values_by_level,
        active.slot_value_pos,
        active.level_idx,
        active.active_slots,
        int(active.active_count),
        np.asarray(channels, dtype=np.int64),
        tolerances.astype(np.float64, copy=False),
    )


def _native_active_expand_protected_region(
    active: NativeActiveTopology,
    protected: np.ndarray,
    passes: int,
) -> Tuple[NativeActiveTopology, np.ndarray, int, int, int, int]:
    """Native (multi-scale) protected-region expand (Part B of Task 6).

    Twin: ``_active_expand_protected_region_counted``. This is an EXACT mirror of the
    twin with only the two native substitutions called for in the plan:
      * ``active, protected = _active_refine_slots(active, refine_slots, protected=protected)``
        -> ``active = _native_active_refine_slots(active, refine_slots, protected=protected)``
        (native returns only ``active``; ``protected`` is mutated in place).
      * ``_active_ensure_2to1_balance_counted`` -> ``_native_active_ensure_2to1_balance``.
    Everything else -- the ``for _ in range(max(0, passes))`` loop,
    ``_build_active_owner_grid``, ``np.flatnonzero(protected)``, the value-agnostic
    ``_active_protected_region_refine_slots_object_neighbors_jit`` decision (reused
    as-is; reads only topology/owner), the break on empty, the counters -- is copied
    verbatim. Returns the SAME 6-tuple so the Task-9 driver gets identical telemetry.
    """
    max_level = int(active.domain["max_level_idx"])
    protected_refine_calls = 0
    protected_refine_parents = 0
    balance_refine_calls = 0
    balance_refine_parents = 0
    for _ in range(max(0, passes)):
        owner = _build_active_owner_grid(active)
        protected_slots = np.flatnonzero(protected).astype(np.int32)
        refine_mask = _active_protected_region_refine_slots_object_neighbors_jit(
            protected_slots,
            protected,
            active.level_idx,
            owner,
            active.tile_ix,
            active.tile_iy,
            active.x_idx,
            active.y_idx,
            max_level,
        )
        refine_slots = np.flatnonzero(refine_mask).astype(np.int32, copy=False)
        if refine_slots.size == 0:
            break
        protected_refine_calls += 1
        protected_refine_parents += int(refine_slots.size)
        active = _native_active_refine_slots(active, refine_slots, protected=protected)
        active, protected, balance_calls, balance_parents = _native_active_ensure_2to1_balance(
            active,
            protected=protected,
        )
        balance_refine_calls += int(balance_calls)
        balance_refine_parents += int(balance_parents)
    return (
        active,
        protected,
        protected_refine_calls,
        protected_refine_parents,
        balance_refine_calls,
        balance_refine_parents,
    )


# Native coarsen (Task 7): the EXACT inverse of the Task-4 native refine slice.
#
# Acceptance mirrors the uniform twin _active_coarsen_acceptance_mask_sequential_jit, with one
# delta -- each child's range is read from its OWN per-level native patch
# value_tuple[child_level][slot_value_pos[child], ch] (shape (Hc, Wc)) instead of the uniform
# fixed-shape values[slot, ch]. All 4 children of a group share child_level, so Hc/Wc match.
# The protected check and _active_parent_side_has_too_fine_neighbor_jit are value-agnostic --
# identical to the twin.
#
# Stitch fill is LOSSLESS (no block-mean): the parent native patch is (tc, 2*Hc, 2*Wc) and each
# child's full (tc, Hc, Wc) patch is placed at quadrant (dst_y0, dst_x0) with dst_y0 = Hc if
# q>=2 else 0, dst_x0 = Wc if q in {1,3} else 0. This is the byte-exact inverse of the T4 slice
# (T4: src_y0 = hm=Hc if q>=2; src_x0 = wm=Wc if q in {1,3}) and matches the object
# QuadCell.coarsen(stitch_only=True) SW/SE/NW/NE layout.
@njit(boundscheck=False, cache=True)
def _native_active_coarsen_once_jit(
    value_tuple,
    slot_value_pos: np.ndarray,
    level_next_pos: np.ndarray,
    level_cap: np.ndarray,
    tile_ix: np.ndarray,
    tile_iy: np.ndarray,
    level_idx: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    active_slots: np.ndarray,
    active_count: int,
    next_slot: int,
    protected: np.ndarray,
    group_keys: np.ndarray,
    sibling_groups: np.ndarray,
    channels: np.ndarray,
    tolerances: np.ndarray,
    owner: np.ndarray,
    max_level: int,
) -> Tuple[np.ndarray, int, int, bool, int, int]:
    capacity = active_slots.shape[0]
    L = level_cap.shape[0] - 1

    # --- Acceptance (native read; mirrors the uniform sequential acceptance jit) ---
    accepted_mask = np.zeros(sibling_groups.shape[0], dtype=np.bool_)
    for group_idx in range(sibling_groups.shape[0]):
        child_slots = sibling_groups[group_idx]
        if (
            protected[int(child_slots[0])]
            or protected[int(child_slots[1])]
            or protected[int(child_slots[2])]
            or protected[int(child_slots[3])]
        ):
            continue

        # All 4 children share level (complete sibling group); read native per-level patches.
        child_level = int(level_idx[int(child_slots[0])])
        reject = False
        for tol_idx in range(channels.shape[0]):
            ch = int(channels[tol_idx])
            tol = float(tolerances[tol_idx])
            first_patch = value_tuple[child_level][int(slot_value_pos[int(child_slots[0])]), ch]
            v_min = first_patch[0, 0]
            v_max = v_min
            for quad in range(4):
                patch = value_tuple[child_level][int(slot_value_pos[int(child_slots[quad])]), ch]
                ph = patch.shape[0]
                pw = patch.shape[1]
                for yy in range(ph):
                    for xx in range(pw):
                        val = patch[yy, xx]
                        if val < v_min:
                            v_min = val
                        elif val > v_max:
                            v_max = val
            if float(v_max - v_min) >= tol:
                reject = True
                break
        if reject:
            continue

        key = group_keys[group_idx]
        if _active_parent_side_has_too_fine_neighbor_jit(
            level_idx,
            owner,
            child_slots,
            int(key[2]),
            int(key[3]),
            int(key[4]),
            int(key[0]),
            int(key[1]),
            max_level,
        ):
            continue

        accepted_mask[group_idx] = True

    # --- Collect accepted groups (slot guard + per-level value tally; pre-mutation) ---
    accepted_groups = np.empty(sibling_groups.shape[0], dtype=np.int32)
    accepted_count = 0
    parent_count = np.zeros(L + 1, dtype=np.int64)  # accepted-parent count per parent level
    for group_idx in range(sibling_groups.shape[0]):
        if not accepted_mask[group_idx]:
            continue
        if next_slot + accepted_count >= capacity:
            return active_slots, active_count, next_slot, False, 1, accepted_count
        accepted_groups[accepted_count] = group_idx
        accepted_count += 1
        parent_count[int(group_keys[group_idx, 2])] += 1

    if accepted_count == 0:
        return active_slots, active_count, next_slot, False, 0, 0

    # Per-level VALUE capacity guard (pre-mutation): accepted parents are appended to
    # value_tuple[pl] at level_next_pos[pl]. Check BEFORE any value mutation so an err leaves
    # the struct untouched (the Python wrapper raises _ArrayFallback; no partial mutation).
    for pl in range(L + 1):
        if int(parent_count[pl]) == 0:
            continue
        if int(level_next_pos[pl]) + int(parent_count[pl]) > int(level_cap[pl]):
            return active_slots, active_count, next_slot, False, 1, accepted_count

    # --- Topology: mark accepted children skipped, clear their protected ---
    skipped = np.zeros(capacity, dtype=np.bool_)
    for accepted_pos in range(accepted_count):
        child_slots = sibling_groups[int(accepted_groups[accepted_pos])]
        for quad in range(4):
            skipped[int(child_slots[quad])] = True

    new_active_slots = np.empty_like(active_slots)
    out_count = 0
    for active_pos in range(active_count):
        slot = int(active_slots[active_pos])
        if skipped[slot]:
            protected[slot] = False
            continue
        new_active_slots[out_count] = slot
        out_count += 1

    # --- Append one parent slot per accepted group; native lossless stitch fill ---
    parent_start = next_slot
    for accepted_pos in range(accepted_count):
        group_idx = int(accepted_groups[accepted_pos])
        key = group_keys[group_idx]
        child_slots = sibling_groups[group_idx]
        parent_slot = parent_start + accepted_pos
        parent_level = int(key[2])

        tile_ix[parent_slot] = int(key[0])
        tile_iy[parent_slot] = int(key[1])
        level_idx[parent_slot] = np.int16(parent_level)
        x_idx[parent_slot] = int(key[3])
        y_idx[parent_slot] = int(key[4])
        protected[parent_slot] = False
        new_active_slots[out_count] = parent_slot
        out_count += 1

        # Allocate the parent native row; child rows are left logically dead (not reclaimed).
        r = int(level_next_pos[parent_level])
        slot_value_pos[parent_slot] = r
        level_next_pos[parent_level] = r + 1

        parent_arr = value_tuple[parent_level]
        child_arr = value_tuple[parent_level + 1]
        # Child patch dims (Hc, Wc); parent is (tc, 2*Hc, 2*Wc) -- the T4 source/dest mirror.
        sample_child = child_arr[int(slot_value_pos[int(child_slots[0])])]
        tc = sample_child.shape[0]
        Hc = sample_child.shape[1]
        Wc = sample_child.shape[2]
        for quad in range(4):
            cpos = int(slot_value_pos[int(child_slots[quad])])
            # child 0=SW, 1=SE, 2=NW, 3=NE; parent row 0 = bottom (low y), col 0 = left (low x).
            dst_y0 = 0
            dst_x0 = 0
            if quad >= 2:
                dst_y0 = Hc
            if quad == 1 or quad == 3:
                dst_x0 = Wc
            for fc in range(tc):
                for yy in range(Hc):
                    for xx in range(Wc):
                        parent_arr[r, fc, dst_y0 + yy, dst_x0 + xx] = child_arr[cpos, fc, yy, xx]

    next_slot = parent_start + accepted_count
    return new_active_slots, out_count, next_slot, True, 0, accepted_count


def _native_active_coarsen_once(
    active: NativeActiveTopology,
    protected: np.ndarray,
    channels: list,
    tolerances: np.ndarray,
) -> Tuple[NativeActiveTopology, np.ndarray, bool, int]:
    """Native (multi-scale) single coarsen pass: accept full sibling groups whose combined
    per-child native range is below tol (and that pass the protected + 2:1 checks) and stitch
    their 4 children back into one parent patch (LOSSLESS, exact inverse of the T4 refine).

    Mirrors the uniform twin ``_active_coarsen_once`` (reuses the value-agnostic
    ``_active_complete_sibling_groups`` + ``_build_active_owner_grid``); the native deltas live
    in ``_native_active_coarsen_once_jit`` (per-level reads, per-level value capacity guard,
    native stitch). Mutates ``active``'s pre-allocated arrays in place and returns the same
    4-tuple as the uniform wrapper.
    """
    group_keys, sibling_groups = _active_complete_sibling_groups(active)
    owner = _build_active_owner_grid(active)
    new_active_slots, active_count, next_slot, changed, err, accepted_count = _native_active_coarsen_once_jit(
        active.values_by_level,
        active.slot_value_pos,
        active.level_next_pos,
        active.level_cap,
        active.tile_ix,
        active.tile_iy,
        active.level_idx,
        active.x_idx,
        active.y_idx,
        active.active_slots,
        int(active.active_count),
        int(active.next_slot),
        protected,
        group_keys,
        sibling_groups,
        np.asarray(channels, dtype=np.int64),
        tolerances.astype(np.float64, copy=False),
        owner,
        int(active.domain["max_level_idx"]),
    )
    if err:
        raise _ArrayFallback("native_level_capacity")
    if not changed:
        return active, protected, False, int(accepted_count)

    active.active_slots = new_active_slots
    active.active_count = int(active_count)
    active.next_slot = int(next_slot)
    return active, protected, True, int(accepted_count)


def _export_native_active_topology(
    active: NativeActiveTopology,
    *,
    C: int,
    T: int,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, Dict[str, Any]]:
    """Export a NativeActiveTopology back to per-level folded 4-D buckets.

    Twin of ``_export_active_topology`` (uniform) for the native multi-scale path.
    Reuses the same Morton ordering and level_idx metadata machinery verbatim; the
    native delta is grouping values by level instead of copying to a flat array.

    Only ``cell_scale_mode == "level_idx"`` is supported (raises ValueError otherwise).

    Parameters
    ----------
    active:
        Live native topology (may have been refined/coarsened since import).
    C, T:
        Channels and timesteps used to validate the internal T*C dimension.

    Returns
    -------
    by_level : Dict[int, ndarray] of shape ``(N_l, T*C, H_l, W_l)`` float32, contiguous.
        Folded 4-D buckets, one entry per level 0..max_level_idx (matching
        ``quadtree_to_tensor_native``'s key set, including empty levels).
    leaf_to_bucket : ndarray of shape ``(N, 2)`` int64.
        Row ``i`` is ``(level_idx, position_in_level_bucket)`` for the i-th
        leaf in Morton order.
    meta : dict
        Metadata dict identical in structure to ``_export_active_topology``'s
        level_idx output (centers, levels, tiles, xy_idx, cell_ids, domain,
        cell_scale_mode).
    """
    if active.cell_scale_mode != "level_idx":
        raise ValueError(
            f"_export_native_active_topology only supports cell_scale_mode='level_idx'; "
            f"got {active.cell_scale_mode!r}."
        )

    # Validate C, T against internal storage.
    tc = T * C
    for lvl in range(int(active.domain["max_level_idx"]) + 1):
        arr = active.values_by_level[lvl]
        if arr.shape[0] > 0:
            assert arr.shape[1] == tc, (
                f"T*C mismatch at level {lvl}: expected {tc}, got {arr.shape[1]}. "
                f"Check C={C}, T={T}."
            )
            break

    # --- Ordering machinery (verbatim from _export_active_topology) ---
    slots = active.active_slots[:active.active_count]
    x0, y0, scale = _active_leaf_fine_bounds(active, slots)
    nx_tiles, ny_tiles = _nx_ny_tiles(active.domain)
    max_level = int(active.domain["max_level_idx"])
    bits = max_level + _ceil_log2(max(nx_tiles, ny_tiles))
    keys = _morton_keys_jit(x0, y0, bits)
    order = np.argsort(keys, kind="stable")
    ordered_slots = slots[order]

    # --- Metadata (level_idx path verbatim from _export_active_topology) ---
    centers, levels, tiles, xy_idx, cell_ids = _export_active_level_idx_metadata_jit(
        active.tile_ix,
        active.tile_iy,
        active.level_idx,
        active.x_idx,
        active.y_idx,
        ordered_slots,
        x0[order],
        y0[order],
        scale[order],
        max_level,
        float(active.domain["xmin"]),
        float(active.domain["xmax"]),
        float(active.domain["ymin"]),
        float(active.domain["ymax"]),
        float(active.domain["tile_width"]),
        float(active.domain["tile_height"]),
        max_level,
        max_level,
        6,
        _ceil_log2(nx_tiles),
        _ceil_log2(ny_tiles),
    )
    meta = {
        "centers": centers,
        "levels": levels,
        "tiles": tiles,
        "xy_idx": xy_idx,
        "cell_ids": cell_ids,
        "domain": dict(active.domain),
        "cell_scale_mode": active.cell_scale_mode,
    }

    # --- Native delta: regroup values by level in Morton order ---
    N = int(ordered_slots.shape[0])
    leaf_to_bucket = np.empty((N, 2), dtype=np.int64)

    # Per-level patch lists and running counters.
    per_level_patches: Dict[int, list] = {lvl: [] for lvl in range(max_level + 1)}
    level_counter = [0] * (max_level + 1)

    for i in range(N):
        s = int(ordered_slots[i])
        lvl = int(active.level_idx[s])
        patch = active.values_by_level[lvl][int(active.slot_value_pos[s])]  # (T*C, H_l, W_l)
        per_level_patches[lvl].append(patch)
        pos = level_counter[lvl]
        leaf_to_bucket[i, 0] = lvl
        leaf_to_bucket[i, 1] = pos
        level_counter[lvl] = pos + 1

    # Build per-level stacked arrays; include empty levels with correct H_l, W_l.
    by_level: Dict[int, np.ndarray] = {}
    for lvl in range(max_level + 1):
        h_l, w_l = _native_patch_hw(active.domain, lvl)
        patches = per_level_patches[lvl]
        if patches:
            by_level[lvl] = np.ascontiguousarray(
                np.stack(patches, axis=0), dtype=np.float32
            )
        else:
            by_level[lvl] = np.empty((0, tc, h_l, w_l), dtype=np.float32)

    return by_level, leaf_to_bucket, meta


def _run_native_active_regrid(
    active: NativeActiveTopology,
    *,
    ranges: np.ndarray,
    channels: list[int],
    tol_frac: Tol,
    max_passes: int,
    coarsen_ratio: float,
    adapt_nearby: int,
) -> Tuple[NativeActiveTopology, Dict[str, float], Dict[str, int]]:
    """Native active-regrid driver: mirrors ``_run_active_regrid`` exactly with native-kernel
    substitutions from the Task-9 substitution table.  The only differences from the uniform
    twin are:
      * ``_compute_active_refine_slots``      -> ``_compute_native_active_refine_slots``
      * ``_active_refine_slots`` (tuple)      -> ``_native_active_refine_slots`` (active only)
      * ``_active_ensure_2to1_balance_counted`` -> ``_native_active_ensure_2to1_balance``
      * ``_active_protected_mask``            -> ``_native_active_protected_mask``
      * ``_active_expand_protected_region_counted`` -> ``_native_active_expand_protected_region``
      * ``_active_coarsen_once``              -> ``_native_active_coarsen_once``
    Everything else — timings/counts keys, tolerance math, loop structure, validate→_ArrayFallback,
    final slot accounting — is copied verbatim from the uniform twin.

    Note on capacity churn: native refine/coarsen append rows to per-level value arrays and never
    reclaim dead rows.  Per-level capacity is churn_slack(3)*tiles*4^l; if a multi-pass regrid's
    churn exceeds that, the native refine/coarsen raises _ArrayFallback (clean fallback).
    """
    timings = {"refine": 0.0, "protect": 0.0, "coarsen": 0.0, "validate": 0.0}
    initial_slots = int(active.active_count)
    counts = {
        "balance_refine_parents": 0,
        "coarsen_passes": 0,
        "coarsen_accepted": 0,
        "final_appended_slots": 0,
        "final_original_slots": initial_slots,
        "initial_slots": initial_slots,
        "peak_slots": int(active.next_slot),
        "protected_refine_parents": 0,
        "refine_calls": 0,
        "refine_parents": 0,
    }
    if not channels:
        return active, timings, counts

    refine_start = time.perf_counter()
    tol_fracs = _normalize_tolerances(tol_frac, len(channels))
    tols_refine = tol_fracs * np.abs(ranges)
    tols_coarsen = tols_refine * float(coarsen_ratio)

    for _ in range(max_passes):
        owner = _build_active_owner_grid(active)
        refine_slots = _compute_native_active_refine_slots(active, channels, tols_refine, owner)
        if refine_slots.size == 0:
            break
        counts["refine_calls"] += 1
        counts["refine_parents"] += int(refine_slots.size)
        active = _native_active_refine_slots(active, refine_slots)
        active, _, balance_calls, balance_parents = _native_active_ensure_2to1_balance(active)
        counts["refine_calls"] += int(balance_calls)
        counts["refine_parents"] += int(balance_parents)
        counts["balance_refine_parents"] += int(balance_parents)
    timings["refine"] = time.perf_counter() - refine_start

    protect_start = time.perf_counter()
    protected = np.zeros(int(active.active_slots.shape[0]), dtype=bool)
    if adapt_nearby > 0:
        protected = _native_active_protected_mask(active, channels, tols_coarsen)
        protected = _active_dilate_protected(active, protected, adapt_nearby)
        (
            active,
            protected,
            protected_calls,
            protected_parents,
            balance_calls,
            balance_parents,
        ) = _native_active_expand_protected_region(active, protected, adapt_nearby)
        counts["refine_calls"] += int(protected_calls) + int(balance_calls)
        counts["refine_parents"] += int(protected_parents) + int(balance_parents)
        counts["protected_refine_parents"] += int(protected_parents)
        counts["balance_refine_parents"] += int(balance_parents)
    timings["protect"] = time.perf_counter() - protect_start

    coarsen_start = time.perf_counter()
    for _ in range(max_passes):
        active, protected, changed, accepted_count = _native_active_coarsen_once(active, protected, channels, tols_coarsen)
        counts["coarsen_passes"] += 1
        counts["coarsen_accepted"] += int(accepted_count)
        if not changed:
            break
    timings["coarsen"] = time.perf_counter() - coarsen_start

    validate_start = time.perf_counter()
    owner = _build_active_owner_grid(active)
    if _find_active_balance_refinements(active, owner).size:
        raise _ArrayFallback("invalid_topology")
    timings["validate"] = time.perf_counter() - validate_start
    counts["peak_slots"] = int(active.next_slot)
    final_slots = active.active_slots[: active.active_count]
    final_original = int(np.count_nonzero(final_slots < initial_slots))
    counts["final_original_slots"] = final_original
    counts["final_appended_slots"] = int(active.active_count) - final_original
    return active, timings, counts


def _native_adapt_channels(C: int, T: int, adapt_on_channels: Optional[Sequence[int]]) -> list:
    """Adapt-channel mapping, identical to ``regrid_native`` (adapt_wavelet.py:641-646).

    Channels drive adaptation in the folded ``T*C`` layout; default is the last
    timestep's physical channels.
    """
    ch_offset = (T - 1) * C
    if adapt_on_channels is not None:
        return [ch + ch_offset for ch in adapt_on_channels]
    return list(range(ch_offset, T * C))


def _unfold_native_buckets_to_5d(
    folded_by_level: Dict[int, np.ndarray], *, C: int, T: int
) -> Dict[int, np.ndarray]:
    """Unfold folded 4-D ``(N_l, T*C, H_l, W_l)`` buckets to 5-D ``(N_l, C, T, H_l, W_l)``.

    Faithful copy of the unflatten block in ``regrid_native`` (adapt_wavelet.py:659-671),
    including the empty-level path.
    """
    new_by_level: Dict[int, np.ndarray] = {}
    for lvl, flat_arr in folded_by_level.items():
        n_l = flat_arr.shape[0]
        if n_l == 0:
            h_l = flat_arr.shape[-2] if flat_arr.ndim >= 3 else 0
            w_l = flat_arr.shape[-1] if flat_arr.ndim >= 3 else 0
            new_by_level[lvl] = np.zeros((0, C, T, h_l, w_l), dtype=flat_arr.dtype)
            continue
        _, tc, h_l, w_l = flat_arr.shape
        new_by_level[lvl] = np.ascontiguousarray(
            flat_arr.reshape(n_l, T, C, h_l, w_l).transpose(0, 2, 1, 3, 4)
        )
    return new_by_level


def array_regrid_native_from_sequence(
    by_level: Dict[int, np.ndarray],
    leaf_to_bucket: np.ndarray,
    meta: Dict[str, Any],
    *,
    C: int,
    T: int,
    tol_frac: Tol,
    cell_scale_mode: str = "level_idx",
    adapt_on_channels: Optional[Sequence[int]] = None,
    adapt_nearby: int = 0,
    allow_coarsening: bool = True,
    max_passes: int = 10,
    capacity: int = 8192,
    coarsen_ratio: float = 0.25,
    array_regrid_mode: str = "parity",
) -> Tuple[Dict[int, np.ndarray], np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """Array twin of object ``regrid_native`` (adapt_wavelet.py:588).

    Regrids native-mode (multi-scale) per-level tensors via the array engine, with a
    clean fallback to the object backend on any unsupported topology / invalid state.

    Returns a 4-tuple ``(new_by_level, new_leaf_to_bucket, new_meta, status)`` (5-D
    buckets ``{lvl: (N_l, C, T, H_l, W_l)}``); ``status`` carries backend + profiling info.

    Notes
    -----
    * Parity vs object ``regrid_native``: byte-exact at ``adapt_nearby=0``. At
      ``adapt_nearby>0`` (the deployed rollout config) the protect/dilate/expand phase
      reuses the uniform array driver's machinery verbatim and therefore inherits its
      array-vs-object non-determinism: single-field outputs land within the documented
      node-coord similarity band (>=0.90; observed >=0.96), not bit-exact. This matches
      the uniform path's accepted behavior; it is not a native-specific regression. See
      ``tests/test_array_regrid_native.py::test_array_regrid_native_an2_within_band``.
    * ``coarsen_ratio`` defaults to 0.25 to match object ``regrid``'s ``tols_coarsen =
      tol_refine * 0.25`` (the ``regrid`` default).
    * ``allow_coarsening=False`` is not supported by the always-coarsen native driver;
      it falls back to the object backend transparently.
    * ``array_regrid_mode`` is validated and echoed into ``status`` for API symmetry with
      the uniform entry, but is accepted-but-ignored here: the native path always runs the
      single object-faithful driver regardless of mode.
    * Inputs are never mutated: the importer copies every patch + the domain, so the
      object fallback operates on the original arrays safely.
    """
    array_regrid_mode = _normalize_array_regrid_mode(array_regrid_mode)
    start = time.perf_counter()
    try:
        if not allow_coarsening:
            # The native driver always runs the coarsen phase; refinement-only is not the
            # paper's rollout config -> fall back to the object backend transparently.
            raise _ArrayFallback("allow_coarsening_unsupported")

        active = _native_active_from_buckets(
            by_level, leaf_to_bucket, meta,
            C=C, T=T, capacity=capacity, cell_scale_mode=cell_scale_mode,
        )
        after_import = time.perf_counter()

        channels = _native_adapt_channels(C, T, adapt_on_channels)
        ranges = _native_active_channel_ranges(active, channels)
        active, run_timings, run_counts = _run_native_active_regrid(
            active,
            ranges=ranges,
            channels=channels,
            tol_frac=tol_frac,
            max_passes=max_passes,
            coarsen_ratio=coarsen_ratio,
            adapt_nearby=adapt_nearby,
        )
        after_run = time.perf_counter()

        folded_by_level, new_l2b, new_meta = _export_native_active_topology(active, C=C, T=T)
        new_by_level = _unfold_native_buckets_to_5d(folded_by_level, C=C, T=T)
        end = time.perf_counter()

        status = {
            "backend": "array",
            "fallback_reason": None,
            "array_regrid_mode": array_regrid_mode,
            "timings": {
                "import": after_import - start,
                "run": after_run - after_import,
                "export": end - after_run,
                "total": end - start,
            },
            "run_timings": run_timings,
            "run_counts": run_counts,
        }
        return new_by_level, new_l2b, new_meta, status
    except (_ArrayFallback, ValueError) as exc:
        from wamrvit.quad.adapt_wavelet import regrid_native

        fallback_reason = exc.reason if isinstance(exc, _ArrayFallback) else "invalid_topology"
        obj_by_level, obj_l2b, obj_meta = regrid_native(
            by_level,
            leaf_to_bucket,
            meta,
            C=C,
            T=T,
            tol_frac=tol_frac,
            cell_scale_mode=cell_scale_mode,
            adapt_on_channels=adapt_on_channels,
            adapt_nearby=adapt_nearby,
            allow_coarsening=allow_coarsening,
            max_passes=max_passes,
        )
        status = {
            "backend": "object",
            "fallback_reason": fallback_reason,
            "array_regrid_mode": array_regrid_mode,
            "timings": {"total": time.perf_counter() - start},
            "run_timings": {},
            "run_counts": {},
        }
        return obj_by_level, obj_l2b, obj_meta, status


def warm_array_regrid_native_kernels(
    *,
    base_patch_h: int,
    base_patch_w: int,
    max_level_idx: int,
    channels: int,
    timesteps: int,
) -> None:
    """JIT-compile all native numba kernels by running a tiny synthetic regrid.

    Call this once at rollout startup so the first real regrid step is not
    JIT-dominated.  Numba specialises on dtype/ndim/layout — not on concrete
    shapes — so the exact ``base_patch_h/w`` values don't matter for
    compilation coverage; only the dtypes and array ranks need to match production.

    If the synthetic topology happens to fall back to the object backend (which
    would only happen for degenerate inputs), the kernels that *did* execute
    before the fallback are still compiled; the function never raises.

    Parameters
    ----------
    base_patch_h, base_patch_w:
        Base (finest-level) patch height and width.  Used only to size the
        synthetic quadtree; any positive integers are valid.  For non-square
        patches the quadtree is built with a square base patch of
        ``max(base_patch_h, base_patch_w)`` because the ``Quadtree`` API
        requires equal tile dimensions — numba specialises on dtype/ndim/layout
        not shape, so this is sufficient for compilation.
    max_level_idx:
        Maximum level index of the target quadtree (0-based depth).
    channels:
        Number of physical channels (C) in the production data.
    timesteps:
        Number of timesteps (T) in the production data.
    """
    # Lazy imports: avoid circular import at module load.
    from wamrvit.quad.quadtree import Quadtree
    from wamrvit.quad.quad_utils import quadtree_to_tensor_native

    C = int(channels)
    T = int(timesteps)
    L = int(max_level_idx)

    # Build a tiny synthetic quadtree covering all levels 0..L.
    # Mirror _make_native_buckets from tests/test_array_regrid_native.py, but
    # self-contained.  Use a square base patch to satisfy the Quadtree API.
    base_p = max(int(base_patch_h), int(base_patch_w))
    tile = float(base_p * (1 << L))   # -> leaf size = tile/2**L = base_p
    span = 2.0 * tile                  # 2x2 tile grid -> 4 root cells

    qt = Quadtree(
        0.0, span, 0.0, span,
        max_level_idx=L, channels=C * T,
        tile_width=tile, tile_height=tile,
        value_storage="native",
    )

    # Fill root cells with deterministic non-constant values so the adapt
    # criterion has something to act on (refine criterion + coarsen acceptance).
    rng = np.random.default_rng(42)
    for row in qt.roots:
        for leaf in row:
            ph, pw = leaf.value.shape[-2:]
            leaf.value = rng.standard_normal((C * T, ph, pw)).astype(np.float32)

    # Refine a corner down to level L so cells exist at ALL levels 0..L.
    node = qt.roots[0][0]
    qt.refine_leaf(node)
    child = node.children[0]
    for _ in range(L - 1):
        qt.refine_leaf(child)
        child = child.children[0]

    # 2:1 balance so the topology is valid for the array engine.
    qt.ensure_2to1_balance()

    # Export to folded 4-D (N_l, T*C, H_l, W_l), then unfold to 5-D.
    folded, l2b, meta = quadtree_to_tensor_native(qt, cell_scale_mode="level_idx")
    by_level_5d: Dict[int, np.ndarray] = {}
    for lvl in range(L + 1):
        if lvl in folded and folded[lvl].shape[0] > 0:
            arr = folded[lvl]
            n, h_l, w_l = arr.shape[0], arr.shape[-2], arr.shape[-1]
            by_level_5d[lvl] = arr.reshape(n, T, C, h_l, w_l).transpose(0, 2, 1, 3, 4).copy()
        else:
            # Empty level: determine patch size from domain geometry.
            base_h = int(base_patch_h)
            base_w = int(base_patch_w)
            h_l = base_h << (L - lvl)
            w_l = base_w << (L - lvl)
            by_level_5d[lvl] = np.zeros((0, C, T, h_l, w_l), dtype=np.float32)

    try:
        array_regrid_native_from_sequence(
            by_level_5d, l2b, meta,
            C=C, T=T,
            tol_frac=0.1,
            cell_scale_mode="level_idx",
            adapt_nearby=1,
            max_passes=4,
            array_regrid_mode="parity",
        )
    except _ArrayFallback:
        # A degenerate synthetic case that triggers fallback still compiles
        # the kernels that ran before the fallback.
        pass


def array_regrid_from_tensor(
    data: np.ndarray,
    meta: Dict[str, Any],
    *,
    cell_scale_mode: Optional[str],
    tol_frac: Tol,
    channel: Channel,
    max_passes: int,
    coarsen_ratio: float = 0.25,
    adapt_nearby: int = 0,
    capacity: int = 8192,
    disable_warnings: bool = True,
    array_regrid_mode: str = "parity",
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    array_regrid_mode = _normalize_array_regrid_mode(array_regrid_mode)
    start = time.perf_counter()
    try:
        top = _import_topology(data, meta, cell_scale_mode)
        after_import = time.perf_counter()
        if array_regrid_mode == "parity":
            top, run_timings, run_counts = _run_array_regrid_parity(
                top,
                tol_frac=tol_frac,
                channel=channel,
                max_passes=max_passes,
                coarsen_ratio=coarsen_ratio,
                adapt_nearby=adapt_nearby,
                capacity=capacity,
            )
        else:
            top, run_timings, run_counts = _run_array_regrid(
                top,
                tol_frac=tol_frac,
                channel=channel,
                max_passes=max_passes,
                coarsen_ratio=coarsen_ratio,
                adapt_nearby=adapt_nearby,
                capacity=capacity,
            )
        after_run = time.perf_counter()
        if isinstance(top, ActiveTopology):
            out, out_meta = _export_active_topology(top)
        else:
            out, out_meta = _export_topology(top)
        end = time.perf_counter()
        return out, out_meta, {
            "backend": "array",
            "fallback_reason": None,
            **_array_regrid_config_status(capacity),
            **_array_regrid_param_status(
                tol_frac=tol_frac,
                channel=channel,
                max_passes=max_passes,
                coarsen_ratio=coarsen_ratio,
                adapt_nearby=adapt_nearby,
            ),
            "array_regrid_mode": array_regrid_mode,
            "timings": {
                "import": after_import - start,
                "run": after_run - after_import,
                "export": end - after_run,
                "total": end - start,
            },
            "run_timings": run_timings,
            "run_counts": run_counts,
        }
    except _ArrayFallback as exc:
        out, out_meta, status = object_regrid_from_tensor(
            data,
            meta,
            cell_scale_mode=cell_scale_mode,
            tol_frac=tol_frac,
            channel=channel,
            max_passes=max_passes,
            coarsen_ratio=coarsen_ratio,
            adapt_nearby=adapt_nearby,
            disable_warnings=disable_warnings,
        )
        status["fallback_reason"] = exc.reason
        status["array_regrid_mode"] = array_regrid_mode
        status.update(_array_regrid_config_status(capacity))
        status.update(
            _array_regrid_param_status(
                tol_frac=tol_frac,
                channel=channel,
                max_passes=max_passes,
                coarsen_ratio=coarsen_ratio,
                adapt_nearby=adapt_nearby,
            )
        )
        return out, out_meta, status
    except ValueError:
        out, out_meta, status = object_regrid_from_tensor(
            data,
            meta,
            cell_scale_mode=cell_scale_mode,
            tol_frac=tol_frac,
            channel=channel,
            max_passes=max_passes,
            coarsen_ratio=coarsen_ratio,
            adapt_nearby=adapt_nearby,
            disable_warnings=disable_warnings,
        )
        status["fallback_reason"] = "invalid_topology"
        status["array_regrid_mode"] = array_regrid_mode
        status.update(_array_regrid_config_status(capacity))
        status.update(
            _array_regrid_param_status(
                tol_frac=tol_frac,
                channel=channel,
                max_passes=max_passes,
                coarsen_ratio=coarsen_ratio,
                adapt_nearby=adapt_nearby,
            )
        )
        return out, out_meta, status
