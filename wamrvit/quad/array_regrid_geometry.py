from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from numba import njit

from wamrvit.quad.array_regrid_types import ActiveTopology, FineBounds, FlatTopology
from wamrvit.quad.quadtree import _ceil_log2
from wamrvit.quad.quadtree_kernels import morton2D_jit


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
