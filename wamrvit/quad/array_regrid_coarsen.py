from __future__ import annotations

from typing import Tuple

import numpy as np
from numba import njit, prange

from wamrvit.quad.array_regrid_geometry import _build_active_owner_grid
from wamrvit.quad.array_regrid_refine import _source_or_workspace_value
from wamrvit.quad.array_regrid_runtime import (
    _COARSEN_PARALLEL_ACCEPT_THRESHOLD, _COARSEN_PARALLEL_FILL_THRESHOLD,
)
from wamrvit.quad.array_regrid_types import (
    ActiveTopology, ArrayFallback, SequenceSourceActiveTopology,
)
from wamrvit.quad.regrid_lineage import PayloadOp


@njit(boundscheck=False, inline="always")
def _stitched_child_sample_jit(
    v_sw: np.ndarray,
    v_se: np.ndarray,
    v_nw: np.ndarray,
    v_ne: np.ndarray,
    channel: int,
    stitched_y: int,
    stitched_x: int,
    patch_h: int,
    patch_w: int,
) -> float:
    """Sample a child from the 2x2 stitched grid, including odd patch sizes.

    A parent patch is formed by stitching four child patches, so a 2x2 block can
    straddle the child seam when ``patch_h`` or ``patch_w`` is odd.  Splitting at
    ``patch_* // 2`` (the old implementation) is incorrect for that case and can
    read one column/row past the child array.  Coordinates here are in the stitched
    ``(2*patch_h, 2*patch_w)`` grid and are mapped back to the appropriate child.
    """
    north = stitched_y >= patch_h
    east = stitched_x >= patch_w
    child_y = stitched_y - patch_h if north else stitched_y
    child_x = stitched_x - patch_w if east else stitched_x
    if north:
        child = v_ne if east else v_nw
    else:
        child = v_se if east else v_sw
    return child[channel, child_y, child_x]

@njit(boundscheck=False)
def _coarsen_quadrants_mean_jit(
    v_sw: np.ndarray,
    v_se: np.ndarray,
    v_nw: np.ndarray,
    v_ne: np.ndarray,
) -> np.ndarray:
    channels, patch_h, patch_w = v_sw.shape
    out = np.empty_like(v_sw)
    for ch in range(channels):
        for yy in range(patch_h):
            stitched_y = 2 * yy
            for xx in range(patch_w):
                stitched_x = 2 * xx
                out[ch, yy, xx] = 0.25 * (
                    _stitched_child_sample_jit(
                        v_sw, v_se, v_nw, v_ne, ch, stitched_y, stitched_x, patch_h, patch_w
                    )
                    + _stitched_child_sample_jit(
                        v_sw, v_se, v_nw, v_ne, ch, stitched_y + 1, stitched_x, patch_h, patch_w
                    )
                    + _stitched_child_sample_jit(
                        v_sw, v_se, v_nw, v_ne, ch, stitched_y, stitched_x + 1, patch_h, patch_w
                    )
                    + _stitched_child_sample_jit(
                        v_sw, v_se, v_nw, v_ne, ch, stitched_y + 1, stitched_x + 1, patch_h, patch_w
                    )
                )
    return out

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

@njit(boundscheck=False, inline="always")
def _fill_one_coarsened_parent_values_jit(
    values: np.ndarray,
    child_slots: np.ndarray,
    parent_slot: int,
) -> None:
    values[parent_slot] = _coarsen_quadrants_mean_jit(
        values[int(child_slots[0])],
        values[int(child_slots[1])],
        values[int(child_slots[2])],
        values[int(child_slots[3])],
    )

@njit(boundscheck=False, cache=True)
def _fill_accepted_coarsened_parent_values_sequential_jit(
    values: np.ndarray,
    sibling_groups: np.ndarray,
    accepted_groups: np.ndarray,
    accepted_count: int,
    parent_start: int,
) -> None:
    for accepted_pos in range(accepted_count):
        group_idx = int(accepted_groups[accepted_pos])
        _fill_one_coarsened_parent_values_jit(
            values,
            sibling_groups[group_idx],
            parent_start + accepted_pos,
        )

@njit(boundscheck=False, parallel=True, cache=True)
def _fill_accepted_coarsened_parent_values_parallel_jit(
    values: np.ndarray,
    sibling_groups: np.ndarray,
    accepted_groups: np.ndarray,
    accepted_count: int,
    parent_start: int,
) -> None:
    for accepted_pos in prange(accepted_count):
        group_idx = int(accepted_groups[accepted_pos])
        _fill_one_coarsened_parent_values_jit(
            values,
            sibling_groups[group_idx],
            parent_start + accepted_pos,
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
def _fill_one_coarsened_parent_from_source_jit(
    source_sequence: np.ndarray,
    values: np.ndarray,
    initial_slots: int,
    sequence_channels: int,
    child_slots: np.ndarray,
    parent_slot: int,
) -> None:
    """Coarsen source/workspace children through the stitched-grid geometry.

    The old quadrant-slice implementation split the *parent* patch at
    ``patch_* // 2``.  That only works for even dimensions; PLI uses a 10x5
    patch, where stitched 2x2 averaging crosses the midpoint seam in the width.
    """
    channels_total = values.shape[1]
    patch_h = values.shape[2]
    patch_w = values.shape[3]
    parent_workspace_slot = parent_slot - initial_slots

    for flat_ch in range(channels_total):
        t_idx = flat_ch // sequence_channels
        ch = flat_ch - t_idx * sequence_channels
        for yy in range(patch_h):
            stitched_y = 2 * yy
            for xx in range(patch_w):
                stitched_x = 2 * xx
                total = 0.0
                for corner in range(4):
                    sy = stitched_y + (corner & 1)
                    sx = stitched_x + ((corner >> 1) & 1)
                    north = sy >= patch_h
                    east = sx >= patch_w
                    child_y = sy - patch_h if north else sy
                    child_x = sx - patch_w if east else sx
                    quad = (1 if east else 0) + (2 if north else 0)
                    child_slot = int(child_slots[quad])
                    total += _source_or_workspace_value(
                        source_sequence,
                        values,
                        initial_slots,
                        sequence_channels,
                        child_slot,
                        flat_ch,
                        child_y,
                        child_x,
                    )
                values[parent_workspace_slot, flat_ch, yy, xx] = 0.25 * total

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
                raise ArrayFallback("capacity_exceeded")
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

def _record_source_coarsen_ops(
    active: SequenceSourceActiveTopology,
    group_keys: np.ndarray,
    sibling_groups: np.ndarray,
    parent_start: int,
    parent_stop: int,
) -> None:
    if active.payload_ops is None:
        return
    groups_by_key = {
        tuple(int(value) for value in group_keys[group_idx]): sibling_groups[group_idx]
        for group_idx in range(group_keys.shape[0])
    }
    for parent_slot in range(int(parent_start), int(parent_stop)):
        key = (
            int(active.tile_ix[parent_slot]), int(active.tile_iy[parent_slot]),
            int(active.level_idx[parent_slot]), int(active.x_idx[parent_slot]),
            int(active.y_idx[parent_slot]),
        )
        children = groups_by_key.get(key)
        if children is None:
            raise RuntimeError(f"Missing payload lineage children for coarsened key {key}.")
        child_ops = tuple(int(child) for child in children)
        if any(active.payload_ops[child] is None for child in child_ops):
            raise RuntimeError(f"Missing child payload lineage for coarsened key {key}.")
        active.payload_ops[parent_slot] = PayloadOp(
            kind="coarsen",
            child_ops=child_ops,
            phase=active.lineage_phase,
        )

def _active_coarsen_once_from_source_conservative(
    active: SequenceSourceActiveTopology,
    protected: np.ndarray,
    channels: list[int],
    tolerances: np.ndarray,
) -> Tuple[SequenceSourceActiveTopology, np.ndarray, bool, int]:
    active.lineage_phase = "coarsen"
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
            parent_start = int(active.next_slot)
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
                raise ArrayFallback("capacity_exceeded")
            if changed:
                _record_source_coarsen_ops(
                    active, single_keys, single_groups, parent_start, int(next_slot)
                )
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
