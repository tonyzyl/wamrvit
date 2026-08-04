from __future__ import annotations

from typing import Tuple

import numpy as np
from numba import njit, prange

from wamrvit.quad.array_regrid_geometry import _build_active_owner_grid
from wamrvit.quad.array_regrid_refine import (
    _active_ensure_2to1_balance_counted,
    _active_ensure_2to1_balance_counted_from_source,
    _active_refine_slots, _active_refine_slots_from_source, _source_or_workspace_value,
)
from wamrvit.quad.array_regrid_runtime import _ACTIVE_PROTECTED_MASK_PARALLEL_THRESHOLD
from wamrvit.quad.array_regrid_types import ActiveTopology, SequenceSourceActiveTopology


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
        active.lineage_phase = "protected"
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
