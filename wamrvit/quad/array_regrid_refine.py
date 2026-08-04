from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from numba import njit, prange

from wamrvit.quad.array_regrid_geometry import (
    _active_leaf_fine_bounds, _build_active_owner_grid, _build_owner_grid,
    _find_active_balance_refinements, _find_balance_refinements, _leaf_fine_bounds,
)
from wamrvit.quad.array_regrid_runtime import (
    _REFINE_PARALLEL_FILL_THRESHOLD, _SOURCE_REFINE_PARALLEL_FILL_THRESHOLD,
    _ensure_capacity,
)
from wamrvit.quad.array_regrid_types import (
    ActiveTopology, ArrayFallback, FlatTopology, SequenceSourceActiveTopology,
)
from wamrvit.quad.quadtree_kernels import resize_patch_bilinear_jit
from wamrvit.quad.regrid_lineage import PayloadOp


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

def _ensure_active_append_capacity(active: ActiveTopology, append_count: int, new_active_count: int) -> None:
    capacity = int(active.active_slots.shape[0])
    if new_active_count > capacity or int(active.next_slot) + int(append_count) > capacity:
        raise ArrayFallback("capacity_exceeded")

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

def _record_source_refine_ops(
    active: SequenceSourceActiveTopology,
    parent_slots: Sequence[int],
    child_start: int,
) -> None:
    if active.payload_ops is None:
        return
    next_child = int(child_start)
    for parent_slot in parent_slots:
        parent_op = active.payload_ops[int(parent_slot)]
        if parent_op is None:
            raise RuntimeError(f"Missing payload lineage for refine parent slot {parent_slot}.")
        parent_op_id = int(parent_slot)
        for quadrant in range(4):
            active.payload_ops[next_child + quadrant] = PayloadOp(
                kind="refine",
                parent_op=parent_op_id,
                quadrant=quadrant,
                phase=active.lineage_phase,
            )
        next_child += 4

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

    refined_parents = [
        int(slot) for slot in active_slots if bool(refine_mask[int(slot)])
    ]
    child_start = int(active.next_slot)
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
    _record_source_refine_ops(active, refined_parents, child_start)
    active.active_slots = new_active_slots
    active.active_count = out_count
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
    previous_phase = active.lineage_phase
    active.lineage_phase = "balance"
    try:
        while True:
            owner = _build_active_owner_grid(active)
            refine_slots = _find_active_balance_refinements(active, owner)
            if refine_slots.size == 0:
                return active, protected, refine_calls, refined_parents
            refine_calls += 1
            refined_parents += int(refine_slots.size)
            active, protected = _active_refine_slots_from_source(
                active, refine_slots, protected=protected
            )
    finally:
        active.lineage_phase = previous_phase

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
                raise ArrayFallback("invalid_topology") from exc
            covered_ancestors = _active_fully_covered_ancestor_slots(active)
            if covered_ancestors.size:
                repair_invalid_slots += int(covered_ancestors.size)
                active, protected = _active_remove_slots(active, protected, covered_ancestors)
                continue
            ancestor_slots = _active_overlapping_ancestor_slots(active)
            if ancestor_slots.size == 0:
                raise ArrayFallback("invalid_topology") from exc
            repair_invalid_slots += int(ancestor_slots.size)
            active, protected = _active_refine_slots_for_repair(active, ancestor_slots, protected)
            repair_calls += 1
            repair_parents += int(ancestor_slots.size)
            continue

        return active, protected, owner, repair_invalid_slots, repair_calls, repair_parents

    raise ArrayFallback("invalid_topology")

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
        raise ArrayFallback("invalid_topology")
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
        raise ArrayFallback("invalid_topology")
    return active, protected, {
        "repair_invalid_slots": int(overlap_invalid) + int(invalid_slots.size),
        "repair_refine_calls": int(overlap_calls) + int(repair_calls),
        "repair_refine_parents": int(overlap_parents) + int(repair_parents),
    }
