from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from numba import njit, prange

from wamrvit.quad.adapt_wavelet import regrid
from wamrvit.quad.quad_utils import quadtree_to_tensor, tensor_to_quadtree
from wamrvit.quad.quadtree import _ceil_log2
from wamrvit.quad.array_regrid_coarsen import (
    _active_coarsen_acceptance_mask_from_source_jit,
    _active_coarsen_acceptance_mask_parallel_jit,
    _active_coarsen_once_conservative,
    _active_coarsen_once_from_source_conservative,
    _active_coarsen_once_from_source_jit,
    _active_coarsen_once_jit,
    _fill_accepted_coarsened_parent_values_from_source_jit,
    _fill_accepted_coarsened_parent_values_jit,
)
from wamrvit.quad.array_regrid_geometry import (
    _active_leaf_fine_bounds,
    _as_numpy,
    _build_active_owner_grid,
    _export_active_level_idx_metadata_jit,
    _export_ordered_topology,
    _export_topology,
    _import_topology,
    _import_topology_metadata,
    _morton_keys_jit,
    _nx_ny_tiles,
)
from wamrvit.quad.array_regrid_protect import (
    _active_dilate_protected,
    _active_expand_protected_region_counted,
    _active_expand_protected_region_counted_from_source,
    _active_protected_mask,
    _active_protected_mask_from_source,
    _active_protected_mask_from_source_jit,
    _active_protected_mask_parallel_jit,
)
from wamrvit.quad.array_regrid_refine import (
    _active_ensure_2to1_balance_counted,
    _active_ensure_2to1_balance_counted_from_source,
    _active_refine_slots,
    _active_refine_slots_from_source,
    _active_refine_slots_from_source_jit,
    _active_refine_slots_jit,
    _compute_active_refine_slots_from_source_object_style,
    _compute_active_refine_slots_object_style,
    _fill_refined_child_values_from_source_jit,
    _fill_refined_child_values_jit,
    _initial_channel_ranges,
    _initial_sequence_channel_ranges,
    _repair_active_balance_or_raise,
    _repair_source_active_balance_or_raise,
)
from wamrvit.quad.array_regrid_runtime import (
    _ACTIVE_PROTECTED_MASK_PARALLEL_THRESHOLD,
    _COARSEN_PARALLEL_ACCEPT_THRESHOLD,
    _COARSEN_PARALLEL_FILL_THRESHOLD,
    _REFINE_PARALLEL_FILL_THRESHOLD,
    _SOURCE_REFINE_PARALLEL_FILL_THRESHOLD,
    _WARM_ARRAY_REGRID_ACTIVE_CAPACITY,
    _array_regrid_config_status,
    _array_regrid_param_status,
    _ensure_capacity,
    _normalize_array_regrid_mode,
    _normalize_channels,
    _normalize_tolerances,
)
from wamrvit.quad.array_regrid_types import (
    ActiveTopology,
    ArrayFallback,
    Channel,
    FlatTopology,
    SequenceSourceActiveTopology,
    Tol,
)
from wamrvit.quad.regrid_lineage import (
    DetectorSpec,
    LineagePlan,
    PayloadOp,
    lineage_plan_from_payload_ops,
)

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
        raise ArrayFallback("capacity_exceeded")
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
    *,
    record_lineage: bool = False,
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

    payload_ops: Optional[list[Optional[PayloadOp]]] = None
    if record_lineage:
        payload_ops = [None] * int(capacity)
        for source_slot in range(int(n_leaves)):
            payload_ops[source_slot] = PayloadOp(kind="source", source_slot=source_slot)

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
        payload_ops=payload_ops,
        lineage_phase="refine",
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


def _ordered_source_active_slots(active: SequenceSourceActiveTopology) -> np.ndarray:
    slots = active.active_slots[:active.active_count]
    x0, y0, _scale = _active_leaf_fine_bounds(active, slots)
    nx_tiles, ny_tiles = _nx_ny_tiles(active.domain)
    max_level = int(active.domain["max_level_idx"])
    bits = max_level + _ceil_log2(max(nx_tiles, ny_tiles))
    keys = _morton_keys_jit(x0, y0, bits)
    return slots[np.argsort(keys, kind="stable")].astype(np.int32, copy=False)


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
    ordered_slots = _ordered_source_active_slots(active)

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
    record_lineage: bool = False,
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
    active, _ = _active_from_sequence_source(
        sequence, top, capacity=capacity, record_lineage=record_lineage
    )
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
    except ArrayFallback as exc:
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


def array_regrid_topology_from_detectors(
    detectors: np.ndarray,
    meta: Dict[str, Any],
    *,
    detector_spec: DetectorSpec,
    cell_scale_mode: Optional[str],
    tol_frac: Tol,
    max_passes: int,
    coarsen_ratio: float = 0.25,
    adapt_nearby: int = 0,
    capacity: int = 8192,
) -> Tuple[np.ndarray, Dict[str, Any], LineagePlan, Dict[str, Any]]:
    """Run CPU topology using only latest-frame detector fields.

    ``detectors`` has shape ``(N,K,H,W)`` and contains no non-detector payload.
    Returned detector values are in final Morton order and provide a same-input
    value gate for the compact topology path.
    """
    start = time.perf_counter()
    detector_array = np.ascontiguousarray(_as_numpy(detectors), dtype=np.float32)
    if detector_array.ndim != 4:
        raise ValueError(
            f"Expected detector shape (N,K,H,W), got {detector_array.shape}."
        )
    detector_count = len(detector_spec.physical_channels)
    if detector_array.shape[1] != detector_count:
        raise ValueError(
            f"Detector tensor has {detector_array.shape[1]} fields but spec has "
            f"{detector_count}."
        )
    sequence = detector_array[:, :, None, :, :]
    top = _import_topology_metadata(int(sequence.shape[0]), meta, cell_scale_mode)
    after_import = time.perf_counter()
    active, run_timings, run_counts = _run_array_regrid_from_sequence_source_parity(
        sequence,
        top,
        tol_frac=tol_frac,
        channel=list(range(detector_count)),
        max_passes=max_passes,
        coarsen_ratio=coarsen_ratio,
        adapt_nearby=adapt_nearby,
        capacity=capacity,
        record_lineage=True,
    )
    after_run = time.perf_counter()
    if active.payload_ops is None:
        raise RuntimeError("Detector topology entry did not record payload lineage.")
    ordered_slots = _ordered_source_active_slots(active)
    plan = lineage_plan_from_payload_ops(
        active.payload_ops,
        next_slot=int(active.next_slot),
        initial_slots=int(active.initial_slots),
        final_slots=ordered_slots,
        patch_shape=(int(detector_array.shape[-2]), int(detector_array.shape[-1])),
    )
    detector_out_sequence, out_meta = _export_source_active_topology(
        active, output_layout="sequence"
    )
    detector_out = np.ascontiguousarray(detector_out_sequence[:, :, 0])
    end = time.perf_counter()
    return detector_out, out_meta, plan, {
        "backend": "array_detector_topology",
        "fallback_reason": None,
        **_array_regrid_config_status(capacity),
        **_array_regrid_param_status(
            tol_frac=tol_frac,
            channel=list(range(detector_count)),
            max_passes=max_passes,
            coarsen_ratio=coarsen_ratio,
            adapt_nearby=adapt_nearby,
        ),
        "detector_physical_channels": list(detector_spec.physical_channels),
        "detector_field_names": list(detector_spec.field_names),
        "detector_bytes": int(detector_array.nbytes),
        "timings": {
            "import": after_import - start,
            "run": after_run - after_import,
            "lineage_export": end - after_run,
            "total": end - start,
        },
        "run_timings": run_timings,
        "run_counts": run_counts,
    }


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
        top, run_timings, run_counts = _run_array_regrid_parity(
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
    except ArrayFallback as exc:
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
