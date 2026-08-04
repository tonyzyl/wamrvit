from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from numba import njit, prange

from wamrvit.quad.array_regrid_coarsen import (
    _active_complete_sibling_groups, _active_parent_side_has_too_fine_neighbor_jit,
)
from wamrvit.quad.array_regrid_geometry import (
    _active_leaf_fine_bounds, _build_active_owner_grid,
    _export_active_level_idx_metadata_jit, _find_active_balance_refinements,
    _import_topology_metadata, _morton_keys_jit, _nx_ny_tiles,
)
from wamrvit.quad.array_regrid_protect import (
    _active_dilate_protected, _active_protected_region_refine_slots_object_neighbors_jit,
)
from wamrvit.quad.array_regrid_refine import (
    _bilinear_sample_patch_jit, _owner_slot_for_physical_point_jit,
)
from wamrvit.quad.array_regrid_runtime import (
    _ensure_capacity, _normalize_array_regrid_mode, _normalize_tolerances,
)
from wamrvit.quad.array_regrid_types import ArrayFallback, Tol
from wamrvit.quad.quadtree import _ceil_log2


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
    # StopIteration escape the entry's `except (ArrayFallback, ValueError)`.
    probe_lvl = next((l for l in range(L + 1) if l in by_level and by_level[l].shape[0] > 0), None)
    if probe_lvl is None:
        raise ArrayFallback("empty_topology")
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
            raise ArrayFallback("native_level_capacity")
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
        raise ArrayFallback("capacity_exceeded")

    # Per-level value capacity guard: children land at level parent_level+1; the kernel writes
    # 4 child rows PER PARENT, so multiply the per-parent tally by 4 (a parent-count guard
    # would be 4x too loose and permit a silent OOB write with boundscheck=False).
    child_levels = active.level_idx[to_refine].astype(np.int64) + 1
    counts = np.bincount(child_levels, minlength=L + 1) * 4
    for lc in range(L + 1):
        if int(counts[lc]) == 0:
            continue
        if int(active.level_next_pos[lc]) + int(counts[lc]) > int(active.level_cap[lc]):
            raise ArrayFallback("native_level_capacity")

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
    # the struct untouched (the Python wrapper raises ArrayFallback; no partial mutation).
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
        raise ArrayFallback("native_level_capacity")
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
    Everything else — timings/counts keys, tolerance math, loop structure, validate→ArrayFallback,
    final slot accounting — is copied verbatim from the uniform twin.

    Note on capacity churn: native refine/coarsen append rows to per-level value arrays and never
    reclaim dead rows.  Per-level capacity is churn_slack(3)*tiles*4^l; if a multi-pass regrid's
    churn exceeds that, the native refine/coarsen raises ArrayFallback (clean fallback).
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
        raise ArrayFallback("invalid_topology")
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
    * ``array_regrid_mode`` is retained for API symmetry, accepts only ``"parity"``,
      and is echoed into ``status``. The removed approximate mode is rejected explicitly.
    * Inputs are never mutated: the importer copies every patch + the domain, so the
      object fallback operates on the original arrays safely.
    """
    array_regrid_mode = _normalize_array_regrid_mode(array_regrid_mode)
    start = time.perf_counter()
    try:
        if not allow_coarsening:
            # The native driver always runs the coarsen phase; refinement-only is not the
            # paper's rollout config -> fall back to the object backend transparently.
            raise ArrayFallback("allow_coarsening_unsupported")

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
    except (ArrayFallback, ValueError) as exc:
        from wamrvit.quad.adapt_wavelet import regrid_native

        fallback_reason = exc.reason if isinstance(exc, ArrayFallback) else "invalid_topology"
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
    except ArrayFallback:
        # A degenerate synthetic case that triggers fallback still compiles
        # the kernels that ran before the fallback.
        pass
