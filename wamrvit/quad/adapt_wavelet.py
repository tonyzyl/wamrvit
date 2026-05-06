import copy
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from numba import njit

from wamrvit.quad.quadtree import (
    Dir,
    QuadCell,
    Quadtree,
    apply_tree_diff,
    compute_tree_diff,
    merge_many_quadtrees,
)


@njit(boundscheck=False, fastmath=True, cache=True)
def bilinear_sample_jit(patch: np.ndarray, px: float, py: float) -> float:
    Ph, Pw = patch.shape

    x_f = int(np.floor(px))
    y_f = int(np.floor(py))
    x_c = x_f + 1
    y_c = y_f + 1

    x_f = max(0, min(Pw - 1, x_f))
    x_c = max(0, min(Pw - 1, x_c))
    y_f = max(0, min(Ph - 1, y_f))
    y_c = max(0, min(Ph - 1, y_c))

    wx = px - np.floor(px)
    wy = py - np.floor(py)

    val_ff = patch[y_f, x_f]
    val_fc = patch[y_f, x_c]
    val_cf = patch[y_c, x_f]
    val_cc = patch[y_c, x_c]

    top = val_ff * (1.0 - wx) + val_fc * wx
    bot = val_cf * (1.0 - wx) + val_cc * wx

    return top * (1.0 - wy) + bot * wy


# Helper: Slice the field for a specific cell (updated for hx/hy)
@njit(boundscheck=False, fastmath=True, cache=True)
def get_cell_slice_bounds_jit(
    cx: float,
    cy: float,
    hx: float,
    hy: float,
    xmin: float,
    ymin: float,
    dx: float,
    dy: float,
    xmax: float,
    ymax: float,
    W: int,
    H: int,
):
    x0 = max(xmin, cx - hx)
    x1 = min(xmax, cx + hx)
    y0 = max(ymin, cy - hy)
    y1 = min(ymax, cy + hy)

    j_start = int(np.floor((x0 - xmin) / dx))
    j_end = int(np.ceil((x1 - xmin) / dx))
    i_start = int(np.floor((y0 - ymin) / dy))
    i_end = int(np.ceil((y1 - ymin) / dy))

    j_start = max(0, min(j_start, W - 1))
    j_end = max(0, min(j_end, W))
    i_start = max(0, min(i_start, H - 1))
    i_end = max(0, min(i_end, H))

    if j_end <= j_start or i_end <= i_start:
        jc = max(0, min(int(np.floor((cx - xmin) / dx)), W - 1))
        ic = max(0, min(int(np.floor((cy - ymin) / dy)), H - 1))
        return ic, ic + 1, jc, jc + 1

    return i_start, i_end, j_start, j_end


@njit(boundscheck=False, fastmath=True, cache=True)
def compute_patch_range_jit(patch: np.ndarray):
    return np.max(patch) - np.min(patch)


class PatchSampler:
    """
    Samples the quadtree value at exact (x,y) using bilinear interpolation
    within the located leaf's patch.
    """

    def __init__(self, source_qt: Quadtree, channel_idx: int):
        self.qt = source_qt
        self.ch = channel_idx

    def sample(self, x: float, y: float) -> float:
        leaf = self.qt.locate_leaf(x, y)
        if leaf is None or leaf.value is None:
            return 0.0

        v = np.asarray(leaf.value)
        if v.ndim != 3:
            return 0.0

        patch = v[self.ch]
        Ph, Pw = patch.shape

        width = 2.0 * leaf.hx
        height = 2.0 * leaf.hy

        u = (x - (leaf.cx - leaf.hx)) / width
        v_norm = (y - (leaf.cy - leaf.hy)) / height

        u = max(0.0, min(1.0, u))
        v_norm = max(0.0, min(1.0, v_norm))

        px = u * Pw - 0.5
        py = v_norm * Ph - 0.5

        return bilinear_sample_jit(patch, px, py)


def combined_error_metric(cell, sampler: PatchSampler, ch_idx: int) -> float:
    """
    Combines:
    1. Internal Variation: Range of values inside the patch.
    2. Boundary Discontinuity: Difference between patch edge and neighbor.
    """
    # A. Internal Variation
    # This catches features smaller than the cell but resolved by the patch
    patch = cell.value[ch_idx]
    # internal_err = patch.max() - patch.min()
    internal_err = compute_patch_range_jit(patch)

    # B. Boundary Discontinuity — sample 4 cardinal neighbors via the sampler;
    # the internal variation check above + this sampling cover both detectors.
    v_center = sampler.sample(cell.cx, cell.cy)
    v_w = sampler.sample(cell.cx - cell.hx * 2.0, cell.cy)
    v_e = sampler.sample(cell.cx + cell.hx * 2.0, cell.cy)
    v_s = sampler.sample(cell.cx, cell.cy - cell.hy * 2.0)
    v_n = sampler.sample(cell.cx, cell.cy + cell.hy * 2.0)

    boundary_err = max(
        abs(v_center - v_w), abs(v_center - v_e), abs(v_center - v_s), abs(v_center - v_n)
    )

    return max(internal_err, boundary_err)


def adapt_on_field(
    qt: Quadtree,
    field_arr: np.ndarray,
    tol_frac: float | Sequence[float],
    *,
    channel: int | Sequence[int] | None = None,
    max_passes: int | None = None,  # <--- NEW ARGUMENT
    update_values: bool = False,  # <--- Added convenience
) -> None:
    """
    Refine based on a provided field (2D or 3D [C,H,W]) using a wavelet-like error.

    Args:
        qt: The target Quadtree.
        field_arr: The dense field data (C, H, W).
        tol_frac: Tolerance fraction (relative to global channel range).
        channel: Specific channels to adapt on.
        max_passes: Maximum number of refinement iterations. If None, defaults
                    to qt.max_level_idx (full resolution).
        update_values: If True, projects the field values onto the leaves
                       after adaptation.
    """
    # 1. Input validation and Setup
    if field_arr.ndim == 2:
        field_arr = field_arr[np.newaxis, ...]  # Ensure (C, H, W)

    C, H, W = field_arr.shape

    # Determine channels to process
    if channel is None:
        selected_channels = list(range(C))
    elif isinstance(channel, int):
        selected_channels = [channel]
    else:
        selected_channels = list(channel)

    # Handle tolerance broadcasting
    if isinstance(tol_frac, (float, int)):
        tols = [float(tol_frac)] * len(selected_channels)
    else:
        tols = list(tol_frac)
        if len(tols) != len(selected_channels):
            raise ValueError("Length of tol_frac must match number of selected channels.")

    # Precompute domain mapping constants
    xmin, xmax = qt.xmin, qt.xmax
    ymin, ymax = qt.ymin, qt.ymax
    dx = (xmax - xmin) / max(W, 1)
    dy = (ymax - ymin) / max(H, 1)

    # Determine refinement passes
    # Default to max_level_idx to ensure we can drill down from root to leaf in one go
    passes_to_run = max_passes if max_passes is not None else qt.max_level_idx

    def get_cell_slice(c: "QuadCell", data: np.ndarray) -> np.ndarray:
        i_start, i_end, j_start, j_end = get_cell_slice_bounds_jit(
            c.cx, c.cy, c.hx, c.hy, xmin, ymin, dx, dy, xmax, ymax, W, H
        )
        return data[i_start:i_end, j_start:j_end]

    # 2. Compute Refinement per Channel on Separate Trees
    temp_trees: list[Quadtree] = []

    for i, ch_idx in enumerate(selected_channels):
        ch_data = field_arr[ch_idx]
        global_range = ch_data.max() - ch_data.min()
        threshold = tols[i] * global_range

        # Initialize a fresh coarse tree matching qt's config (including tile dims)
        qt_ch = Quadtree(
            xmin,
            xmax,
            ymin,
            ymax,
            max_level_idx=qt.max_level_idx,
            channels=0,
            tile_width=qt.tile_width,
            tile_height=qt.tile_height,
        )

        def predicate(cell: "QuadCell") -> bool:
            region = get_cell_slice(cell, ch_data)
            if region.size == 0:
                return False
            local_range = np.max(region) - np.min(region)
            return local_range > threshold

        # Iteratively refine up to max_passes
        for _ in range(passes_to_run):
            count = qt_ch.refine_where(predicate)
            if count == 0:
                break

        temp_trees.append(qt_ch)

    # 3. Merge Structures
    if not temp_trees:
        return

    target_topology = merge_many_quadtrees(temp_trees, mode="stack")

    # 4. Apply Topology to Original Tree (In-Place)
    diff = compute_tree_diff(qt, target_topology, include_values=False)

    apply_tree_diff(
        qt, diff, update_values=False, allow_refine=True, allow_coarsen=True, maintain_balance=True
    )

    # 5. Optional: Update values from field
    if update_values:
        qt.assign_from_array(field_arr)


def channel_range(qt: Quadtree, ch: int) -> float:
    vmin, vmax = np.inf, -np.inf
    for leaf in qt._iter_all_leaves():
        if getattr(leaf, "value", None) is None:
            continue
        v = np.asarray(leaf.value)
        if v.ndim == 3 and ch < v.shape[0]:
            patch = v[ch]
            curr_min, curr_max = patch.min(), patch.max()
        elif v.ndim == 0 and ch == 0:
            curr_min = curr_max = float(v)
        else:
            continue

        if curr_min < vmin:
            vmin = float(curr_min)
        if curr_max > vmax:
            vmax = float(curr_max)

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0.0
    return float(vmax - vmin)


def expand_fine_region(qt: "Quadtree", adapt_nearby: int) -> "Quadtree":
    """Pad refined regions by ``adapt_nearby`` cells of buffer.

    For each leaf at ``level_idx > 0``, force any same-or-coarser-level neighbors
    within ``adapt_nearby`` cells (in any direction) to be refined to at least
    the leaf's level. Used by ``regrid`` Phase 2 to give the predictor context
    around sharp features (shock fronts, interfaces) instead of cliffs at the
    refined region's boundary.

    Returns a deep copy; the input ``qt`` is not modified.
    """
    new_qt = copy.deepcopy(qt)

    if adapt_nearby <= 0:
        return new_qt

    # 1. Enforce initial 2:1 balance to ensure stable boundaries
    while new_qt.ensure_2to1_balance() > 0:
        pass

    # 2. Iteratively expand the fine regions outward layer by layer
    for _ in range(adapt_nearby):
        # Get ALL leaves that are currently refined (level_idx > 0)
        refined_leaves = [leaf for leaf in new_qt._iter_all_leaves() if leaf.level_idx > 0]

        if not refined_leaves:
            break

        to_refine_uids: set[int] = set()
        to_refine_leaves: list[QuadCell] = []

        for leaf in refined_leaves:
            for direction in Dir:
                neighbors = new_qt.face_neighbors(leaf, direction)
                for neighbor_leaf in neighbors:
                    # Target immediate neighbors that are coarser than the current leaf
                    if neighbor_leaf.level_idx < leaf.level_idx:
                        uid = int(new_qt.cell_uid_64bit(neighbor_leaf))
                        if uid not in to_refine_uids:
                            to_refine_uids.add(uid)
                            to_refine_leaves.append(neighbor_leaf)

        # 3. Refine those coarser boundary neighbors
        for leaf in to_refine_leaves:
            # Safety check: never exceed the global max_level_idx preset
            if leaf.is_leaf() and leaf.level_idx < new_qt.max_level_idx:
                new_qt.refine_leaf(leaf)

        # 4. Maintain 2:1 balance after stepping outward
        # This guarantees the quadtree remains valid and smooths out the new boundaries.
        while new_qt.ensure_2to1_balance() > 0:
            pass

        # qt.mark_boundary_cells()  # disabled: is_boundary flag unread by active pipeline

    return new_qt


def regrid(
    qt: "Quadtree",
    tol_frac: float | Sequence[float] | np.ndarray,
    *,
    channel: int | Sequence[int] | None = None,
    max_passes: int = 99,
    coarsen_ratio: float = 0.25,
    adapt_nearby: int = 0,
    allow_coarsening: bool = True,
    disable_warnings: bool = False,
) -> None:
    """Adaptively re-mesh ``qt`` in place via a 3-phase refine/protect/coarsen pass.

    Phases:
        1. **Refinement**: split leaves whose combined error metric (internal
           variation + boundary discontinuity) exceeds ``tol_frac``.
        2. **Protection buffer** (only if ``adapt_nearby > 0``): expand each
           refined region outward by ``adapt_nearby`` cells so neighborhoods
           around features stay resolved at the same level.
        3. **Coarsening** (only if ``allow_coarsening``): merge sibling leaves
           whose combined value range falls below ``coarsen_ratio * tol_frac``.
           Skipped entirely when ``allow_coarsening=False``.

    The 2:1 balance constraint is enforced after refinement; coarsening uses a
    predicate that rejects any merge that would break it.

    Args:
        qt: Quadtree to mutate (this function does NOT return a new tree).
        tol_frac: per-channel refinement tolerance, broadcast against
            ``channel_range(qt, ch)``. Scalar applies uniformly; sequence must
            match the number of selected channels.
        channel: channel index/indices that drive adaptation. ``None`` = all.
        max_passes: cap on iterations within each phase (refine and coarsen).
        coarsen_ratio: coarsen when value range < ``coarsen_ratio * tol_frac``.
        adapt_nearby: protection buffer width in cells of the refined level.
        allow_coarsening: if False, skip phase 3 (refinement-only regrid).
        disable_warnings: suppress the "reached max_passes" warning.
    """
    C = max(0, int(getattr(qt, "channels", 0)))
    if C == 0:
        return

    # --- 1. Channel Selection ---
    channels_to_use = (
        list(range(C))
        if channel is None
        else ([channel] if isinstance(channel, int) else list(channel))
    )
    channels_to_use = [c for c in channels_to_use if 0 <= c < C]
    if not channels_to_use:
        return

    # --- 2. Tolerance Calculation ---
    tols_input = (
        [float(tol_frac)] * len(channels_to_use)
        if isinstance(tol_frac, (float, int))
        else [float(t) for t in tol_frac]
    )
    if len(tols_input) != len(channels_to_use):
        raise ValueError("The length of tol_frac must match the number of specified channels.")

    tols_refine, tols_coarsen = [], []
    for i, ch in enumerate(channels_to_use):
        abs_tol = tols_input[i] * abs(channel_range(qt, ch))
        tols_refine.append(abs_tol)
        tols_coarsen.append(abs_tol * coarsen_ratio)

    # --- Helper: Clear flags ---
    def clear_flags(node):
        node._protected = False
        if not node.is_leaf():
            for c in node.children:
                if c is not None:
                    clear_flags(c)

    for row in qt.roots:
        for r in row:
            clear_flags(r)

    # --- Monkey-Patch: Ensure protection flags survive tree balancing ---
    # When a cell refines, its children need to inherit the protection flag.
    original_refine = qt.roots[0][0].__class__.refine  # Robust way to get QuadCell.refine

    def patched_refine(self, init_child=None, interpolation="bilinear"):
        def combo_init(p, c):
            c._protected = getattr(p, "_protected", False)
            if init_child:
                init_child(p, c)

        original_refine(self, init_child=combo_init, interpolation=interpolation)

    qt.roots[0][0].__class__.refine = patched_refine

    try:
        # ==========================================
        # PHASE 1: REFINEMENT (Drill down on features)
        # ==========================================
        for pass_idx in range(max_passes):
            changed = 0
            samplers = [PatchSampler(qt, ch) for ch in channels_to_use]

            def _refine_pred(cell: "QuadCell") -> bool:
                if cell.level_idx >= qt.max_level_idx:
                    return False
                for i, ch in enumerate(channels_to_use):
                    if combined_error_metric(cell, samplers[i], ch) > tols_refine[i]:
                        return True
                return False

            changed += qt.refine_where(_refine_pred)
            changed += qt.ensure_2to1_balance()
            if changed == 0:
                break

        # ==========================================
        # PHASE 2: COMPUTE & EXPAND PROTECTION BUFFER
        # ==========================================
        if adapt_nearby > 0:
            # 1. Flag initial "Anchors" (cells with high variation)
            for leaf in qt._iter_all_leaves():
                if leaf.value is None:
                    continue
                for i, ch in enumerate(channels_to_use):
                    patch = leaf.value[ch]
                    if (np.max(patch) - np.min(patch)) >= tols_coarsen[i]:
                        leaf._protected = True
                        break

            # 2. Dilate the boolean flag outward (Extremely fast graph step)
            for _ in range(adapt_nearby):
                new_protected = []
                for leaf in qt._iter_all_leaves():
                    if getattr(leaf, "_protected", False):
                        for d in Dir:
                            for n in qt.face_neighbors(leaf, d):
                                if not getattr(n, "_protected", False):
                                    new_protected.append(n)
                for n in new_protected:
                    n._protected = True

            # 3. Physically step-up the resolution of the protected buffer
            for _ in range(adapt_nearby):
                to_refine = []
                for leaf in qt._iter_all_leaves():
                    # Only expand from protected cells that are already refined
                    if getattr(leaf, "_protected", False) and leaf.level_idx > 0:
                        for d in Dir:
                            for n in qt.face_neighbors(leaf, d):
                                if getattr(n, "_protected", False) and n.level_idx < leaf.level_idx:
                                    to_refine.append(n)

                # Fast deduplication using Python memory IDs
                unique_to_refine = {id(n): n for n in to_refine}.values()
                expanded = 0
                for n in unique_to_refine:
                    if n.is_leaf() and n.level_idx < qt.max_level_idx:
                        qt.refine_leaf(n)
                        expanded += 1

                while True:
                    bal = qt.ensure_2to1_balance()
                    if bal == 0:
                        break
                    expanded += bal

                if expanded == 0:
                    break

        # ==========================================
        # PHASE 3: COARSENING (Clean up flat regions)
        # ==========================================
        if not allow_coarsening:
            return

        def _coarsen_pred(node: "QuadCell") -> bool:
            # 1. Reject if any child is part of the protected buffer!
            if adapt_nearby > 0:
                for c in node.children:
                    if c is not None and getattr(c, "_protected", False):
                        return False

            # 2. Balance-Aware Lookahead (PREVENTS THE DEADLOCK)
            # 'node' is at level_idx L. If coarsened, it becomes a leaf at level_idx L.
            # To maintain 2:1 balance, NO adjacent leaf can be at level_idx L + 2 or higher.
            for c in node.children:
                if c is not None:
                    for d in Dir:
                        for neighbor in qt.face_neighbors(c, d):
                            # If the neighbor is outside this 2x2 block and is too fine:
                            if neighbor.parent != node and neighbor.level_idx >= node.level_idx + 2:
                                return False

            # 3. Reject if data variation is too high
            for i, ch in enumerate(channels_to_use):
                vmin, vmax = np.inf, -np.inf
                for c in node.children:
                    if c is not None and c.value is not None:
                        patch = c.value[ch]
                        pmin, pmax = np.min(patch), np.max(patch)
                        if pmin < vmin:
                            vmin = pmin
                        if pmax > vmax:
                            vmax = pmax
                if (vmax - vmin) >= tols_coarsen[i]:
                    return False

            return True

        # Run the coarsening loop.
        # pass_idx is initialized so the post-loop warning check is well-defined
        # even when max_passes == 0 (no iterations).
        pass_idx = -1
        for pass_idx in range(max_passes):
            changed = qt.coarsen_where(_coarsen_pred)

            # We still run ensure_2to1_balance just in case, but it will do
            # almost zero work now because _coarsen_pred prevents illegal moves.
            # changed += qt.ensure_2to1_balance()

            if changed == 0:
                break

        if pass_idx == max_passes - 1 and not disable_warnings:
            warnings.warn(
                f"Reached max_passes={max_passes} during coarsening. "
                "Consider increasing max_passes."
            )

    finally:
        # Always restore the original refine method and clean up memory
        qt.roots[0][0].__class__.refine = original_refine
        for row in qt.roots:
            for r in row:
                clear_flags(r)

        # qt.mark_boundary_cells()  # disabled: is_boundary flag unread by active pipeline


def regrid_native(
    curr_by_level: dict[int, np.ndarray],
    leaf_to_bucket: np.ndarray,
    meta: dict[str, Any],
    *,
    C: int,
    T: int,
    tol_frac: float,
    cell_scale_mode: str = "area",
    adapt_on_channels: Sequence[int] | None = None,
    adapt_nearby: int = 0,
    allow_coarsening: bool = True,
    max_passes: int = 10,
) -> tuple[dict[int, np.ndarray], np.ndarray, dict[str, Any]]:
    """Regrid native-mode (multi-scale) per-level tensors via exact refine/coarsen.

    Operates entirely in numpy.  Callers handle torch <-> numpy conversion.

    Args:
        curr_by_level: ``{lvl: (N_l, C, T, H_l, W_l)}`` current per-level tensors.
        leaf_to_bucket: ``(N, 2)`` mapping from canonical leaf order to per-level
            bucket positions.
        meta: Quadtree metadata dict (``centers``, ``levels``, ``domain``).
        C: Number of physical channels.
        T: Number of time frames.
        tol_frac: Relative tolerance for the wavelet-based regrid.
        cell_scale_mode: Scale mode for center/level encoding.
        adapt_on_channels: Physical channels to drive adaptation (indices into C).
            If *None*, the last time frame's channels are used.
        adapt_nearby: Number of expansion layers around refined regions.
        allow_coarsening: Whether the coarsen phase of ``regrid`` is enabled.
        max_passes: Maximum refinement/coarsening passes.

    Returns:
        ``(new_by_level, new_leaf_to_bucket, new_meta)`` -- all numpy, with
        ``new_by_level`` shaped ``{lvl: (N_l, C, T, H_l, W_l)}``.
    """
    from wamrvit.quad.quad_utils import quadtree_to_tensor_native, tensor_to_quadtree_native

    # Flatten (N_l, C, T, H_l, W_l) -> (N_l, T*C, H_l, W_l) for wavelet adaptation.
    flat_by_level: dict[int, np.ndarray] = {}
    for lvl, arr in curr_by_level.items():
        if arr.shape[0] == 0:
            flat_by_level[lvl] = arr[:, :0]
            continue
        n, c, t, h, w = arr.shape
        flat_by_level[lvl] = arr.transpose(0, 2, 1, 3, 4).reshape(n, t * c, h, w)

    qt = tensor_to_quadtree_native(
        flat_by_level, leaf_to_bucket, meta, cell_scale_mode=cell_scale_mode
    )

    # Adapt using last-timestep channels.
    ch_offset = (T - 1) * C
    use_channels = (
        [ch + ch_offset for ch in adapt_on_channels]
        if adapt_on_channels is not None
        else list(range(ch_offset, T * C))
    )
    regrid(
        qt,
        tol_frac=tol_frac,
        channel=use_channels,
        max_passes=max_passes,
        adapt_nearby=adapt_nearby,
        allow_coarsening=allow_coarsening,
        disable_warnings=True,
    )

    new_buckets, new_l2b, new_meta = quadtree_to_tensor_native(qt, cell_scale_mode=cell_scale_mode)

    # Unflatten T*C -> (N_l, C, T, H_l, W_l) per level.
    new_by_level: dict[int, np.ndarray] = {}
    for lvl, flat_arr in new_buckets.items():
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

    return new_by_level, new_l2b, new_meta
