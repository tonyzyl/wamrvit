from typing import Any, Literal, Optional, Union

import numpy as np
import torch

from wamrvit.quad.quadtree import (
    Quadtree,
    TreeDiff,
    apply_tree_diff,
    compute_tree_diff,
)


def random_resample_quadtree(
    qt: Quadtree,
    coarsen_ratio: float = 0.1,
    refine_ratio: float = 0.1,
    rng: np.random.Generator | None = None,
) -> None:
    """
    Randomly perturb quadtree topology as a data augmentation.
    Coarsens some refined cells and refines some coarse cells,
    then restores 2:1 balance.

    Must be called *before* assign_from_array so data is filled
    into the perturbed structure.

    Args:
        qt: Quadtree to mutate in-place.
        coarsen_ratio: Fraction of coarsenable parents to coarsen.
        refine_ratio: Fraction of refinable leaves to refine.
        rng: NumPy random generator for reproducibility.
    """
    if coarsen_ratio <= 0 and refine_ratio <= 0:
        return

    if rng is None:
        rng = np.random.default_rng()

    # --- Coarsen phase ---
    if coarsen_ratio > 0:
        # Collect internal nodes whose 4 children are all leaves and above level 0
        coarsenable = []
        for leaf in qt._iter_all_leaves():
            p = leaf.parent
            if p is None:
                continue
            # Check: all 4 siblings are leaves and above root level
            if (
                all(c is not None and c.is_leaf() for c in p.children)
                and p.children[0].level_idx > 0
            ):
                coarsenable.append(p)

        # Deduplicate (each parent seen up to 4 times via its children)
        seen_ids: set = set()
        unique: list = []
        for p in coarsenable:
            pid = id(p)
            if pid not in seen_ids:
                seen_ids.add(pid)
                unique.append(p)
        coarsenable = unique

        n_coarsen = max(1, int(len(coarsenable) * coarsen_ratio))
        if coarsenable:
            chosen = rng.choice(
                len(coarsenable), size=min(n_coarsen, len(coarsenable)), replace=False
            )
            for idx in chosen:
                qt.coarsen_node(coarsenable[idx])

    # --- Refine phase ---
    if refine_ratio > 0:
        refinable = [leaf for leaf in qt._iter_all_leaves() if leaf.level_idx < qt.max_level_idx]
        n_refine = max(1, int(len(refinable) * refine_ratio))
        if refinable:
            chosen = rng.choice(len(refinable), size=min(n_refine, len(refinable)), replace=False)
            for idx in chosen:
                qt.refine_leaf(refinable[idx])

    # --- Restore 2:1 balance ---
    while qt.ensure_2to1_balance() > 0:
        pass

    # qt.mark_boundary_cells()  # disabled: is_boundary flag unread by active pipeline


def quadtree_to_tensor(
    qt: Quadtree,
    *,
    order: Literal["morton", "uid", "raster"] = "morton",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    cell_scale_mode: str | None = None,
    return_tensor: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Export a uniform-storage Quadtree as a flat (N, C, Ph, Pw) leaf-patch tensor.

    N is the number of leaves at the time of export; Ph, Pw are the quadtree's
    patch dimensions (constant across leaves in uniform mode). Leaves are
    serialised in ``order`` — Morton (default) is the canonical choice and the
    one matched by ``tensor_to_quadtree``.

    Returns:
        (X, meta) where X has shape ``(N, C, Ph, Pw)`` and meta carries
        ``centers, levels, tiles, xy_idx, cell_ids, domain`` — everything
        ``tensor_to_quadtree`` needs to reconstruct the same topology.

    Args:
        order: leaf traversal order. Use the same order on the inverse call.
        device, dtype: only used when ``return_tensor=True``.
        cell_scale_mode: if set (e.g., ``"area"``), normalize centers to [-1, 1].
        return_tensor: True returns a torch tensor; False returns numpy.
    """
    exp = qt.export_leaves(order=order, cell_scale_mode=cell_scale_mode)
    values = exp["values"]
    assert values is not None, "Quadtree has 0 channels; cannot extract values tensor."

    # 1. Capture exact domain parameters from the quadtree instance
    domain_meta = qt.get_domain_meta()

    if return_tensor:
        x = torch.as_tensor(values, device=device, dtype=dtype if dtype else torch.float32)
        meta = {
            "centers": torch.as_tensor(exp["centers"], device=device, dtype=torch.float32),
            "levels": torch.as_tensor(exp["levels"], device=device, dtype=torch.int16),
            "tiles": torch.as_tensor(exp["tiles"], device=device, dtype=torch.int16),
            "xy_idx": torch.as_tensor(exp["xy_idx"], device=device, dtype=torch.int32),
            "cell_ids": torch.as_tensor(
                exp["ids"].astype(np.int64), device=device, dtype=torch.long
            ),
            "domain": domain_meta,
        }
    else:
        x = values
        meta = {
            "centers": exp["centers"],
            "levels": exp["levels"],
            "tiles": exp["tiles"],
            "xy_idx": exp["xy_idx"],
            "cell_ids": exp["ids"],
            "domain": domain_meta,
        }

    return x, meta


# save/load quadtree remain effectively unchanged as they just serialize numpy arrays
def save_quadtree(qt: Quadtree, path: str) -> None:
    coarse_qt = Quadtree(
        qt.xmin,
        qt.xmax,
        qt.ymin,
        qt.ymax,
        max_level_idx=qt.max_level_idx,
        channels=qt.channels,
        tile_width=qt.tile_width,
        tile_height=qt.tile_height,
    )
    diff = compute_tree_diff(coarse_qt, qt, include_values=True)
    save_dict = {
        "xmin": diff.xmin,
        "xmax": diff.xmax,
        "ymin": diff.ymin,
        "ymax": diff.ymax,
        "nx_tiles": diff.nx_tiles,
        "ny_tiles": diff.ny_tiles,
        "max_level_idx": diff.max_level_idx,
        "channels": diff.channels,
        "ids": diff.ids,
        "tuples": diff.tuples,
    }
    if diff.values is not None:
        save_dict["values"] = diff.values
    if diff.ops is not None:
        save_dict["ops"] = diff.ops

    np.savez(path, **save_dict)


def load_quadtree(path: str) -> Quadtree:
    with np.load(path) as data:
        diff = TreeDiff(
            xmin=data["xmin"].item(),
            xmax=data["xmax"].item(),
            ymin=data["ymin"].item(),
            ymax=data["ymax"].item(),
            nx_tiles=data["nx_tiles"].item(),
            ny_tiles=data["ny_tiles"].item(),
            max_level_idx=data["max_level_idx"].item(),
            channels=data["channels"].item(),
            ids=data["ids"],
            tuples=data["tuples"],
            values=data.get("values"),
            ops=data.get("ops"),
        )

    # Infer tile dims
    tw = (diff.xmax - diff.xmin) / max(1, diff.nx_tiles)
    th = (diff.ymax - diff.ymin) / max(1, diff.ny_tiles)

    qt = Quadtree(
        xmin=diff.xmin,
        xmax=diff.xmax,
        ymin=diff.ymin,
        ymax=diff.ymax,
        max_level_idx=diff.max_level_idx,
        channels=diff.channels,
        tile_width=tw,
        tile_height=th,
    )

    apply_tree_diff(qt, diff, update_values=True)
    return qt


def tensor_to_quadtree(
    data: Union[np.ndarray, "torch.Tensor"],
    meta: dict[str, Any],
    cell_scale_mode: str | None = "area",  # Explicit control kwarg
) -> "Quadtree":
    """Inverse of ``quadtree_to_tensor``: rebuild a uniform-storage Quadtree.

    The leaf order in ``data`` must match the order used at export (typically
    Morton); the meta dict must carry ``centers``, ``levels``, and ``domain``
    (anything ``quadtree_to_tensor`` writes is sufficient). If
    ``cell_scale_mode`` is set, centers are assumed normalized and rescaled to
    absolute domain coordinates before reconstruction.

    Topology is rebuilt from (centers, levels) and values copied row-by-row;
    the resulting Quadtree is independent of the input arrays.
    """
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()

    centers = meta["centers"]
    if hasattr(centers, "detach"):
        centers = centers.detach().cpu().numpy()

    levels = meta["levels"]
    if hasattr(levels, "detach"):
        levels = levels.detach().cpu().numpy()

    C = data.shape[1]
    domain = meta["domain"]

    qt = Quadtree(
        xmin=domain["xmin"],
        xmax=domain["xmax"],
        ymin=domain["ymin"],
        ymax=domain["ymax"],
        max_level_idx=domain["max_level_idx"],
        channels=C,
        tile_width=domain["tile_width"],
        tile_height=domain["tile_height"],
    )

    # --- Explicit Coordinate Handling via Kwarg ---
    if cell_scale_mode is not None:
        Lx = domain["xmax"] - domain["xmin"]
        Ly = domain["ymax"] - domain["ymin"]

        # Scale X and Y independently to match the export_leaves normalization
        abs_cx = (centers[:, 0] * Lx) + domain["xmin"]
        abs_cy = (centers[:, 1] * Ly) + domain["ymin"]
    else:
        abs_cx = centers[:, 0]
        abs_cy = centers[:, 1]

    order = np.argsort(levels)

    for i in order:
        cx, cy = abs_cx[i], abs_cy[i]
        target_lvl = levels[i]

        leaf = qt.locate_leaf(cx, cy)
        if leaf is None:
            raise ValueError(f"Coordinate ({cx}, {cy}) is outside the Quadtree domain bounds.")

        while leaf.level_idx < target_lvl:
            qt.refine_leaf(leaf)
            leaf = qt.locate_leaf(cx, cy)

        if leaf.level_idx == target_lvl:
            leaf.value = data[i].copy()

    # qt.mark_boundary_cells()  # disabled: is_boundary flag unread by active pipeline

    return qt


def quadtree_to_uniform(
    qt: "Quadtree",
    return_tensor: bool = False,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Converts a Quadtree into a uniform, max-resolution dense grid.
    Returns an array (or tensor) of shape (C, H, W).
    """
    # Calculate the scale factor from the root tile to the finest minimal cell
    max_scale = 1 << qt.max_level_idx

    # Calculate full uniform dimensions at max resolution
    H = qt.ny_tiles * qt.patch_height * max_scale
    W = qt.nx_tiles * qt.patch_width * max_scale
    C = qt.channels

    # Initialize the empty dense array
    out = np.zeros((C, H, W), dtype=np.float32)

    for leaf in qt._iter_all_leaves():
        val = leaf.value
        if val is None:
            continue

        # Determine how much this leaf needs to be scaled up to match the max resolution
        scale = max(0, qt.max_level_idx - leaf.level_idx)

        # Get the logical base coordinates of the leaf in terms of minimal cells
        x0, y0 = qt._leaf_fine_coords(leaf)

        # Map to actual pixel indices in the dense grid
        px_start = x0 * qt.patch_width
        py_start = y0 * qt.patch_height

        px_end = px_start + ((1 << scale) * qt.patch_width)
        py_end = py_start + ((1 << scale) * qt.patch_height)

        if scale > 0:
            # SCORING ONLY: nearest-neighbor upsample of coarse cells onto the dense grid
            # for the uniform-adaptive full-field metric. NOTE: refine() value-fill is now
            # bilinear (object/array byte-parity), so this no longer mirrors refine -- kept
            # nearest deliberately for now. It is applied IDENTICALLY to both regrid backends,
            # so it does not affect the object-vs-array comparison (only absolute metric scale).
            val_up = np.repeat(np.repeat(val, 1 << scale, axis=1), 1 << scale, axis=2)
            out[:, py_start:py_end, px_start:px_end] = val_up
        else:
            # Already at maximum resolution
            out[:, py_start:py_end, px_start:px_end] = val

    if return_tensor:
        return torch.as_tensor(out, device=device, dtype=dtype if dtype else torch.float32)

    return out


def tensor_to_uniform(
    data: Union[np.ndarray, "torch.Tensor"],
    meta: dict[str, Any],
    cell_scale_mode: str | None = "area",  # Explicit control kwarg
    return_tensor: bool = False,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Converts a flat batch of patches directly into a uniform, max-resolution dense grid,
    bypassing explicit Quadtree construction.
    Returns an array (or tensor) of shape (C, H, W).
    """
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()

    centers = meta["centers"]
    if hasattr(centers, "detach"):
        centers = centers.detach().cpu().numpy()

    levels = meta["levels"]
    if hasattr(levels, "detach"):
        levels = levels.detach().cpu().numpy()

    C = data.shape[1]
    domain = meta["domain"]

    # --- Reconstruct target grid dimensions ---
    Lx = domain["xmax"] - domain["xmin"]
    Ly = domain["ymax"] - domain["ymin"]

    nx_tiles = int(round(Lx / domain["tile_width"]))
    ny_tiles = int(round(Ly / domain["tile_height"]))

    max_lvl = domain["max_level_idx"]
    factor = 1 << max_lvl

    patch_width = int(round(domain["tile_width"] / factor))
    patch_height = int(round(domain["tile_height"] / factor))

    # Full uniform dimensions at max resolution
    H = ny_tiles * patch_height * factor
    W = nx_tiles * patch_width * factor

    out = np.zeros((C, H, W), dtype=np.float32)

    # --- Explicit Coordinate Handling via Kwarg ---
    if cell_scale_mode is not None:
        # Scale X and Y independently to match the export_leaves normalization
        abs_cx = (centers[:, 0] * Lx) + domain["xmin"]
        abs_cy = (centers[:, 1] * Ly) + domain["ymin"]
    else:
        abs_cx = centers[:, 0]
        abs_cy = centers[:, 1]

    # Pre-calculate pixel mappings per tile to avoid floating-point drift
    pixels_per_tile_x = patch_width * factor
    pixels_per_tile_y = patch_height * factor

    # Sort by level so lower resolutions (parents) are drawn first,
    # ensuring children correctly overwrite them if overlaps exist.
    order = np.argsort(levels)

    for i in order:
        cx, cy = abs_cx[i], abs_cy[i]
        lvl = int(levels[i])

        # 1. Determine the half-dimensions of this cell in physical units
        hx = (domain["tile_width"] / (1 << lvl)) * 0.5
        hy = (domain["tile_height"] / (1 << lvl)) * 0.5

        x_tl = cx - hx
        y_tl = cy - hy

        # 2. Map physical top-left coordinate to pixel indices using robust rounding
        px_start = int(round((x_tl - domain["xmin"]) / domain["tile_width"] * pixels_per_tile_x))
        py_start = int(round((y_tl - domain["ymin"]) / domain["tile_height"] * pixels_per_tile_y))

        # 3. Determine the pixel width/height of this specific patch
        scale = max_lvl - lvl
        pw_scaled = patch_width * (1 << scale)
        ph_scaled = patch_height * (1 << scale)

        px_end = px_start + pw_scaled
        py_end = py_start + ph_scaled

        val = data[i]

        # 4. Upsample if the patch isn't at the maximum resolution
        if scale > 0:
            # SCORING ONLY: nearest-neighbor upsample of coarse cells onto the dense grid
            # for the uniform-adaptive full-field metric. NOTE: refine() value-fill is now
            # bilinear (object/array byte-parity), so this no longer mirrors refine -- kept
            # nearest deliberately for now. It is applied IDENTICALLY to both regrid backends,
            # so it does not affect the object-vs-array comparison (only absolute metric scale).
            val = np.repeat(np.repeat(val, 1 << scale, axis=1), 1 << scale, axis=2)

        out[:, py_start:py_end, px_start:px_end] = val

    if return_tensor:
        import torch

        return torch.as_tensor(out, device=device, dtype=dtype if dtype else torch.float32)

    return out


# ---------------------------------------------------------------------------
# Native-resolution variants (value_storage="native")
# ---------------------------------------------------------------------------


def quadtree_to_tensor_native(
    qt: Quadtree,
    *,
    order: Literal["morton", "uid", "raster"] = "morton",
    cell_scale_mode: str | None = None,
) -> tuple[dict[int, np.ndarray], np.ndarray, dict[str, Any]]:
    """
    Export a native-storage Quadtree into per-level value buckets.

    Returns:
        values_by_level: Dict[level_idx -> ndarray of shape (N_l, C, Ph*s, Pw*s)]
                         where s = 2**(max_level_idx - level_idx).
        leaf_to_bucket: ndarray of shape (N, 2) with columns (level_idx, position_in_bucket).
                        Row k corresponds to the k-th leaf in the canonical `order`.
        meta: dict with 'centers', 'levels', 'tiles', 'xy_idx', 'cell_ids', 'domain'.
              Matches quadtree_to_tensor's contract except there is no flat 'values' tensor.
    """
    if qt.value_storage != "native":
        raise ValueError(
            f"quadtree_to_tensor_native requires value_storage='native'; "
            f"got value_storage={qt.value_storage!r}."
        )

    exp = qt.export_leaves(order=order, cell_scale_mode=cell_scale_mode)
    ordered = qt.ordered_leaves(order)

    L = qt.max_level_idx
    C = qt.channels
    Ph, Pw = qt.patch_height, qt.patch_width

    N = len(ordered)
    leaf_to_bucket = np.empty((N, 2), dtype=np.int64)
    # Group leaf indices by level.
    per_level_indices: dict[int, list] = {lvl: [] for lvl in range(L + 1)}
    for k, c in enumerate(ordered):
        per_level_indices[c.level_idx].append(k)

    values_by_level: dict[int, np.ndarray] = {}
    for lvl in range(L + 1):
        s = 1 << (L - lvl)
        h = Ph * s
        w = Pw * s
        idxs = per_level_indices[lvl]
        bucket = np.zeros((len(idxs), C, h, w), dtype=np.float32)
        for pos, k in enumerate(idxs):
            leaf = ordered[k]
            if leaf.value is not None:
                bucket[pos] = leaf.value
            leaf_to_bucket[k, 0] = lvl
            leaf_to_bucket[k, 1] = pos
        values_by_level[lvl] = bucket

    meta: dict[str, Any] = {
        "centers": exp["centers"],
        "levels": exp["levels"],
        "tiles": exp["tiles"],
        "xy_idx": exp["xy_idx"],
        "cell_ids": exp["ids"],
        "domain": qt.get_domain_meta(),
    }

    return values_by_level, leaf_to_bucket, meta


def tensor_to_quadtree_native(
    values_by_level: dict[int, np.ndarray],
    leaf_to_bucket: np.ndarray,
    meta: dict[str, Any],
    *,
    order: Literal["morton", "uid", "raster"] = "morton",
    cell_scale_mode: str | None = "area",
) -> Quadtree:
    """
    Inverse of quadtree_to_tensor_native: rebuild a native-storage Quadtree
    from per-level buckets plus `leaf_to_bucket`.

    The meta dict must contain 'centers', 'levels', and 'domain'. Leaves are
    located via centers (as in tensor_to_quadtree), then each leaf's native-res
    value is copied from values_by_level[level][position].
    """
    centers = meta["centers"]
    if hasattr(centers, "detach"):
        centers = centers.detach().cpu().numpy()

    levels = meta["levels"]
    if hasattr(levels, "detach"):
        levels = levels.detach().cpu().numpy()

    if hasattr(leaf_to_bucket, "detach"):
        leaf_to_bucket = leaf_to_bucket.detach().cpu().numpy()

    # Infer channels from the first non-empty bucket.
    C = 0
    for lvl in sorted(values_by_level.keys()):
        arr = values_by_level[lvl]
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
            values_by_level[lvl] = arr
        if arr.shape[0] > 0:
            C = int(arr.shape[1])
            break

    domain = meta["domain"]
    qt = Quadtree(
        xmin=domain["xmin"],
        xmax=domain["xmax"],
        ymin=domain["ymin"],
        ymax=domain["ymax"],
        max_level_idx=domain["max_level_idx"],
        channels=C,
        tile_width=domain["tile_width"],
        tile_height=domain["tile_height"],
        value_storage="native",
    )

    if cell_scale_mode is not None:
        Lx = domain["xmax"] - domain["xmin"]
        Ly = domain["ymax"] - domain["ymin"]
        abs_cx = (centers[:, 0] * Lx) + domain["xmin"]
        abs_cy = (centers[:, 1] * Ly) + domain["ymin"]
    else:
        abs_cx = centers[:, 0]
        abs_cy = centers[:, 1]

    # Refine to target levels first (coarse-to-fine).
    refine_order = np.argsort(levels)
    for i in refine_order:
        cx, cy = abs_cx[i], abs_cy[i]
        target_lvl = int(levels[i])
        leaf = qt.locate_leaf(cx, cy)
        if leaf is None:
            raise ValueError(f"Coordinate ({cx}, {cy}) outside Quadtree domain.")
        while leaf.level_idx < target_lvl:
            qt.refine_leaf(leaf)
            leaf = qt.locate_leaf(cx, cy)

    # Assign values from buckets.
    for i in range(len(levels)):
        cx, cy = abs_cx[i], abs_cy[i]
        leaf = qt.locate_leaf(cx, cy)
        if leaf is None or leaf.level_idx != int(levels[i]):
            continue
        lvl = int(leaf_to_bucket[i, 0])
        pos = int(leaf_to_bucket[i, 1])
        leaf.value = values_by_level[lvl][pos].astype(float).copy()

    # qt.mark_boundary_cells()  # disabled: is_boundary flag unread by active pipeline
    return qt


def tensor_to_uniform_native(
    values_by_level: dict[int, np.ndarray],
    leaf_to_bucket: np.ndarray,
    meta: dict[str, Any],
    *,
    return_tensor: bool = False,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Tile per-level native-resolution patches directly onto the finest uniform grid.
    No upsampling: each leaf already stores its native-res region.
    """
    centers = meta["centers"]
    if hasattr(centers, "detach"):
        centers = centers.detach().cpu().numpy()
    levels = meta["levels"]
    if hasattr(levels, "detach"):
        levels = levels.detach().cpu().numpy()
    if hasattr(leaf_to_bucket, "detach"):
        leaf_to_bucket = leaf_to_bucket.detach().cpu().numpy()

    domain = meta["domain"]
    Lx = domain["xmax"] - domain["xmin"]
    Ly = domain["ymax"] - domain["ymin"]
    max_lvl = domain["max_level_idx"]
    factor = 1 << max_lvl

    nx_tiles = int(round(Lx / domain["tile_width"]))
    ny_tiles = int(round(Ly / domain["tile_height"]))
    patch_width = int(round(domain["tile_width"] / factor))
    patch_height = int(round(domain["tile_height"] / factor))

    # Determine channel count from buckets.
    C = 0
    for lvl_key in sorted(values_by_level.keys()):
        arr = values_by_level[lvl_key]
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
            values_by_level[lvl_key] = arr
        if arr.shape[0] > 0:
            C = int(arr.shape[1])
            break

    H = ny_tiles * patch_height * factor
    W = nx_tiles * patch_width * factor
    out = np.zeros((C, H, W), dtype=np.float32)

    abs_cx = (centers[:, 0] * Lx) + domain["xmin"]
    abs_cy = (centers[:, 1] * Ly) + domain["ymin"]
    pixels_per_tile_x = patch_width * factor
    pixels_per_tile_y = patch_height * factor

    order = np.argsort(levels)
    for i in order:
        cx, cy = abs_cx[i], abs_cy[i]
        lvl = int(levels[i])
        hx = (domain["tile_width"] / (1 << lvl)) * 0.5
        hy = (domain["tile_height"] / (1 << lvl)) * 0.5
        x_tl = cx - hx
        y_tl = cy - hy
        px_start = int(round((x_tl - domain["xmin"]) / domain["tile_width"] * pixels_per_tile_x))
        py_start = int(round((y_tl - domain["ymin"]) / domain["tile_height"] * pixels_per_tile_y))
        scale = max_lvl - lvl
        pw_native = patch_width * (1 << scale)
        ph_native = patch_height * (1 << scale)

        bucket_lvl = int(leaf_to_bucket[i, 0])
        bucket_pos = int(leaf_to_bucket[i, 1])
        val = values_by_level[bucket_lvl][bucket_pos]  # (C, ph_native, pw_native)

        out[:, py_start : py_start + ph_native, px_start : px_start + pw_native] = val

    if return_tensor:
        return torch.as_tensor(out, device=device, dtype=dtype if dtype else torch.float32)
    return out


def scatter_native_to_uniform_torch(
    values_by_level: dict[int, "torch.Tensor"],
    leaf_to_bucket: "torch.Tensor",
    centers: "torch.Tensor",
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    tile_h: float,
    tile_w: float,
    max_level_idx: int,
    base_patch_h: int,
    base_patch_w: int,
) -> "torch.Tensor":
    """Differentiable native-to-uniform scatter.

    Mirrors :func:`tensor_to_uniform_native` in pure PyTorch so it can be
    used inside a training loss. Gradients flow from the uniform output back
    into the per-level bucket tensors via advanced-index assignment.

    Args:
        values_by_level: ``{lvl: (N_l, C, T, Ph·s, Pw·s)}`` where
            ``s = 2^(max_level_idx - lvl)``. May contain empty buckets.
        leaf_to_bucket: ``(N, 2)`` long — ``[:, 0]`` is the level,
            ``[:, 1]`` is the leaf's position within its per-level bucket.
        centers: ``(N, >=2)`` float, ``centers[:, 0:2]`` normalized to
            ``[0, 1]`` over the full domain (matches the ``cell_scale_mode``
            branch at :func:`tensor_to_quadtree_native`).
        xmin/xmax/ymin/ymax/tile_h/tile_w/max_level_idx/base_patch_h/base_patch_w:
            Scatter geometry. Same semantics as ``meta["domain"]`` in the
            numpy twin.

    Returns:
        ``(1, C, T, H_finest, W_finest)`` tensor on the same device/dtype
        as the first non-empty bucket. ``H_finest`` and ``W_finest`` are
        derived from the tile geometry; see below.
    """
    Lx = xmax - xmin
    Ly = ymax - ymin
    factor = 1 << max_level_idx
    nx_tiles = int(round(Lx / tile_w))
    ny_tiles = int(round(Ly / tile_h))
    H_finest = ny_tiles * base_patch_h * factor
    W_finest = nx_tiles * base_patch_w * factor

    # Pick up device/dtype/C/T from the first non-empty bucket.
    ref = None
    for lvl in range(max_level_idx + 1):
        b = values_by_level.get(lvl)
        if b is not None and b.shape[0] > 0:
            ref = b
            break
    if ref is None:
        # No leaves at any level — return a zero tensor with (unknown) minimal
        # shape. This shouldn't occur in practice; guard for safety.
        any_b = next(iter(values_by_level.values()))
        return any_b.new_zeros((1, any_b.shape[1], any_b.shape[2], H_finest, W_finest))
    C = int(ref.shape[1])
    T = int(ref.shape[2])
    device = ref.device
    dtype = ref.dtype

    uniform = torch.zeros(1, C, T, H_finest, W_finest, device=device, dtype=dtype)

    pixels_per_tile_x = base_patch_w * factor
    pixels_per_tile_y = base_patch_h * factor

    for lvl in range(max_level_idx + 1):
        bucket = values_by_level.get(lvl)
        if bucket is None or bucket.shape[0] == 0:
            continue

        mask = leaf_to_bucket[:, 0] == lvl
        if not bool(mask.any()):
            continue
        bucket_pos = leaf_to_bucket[mask, 1].long()  # (N_l,)
        # Reorder bucket values into mask-order (matches the order of centers[mask]).
        values_at_mask = bucket[bucket_pos]  # (N_l, C, T, H_l, W_l)

        depth = max_level_idx - lvl
        H_l = base_patch_h << depth
        W_l = base_patch_w << depth

        # Normalized → absolute, then → integer pixel top-left.
        abs_cx = centers[mask, 0] * Lx + xmin  # (N_l,)
        abs_cy = centers[mask, 1] * Ly + ymin
        hx_lvl = (tile_w / (1 << lvl)) * 0.5  # half-width of a level-ℓ cell
        hy_lvl = (tile_h / (1 << lvl)) * 0.5
        px_start = torch.round(
            (abs_cx - hx_lvl - xmin) / tile_w * pixels_per_tile_x
        ).long()  # (N_l,)
        py_start = torch.round((abs_cy - hy_lvl - ymin) / tile_h * pixels_per_tile_y).long()

        y_offsets = torch.arange(H_l, device=device)
        x_offsets = torch.arange(W_l, device=device)
        py_grid = (py_start[:, None, None] + y_offsets[None, :, None]).expand(-1, H_l, W_l)
        px_grid = (px_start[:, None, None] + x_offsets[None, None, :]).expand(-1, H_l, W_l)

        # Source reshape: (N_l, C, T, H_l, W_l) → (C, T, N_l*H_l*W_l).
        src = values_at_mask.permute(1, 2, 0, 3, 4).reshape(C, T, -1)

        # Advanced-index assignment: differentiable wrt `src`.
        uniform[0, :, :, py_grid.reshape(-1), px_grid.reshape(-1)] = src

    return uniform
