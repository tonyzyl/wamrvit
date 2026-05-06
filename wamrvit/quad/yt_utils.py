import numba as nb
import numpy as np
import torch
import unyt

# import amrex.space2d as amr2d
from yt.frontends.amrex.data_structures import BoxlibDataset


def get_amrex_grid_edges_unyt(
    ds: BoxlibDataset,
) -> tuple[tuple[unyt.unyt_array, unyt.unyt_array], ...]:
    """Return per-grid (LeftEdge, RightEdge) as unyt arrays for an AMReX dataset."""
    return tuple((grid.LeftEdge, grid.RightEdge) for grid in ds.index.grids)


def get_amrex_grid_edges(ds: BoxlibDataset) -> tuple[np.ndarray, np.ndarray]:
    """Return per-grid LeftEdge and RightEdge as numpy arrays (float64) with shape (G,3).

    Useful for serialization/object-store and cross-process usage. Values are in the
    dataset's code-length unit (no conversion applied here).
    """
    left_list = []
    right_list = []
    for grid in ds.index.grids:
        left_list.append(np.asarray(grid.LeftEdge, dtype=np.float64))
        right_list.append(np.asarray(grid.RightEdge, dtype=np.float64))
    return np.stack(left_list, axis=0), np.stack(right_list, axis=0)


def get_vit_2D_patch_centers(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    W: int,
    H: int,
    patch: int = 32,
    mode: str = "coord",  # "coord" or "index"
    normalize: bool = False,
) -> torch.Tensor:
    """
    Compute ViT patch centers for a 2D global grid.

    Parameters
    ----------
    x0, x1 : float
        Physical domain bounds in x.
    y0, y1 : float
        Physical domain bounds in y.
    W, H : int
        Global grid resolution.
    patch : int
        Patch size (default 32).
    mode : str
        "coord" -> physical (x, y)
        "index" -> patch (i, j)
    normalize : bool
        If True:
          - coord mode: normalize to [0, 1]
          - index mode: normalize to [0, 1]

    Returns
    -------
    centers : np.ndarray
        Shape (N_patches, 2), dtype float32
    """

    assert W % patch == 0, "W must be divisible by patch"
    assert H % patch == 0, "H must be divisible by patch"
    assert mode in ("coord", "index")

    nW = W // patch
    nH = H // patch

    # --------------------------------------------------
    # INDEX-BASED CENTERS (i, j)
    # --------------------------------------------------
    if mode == "index":
        j = np.arange(nW, dtype=np.float32)  # x-direction (columns)
        i = np.arange(nH, dtype=np.float32)  # y-direction (rows)

        jj, ii = np.meshgrid(j, i, indexing="xy")
        centers = np.stack([jj, ii], axis=-1)  # (nH, nW, 2)

        if normalize:
            centers[..., 0] /= nW - 1
            centers[..., 1] /= nH - 1

    # --------------------------------------------------
    # COORDINATE-BASED CENTERS (x, y)
    # --------------------------------------------------
    else:
        x_edges = np.linspace(x0, x1, W + 1, dtype=np.float32)
        y_edges = np.linspace(y0, y1, H + 1, dtype=np.float32)

        x_centers = 0.5 * (x_edges[0::patch][:nW] + x_edges[patch::patch])
        y_centers = 0.5 * (y_edges[0::patch][:nH] + y_edges[patch::patch])

        xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")
        centers = np.stack([xx, yy], axis=-1)  # (nH, nW, 2)

        if normalize:
            centers[..., 0] = (centers[..., 0] - x0) / (x1 - x0)
            centers[..., 1] = (centers[..., 1] - y0) / (y1 - y0)

    # --------------------------------------------------
    # FLATTEN TO TOKEN LIST
    # --------------------------------------------------
    centers = centers.reshape(nH * nW, 2).astype(np.float32)

    return torch.from_numpy(centers)


def get_array_from_arbitrary_grid(
    ds: BoxlibDataset,
    left_edge: unyt.unyt_array,
    right_edge: unyt.unyt_array,
    active_dims: tuple[int, ...],
    fields_name: list[str] | str | None = None,
    dim: int = 2,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Get a numpy array containing the specified fields from an arbitrary grid.

    Args:
        ds: The AMReX dataset.
        left_edge: Left edge of the arbitrary grid.
        right_edge: Right edge of the arbitrary grid.
        fields_name: List of field names to extract (str or list of str).
        dim: Dimension of the data to extract, use the first `dim` of ActiveDimensions.
    Returns:
        A numpy.ndarray containing field data with shape (C, H, W).
    """
    out = np.empty((len(fields_name), *active_dims[:dim]), dtype=dtype)
    for field_idx, field in enumerate(fields_name):
        ag = ds.arbitrary_grid(left_edge, right_edge, active_dims)
        arr = ag[field].v
        if dim == 2:
            out[field_idx] = np.asarray(arr).squeeze(-1).astype(dtype)
        else:
            out[field_idx] = np.asarray(arr).astype(dtype)  # ActiveDimensions shape

    return out


def get_amrex_array_from_edges(
    ds: BoxlibDataset,
    edges: tuple[tuple[unyt.unyt_array, unyt.unyt_array], ...] | tuple[np.ndarray, np.ndarray],
    fields_name: list[str] | str | None = None,
    dim: int = 2,
) -> np.ndarray:
    """Get a numpy array containing the specified fields from an AMReX dataset.

    Args:
        ds: The AMReX dataset.
        edges: Tuple of (LeftEdge, RightEdge) pairs for each grid.
        fields_name: List of field names to extract (str or list of str).
        dim: Dimension of the data to extract, use the first `dim` of ActiveDimensions.
    Returns:
        A numpy.ndarray containing concatenated field data from all grids with shape (N, C, H, W).
    """
    # Get H, W from ActiveDimensions (assume uniform sub-grid shape per grid)
    tensor_shape = ds.index.grids[0].ActiveDimensions[:dim]
    active_dims = ds.index.grids[0].ActiveDimensions
    if isinstance(fields_name, str):
        fields_name = [fields_name]
    if fields_name is None:
        # field_list returns [("boxlib", "density"), ("boxlib", "momentum_x"), ...]
        fields_name = [f[1] for f in ds.field_list if f[0] == "boxlib"]

    # Normalize edges to unyt tuple-of-tuples if numpy pair was provided
    if isinstance(edges, tuple) and len(edges) == 2 and isinstance(edges[0], np.ndarray):
        # edges = (left_edges_np, right_edges_np) with shape (G,3)
        left_np, right_np = edges
        unit = ds.length_unit
        edges_unyt = []
        for le, re in zip(left_np, right_np):
            le_u = ds.arr(np.asarray(le, dtype=np.float64), unit)
            re_u = ds.arr(np.asarray(re, dtype=np.float64), unit)
            edges_unyt.append((le_u, re_u))
        edges_seq = tuple(edges_unyt)
    else:
        edges_seq = edges  # assume already unyt tuples

    out = np.empty((len(edges_seq), len(fields_name), *tensor_shape), dtype=np.float32)
    # Loop over fields via arbitrary_grid
    for patch_idx in range(len(edges_seq)):
        left_edge, right_edge = edges_seq[patch_idx]
        out[patch_idx] = get_array_from_arbitrary_grid(
            ds,
            left_edge,
            right_edge,
            active_dims=active_dims,
            fields_name=fields_name,
            dim=dim,
            dtype=np.float32,
        )
        # ag = ds.arbitrary_grid(left_edge, right_edge, active_dims)
        # for field_idx, field in enumerate(fields_name):
        # arr = ag[field].v
        # if dim == 2:
        # out[patch_idx, field_idx] = np.asarray(arr).squeeze(-1).astype(np.float32)
        # else:
        # out[patch_idx, field_idx] = np.asarray(arr).astype(np.float32)  # ActiveDimensions shape

    return out


@nb.njit(parallel=False, fastmath=True)
def remap_kernel(
    out,
    src_fields,
    src_le,
    src_dx,
    src_shape,
    tgt_le,
    tgt_dx,
    H,
    W,
):
    # Source patches must be sorted coarse→fine before calling this kernel
    # (extract_amr_patches does this). Last-write-wins then gives the
    # "finest available value per target cell" guarantee.
    Nt, Nf = out.shape[0], out.shape[1]
    Ns = src_fields.shape[0]

    for gi in nb.prange(Nt):
        gt_le_x = tgt_le[gi, 0]
        gt_le_y = tgt_le[gi, 1]
        gt_dx_x = tgt_dx[gi, 0]
        gt_dx_y = tgt_dx[gi, 1]

        for gs in range(Ns):
            gs_le_x = src_le[gs, 0]
            gs_le_y = src_le[gs, 1]
            gs_dx_x = src_dx[gs, 0]
            gs_dx_y = src_dx[gs, 1]

            # --- Resolution ratio (target_dx / source_dx) ---
            ratio_f = gt_dx_x / gs_dx_x

            # --- Overlap box ---
            ox0 = max(gt_le_x, gs_le_x)
            oy0 = max(gt_le_y, gs_le_y)

            ox1 = min(
                gt_le_x + W * gt_dx_x,
                gs_le_x + src_shape[gs, 1] * gs_dx_x,
            )
            oy1 = min(
                gt_le_y + H * gt_dx_y,
                gs_le_y + src_shape[gs, 0] * gs_dx_y,
            )

            if ox0 >= ox1 or oy0 >= oy1:
                continue

            # --- Indices ---
            i0s = int((ox0 - gs_le_x) / gs_dx_x)
            j0s = int((oy0 - gs_le_y) / gs_dx_y)

            i0t = int((ox0 - gt_le_x) / gt_dx_x)
            j0t = int((oy0 - gt_le_y) / gt_dx_y)

            if ratio_f >= 1.0:
                # Source at target resolution (ratio==1, copy) or finer
                # (ratio>1, block-mean restriction).
                ratio = int(ratio_f + 0.5)
                if abs(ratio - ratio_f) > 1e-6:
                    continue

                for fi in range(Nf):
                    src = src_fields[gs, fi]

                    if ratio == 1:
                        for j in range(j0t, H):
                            js = j0s + (j - j0t)
                            if js >= src_shape[gs, 0]:
                                break
                            for i in range(i0t, W):
                                is_ = i0s + (i - i0t)
                                if is_ >= src_shape[gs, 1]:
                                    break
                                out[gi, fi, j, i] = src[js, is_]

                    else:
                        # Restriction
                        for j in range(j0t, H):
                            js = j0s + (j - j0t) * ratio
                            if js + ratio > src_shape[gs, 0]:
                                break

                            for i in range(i0t, W):
                                is_ = i0s + (i - i0t) * ratio
                                if is_ + ratio > src_shape[gs, 1]:
                                    break

                                acc = 0.0
                                for jj in range(ratio):
                                    for ii in range(ratio):
                                        acc += src[js + jj, is_ + ii]

                                out[gi, fi, j, i] = acc / (ratio * ratio)
            else:
                # Source coarser than target → piecewise-constant prolongation.
                # Each source cell stamps an (inv_ratio × inv_ratio) block of
                # target cells with its value; subsequent finer source patches
                # in the coarse→fine traversal overwrite where they overlap.
                inv_ratio_f = 1.0 / ratio_f
                inv_ratio = int(inv_ratio_f + 0.5)
                if abs(inv_ratio - inv_ratio_f) > 1e-6:
                    continue

                src_h_max = src_shape[gs, 0] - j0s
                src_w_max = src_shape[gs, 1] - i0s

                for fi in range(Nf):
                    src = src_fields[gs, fi]
                    for j_src in range(src_h_max):
                        j_t_base = j0t + j_src * inv_ratio
                        if j_t_base >= H:
                            break
                        for jj in range(inv_ratio):
                            j_t = j_t_base + jj
                            if j_t >= H:
                                break
                            for i_src in range(src_w_max):
                                i_t_base = i0t + i_src * inv_ratio
                                if i_t_base >= W:
                                    break
                                v = src[j0s + j_src, i0s + i_src]
                                for ii in range(inv_ratio):
                                    i_t = i_t_base + ii
                                    if i_t >= W:
                                        break
                                    out[gi, fi, j_t, i_t] = v


def remap_amr_patches(
    ds_src, ds_tgt, fields, dtype=np.float32, return_tensor=True, cell_scale_mode="area"
):
    """
    Args:
        ds_src: Source AMReX dataset (yt BoxlibDataset), extract data.
        ds_tgt: Target AMReX grids.
    """

    if isinstance(fields, str):
        fields = [fields]

    # --- Target grids ---
    tgt_grids = ds_tgt.index.grids
    Nt = len(tgt_grids)
    W, H = tgt_grids[0].ActiveDimensions[:2]

    tgt_le = np.zeros((Nt, 2), dtype=np.float64)
    tgt_dx = np.zeros((Nt, 2), dtype=np.float64)

    for i, g in enumerate(tgt_grids):
        tgt_le[i] = g.LeftEdge.d[:2]
        tgt_dx[i] = g.dds[:2].d

    # --- Source grids ---
    src_grids = sorted(ds_src.index.grids, key=lambda g: g.Level)
    Ns = len(src_grids)
    Nf = len(fields)

    src_le = np.zeros((Ns, 2), dtype=np.float64)
    src_dx = np.zeros((Ns, 2), dtype=np.float64)
    src_shape = np.zeros((Ns, 2), dtype=np.int64)
    src_fields = np.zeros((Ns, Nf, H, W), dtype=dtype)

    for gs, g in enumerate(src_grids):
        src_le[gs] = g.LeftEdge.d[:2]
        src_dx[gs] = g.dds[:2].d
        src_shape[gs] = (H, W)

        for fi, f in enumerate(fields):
            arr = g[f].d
            if arr.ndim == 3:
                arr = arr[..., 0]
            src_fields[gs, fi] = arr.astype(dtype, copy=False)

    out = np.zeros((Nt, Nf, H, W), dtype=dtype)

    remap_kernel(out, src_fields, src_le, src_dx, src_shape, tgt_le, tgt_dx, H, W)

    return torch.from_numpy(out) if return_tensor else out


def extract_amr_patches(
    ds,
    fields,
    dtype=np.float32,
    return_tensor=True,
):
    """
    Extracts raw field data from AMR grids in a single pass.
    Returns a dict containing 'data' and raw geometry.
    """
    if isinstance(fields, str):
        fields = [fields]

    # Sort grids (critical for consistency)
    grids = sorted(ds.index.grids, key=lambda g: g.Level)

    N = len(grids)
    C = len(fields)

    # 1. Pre-allocate arrays
    # Assuming uniform tile sizes
    W, H = grids[0].ActiveDimensions[:2]

    out = np.zeros((N, C, H, W), dtype=dtype)
    le = np.zeros((N, 2), dtype=np.float64)  # Left Edge (x, y)
    dx = np.zeros((N, 2), dtype=np.float64)  # Cell Size (dx, dy)
    shapes = np.array([H, W], dtype=np.int64)[None, :].repeat(N, axis=0)

    # 2. Main Loop: Extract Data & Calculate Raw Geometry
    for i, g in enumerate(grids):
        # Optimization: Bulk read all fields
        g.get_data(fields)

        # --- Geometry ---
        lx, ly = g.LeftEdge.d[0], g.LeftEdge.d[1]

        le[i] = [lx, ly]
        dx[i] = g.dds.d[:2]

        # --- Data Copy ---
        for fi, f in enumerate(fields):
            arr = g[f].d
            if arr.ndim == 3:
                arr = arr[..., 0]
            out[i, fi] = arr.astype(dtype, copy=False)

        # g.field_data.clear() # Uncomment if RAM is tight

    # 3. Return
    if return_tensor:
        out = torch.from_numpy(out)

    return {
        "data": out,  # (N, C, H, W)
        "left_edge": le,
        "delta_x": dx,
        "shapes": shapes,
    }


def remap_amrex_to_regular(
    ds_src,
    fields,
    tgt_geometry: tuple[np.ndarray, np.ndarray, int, int],
    dtype=np.float32,
    return_tensor=True,
):
    """
    Optimized remapper.
    Args:
        tgt_geometry: Tuple (tgt_le, tgt_dx, H, W) pre-calculated from the reference.

    returns: Array or Tensor of shape (1, C, H, W) containing the remapped data on the regular grid.
    """
    tgt_le, tgt_dx, H, W = tgt_geometry

    # --- 1. Fast Source Extraction ---
    # We reuse the optimized loader logic to get raw pointers (N, C, h, w)
    # This handles the bulk IO and caching internally.
    src_batch = extract_amr_patches(ds_src, fields, return_tensor=False)

    src_data = src_batch["data"]  # (Ns, C, maxH, maxW)
    src_le = src_batch["left_edge"]  # (Ns, 2)
    src_dx = src_batch["delta_x"]  # (Ns, 2)
    src_shapes = src_batch["shapes"]  # (Ns, 2)

    # --- 2. Allocate Output ---
    C = len(fields)
    out = np.zeros((1, C, H, W), dtype=dtype)

    # --- 3. Run Kernel ---
    remap_kernel(
        out,
        src_data,
        src_le,
        src_dx,
        src_shapes,
        tgt_le,  # (1, 2)
        tgt_dx,  # (1, 2)
        H,
        W,
    )

    return torch.from_numpy(out) if return_tensor else out


def make_regular_target_from_amrex(ds, domain_from="domain", target_level=0):
    """
    Constructs a regular target grid geometry at the simulation refinement
    level given by ``target_level``. Returns (tgt_le, tgt_dx, H, W) compatible
    with remap_kernel. Default (``target_level=0``) reproduces the legacy
    coarsest-grid projection; ``target_level=k`` projects onto a grid that is
    2^k finer per dimension than level 0.
    """
    # 1. Pixel size = dx of the requested simulation level
    target_grids = [g for g in ds.index.grids if g.Level == target_level]
    if not target_grids:
        raise ValueError(
            f"No AMReX grids at level {target_level}; available levels: "
            f"{sorted({g.Level for g in ds.index.grids})}"
        )
    dx_tgt = target_grids[0].dds.d[:2]  # (dx, dy)

    # 2. Determine Bounding Box (always from full simulation domain unless
    # caller specifically asked for the finest level's bbox)
    if domain_from == "finest":
        finest_level = ds.index.max_level_idx
        grids = [g for g in ds.index.grids if g.Level == finest_level]
        if not grids:
            grids = ds.index.grids
    else:
        grids = ds.index.grids

    all_le = np.array([g.LeftEdge.d[:2] for g in grids])
    all_re = np.array([g.RightEdge.d[:2] for g in grids])

    le = np.min(all_le, axis=0)
    re = np.max(all_re, axis=0)

    # 3. Compute Dimensions (H, W) at target_level resolution
    W = int(np.round((re[0] - le[0]) / dx_tgt[0]))
    H = int(np.round((re[1] - le[1]) / dx_tgt[1]))

    # 4. Format for Kernel (Nt=1)
    tgt_le = le[None, :].astype(np.float64)  # Shape (1, 2)
    tgt_dx = dx_tgt[None, :].astype(np.float64)  # Shape (1, 2)

    return tgt_le, tgt_dx, H, W


def make_regular_centers(H, W, p, device):
    """
    Compute regular grid patch centers for ViT, values in [0, 1] relative to the global domain.
    """
    if isinstance(p, int):
        p = (p, p)
    H_p, W_p = H // p[0], W // p[1]

    ys = (torch.arange(H_p, device=device) + 0.5) * p[0] / H
    xs = (torch.arange(W_p, device=device) + 0.5) * p[1] / W

    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    centers = torch.stack([xx, yy], dim=-1)  # (H_p, W_p, 2)

    # Optional: add scale (patch size)
    # h = torch.full((H_p, W_p, 1), p[0] / H, device=device)
    # centers = torch.cat([centers, h], dim=-1)

    return centers.view(-1, 2)  # (N, 2)
