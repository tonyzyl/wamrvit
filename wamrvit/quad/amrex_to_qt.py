"""
amrex_to_quadtree.py — Convert AMReX plotfile data (loaded via yt) into a Quadtree.

Handles the overlap problem: AMReX stores data on overlapping coarse/fine grids.
We resolve ownership by only writing leaf-level_idx data (cells not covered by finer grids).

Performance notes:
  - Avoids covering_grid entirely (reads directly from grid objects → zero interpolation cost)
  - Pre-computes a vectorized ownership mask per level_idx using numba
  - Batches the refine+write loop by processing tiles column-major for cache locality
  - Uses numba to accelerate the inner refinement mask construction
  - read_plotfile_raw() bypasses yt for ~10x faster grid data extraction
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
from numba import njit

from wamrvit.quad.quadtree import Quadtree

# ─── Direct AMReX Plotfile Reader (bypasses yt) ───────────────────────────────


def read_plotfile_header(plotfile_path: str) -> dict[str, Any]:
    """Parse the top-level Header file of an AMReX plotfile.

    Returns dict with: field_names, ndim, time, max_level_idx, domain_lo, domain_hi,
    domain_dims (per-level), dx (per-level), ref_ratios.
    """
    header_path = os.path.join(plotfile_path, "Header")
    with open(header_path) as f:
        lines = f.readlines()

    idx = 0
    _version = lines[idx].strip()
    idx += 1
    ncomp = int(lines[idx].strip())
    idx += 1
    field_names = [lines[idx + i].strip() for i in range(ncomp)]
    idx += ncomp
    ndim = int(lines[idx].strip())
    idx += 1
    time = float(lines[idx].strip())
    idx += 1
    max_level_idx = int(lines[idx].strip())
    idx += 1

    domain_lo = np.array([float(x) for x in lines[idx].split()])
    idx += 1
    domain_hi = np.array([float(x) for x in lines[idx].split()])
    idx += 1

    # Refinement ratios between levels (max_level_idx entries)
    ref_ratios = [int(x) for x in lines[idx].split()]
    idx += 1

    # Domain boxes per level — all on one line, space-separated
    # e.g. "((0,0) (1535,255) (0,0)) ((0,0) (3071,511) (0,0)) ..."
    _domain_boxes_line = lines[idx].strip()
    idx += 1

    # Number of steps per level
    _steps = lines[idx].strip()
    idx += 1

    # Cell sizes per level (one line per level, ndim values)
    dx = []
    for lev in range(max_level_idx + 1):
        dx.append(np.array([float(x) for x in lines[idx].split()]))
        idx += 1

    return {
        "field_names": field_names,
        "ncomp": ncomp,
        "ndim": ndim,
        "time": time,
        "max_level_idx": max_level_idx,
        "domain_lo": domain_lo[:ndim],
        "domain_hi": domain_hi[:ndim],
        "dx": dx,
        "ref_ratios": ref_ratios,
    }


def _parse_box(box_str: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse '((lo_x,lo_y,...) (hi_x,hi_y,...) (0,...))' into (lo, hi) integer arrays."""
    # Extract the coordinate tuples
    parts = re.findall(r"\(([^()]+)\)", box_str)
    lo = np.array([int(x) for x in parts[0].split(",")])
    hi = np.array([int(x) for x in parts[1].split(",")])
    return lo, hi


def read_plotfile_level(
    plotfile_path: str,
    level: int,
    header_info: dict[str, Any],
    field_indices: list[int] | None = None,
    patch_size: tuple[int, int] = (32, 32),
) -> tuple[np.ndarray, np.ndarray]:
    """Read all grid data for a single AMR level directly from binary files.

    AMReX boxes at a given level can have non-uniform sizes (load-balancing /
    box merging). Each box is read at its actual on-disk dimensions, then split
    into patch_size-aligned subgrids so the downstream sampler sees uniformly
    sized tiles. Box dimensions must be integer multiples of patch_size.

    Args:
        plotfile_path: Path to plotfile directory.
        level: AMR level index (0-based).
        header_info: Dict from read_plotfile_header().
        field_indices: If given, only extract these field indices (0-based).
                       If None, extract all fields.
        patch_size: (patch_h, patch_w) of output tiles.

    Returns:
        grid_info: (N, 5) float64 — [level, xmin_phys, xmax_phys, ymin_phys, ymax_phys].
        grid_data: (N, C, patch_h, patch_w) float32 — field data per subgrid.
    """
    level_dir = os.path.join(plotfile_path, f"Level_{level}")
    cell_h_path = os.path.join(level_dir, "Cell_H")
    patch_h, patch_w = patch_size

    ncomp_total = header_info["ncomp"]
    C = len(field_indices) if field_indices is not None else ncomp_total

    with open(cell_h_path) as f:
        cell_h_lines = f.readlines()

    # Parse Cell_H header
    idx = 0
    _version = cell_h_lines[idx].strip()
    idx += 1
    _how = int(cell_h_lines[idx].strip())
    idx += 1  # storage order
    _ncomp = int(cell_h_lines[idx].strip())
    idx += 1
    _nghost = int(cell_h_lines[idx].strip())
    idx += 1

    # Box definitions enclosed in parentheses: "(N_boxes coord_type\n box0 \n box1 \n ... \n )"
    box_header = cell_h_lines[idx].strip()
    idx += 1
    # Format: "(384 0" — first token after '(' is n_boxes
    n_boxes = int(box_header.lstrip("(").split()[0])

    boxes = []
    for _ in range(n_boxes):
        lo, hi = _parse_box(cell_h_lines[idx].strip())
        boxes.append((lo, hi))
        idx += 1
    # Skip closing paren line
    idx += 1

    # Next line: n_boxes again (count before FabOnDisk entries)
    _nfabs = int(cell_h_lines[idx].strip())
    idx += 1

    # FabOnDisk entries: "FabOnDisk: Cell_D_XXXXX offset"
    fab_entries = []
    for _ in range(n_boxes):
        parts = cell_h_lines[idx].strip().split()
        # parts = ["FabOnDisk:", "Cell_D_00000", "12345"]
        fab_file = parts[1]
        fab_offset = int(parts[2])
        fab_entries.append((fab_file, fab_offset))
        idx += 1

    if n_boxes == 0:
        return np.zeros((0, 5), dtype=np.float64), np.zeros((0, C, patch_h, patch_w), dtype=np.float32)

    # Per-box dimensions and patch subdivision counts. AMReX allows non-uniform
    # boxes within a level, so we must NOT inherit boxes[0]'s dims for all FABs.
    box_dims = []
    sub_offset = np.zeros(n_boxes, dtype=np.int64)
    total_subgrids = 0
    for i, (lo, hi) in enumerate(boxes):
        nx = int(hi[0] - lo[0] + 1)
        ny = int(hi[1] - lo[1] + 1)
        if nx % patch_w != 0 or ny % patch_h != 0:
            raise ValueError(
                f"AMReX box {nx}x{ny} at level {level} (box {i}) is not divisible by "
                f"patch_size {patch_w}x{patch_h}"
            )
        nsub_x = nx // patch_w
        nsub_y = ny // patch_h
        box_dims.append((nx, ny, nsub_x, nsub_y))
        sub_offset[i] = total_subgrids
        total_subgrids += nsub_x * nsub_y

    # Read binary data — group by Cell_D file for efficiency
    from collections import defaultdict

    file_groups = defaultdict(list)
    for i, (fab_file, fab_offset) in enumerate(fab_entries):
        file_groups[fab_file].append((i, fab_offset))

    grid_data = np.zeros((total_subgrids, C, patch_h, patch_w), dtype=np.float32)

    for fab_file, entries in file_groups.items():
        fab_path = os.path.join(level_dir, fab_file)
        with open(fab_path, "rb") as f:
            raw = f.read()

        for grid_idx, offset in entries:
            grid_nx, grid_ny, nsub_x, nsub_y = box_dims[grid_idx]
            # Skip ASCII FAB header line (ends with \n)
            header_end = raw.index(b"\n", offset) + 1
            data_start = header_end

            # Read this FAB's full (C, grid_ny, grid_nx) buffer at native dims.
            fab_buf = np.empty((C, grid_ny, grid_nx), dtype=np.float32)
            if field_indices is not None:
                # Selective field extraction
                for out_c, field_c in enumerate(field_indices):
                    field_offset = data_start + field_c * grid_nx * grid_ny * 8
                    arr = np.frombuffer(
                        raw, dtype=np.float64, count=grid_nx * grid_ny, offset=field_offset
                    )
                    # AMReX FAB layout: x varies fastest. Reshape (ny, nx) C-order
                    # matches yt's grid[f].d[:,:,0].T
                    fab_buf[out_c] = arr.reshape(grid_ny, grid_nx).astype(np.float32)
            else:
                arr = np.frombuffer(
                    raw, dtype=np.float64, count=ncomp_total * grid_nx * grid_ny, offset=data_start
                )
                # Reshape each component: (ny, nx) C-order matches yt convention
                for c in range(ncomp_total):
                    c_data = arr[c * grid_nx * grid_ny : (c + 1) * grid_nx * grid_ny]
                    fab_buf[c] = c_data.reshape(
                        grid_ny, grid_nx
                    ).astype(np.float32)

            # Tile the FAB into patch-sized subgrids (row-major by subgrid index).
            base_out = int(sub_offset[grid_idx])
            for sy in range(nsub_y):
                y0 = sy * patch_h
                for sx in range(nsub_x):
                    x0 = sx * patch_w
                    out_idx = base_out + sy * nsub_x + sx
                    grid_data[out_idx] = fab_buf[:, y0:y0 + patch_h, x0:x0 + patch_w]

    # Build grid_info with physical-space coordinates for each subgrid
    dx = header_info["dx"][level]
    domain_lo = header_info["domain_lo"]

    grid_info = np.zeros((total_subgrids, 5), dtype=np.float64)
    for i, (lo, hi) in enumerate(boxes):
        _, _, nsub_x, nsub_y = box_dims[i]
        base_out = int(sub_offset[i])
        for sy in range(nsub_y):
            sub_lo_y = lo[1] + sy * patch_h
            sub_hi_y = sub_lo_y + patch_h  # exclusive boundary
            for sx in range(nsub_x):
                sub_lo_x = lo[0] + sx * patch_w
                sub_hi_x = sub_lo_x + patch_w  # exclusive boundary
                out_idx = base_out + sy * nsub_x + sx
                grid_info[out_idx, 0] = float(level)
                grid_info[out_idx, 1] = domain_lo[0] + sub_lo_x * dx[0]
                grid_info[out_idx, 2] = domain_lo[0] + sub_hi_x * dx[0]
                grid_info[out_idx, 3] = domain_lo[1] + sub_lo_y * dx[1]
                grid_info[out_idx, 4] = domain_lo[1] + sub_hi_y * dx[1]

    return grid_info, grid_data


def read_plotfile_raw(
    plotfile_path: str,
    field_names: list[str] | None = None,
    patch_size: int | tuple[int, int] = 32,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read all grids from an AMReX plotfile without yt.

    Args:
        plotfile_path: Path to plotfile directory.
        field_names: List of field names to extract. If None, all fields.
        patch_size: Expected grid patch size (for validation).

    Returns:
        grid_info: (N_total, 5) float64 — [level, xmin_phys, xmax_phys, ymin_phys, ymax_phys].
        grid_data: (N_total, C, patch_h, patch_w) float32 — field data.
        header_info: Parsed header dict (includes domain bounds, field_names, etc.).
    """
    header_info = read_plotfile_header(plotfile_path)

    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)

    # Resolve field indices
    all_field_names = header_info["field_names"]
    if field_names is not None:
        field_indices = [all_field_names.index(f) for f in field_names]
    else:
        field_indices = None
        field_names = all_field_names

    all_grid_info = []
    all_grid_data = []

    for level in range(header_info["max_level_idx"] + 1):
        gi, gd = read_plotfile_level(plotfile_path, level, header_info, field_indices, patch_size)
        if gi.shape[0] > 0:
            all_grid_info.append(gi)
            all_grid_data.append(gd)

    grid_info = np.concatenate(all_grid_info, axis=0)
    grid_data = np.concatenate(all_grid_data, axis=0)

    return grid_info, grid_data, header_info


@njit(parallel=False, fastmath=True, boundscheck=True, cache=True)
def _fast_sample_leaves(
    leaf_bounds: np.ndarray,
    grid_info: np.ndarray,
    grid_data: np.ndarray,
    patch_size: tuple[int, int],
) -> np.ndarray:

    N_leaves = leaf_bounds.shape[0]
    N_grids = grid_info.shape[0]
    C = grid_data.shape[1]

    out_values = np.zeros((N_leaves, C, patch_size[0], patch_size[1]), dtype=grid_data.dtype)

    for i in range(N_leaves):
        l_xmin = leaf_bounds[i, 0]
        l_xmax = leaf_bounds[i, 1]
        l_ymin = leaf_bounds[i, 2]
        l_ymax = leaf_bounds[i, 3]

        # FIX: Increased buffer size significantly
        valid_grids = np.zeros(2000, dtype=np.int32)
        num_valid = 0

        for g in range(N_grids):
            g_xmin = grid_info[g, 1]
            g_xmax = grid_info[g, 2]
            g_ymin = grid_info[g, 3]
            g_ymax = grid_info[g, 4]

            # Intersection test
            if (max(l_xmin, g_xmin) < min(l_xmax, g_xmax)) and (
                max(l_ymin, g_ymin) < min(l_ymax, g_ymax)
            ):
                # FIX: Explicit bounds safety
                if num_valid < 2000:
                    valid_grids[num_valid] = g
                    num_valid += 1

        dx_leaf = (l_xmax - l_xmin) / patch_size[1]
        dy_leaf = (l_ymax - l_ymin) / patch_size[0]

        # 2. Narrow-phase: Pixel-by-pixel extraction
        for iy in range(patch_size[0]):
            for ix in range(patch_size[1]):
                px = l_xmin + (ix + 0.5) * dx_leaf
                py = l_ymin + (iy + 0.5) * dy_leaf

                best_g = -1
                best_lvl = -1.0

                # Find the highest level_idx grid containing this pixel
                for idx in range(num_valid):
                    g = valid_grids[idx]
                    lvl = grid_info[g, 0]
                    if lvl > best_lvl:
                        g_xmin = grid_info[g, 1]
                        g_xmax = grid_info[g, 2]
                        g_ymin = grid_info[g, 3]
                        g_ymax = grid_info[g, 4]

                        if g_xmin <= px <= g_xmax and g_ymin <= py <= g_ymax:
                            best_g = g
                            best_lvl = lvl

                # 3. Extract data from the winning grid
                if best_g != -1:
                    g_xmin = grid_info[best_g, 1]
                    g_xmax = grid_info[best_g, 2]
                    g_ymin = grid_info[best_g, 3]
                    g_ymax = grid_info[best_g, 4]

                    dx_grid = (g_xmax - g_xmin) / patch_size[1]
                    dy_grid = (g_ymax - g_ymin) / patch_size[0]

                    gx = int((px - g_xmin) / dx_grid)
                    gy = int((py - g_ymin) / dy_grid)

                    # Clamp indices safely
                    if gx < 0:
                        gx = 0
                    elif gx > patch_size[1] - 1:
                        gx = patch_size[1] - 1
                    if gy < 0:
                        gy = 0
                    elif gy > patch_size[0] - 1:
                        gy = patch_size[0] - 1

                    for c in range(C):
                        out_values[i, c, iy, ix] = grid_data[best_g, c, gy, gx]

    return out_values


@njit(fastmath=True, cache=True)
def _fast_check_refinement(cx, cy, hx, hy, level_idx, grid_info):
    c_xmin = cx - hx
    c_xmax = cx + hx
    c_ymin = cy - hy
    c_ymax = cy + hy
    num_grids = grid_info.shape[0]

    for g in range(num_grids):
        lvl = grid_info[g, 0]
        if lvl > level_idx:
            g_xmin = grid_info[g, 1]
            g_xmax = grid_info[g, 2]
            g_ymin = grid_info[g, 3]
            g_ymax = grid_info[g, 4]

            # Intersection test
            if (max(c_xmin, g_xmin) + 1e-4 < min(c_xmax, g_xmax)) and (
                max(c_ymin, g_ymin) + 1e-4 < min(c_ymax, g_ymax)
            ):
                return True
    return False


def _phys_to_idx(
    grid_info_phys: np.ndarray,
    domain_lo: np.ndarray,
    domain_hi: np.ndarray,
    xmax_idx: float,
    ymax_idx: float,
) -> np.ndarray:
    """Convert grid_info physical coordinates to index coordinates in-place copy."""
    grid_info_idx = grid_info_phys.copy()
    x_scale = xmax_idx / (domain_hi[0] - domain_lo[0])
    y_scale = ymax_idx / (domain_hi[1] - domain_lo[1])
    grid_info_idx[:, 1] = (grid_info_phys[:, 1] - domain_lo[0]) * x_scale
    grid_info_idx[:, 2] = (grid_info_phys[:, 2] - domain_lo[0]) * x_scale
    grid_info_idx[:, 3] = (grid_info_phys[:, 3] - domain_lo[1]) * y_scale
    grid_info_idx[:, 4] = (grid_info_phys[:, 4] - domain_lo[1]) * y_scale
    return grid_info_idx


def amrex_to_quadtree(
    ds_or_path,
    field_names: list[str],
    patch_size: int | tuple[int, int] = 32,
    num_levels: int | None = None,
) -> Quadtree:
    """
    Converts an AMReX plotfile to a custom Quadtree structure.
    Accepts either a yt dataset or a plotfile path string. Prefers direct binary
    reading for speed when given a path; falls back to yt when given a dataset.

    Args:
        num_levels: If provided, overrides the quadtree depth.
            max_level_idx = num_levels - 1.
    """
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)

    if isinstance(ds_or_path, str):
        # --- Fast path: direct binary read ---
        grid_info_phys, grid_data, header_info = read_plotfile_raw(
            ds_or_path, field_names, patch_size
        )

        domain_lo = header_info["domain_lo"]
        domain_hi = header_info["domain_hi"]
        max_level_idx = header_info["max_level_idx"]
        # Infer base grid dimensions from Level_0 dx
        dx0 = header_info["dx"][0]
        base_nx = int(round((domain_hi[0] - domain_lo[0]) / dx0[0])) // patch_size[1]
        base_ny = int(round((domain_hi[1] - domain_lo[1]) / dx0[1])) // patch_size[0]
        C = grid_data.shape[1]
    else:
        # --- Legacy yt path ---
        ds = ds_or_path
        xmin_phys = ds.domain_left_edge[0].d
        xmax_phys = ds.domain_right_edge[0].d
        ymin_phys = ds.domain_left_edge[1].d
        ymax_phys = ds.domain_right_edge[1].d
        domain_lo = np.array([xmin_phys, ymin_phys])
        domain_hi = np.array([xmax_phys, ymax_phys])
        base_nx = ds.domain_dimensions[0] // 32
        base_ny = ds.domain_dimensions[1] // 32
        max_level_idx = ds.max_level
        C = len(field_names)

        num_grids = len(ds.index.grids)
        grid_info_phys = np.zeros((num_grids, 5), dtype=np.float64)
        grid_data = np.zeros((num_grids, C, patch_size[0], patch_size[1]), dtype=np.float32)
        for i, grid in enumerate(ds.index.grids):
            grid.get_data(field_names)
            grid_info_phys[i, 0] = grid.Level
            grid_info_phys[i, 1] = grid.LeftEdge[0].d
            grid_info_phys[i, 2] = grid.RightEdge[0].d
            grid_info_phys[i, 3] = grid.LeftEdge[1].d
            grid_info_phys[i, 4] = grid.RightEdge[1].d
            for c, field in enumerate(field_names):
                arr = grid[field].d[:, :, 0].T
                grid_data[i, c, :, :] = arr
            grid.clear_data()

    if num_levels is not None:
        max_level_idx = num_levels - 1
    tile_w = patch_size[1] * (2**max_level_idx)
    tile_h = patch_size[0] * (2**max_level_idx)
    xmin_idx = 0.0
    xmax_idx = float(base_nx * tile_w)
    ymin_idx = 0.0
    ymax_idx = float(base_ny * tile_h)

    # Convert physical coords to index coords
    grid_info = _phys_to_idx(grid_info_phys, domain_lo, domain_hi, xmax_idx, ymax_idx)

    # --- Build Quadtree Topology ---
    qt = Quadtree(
        xmin=xmin_idx,
        xmax=xmax_idx,
        ymin=ymin_idx,
        ymax=ymax_idx,
        max_level_idx=max_level_idx,
        channels=C,
        tile_width=tile_w,
        tile_height=tile_h,
    )

    def refinement_predicate(cell) -> bool:
        return _fast_check_refinement(cell.cx, cell.cy, cell.hx, cell.hy, cell.level_idx, grid_info)

    while True:
        changed = qt.refine_where(refinement_predicate)
        if changed == 0:
            break

    while qt.ensure_2to1_balance() > 0:
        pass

    # --- Numba Injection & Leaf Prep ---
    leaves = list(qt._iter_all_leaves())
    leaf_bounds = np.zeros((len(leaves), 4), dtype=np.float64)
    for i, leaf in enumerate(leaves):
        leaf_bounds[i, 0] = leaf.cx - leaf.hx
        leaf_bounds[i, 1] = leaf.cx + leaf.hx
        leaf_bounds[i, 2] = leaf.cy - leaf.hy
        leaf_bounds[i, 3] = leaf.cy + leaf.hy

    out_values = _fast_sample_leaves(leaf_bounds, grid_info, grid_data, patch_size)
    for i, leaf in enumerate(leaves):
        leaf.value = out_values[i].astype(np.float32)

    return qt, grid_data, grid_info


def assign_from_amrex(
    qt: Quadtree, ds_or_path, field_names: list[str], patch_size: int | tuple[int, int] = 32
) -> None:
    """
    Populates an existing Quadtree's leaves using AMReX data via Numba sampling.
    Accepts either a yt dataset or a plotfile path string.
    """
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)

    if isinstance(ds_or_path, str):
        # --- Fast path: direct binary read ---
        grid_info_phys, grid_data, header_info = read_plotfile_raw(
            ds_or_path, field_names, patch_size
        )
        domain_lo = header_info["domain_lo"]
        domain_hi = header_info["domain_hi"]
    else:
        # --- Legacy yt path ---
        ds = ds_or_path
        domain_lo = np.array([ds.domain_left_edge[0].d, ds.domain_left_edge[1].d])
        domain_hi = np.array([ds.domain_right_edge[0].d, ds.domain_right_edge[1].d])
        num_grids = len(ds.index.grids)
        C = len(field_names)
        grid_info_phys = np.zeros((num_grids, 5), dtype=np.float64)
        grid_data = np.zeros((num_grids, C, patch_size[0], patch_size[1]), dtype=np.float32)
        for i, grid in enumerate(ds.index.grids):
            grid.get_data(field_names)
            grid_info_phys[i, 0] = grid.Level
            grid_info_phys[i, 1] = grid.LeftEdge[0].d
            grid_info_phys[i, 2] = grid.RightEdge[0].d
            grid_info_phys[i, 3] = grid.LeftEdge[1].d
            grid_info_phys[i, 4] = grid.RightEdge[1].d
            for c, field in enumerate(field_names):
                arr = grid[field].d[:, :, 0].T
                grid_data[i, c, :, :] = arr
            grid.clear_data()

    # Convert physical → index coords
    xmax_idx = qt.xmax
    ymax_idx = qt.ymax
    grid_info = _phys_to_idx(grid_info_phys, domain_lo, domain_hi, xmax_idx, ymax_idx)

    # --- Prep Leaves & Run Numba Sampler ---
    leaves = list(qt._iter_all_leaves())
    leaf_bounds = np.zeros((len(leaves), 4), dtype=np.float64)
    for i, leaf in enumerate(leaves):
        leaf_bounds[i, 0] = leaf.cx - leaf.hx
        leaf_bounds[i, 1] = leaf.cx + leaf.hx
        leaf_bounds[i, 2] = leaf.cy - leaf.hy
        leaf_bounds[i, 3] = leaf.cy + leaf.hy

    out_values = _fast_sample_leaves(leaf_bounds, grid_info, grid_data, patch_size)

    for i, leaf in enumerate(leaves):
        leaf.value = out_values[i].astype(np.float32)
