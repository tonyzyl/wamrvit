import os
import datetime
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import warnings
from collections import OrderedDict
from typing import Any, Optional, List, Callable, Dict, Union, Tuple

import h5py

from wamrvit.quad.quadtree import Quadtree, TreeDiff, compute_tree_diff, apply_tree_diff
from wamrvit.quad.quad_utils import quadtree_to_tensor, quadtree_to_tensor_native
from wamrvit.quad.adapt_wavelet import adapt_on_field, expand_fine_region, regrid
from wamrvit.quad.amrex_to_qt import amrex_to_quadtree, assign_from_amrex
from wamrvit.dataloader.utils import get_amrex_field_names, parse_hdf5_virtual_path


def _import_yt_utils():
    """Lazy import of yt_utils (and yt) — only needed by YTAmReXRegularLoader."""
    import yt  # noqa: F401
    from wamrvit.quad.yt_utils import (
        extract_amr_patches, remap_kernel, remap_amrex_to_regular,
        make_regular_target_from_amrex, make_regular_centers,
    )
    return yt, extract_amr_patches, remap_kernel, remap_amrex_to_regular, make_regular_target_from_amrex, make_regular_centers

class YTAmReXRegularLoader:
    def __init__(
        self,
        field_names: Optional[List[str]] = None,
        domain_from: str = "domain", # "domain" or "finest"
        target_level: int = 0,  # 0 = sim coarsest (legacy); k = 2^k finer per dim
    ):
        self.field_names = field_names
        self.domain_from = domain_from
        self.target_level = target_level

    def __call__(
        self,
        input_paths: List[str],
        target_paths: List[str]
    ) -> Dict[str, np.ndarray]:

        yt, _, _, remap_amrex_to_regular, make_regular_target_from_amrex, _ = _import_yt_utils()

        # 1. Load Reference Grid (Last Input)
        # We define the Target Geometry (H, W, Viewport) based on this snapshot.
        # All other snapshots will be remapped to match this viewport.
        ds_ref = yt.load(input_paths[-1])
        
        if self.field_names is None:
            field_names = get_amrex_field_names(ds_ref)
        else:
            field_names = self.field_names

        # 2. Compute Target Geometry ONCE
        # This returns (le, dx, H, W) for the single uniform grid we want as output
        ref_geometry = make_regular_target_from_amrex(
            ds_ref,
            domain_from=self.domain_from,
            target_level=self.target_level,
        )

        # Prepare lists
        input_arr = []
        output_arr = []

        # --- PROCESS INPUTS ---
        for p in input_paths:
            ds = yt.load(p)
            
            out_tensor = remap_amrex_to_regular(
                ds, 
                field_names, 
                tgt_geometry=ref_geometry, 
                return_tensor=False
            )
            # Result is (1, C, H, W), remove the singleton batch dim for stacking
            input_arr.append(out_tensor[0]) 

        for p in target_paths:
            ds = yt.load(p)
            
            out_tensor = remap_amrex_to_regular(
                ds, 
                field_names, 
                tgt_geometry=ref_geometry, 
                return_tensor=False
            )
            output_arr.append(out_tensor[0])

        # 3. Stack and Return
        # Shape: (T, C, H, W) 
        return {
            "input": np.stack(input_arr),   # (T_in, C, H, W)
            "target": np.stack(output_arr), # (T_out, C, H, W)
        }

        
def _extract_inputs_targets(
    *,
    qt_ref,
    input_paths: List[str],
    target_paths: List[str],
    field_names: List[str],
    mode: str,
    cell_scale_mode: str,
    return_src: bool,
    value_storage: str,
    amrex_patch_size=None,
    hdf5_loader=None,
) -> Dict[str, Any]:
    """Shared input/target extraction for AdaptiveLoader and FillAndExportMapper.

    In 'uniform' mode returns 'input'/'target' as (T, N, C, Ph, Pw) stacks.
    In 'native' mode returns 'input_by_level'/'target_by_level' as
    {lvl: (T, N_l, C, Ph*s, Pw*s)} plus 'leaf_to_bucket' (N, 2).
    """
    def _assign(p: str):
        if mode == "npz":
            arr_all = np.load(p)
            arr_all = np.stack([arr_all[k] for k in field_names], axis=0)
            qt_ref.assign_from_array(arr_all)
            return arr_all
        elif mode == "hdf5":
            sf, vf, fo, cache = hdf5_loader
            arr_all = _load_hdf5_frame(p, sf, vf, fo, cache)
            qt_ref.assign_from_array(arr_all)
            return arr_all
        elif mode == "amrex":
            assign_from_amrex(qt_ref, p, field_names, patch_size=amrex_patch_size)
            return None

    # Uniform path: original flat (T, N, C, Ph, Pw) outputs.
    if value_storage == "uniform":
        input_arr = []
        input_src_arr = [] if return_src else None
        ref_meta = None
        for p in input_paths:
            src = _assign(p)
            arr, meta = quadtree_to_tensor(
                qt_ref, return_tensor=False, cell_scale_mode=cell_scale_mode,
            )
            input_arr.append(arr)
            if return_src:
                input_src_arr.append(src if mode in ("npz", "hdf5") else p)
            if ref_meta is None:
                ref_meta = meta

        output_arr = []
        output_src_arr = [] if return_src else None
        for p in target_paths:
            tgt = _assign(p)
            arr, _ = quadtree_to_tensor(
                qt_ref, return_tensor=False, cell_scale_mode=cell_scale_mode,
            )
            output_arr.append(arr)
            if return_src:
                output_src_arr.append(tgt if mode in ("npz", "hdf5") else p)

        result: Dict[str, Any] = {
            "input": np.stack(input_arr),
            "target": np.stack(output_arr),
            "meta": ref_meta,
            "input_src": np.stack(input_src_arr) if (return_src and mode in ("npz", "hdf5")) else None,
            "target_src": np.stack(output_src_arr) if (return_src and mode in ("npz", "hdf5")) else None,
        }
        if return_src and mode == "amrex":
            result["input_src_paths"] = input_src_arr
            result["target_src_paths"] = output_src_arr
        return result

    # Native path: per-level buckets stacked over time.
    assert mode != "amrex", "native value_storage not supported for AMReX."
    L = qt_ref.max_level_idx
    input_buckets_per_t: List[Dict[int, np.ndarray]] = []
    target_buckets_per_t: List[Dict[int, np.ndarray]] = []
    input_src_arr = [] if return_src else None
    ref_meta = None
    ref_l2b = None
    for p in input_paths:
        src = _assign(p)
        buckets, l2b, meta = quadtree_to_tensor_native(
            qt_ref, cell_scale_mode=cell_scale_mode,
        )
        input_buckets_per_t.append(buckets)
        if ref_meta is None:
            ref_meta = meta
            ref_l2b = l2b
        if return_src:
            input_src_arr.append(src)

    output_src_arr = [] if return_src else None
    for p in target_paths:
        tgt = _assign(p)
        buckets, _l2b, _meta = quadtree_to_tensor_native(
            qt_ref, cell_scale_mode=cell_scale_mode,
        )
        target_buckets_per_t.append(buckets)
        if return_src:
            output_src_arr.append(tgt)

    input_by_level = {
        lvl: np.stack([bt[lvl] for bt in input_buckets_per_t], axis=0) for lvl in range(L + 1)
    }
    target_by_level = {
        lvl: np.stack([bt[lvl] for bt in target_buckets_per_t], axis=0) for lvl in range(L + 1)
    }

    result = {
        "input_by_level": input_by_level,
        "target_by_level": target_by_level,
        "leaf_to_bucket": ref_l2b,
        "meta": ref_meta,
        "input_src": np.stack(input_src_arr) if (return_src and mode in ("npz", "hdf5")) else None,
        "target_src": np.stack(output_src_arr) if (return_src and mode in ("npz", "hdf5")) else None,
    }
    return result


class AdaptiveLoader:
    """
    Args:
        mode (str): Data loading mode. One of 'npz', 'amrex', or 'hdf5'.
        tile_width (int): Width of each root tile.
        tile_height (int): Height of each root tile.
        adapt_nearby (int): Number of nearby cells to also refine around each refined cell.
        amrex_patch_size (Union[int, Tuple[int, int]]): Patch size of each cell, can be (H, W) or int (square).
        scalar_fields (dict): For mode='hdf5'. Maps channel name to [group, dataset].
        vector_fields (dict): For mode='hdf5'. Maps channel name to [group, dataset, component_idx].
        field_order (list): For mode='hdf5'. Channel stacking order.
    """
    def __init__(
        self,
        mode: str = "npz",  # Added mode kwarg
        field_names: Optional[List[str]] = None,
        cell_scale_mode: str = "area",
        num_levels: int = 3,
        tile_width: int = 40,
        tile_height: int = 40,
        adapt_nearby: int = 1,
        tol_frac: float = 0.01,
        adapt_on_channels: Optional[List[int]] = None,
        amrex_patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        return_src: bool = False,
        return_regular: bool = None,  # deprecated alias for return_src
        resample_coarsen_ratio: float = 0.0,
        resample_refine_ratio: float = 0.0,
        # HDF5-specific params
        scalar_fields: Optional[Dict] = None,
        vector_fields: Optional[Dict] = None,
        field_order: Optional[List[str]] = None,
        hdf5_cache_size: int = 4,
        value_storage: str = "uniform",
    ):
        assert mode in ["npz", "amrex", "hdf5"], "mode must be 'npz', 'amrex', or 'hdf5'"
        if value_storage not in ("uniform", "native"):
            raise ValueError(f"value_storage must be 'uniform' or 'native'; got {value_storage!r}.")
        if value_storage == "native" and mode == "amrex":
            raise ValueError("value_storage='native' is not supported with mode='amrex'.")
        self.value_storage = value_storage
        if return_regular is not None:
            warnings.warn("'return_regular' is deprecated. Use 'return_src' instead.", DeprecationWarning)
            return_src = return_regular
        self.mode = mode
        self.field_names = field_names
        self.cell_scale_mode = cell_scale_mode
        self.num_levels = num_levels
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.adapt_nearby = adapt_nearby
        self.tol_frac = tol_frac
        self.adapt_on_channels = adapt_on_channels
        self.amrex_patch_size = amrex_patch_size
        self.return_src = return_src
        self.resample_coarsen_ratio = resample_coarsen_ratio
        self.resample_refine_ratio = resample_refine_ratio

        if amrex_patch_size is not None and mode != "amrex":
            raise ValueError("amrex_patch_size is only applicable when mode='amrex'")

        # HDF5 mode setup
        if mode == "hdf5":
            if scalar_fields is None and vector_fields is None:
                raise ValueError("mode='hdf5' requires at least one of scalar_fields or vector_fields")
            if field_order is None:
                raise ValueError("mode='hdf5' requires field_order")
            self._hdf5_scalar_fields = {k: tuple(v) for k, v in (scalar_fields or {}).items()}
            self._hdf5_vector_fields = {k: (v[0], v[1], int(v[2])) for k, v in (vector_fields or {}).items()}
            self._hdf5_field_order = field_order
            self._hdf5_cache = _HDF5Cache(max_size=hdf5_cache_size)

            
    def __call__(
        self, 
        input_paths: List[str], 
        target_paths: List[str]
    ) -> Dict[str, np.ndarray]:

        ref_path = input_paths[-1]

        # ==========================================
        # 1. Build Quadtree Topology from Reference
        # ==========================================
        if self.mode == "npz":
            ref = np.load(ref_path)
            field_names = self.field_names if self.field_names else list(ref.keys())
            ref = np.stack([ref[k] for k in field_names], axis=0) # (C, H, W)

            qt_extent = (0, ref.shape[2], 0, ref.shape[1]) # (x_min, x_max, y_min, y_max)
            qt_ref = Quadtree(*qt_extent, max_level_idx=self.num_levels-1, channels=len(field_names),
                              tile_width=self.tile_width, tile_height=self.tile_height,
                              value_storage=self.value_storage)

            adapt_on_field(qt_ref, ref, tol_frac=self.tol_frac, channel=self.adapt_on_channels,
                           max_passes=self.num_levels-1)

        elif self.mode == "hdf5":
            ref = _load_hdf5_frame(ref_path, self._hdf5_scalar_fields, self._hdf5_vector_fields,
                                   self._hdf5_field_order, self._hdf5_cache)  # (C, H, W)
            field_names = self._hdf5_field_order

            qt_extent = (0, ref.shape[2], 0, ref.shape[1])
            qt_ref = Quadtree(*qt_extent, max_level_idx=self.num_levels-1, channels=len(field_names),
                              tile_width=self.tile_width, tile_height=self.tile_height,
                              value_storage=self.value_storage)

            adapt_on_field(qt_ref, ref, tol_frac=self.tol_frac, channel=self.adapt_on_channels,
                           max_passes=self.num_levels-1)

        elif self.mode == "amrex":
            # For AMReX, we build topology directly from the AMR hierarchy
            #ref_ds = yt.load(ref_path)
            #field_names = self.field_names if self.field_names else [f[1] for f in ref_ds.field_list]
            #qt_ref, _, _ = amrex_to_quadtree(ref_ds, field_names, patch_size=self.amrex_patch_size)

            if self.field_names:
                field_names = self.field_names
            else:
                from wamrvit.quad.amrex_to_qt import read_plotfile_header
                header_info = read_plotfile_header(ref_path)
                field_names = header_info["field_names"]
            # Pass path string to use fast direct binary reader
            qt_ref, _, _ = amrex_to_quadtree(ref_path, field_names, patch_size=self.amrex_patch_size,
                                             num_levels=self.num_levels)

            if self.adapt_on_channels is not None:
                # regrid expects tol_frac
                regrid(qt_ref, channel=self.adapt_on_channels, tol_frac=self.tol_frac, max_passes=10, disable_warnings=True)

        # Expand fine regions universally
        if self.adapt_nearby > 0:
            qt_ref = expand_fine_region(qt_ref, self.adapt_nearby)

        # Optional: randomly perturb quadtree topology for augmentation
        if self.resample_coarsen_ratio > 0.0 or self.resample_refine_ratio > 0.0:
            from wamrvit.quad.quad_utils import random_resample_quadtree
            random_resample_quadtree(qt_ref, self.resample_coarsen_ratio, self.resample_refine_ratio)

        # qt_ref.mark_boundary_cells()  # disabled: is_boundary flag unread by active pipeline

        # ==========================================
        # 2/3. Extract Data via mode-appropriate export
        # ==========================================
        result = _extract_inputs_targets(
            qt_ref=qt_ref,
            input_paths=input_paths,
            target_paths=target_paths,
            field_names=field_names,
            mode=self.mode,
            cell_scale_mode=self.cell_scale_mode,
            return_src=self.return_src,
            value_storage=self.value_storage,
            amrex_patch_size=self.amrex_patch_size,
            hdf5_loader=(
                (self._hdf5_scalar_fields, self._hdf5_vector_fields,
                 self._hdf5_field_order, self._hdf5_cache)
                if self.mode == "hdf5" else None
            ),
        )
        return result


class NpzRegularLoader:
    """
    Load NPZ snapshots that are already on a regular grid.

    Optional coarse-grid loading is provided via `coarsen_factor`:
      - 1 keeps the original (finest) grid unchanged.
      - >1 downsamples each channel with either block-average or nearest.
    """
    def __init__(
        self,
        field_names: Optional[List[str]] = None,
        coarsen_factor: int = 1,
        coarsen_mode: str = "mean",  # "mean" or "nearest"
    ):
        # Keep only the fields requested by config. If None, all NPZ keys are used.
        self.field_names = field_names
        # A single integer factor keeps config simple and explicit.
        self.coarsen_factor = int(coarsen_factor)
        self.coarsen_mode = coarsen_mode

        if self.coarsen_factor < 1:
            raise ValueError(f"coarsen_factor must be >= 1, got {self.coarsen_factor}")
        if self.coarsen_mode not in {"mean", "nearest"}:
            raise ValueError(f"coarsen_mode must be 'mean' or 'nearest', got {self.coarsen_mode}")

    def _coarsen(self, arr: np.ndarray) -> np.ndarray:
        """
        Coarsen a single snapshot array with shape (C, H, W).
        """
        # Factor 1 is the no-op path (finest grid).
        if self.coarsen_factor == 1:
            return arr

        c, h, w = arr.shape
        f = self.coarsen_factor
        out_h = h // f
        out_w = w // f
        if out_h == 0 or out_w == 0:
            raise ValueError(f"coarsen_factor={f} is too large for shape {(h, w)}")

        # Use only full blocks so behavior is deterministic and fast.
        h_trim = out_h * f
        w_trim = out_w * f
        arr_trim = arr[:, :h_trim, :w_trim]

        if self.coarsen_mode == "mean":
            # Block-average downsampling: (C, H, W) -> (C, H/f, W/f).
            arr_view = arr_trim.reshape(c, out_h, f, out_w, f)
            return arr_view.mean(axis=(2, 4))

        # Nearest-neighbor downsampling via stride sampling.
        return arr_trim[:, ::f, ::f]

    def _load_single(self, path: str) -> np.ndarray:
        """
        Load one NPZ file and return a (C, H, W) array on the chosen grid level.
        """
        with np.load(path) as data:
            if self.field_names is None:
                # Preserve NPZ key order when no explicit field list is provided.
                field_names = list(data.keys())
            else:
                field_names = self.field_names

            # Stack selected channels into the expected model layout.
            arr = np.stack([data[k] for k in field_names], axis=0)

        return self._coarsen(arr)

    def __call__(
        self,
        input_paths: List[str],
        target_paths: List[str]
    ) -> Dict[str, np.ndarray]:
        # Keep the regular loader contract consistent with YTAmReXRegularLoader.
        input_arr = [self._load_single(p) for p in input_paths]
        target_arr = [self._load_single(p) for p in target_paths]

        return {
            # Shape is (T, C, H, W); Seq2SeqMapper will transpose to (B, C, T, H, W).
            "input": np.stack(input_arr),
            "target": np.stack(target_arr),
        }


# ---------------------------------------------------------------------------
# HDF5 loading utilities (for datasets like The Well / turbulent_radiative_layer)
# ---------------------------------------------------------------------------

class _HDF5Cache:
    """Bounded LRU cache for HDF5 file data loaded into memory."""

    def __init__(self, max_size: int = 4):
        self.max_size = max_size
        self._cache: OrderedDict[str, Dict[str, np.ndarray]] = OrderedDict()

    def get(
        self,
        filepath: str,
        scalar_fields: Dict[str, Tuple[str, str]],
        vector_fields: Dict[str, Tuple[str, str, int]],
    ) -> Dict[str, np.ndarray]:
        """Return cached arrays for *filepath*, loading on first access."""
        if filepath in self._cache:
            self._cache.move_to_end(filepath)
            return self._cache[filepath]

        data: Dict[str, np.ndarray] = {}
        raw_vector_keys = set()
        with h5py.File(filepath, "r") as f:
            for name, (group, dataset) in scalar_fields.items():
                data[name] = f[group][dataset][:]  # (n_traj, n_steps, H, W)
            for name, (group, dataset, _comp) in vector_fields.items():
                key = f"{group}/{dataset}"
                if key not in data:
                    data[key] = f[group][dataset][:]  # (n_traj, n_steps, n_comp, H, W)
                raw_vector_keys.add(key)

        # Slice per-channel components from vector fields, then discard the raw arrays
        for name, (group, dataset, comp) in vector_fields.items():
            key = f"{group}/{dataset}"
            data[name] = data[key][..., comp]  # (n_traj, n_steps, H, W)
        for key in raw_vector_keys:
            del data[key]

        self._cache[filepath] = data
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        return data


def _load_hdf5_frame(
    virtual_path: str,
    scalar_fields: Dict[str, Tuple[str, str]],
    vector_fields: Dict[str, Tuple[str, str, int]],
    field_order: List[str],
    cache: _HDF5Cache,
) -> np.ndarray:
    """Load a single (C, H, W) frame from an HDF5 virtual path."""
    filepath, traj_idx, frame_idx = parse_hdf5_virtual_path(virtual_path)
    data = cache.get(filepath, scalar_fields, vector_fields)

    channels = []
    for name in field_order:
        channels.append(data[name][traj_idx, frame_idx])  # (H, W)
    return np.stack(channels, axis=0)  # (C, H, W)


class HDF5RegularLoader:
    """
    Regular-grid loader for HDF5 datasets that store multiple trajectories
    and timesteps per file (e.g. The Well datasets).

    Virtual paths in the form 'filepath:::traj_idx:::frame_idx' are parsed
    to extract the correct slice.
    """

    def __init__(
        self,
        scalar_fields: Dict[str, List],
        vector_fields: Dict[str, List],
        field_order: List[str],
        cache_size: int = 4,
    ):
        # Convert list values from YAML to tuples for cleaner access
        self.scalar_fields = {k: tuple(v) for k, v in scalar_fields.items()}
        self.vector_fields = {k: (v[0], v[1], int(v[2])) for k, v in vector_fields.items()}
        self.field_order = field_order
        self._cache = _HDF5Cache(max_size=cache_size)

    def __call__(
        self,
        input_paths: List[str],
        target_paths: List[str],
    ) -> Dict[str, np.ndarray]:
        input_arr = [
            _load_hdf5_frame(p, self.scalar_fields, self.vector_fields, self.field_order, self._cache)
            for p in input_paths
        ]
        target_arr = [
            _load_hdf5_frame(p, self.scalar_fields, self.vector_fields, self.field_order, self._cache)
            for p in target_paths
        ]
        return {
            "input": np.stack(input_arr),   # (T_in, C, H, W)
            "target": np.stack(target_arr), # (T_out, C, H, W)
        }


# ---------------------------------------------------------------------------
# Two-stage pipeline components for --cache_topology
# ---------------------------------------------------------------------------

class TopologyBuilder:
    """
    Stage 1 Ray Data mapper: builds deterministic quadtree topology from the
    reference frame and serializes it as lightweight TreeDiff columns.

    The output preserves all original columns and adds topo_* columns.
    """

    def __init__(
        self,
        mode: str = "npz",
        field_names: Optional[List[str]] = None,
        num_levels: int = 3,
        tile_width: int = 40,
        tile_height: int = 40,
        adapt_nearby: int = 1,
        tol_frac: float = 0.01,
        adapt_on_channels: Optional[List[int]] = None,
        amrex_patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        # HDF5 params
        scalar_fields: Optional[Dict] = None,
        vector_fields: Optional[Dict] = None,
        field_order: Optional[List[str]] = None,
        hdf5_cache_size: int = 4,
    ):
        assert mode in ["npz", "amrex", "hdf5"], f"mode must be 'npz', 'amrex', or 'hdf5', got {mode}"
        self.mode = mode
        self.field_names = field_names
        self.num_levels = num_levels
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.adapt_nearby = adapt_nearby
        self.tol_frac = tol_frac
        self.adapt_on_channels = adapt_on_channels
        self.amrex_patch_size = amrex_patch_size

        if mode == "hdf5":
            if scalar_fields is None and vector_fields is None:
                raise ValueError("mode='hdf5' requires at least one of scalar_fields or vector_fields")
            if field_order is None:
                raise ValueError("mode='hdf5' requires field_order")
            self._hdf5_scalar_fields = {k: tuple(v) for k, v in (scalar_fields or {}).items()}
            self._hdf5_vector_fields = {k: (v[0], v[1], int(v[2])) for k, v in (vector_fields or {}).items()}
            self._hdf5_field_order = field_order
            self._hdf5_cache = _HDF5Cache(max_size=hdf5_cache_size)

    def _build_topology(self, ref_path: str) -> Tuple["Quadtree", List[str]]:
        """Build adapted quadtree topology from a reference path. Returns (qt, field_names)."""
        if self.mode == "npz":
            ref = np.load(ref_path)
            field_names = self.field_names if self.field_names else list(ref.keys())
            ref = np.stack([ref[k] for k in field_names], axis=0)

            qt_extent = (0, ref.shape[2], 0, ref.shape[1])
            qt = Quadtree(*qt_extent, max_level_idx=self.num_levels - 1, channels=len(field_names),
                          tile_width=self.tile_width, tile_height=self.tile_height)
            adapt_on_field(qt, ref, tol_frac=self.tol_frac, channel=self.adapt_on_channels,
                           max_passes=self.num_levels - 1)

        elif self.mode == "hdf5":
            ref = _load_hdf5_frame(ref_path, self._hdf5_scalar_fields, self._hdf5_vector_fields,
                                   self._hdf5_field_order, self._hdf5_cache)
            field_names = self._hdf5_field_order

            qt_extent = (0, ref.shape[2], 0, ref.shape[1])
            qt = Quadtree(*qt_extent, max_level_idx=self.num_levels - 1, channels=len(field_names),
                          tile_width=self.tile_width, tile_height=self.tile_height)
            adapt_on_field(qt, ref, tol_frac=self.tol_frac, channel=self.adapt_on_channels,
                           max_passes=self.num_levels - 1)

        elif self.mode == "amrex":
            if self.field_names:
                field_names = self.field_names
            else:
                from wamrvit.quad.amrex_to_qt import read_plotfile_header
                header_info = read_plotfile_header(ref_path)
                field_names = header_info["field_names"]

            qt, _, _ = amrex_to_quadtree(ref_path, field_names, patch_size=self.amrex_patch_size,
                                         num_levels=self.num_levels)
            if self.adapt_on_channels is not None:
                regrid(qt, channel=self.adapt_on_channels, tol_frac=self.tol_frac,
                       max_passes=10, disable_warnings=True)

        if self.adapt_nearby > 0:
            qt = expand_fine_region(qt, self.adapt_nearby)

        # qt.mark_boundary_cells()  # disabled: is_boundary flag unread by active pipeline
        return qt, field_names

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        ref_path = batch["input_paths"][0][-1]
        qt, field_names = self._build_topology(ref_path)

        # Create coarse baseline for diff computation
        coarse = Quadtree(qt.xmin, qt.xmax, qt.ymin, qt.ymax,
                          max_level_idx=qt.max_level_idx, channels=qt.channels,
                          tile_width=qt.tile_width, tile_height=qt.tile_height)
        diff = compute_tree_diff(coarse, qt, include_values=False)

        # Serialize TreeDiff fields as numpy arrays
        result = dict(batch)
        result["topo_ids"] = np.array([diff.ids])              # (1, N_diff)
        result["topo_tuples"] = np.array([diff.tuples])        # (1, N_diff, 5)
        result["topo_ops"] = np.array([diff.ops])              # (1, N_diff)
        result["topo_xmin"] = np.array([diff.xmin], dtype=np.float64)
        result["topo_xmax"] = np.array([diff.xmax], dtype=np.float64)
        result["topo_ymin"] = np.array([diff.ymin], dtype=np.float64)
        result["topo_ymax"] = np.array([diff.ymax], dtype=np.float64)
        result["topo_nx_tiles"] = np.array([diff.nx_tiles], dtype=np.int32)
        result["topo_ny_tiles"] = np.array([diff.ny_tiles], dtype=np.int32)
        result["topo_max_level_idx"] = np.array([diff.max_level_idx], dtype=np.int32)
        result["topo_channels"] = np.array([diff.channels], dtype=np.int32)
        return result


class FillAndExportMapper:
    """
    Stage 2 loader: reconstructs quadtree from a cached TreeDiff, optionally
    applies augmentation, fills values from disk, and exports tensors.

    Drop-in replacement for AdaptiveLoader in the Seq2SeqMapper(loader=...) slot,
    except __call__ takes an additional ``topo_diff`` argument.
    """

    def __init__(
        self,
        mode: str = "npz",
        field_names: Optional[List[str]] = None,
        cell_scale_mode: str = "area",
        num_levels: int = 3,
        amrex_patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        return_src: bool = False,
        augment: bool = True,
        resample_coarsen_ratio: float = 0.0,
        resample_refine_ratio: float = 0.0,
        # HDF5 params
        scalar_fields: Optional[Dict] = None,
        vector_fields: Optional[Dict] = None,
        field_order: Optional[List[str]] = None,
        hdf5_cache_size: int = 4,
        value_storage: str = "uniform",
    ):
        assert mode in ["npz", "amrex", "hdf5"]
        if value_storage not in ("uniform", "native"):
            raise ValueError(f"value_storage must be 'uniform' or 'native'; got {value_storage!r}.")
        if value_storage == "native" and mode == "amrex":
            raise ValueError("value_storage='native' is not supported with mode='amrex'.")
        self.value_storage = value_storage
        self.mode = mode
        self.field_names = field_names
        self.cell_scale_mode = cell_scale_mode
        self.num_levels = num_levels
        self.amrex_patch_size = amrex_patch_size
        self.return_src = return_src
        self.augment = augment
        self.resample_coarsen_ratio = resample_coarsen_ratio
        self.resample_refine_ratio = resample_refine_ratio

        if mode == "hdf5":
            self._hdf5_scalar_fields = {k: tuple(v) for k, v in (scalar_fields or {}).items()}
            self._hdf5_vector_fields = {k: (v[0], v[1], int(v[2])) for k, v in (vector_fields or {}).items()}
            self._hdf5_field_order = field_order
            self._hdf5_cache = _HDF5Cache(max_size=hdf5_cache_size)

    def __call__(
        self,
        input_paths: List[str],
        target_paths: List[str],
        topo_diff: "TreeDiff",
    ) -> Dict[str, Any]:
        # 1. Reconstruct quadtree from cached topology
        tw = (topo_diff.xmax - topo_diff.xmin) / max(1, topo_diff.nx_tiles)
        th = (topo_diff.ymax - topo_diff.ymin) / max(1, topo_diff.ny_tiles)
        qt_ref = Quadtree(topo_diff.xmin, topo_diff.xmax, topo_diff.ymin, topo_diff.ymax,
                          max_level_idx=topo_diff.max_level_idx, channels=topo_diff.channels,
                          tile_width=tw, tile_height=th, value_storage=self.value_storage)
        apply_tree_diff(qt_ref, topo_diff, update_values=False)

        # 2. Optional augmentation (train only)
        if self.augment and (self.resample_coarsen_ratio > 0.0 or self.resample_refine_ratio > 0.0):
            from wamrvit.quad.quad_utils import random_resample_quadtree
            random_resample_quadtree(qt_ref, self.resample_coarsen_ratio, self.resample_refine_ratio)

        # qt_ref.mark_boundary_cells()  # disabled: is_boundary flag unread by active pipeline

        # Resolve field names
        if self.mode == "hdf5":
            field_names = self._hdf5_field_order
        elif self.field_names:
            field_names = self.field_names
        else:
            # Infer from first input file
            if self.mode == "npz":
                field_names = list(np.load(input_paths[0]).keys())
            elif self.mode == "amrex":
                from wamrvit.quad.amrex_to_qt import read_plotfile_header
                field_names = read_plotfile_header(input_paths[0])["field_names"]

        # 3/4. Fill inputs + targets via shared helper
        result = _extract_inputs_targets(
            qt_ref=qt_ref,
            input_paths=input_paths,
            target_paths=target_paths,
            field_names=field_names,
            mode=self.mode,
            cell_scale_mode=self.cell_scale_mode,
            return_src=self.return_src,
            value_storage=self.value_storage,
            amrex_patch_size=self.amrex_patch_size,
            hdf5_loader=(
                (self._hdf5_scalar_fields, self._hdf5_vector_fields,
                 self._hdf5_field_order, self._hdf5_cache)
                if self.mode == "hdf5" else None
            ),
        )
        return result
