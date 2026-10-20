import os
import time
import numpy as np
import torch
import warnings
from collections import OrderedDict
from typing import Any, Optional, List, Dict, Union, Tuple

import h5py

from wamrvit.quad.quadtree import (
    Quadtree,
    TreeDiff,
    apply_tree_diff,
    compute_tree_diff,
    export_refine_only_tree_diff,
)
from wamrvit.quad.quad_utils import quadtree_to_tensor, quadtree_to_tensor_native
from wamrvit.quad.adapt_wavelet import adapt_on_field, expand_fine_region, regrid
from wamrvit.quad.amrex_to_qt import (
    _import_yt_utils,
    amrex_leaf_geometry,
    amrex_to_quadtree,
    assign_from_amrex,
    assign_from_amrex_decoded,
    read_plotfile_header,
    read_plotfile_raw,
    sample_amrex_decoded_into,
)
from wamrvit.dataloader.selection.sources import parse_hdf5_virtual_path


def get_amrex_field_names(yt_obj) -> List[str]:
    return [i[1] for i in yt_obj.field_list]


class YTAmReXRegularLoader:
    @staticmethod
    def native_time_reader(params):
        """Provide metadata-only AMReX clocks to selection.sources.source_time_reader."""
        from wamrvit.dataloader.selection.sources import amrex_times

        return amrex_times

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

        
def _extract_amrex_uniform_direct(
    *,
    qt_ref: Quadtree,
    input_paths: List[str],
    target_paths: List[str],
    field_names: List[str],
    cell_scale_mode: str,
    return_src: bool,
    amrex_patch_size: int | Tuple[int, int],
    profile: Optional[Dict[str, float]],
    amrex_frame_refs: Optional[Dict[str, Any]],
    pin_output: bool,
    source_normalization: Optional[Dict[str, Any]],
    direct_geometry: Optional[Dict[str, Any]],
    raster_sampler: bool,
    max_output_bytes: Optional[int],
    source_access_trace: Optional[Any],
    transport_context: Optional[Dict[str, int]],
) -> Dict[str, Any]:
    """Sample AMReX frames directly into time-major model-value buffers.

    The legacy path writes every sampled patch into ``QuadCell.value`` and then
    copies it back out through ``quadtree_to_tensor`` for every frame. This path
    exports ordered geometry once and lets the Numba sampler fill final
    time-major arrays without mutating leaf objects.
    """
    if isinstance(amrex_patch_size, int):
        patch_size = (amrex_patch_size, amrex_patch_size)
    else:
        patch_size = tuple(amrex_patch_size)

    geometry_started = time.perf_counter() if profile is not None else None
    if direct_geometry is None:
        if qt_ref is None:
            raise ValueError("Direct AMReX export requires topology geometry.")
        exported = qt_ref.export_leaves(
            order="morton",
            cell_scale_mode=cell_scale_mode,
            include_values=False,
            include_bounds=True,
        )
        domain = qt_ref.get_domain_meta()
    else:
        exported = direct_geometry
        domain = exported["domain"]
        if float(domain["xmin"]) != 0.0 or float(domain["ymin"]) != 0.0:
            raise ValueError(
                "Direct AMReX TreeDiff geometry requires a zero-origin domain."
            )
    leaf_bounds = exported["bounds"]
    meta = {
        "centers": exported["centers"],
        "levels": exported["levels"],
        "tiles": exported["tiles"],
        "xy_idx": exported["xy_idx"],
        "cell_ids": exported["ids"],
        "domain": domain,
    }
    if profile is not None:
        profile["direct_geometry_s"] = time.perf_counter() - geometry_started

    # Allocate the final model layout once. Each fixed-time view remains
    # writable by the sampler, while the complete result is C-contiguous for
    # in-place normalization and the subsequent pinned-memory copy.
    model_prefix = (1, leaf_bounds.shape[0], len(field_names))
    value_elements = (
        int(np.prod(model_prefix))
        * (len(input_paths) + len(target_paths))
        * patch_size[0]
        * patch_size[1]
    )
    metadata_bytes = int(exported["centers"].nbytes + exported["levels"].nbytes)
    required_output_bytes = value_elements * np.dtype(np.float32).itemsize + metadata_bytes
    if max_output_bytes is not None and required_output_bytes > max_output_bytes:
        raise MemoryError(
            "Direct AMReX payload requires at least "
            f"{required_output_bytes / 1024**3:.3f} GiB before allocation; "
            f"the streaming slot allows {max_output_bytes / 1024**3:.3f} GiB."
        )

    def allocate(time_steps: int) -> np.ndarray:
        shape = (*model_prefix, time_steps, patch_size[0], patch_size[1])
        if pin_output:
            return torch.empty(shape, dtype=torch.float32, pin_memory=True).numpy()
        return np.empty(shape, dtype=np.float32)

    allocation_started = time.perf_counter() if profile is not None else None
    input_values = allocate(len(input_paths))
    target_values = allocate(len(target_paths))
    if profile is not None:
        profile["direct_allocation_s"] = time.perf_counter() - allocation_started

    if profile is not None:
        profile["amrex_frame_cache_hits"] = 0.0
        profile["amrex_frame_cache_misses"] = 0.0

    def fill_frame(
        path: str, output: np.ndarray, source_position: int
    ) -> None:
        frame_started = time.perf_counter() if profile is not None else None
        access_started_ns = (
            time.monotonic_ns() if source_access_trace is not None else None
        )
        frame_ref = (
            amrex_frame_refs.get(os.path.abspath(path))
            if amrex_frame_refs is not None
            else None
        )
        if frame_ref is None:
            read_started = time.perf_counter() if profile is not None else None
            grid_info, grid_data, header = read_plotfile_raw(
                path, field_names, patch_size
            )
            domain_lo = header["domain_lo"]
            domain_hi = header["domain_hi"]
            source_ready_ns = (
                time.monotonic_ns() if source_access_trace is not None else None
            )
            if profile is not None:
                profile["amrex_read_s"] = profile.get("amrex_read_s", 0.0) + (
                    time.perf_counter() - read_started
                )
                profile["amrex_frame_cache_misses"] = profile.get(
                    "amrex_frame_cache_misses", 0.0
                ) + 1.0
            if source_normalization is not None:
                normalize_started = (
                    time.perf_counter() if profile is not None else None
                )
                mean = np.asarray(source_normalization["mean"], dtype=np.float32)
                std = np.asarray(source_normalization["std"], dtype=np.float32)
                stats_shape = (1, grid_data.shape[1], 1, 1)
                np.subtract(grid_data, mean.reshape(stats_shape), out=grid_data)
                np.divide(
                    grid_data,
                    std.reshape(stats_shape) + source_normalization.get("eps", 0.0),
                    out=grid_data,
                )
                if profile is not None:
                    profile["amrex_source_normalize_s"] = profile.get(
                        "amrex_source_normalize_s", 0.0
                    ) + (time.perf_counter() - normalize_started)
            normalize_ready_ns = (
                time.monotonic_ns() if source_access_trace is not None else None
            )
        else:
            import ray

            fetch_started = time.perf_counter() if profile is not None else None
            decoded = (
                ray.get(frame_ref)
                if isinstance(frame_ref, ray.ObjectRef)
                else frame_ref
            )
            grid_info, grid_data, domain_lo, domain_hi = decoded
            source_ready_ns = (
                time.monotonic_ns() if source_access_trace is not None else None
            )
            normalize_ready_ns = source_ready_ns
            if profile is not None:
                profile["amrex_frame_fetch_s"] = profile.get(
                    "amrex_frame_fetch_s", 0.0
                ) + (time.perf_counter() - fetch_started)
                profile["amrex_frame_cache_hits"] = profile.get(
                    "amrex_frame_cache_hits", 0.0
                ) + 1.0

        sample_amrex_decoded_into(
            leaf_bounds,
            float(domain["xmax"]),
            float(domain["ymax"]),
            grid_info,
            grid_data,
            domain_lo,
            domain_hi,
            output,
            patch_size,
            profile,
            raster_sampler=raster_sampler,
        )
        if source_access_trace is not None:
            source_access_trace.record(
                os.path.abspath(path),
                cache_hit=frame_ref is not None,
                start_ns=access_started_ns,
                source_ready_ns=source_ready_ns,
                normalize_ready_ns=normalize_ready_ns,
                end_ns=time.monotonic_ns(),
                context=transport_context,
                source_position=source_position,
            )
        if profile is not None:
            elapsed = time.perf_counter() - frame_started
            profile["source_assign_s"] = profile.get("source_assign_s", 0.0) + elapsed
            profile["direct_frame_s"] = profile.get("direct_frame_s", 0.0) + elapsed

    for time_idx, path in enumerate(input_paths):
        fill_frame(
            str(path),
            input_values[0, :, :, time_idx, :, :],
            time_idx,
        )
    input_count = len(input_paths)
    for time_idx, path in enumerate(target_paths):
        fill_frame(
            str(path),
            target_values[0, :, :, time_idx, :, :],
            input_count + time_idx,
        )

    result: Dict[str, Any] = {
        "input": input_values,
        "target": target_values,
        "meta": meta,
        "_model_layout": True,
        "_already_transformed": source_normalization is not None,
        "input_src": None,
        "target_src": None,
    }
    if return_src:
        result["input_src_paths"] = [str(path) for path in input_paths]
        result["target_src_paths"] = [str(path) for path in target_paths]
    return result


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
    profile: Optional[Dict[str, float]] = None,
    amrex_frame_refs: Optional[Dict[str, Any]] = None,
    direct_amrex_export: bool = False,
    direct_amrex_pin_output: bool = False,
    amrex_source_normalization: Optional[Dict[str, Any]] = None,
    direct_amrex_geometry: Optional[Dict[str, Any]] = None,
    direct_amrex_raster_sampler: bool = False,
    max_output_bytes: Optional[int] = None,
    source_access_trace: Optional[Any] = None,
    transport_context: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Shared input/target extraction for AdaptiveLoader and FillAndExportMapper.

    In 'uniform' mode returns 'input'/'target' as (T, N, C, Ph, Pw) stacks.
    In 'native' mode returns 'input_by_level'/'target_by_level' as
    {lvl: (T, N_l, C, Ph*s, Pw*s)} plus 'leaf_to_bucket' (N, 2).
    """
    if direct_amrex_export:
        if mode != "amrex" or value_storage != "uniform":
            raise ValueError(
                "Direct AMReX export requires mode='amrex' and "
                "value_storage='uniform'."
            )
        if amrex_patch_size is None:
            raise ValueError("Direct AMReX export requires amrex_patch_size.")
        return _extract_amrex_uniform_direct(
            qt_ref=qt_ref,
            input_paths=input_paths,
            target_paths=target_paths,
            field_names=field_names,
            cell_scale_mode=cell_scale_mode,
            return_src=return_src,
            amrex_patch_size=amrex_patch_size,
            profile=profile,
            amrex_frame_refs=amrex_frame_refs,
            pin_output=direct_amrex_pin_output,
            source_normalization=amrex_source_normalization,
            direct_geometry=direct_amrex_geometry,
            raster_sampler=direct_amrex_raster_sampler,
            max_output_bytes=max_output_bytes,
            source_access_trace=source_access_trace,
            transport_context=transport_context,
        )

    def _assign(p: str):
        started = time.perf_counter() if profile is not None else None
        if mode == "npz":
            arr_all = np.load(p)
            arr_all = np.stack([arr_all[k] for k in field_names], axis=0)
            qt_ref.assign_from_array(arr_all)
            result = arr_all
        elif mode == "hdf5":
            sf, vf, fo, cache = hdf5_loader
            arr_all = _load_hdf5_frame(p, sf, vf, fo, cache)
            qt_ref.assign_from_array(arr_all)
            result = arr_all
        elif mode == "amrex":
            frame_ref = (
                amrex_frame_refs.get(os.path.abspath(p))
                if amrex_frame_refs is not None
                else None
            )
            if frame_ref is None:
                assign_from_amrex(
                    qt_ref,
                    p,
                    field_names,
                    patch_size=amrex_patch_size,
                    profile=profile,
                )
            else:
                import ray

                fetch_started = (
                    time.perf_counter() if profile is not None else None
                )
                decoded = (
                    ray.get(frame_ref)
                    if isinstance(frame_ref, ray.ObjectRef)
                    else frame_ref
                )
                if profile is not None:
                    profile["amrex_frame_fetch_s"] = profile.get(
                        "amrex_frame_fetch_s", 0.0
                    ) + (time.perf_counter() - fetch_started)
                    profile["amrex_frame_cache_hits"] = profile.get(
                        "amrex_frame_cache_hits", 0.0
                    ) + 1.0
                assign_from_amrex_decoded(
                    qt_ref,
                    *decoded,
                    patch_size=amrex_patch_size,
                    profile=profile,
                )
            result = None
        if profile is not None:
            profile["source_assign_s"] = profile.get("source_assign_s", 0.0) + (
                time.perf_counter() - started
            )
        return result

    def _export_uniform():
        started = time.perf_counter() if profile is not None else None
        result = quadtree_to_tensor(
            qt_ref, return_tensor=False, cell_scale_mode=cell_scale_mode,
        )
        if profile is not None:
            profile["tensor_export_s"] = profile.get("tensor_export_s", 0.0) + (
                time.perf_counter() - started
            )
        return result

    # Uniform path: original flat (T, N, C, Ph, Pw) outputs.
    if value_storage == "uniform":
        input_arr = []
        input_src_arr = [] if return_src else None
        ref_meta = None
        for p in input_paths:
            src = _assign(p)
            arr, meta = _export_uniform()
            input_arr.append(arr)
            if return_src:
                input_src_arr.append(src if mode in ("npz", "hdf5") else p)
            if ref_meta is None:
                ref_meta = meta

        output_arr = []
        output_src_arr = [] if return_src else None
        for p in target_paths:
            tgt = _assign(p)
            arr, _ = _export_uniform()
            output_arr.append(arr)
            if return_src:
                output_src_arr.append(tgt if mode in ("npz", "hdf5") else p)

        stack_started = time.perf_counter() if profile is not None else None
        stacked_input = np.stack(input_arr)
        stacked_target = np.stack(output_arr)
        if profile is not None:
            profile["stack_s"] = time.perf_counter() - stack_started
        result: Dict[str, Any] = {
            "input": stacked_input,
            "target": stacked_target,
            "meta": ref_meta,
            "input_src": np.stack(input_src_arr) if (return_src and mode in ("npz", "hdf5")) else None,
            "target_src": np.stack(output_src_arr) if (return_src and mode in ("npz", "hdf5")) else None,
        }
        if return_src and mode == "amrex":
            result["input_src_paths"] = input_src_arr
            result["target_src_paths"] = output_src_arr
        return result

    # Native path: per-level buckets stacked over time.
    if mode == "amrex":
        raise ValueError("native value_storage not supported for AMReX.")
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
        return_leaf_geometry (bool): For AMReX, return physical ``(H_x,H_y)``
            support for each leaf and its level-zero root. Defaults to False.
    """
    @staticmethod
    def native_time_reader(params):
        """Provide supported native clocks to selection.sources.source_time_reader without payload IO."""
        from wamrvit.dataloader.selection.sources import amrex_times

        mode = params.get("mode", "npz")
        if mode != "amrex":
            raise ValueError(f"Native timestamps are not supported by AdaptiveLoader mode={mode!r}.")
        return amrex_times

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
        return_leaf_geometry: bool = False,
    ):
        if mode not in ["npz", "amrex", "hdf5"]:
            raise ValueError("mode must be 'npz', 'amrex', or 'hdf5'")
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
        self.return_leaf_geometry = bool(return_leaf_geometry)

        if self.return_leaf_geometry and mode != "amrex":
            raise ValueError(
                "return_leaf_geometry is currently supported only for mode='amrex'."
            )
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
        if self.return_leaf_geometry:
            result["meta"].update(
                amrex_leaf_geometry(result["meta"], read_plotfile_header(str(ref_path)))
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
        if mode not in ["npz", "amrex", "hdf5"]:
            raise ValueError(f"mode must be 'npz', 'amrex', or 'hdf5', got {mode}")
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

        # Keep path descriptors Arrow-native. Older caches retained NumPy
        # object arrays here, which Ray persisted as unsafe pickled objects.
        result = dict(batch)
        result["input_paths"] = [
            [str(path) for path in paths] for paths in batch["input_paths"]
        ]
        result["target_paths"] = [
            [str(path) for path in paths] for paths in batch["target_paths"]
        ]
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
        profile_data_transport: bool = False,
        amrex_frame_refs: Optional[Dict[str, Any]] = None,
        direct_amrex_export: bool = False,
        direct_amrex_pin_output: bool = False,
        amrex_source_normalization: Optional[Dict[str, Any]] = None,
        direct_amrex_topology_geometry: bool = False,
        direct_amrex_raster_sampler: bool = False,
        max_output_bytes: Optional[int] = None,
        amrex_source_trace_dir: Optional[str] = None,
        amrex_source_trace_rank: int = 0,
        return_leaf_geometry: bool = False,
    ):
        if mode not in ["npz", "amrex", "hdf5"]:
            raise ValueError(f"mode must be 'npz', 'amrex', or 'hdf5', got {mode}")
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
        self.profile_data_transport = profile_data_transport
        self.amrex_frame_refs = amrex_frame_refs
        self.direct_amrex_export = direct_amrex_export
        self.direct_amrex_pin_output = direct_amrex_pin_output
        self.amrex_source_normalization = amrex_source_normalization
        self.direct_amrex_topology_geometry = direct_amrex_topology_geometry
        self.direct_amrex_raster_sampler = direct_amrex_raster_sampler
        self.max_output_bytes = max_output_bytes
        self.return_leaf_geometry = bool(return_leaf_geometry)
        if self.return_leaf_geometry and mode != "amrex":
            raise ValueError(
                "return_leaf_geometry is currently supported only for mode='amrex'."
            )
        self._transport_context: Optional[Dict[str, int]] = None
        self._source_access_trace = None
        if amrex_source_trace_dir is not None:
            from wamrvit.dataloader.transport.tracing import SourceAccessTrace

            self._source_access_trace = SourceAccessTrace(
                amrex_source_trace_dir,
                rank=amrex_source_trace_rank,
            )
        if self.direct_amrex_pin_output and not self.direct_amrex_export:
            raise ValueError(
                "direct_amrex_pin_output requires direct_amrex_export."
            )
        if self.amrex_source_normalization is not None and not self.direct_amrex_export:
            raise ValueError(
                "amrex_source_normalization requires direct_amrex_export."
            )
        if self.direct_amrex_topology_geometry and not self.direct_amrex_export:
            raise ValueError(
                "direct_amrex_topology_geometry requires direct_amrex_export."
            )
        if self.direct_amrex_raster_sampler and not self.direct_amrex_export:
            raise ValueError(
                "direct_amrex_raster_sampler requires direct_amrex_export."
            )
        if self._source_access_trace is not None and not self.direct_amrex_export:
            raise ValueError(
                "AMReX source-access tracing requires direct_amrex_export."
            )
        if self.direct_amrex_topology_geometry and (
            self.resample_coarsen_ratio > 0.0 or self.resample_refine_ratio > 0.0
        ):
            raise ValueError(
                "direct_amrex_topology_geometry does not support random resampling."
            )

        if mode == "hdf5":
            self._hdf5_scalar_fields = {k: tuple(v) for k, v in (scalar_fields or {}).items()}
            self._hdf5_vector_fields = {k: (v[0], v[1], int(v[2])) for k, v in (vector_fields or {}).items()}
            self._hdf5_field_order = field_order
            self._hdf5_cache = _HDF5Cache(max_size=hdf5_cache_size)

    def set_transport_context(self, context: Dict[str, int]) -> None:
        self._transport_context = dict(context)

    def close(self) -> None:
        if self._source_access_trace is not None:
            self._source_access_trace.close()

    def __call__(
        self,
        input_paths: List[str],
        target_paths: List[str],
        topo_diff: "TreeDiff",
    ) -> Dict[str, Any]:
        profile = {} if self.profile_data_transport else None
        topology_started = time.perf_counter() if profile is not None else None

        # 1. Reconstruct the object quadtree unless the refine-only cached diff
        # can export identical geometry directly.
        direct_geometry = None
        if self.direct_amrex_topology_geometry:
            qt_ref = None
            direct_geometry = export_refine_only_tree_diff(
                topo_diff,
                cell_scale_mode=self.cell_scale_mode,
                include_bounds=True,
            )
        else:
            tw = (topo_diff.xmax - topo_diff.xmin) / max(1, topo_diff.nx_tiles)
            th = (topo_diff.ymax - topo_diff.ymin) / max(1, topo_diff.ny_tiles)
            qt_ref = Quadtree(
                topo_diff.xmin,
                topo_diff.xmax,
                topo_diff.ymin,
                topo_diff.ymax,
                max_level_idx=topo_diff.max_level_idx,
                channels=topo_diff.channels,
                tile_width=tw,
                tile_height=th,
                value_storage=self.value_storage,
            )
            apply_tree_diff(qt_ref, topo_diff, update_values=False)

            # 2. Optional augmentation (train only)
            if self.augment and (
                self.resample_coarsen_ratio > 0.0
                or self.resample_refine_ratio > 0.0
            ):
                from wamrvit.quad.quad_utils import random_resample_quadtree

                random_resample_quadtree(
                    qt_ref,
                    self.resample_coarsen_ratio,
                    self.resample_refine_ratio,
                )

            # qt_ref.mark_boundary_cells()  # disabled: flag unread by active pipeline
        if profile is not None:
            profile["topology_reconstruct_s"] = (
                time.perf_counter() - topology_started
            )

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
            profile=profile,
            amrex_frame_refs=self.amrex_frame_refs,
            direct_amrex_export=self.direct_amrex_export,
            direct_amrex_pin_output=self.direct_amrex_pin_output,
            amrex_source_normalization=self.amrex_source_normalization,
            direct_amrex_geometry=direct_geometry,
            direct_amrex_raster_sampler=self.direct_amrex_raster_sampler,
            max_output_bytes=self.max_output_bytes,
            source_access_trace=self._source_access_trace,
            transport_context=self._transport_context,
        )
        if self.return_leaf_geometry:
            result["meta"].update(
                amrex_leaf_geometry(
                    result["meta"],
                    read_plotfile_header(str(input_paths[-1])),
                )
            )
        if profile is not None:
            result["_transport_profile"] = profile
        return result
