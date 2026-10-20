"""Sequence-to-batch and cached/sample mapping.

Seq2SeqMapper loads uncached windows; SampleMapper supplies shared references
to CachedSeq2SeqMapper for either global Ray or worker-local delivery.
"""

import copy
import time
import numpy as np
from typing import Any, Optional, Callable, Dict

from wamrvit.quad.quadtree import TreeDiff
from wamrvit.dataloader.payload.transforms import apply_transform_src


_TRANSPORT_CONTROL_COLUMNS = (
    "sample_index",
    "traj_idx",
    "frame_idx",
)


def _copy_transport_control_columns(
    result: Dict[str, np.ndarray], batch: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Preserve descriptor transport tags through both sequence mappers."""
    for name in _TRANSPORT_CONTROL_COLUMNS:
        if name in batch:
            result[name] = np.asarray(batch[name])
    return result


class Seq2SeqMapper:
    """Load and assemble sequence batches for uncached training and rollout mapping.

    Used by: training/rollout Ray map_batches; CachedSeq2SeqMapper reuses assembly.
    """

    def __init__(
        self,
        loader: Callable,
        transform: Optional[Callable] = None,
        ):
        """
        Args:
            loader (Callable): Return with time dimension first, e.g. (T, C, H, W).
            transform (Callable, optional): Function to apply to the batch of data after loading.

        return in (B, C, T, H, W) format for video models
        """
        self.loader = loader
        self.transform = transform

    @staticmethod
    def _assemble_adaptive_result(
        return_dict: Dict[str, Any],
        transform: Optional[Callable] = None,
    ) -> Dict[str, np.ndarray]:
        """Shared logic for assembling adaptive (ndim==5) output from a single loader return_dict.

        Handles transpose (T,N,C,H,W) -> (1,N,C,T,H,W), transform application,
        and metadata flattening into Ray-compatible numpy columns.
        Supports both uniform and native (multi-scale) value storage.
        """
        meta = return_dict.get("meta")

        # Native-storage branch: per-level columns instead of a flat input/target.
        if "input_by_level" in return_dict:
            return Seq2SeqMapper._assemble_native_adaptive_result(return_dict, transform)

        # Per-sample augmentation state (e.g. PeriodicRoll): freeze one shift
        # so input/target/target_src stay aligned across the transform calls below.
        if transform is not None and hasattr(transform, "reset"):
            transform.reset()

        input_data = return_dict["input"]
        target_data = return_dict["target"]
        model_layout = bool(return_dict.get("_model_layout", False))
        already_transformed = bool(return_dict.get("_already_transformed", False))

        if model_layout:
            # Direct AMReX export already emits (1, N, C, T, H, W).
            inputs_np = input_data
            targets_np = target_data
        else:
            # (T, N, C, H, W) -> (1, N, C, T, H, W). Adding the singleton
            # dimension as a view avoids copying multi-GiB adaptive values
            # before the transform writes its final contiguous output.
            inputs_np = input_data[np.newaxis].transpose(0, 2, 3, 1, 4, 5)
            targets_np = target_data[np.newaxis].transpose(0, 2, 3, 1, 4, 5)

        centers_np = meta["centers"][np.newaxis] if meta is not None else None
        levels_np = meta["levels"][np.newaxis] if meta is not None else None

        targets_src_np = None
        target_src_paths_np = None
        if return_dict.get("target_src") is not None:
            # target_src is (T, C, H, W) — no N dim, so no 6D transpose.
            targets_src_np = np.stack([return_dict["target_src"]])  # (1, T, C, H, W)
        if return_dict.get("target_src_paths") is not None:
            target_src_paths_np = np.empty(1, dtype=object)
            target_src_paths_np[0] = return_dict["target_src_paths"]

        if transform and not already_transformed:
            if model_layout and hasattr(transform, "transform_inplace"):
                inputs_np = transform.transform_inplace(inputs_np)
                targets_np = transform.transform_inplace(targets_np)
            else:
                inputs_np = transform(inputs_np)
                targets_np = transform(targets_np)
            if targets_src_np is not None:
                # target_src is (1, T, C, H, W) — different layout from input/target,
                # needs the T↔C transpose dance for NormalizeArray.
                targets_src_np = apply_transform_src(transform, targets_src_np)
        elif already_transformed and targets_src_np is not None:
            raise ValueError(
                "Pre-normalized adaptive values do not support target_src."
            )

        result: Dict[str, Any] = {
            "input": inputs_np,
            "target": targets_np,
        }
        if meta is not None:
            result["centers"] = centers_np
            result["levels"] = levels_np
            result["xmin"] = np.array([meta["domain"]["xmin"]])
            result["xmax"] = np.array([meta["domain"]["xmax"]])
            result["ymin"] = np.array([meta["domain"]["ymin"]])
            result["ymax"] = np.array([meta["domain"]["ymax"]])
            result["max_level_idx"] = np.array([meta["domain"]["max_level_idx"]])
            result["tile_width"] = np.array([meta["domain"]["tile_width"]])
            result["tile_height"] = np.array([meta["domain"]["tile_height"]])
            if meta.get("leaf_support_xy") is not None:
                # (N, 2)/(2,) -> Ray row layout (1, N, 2)/(1, 2).
                result["leaf_support_xy"] = meta["leaf_support_xy"][np.newaxis]
                result["root_support_xy"] = meta["root_support_xy"][np.newaxis]

        if targets_src_np is not None:
            result["target_src"] = targets_src_np
        if target_src_paths_np is not None:
            result["target_src_paths"] = target_src_paths_np

        return result


    @staticmethod
    def _assemble_native_adaptive_result(
        return_dict: Dict[str, Any],
        transform: Optional[Callable] = None,
    ) -> Dict[str, np.ndarray]:
        """Assemble per-level Ray Data columns for value_storage='native'.

        Emits:
          - input_level_{l} / target_level_{l}: (1, N_l, C, T, H_l, W_l)
          - leaf_to_bucket: (1, N, 2)
          - centers, levels, domain scalars (same as uniform)
        """
        input_by_level = return_dict["input_by_level"]     # {lvl: (T_in, N_l, C, H_l, W_l)}
        target_by_level = return_dict["target_by_level"]   # {lvl: (T_out, N_l, C, H_l, W_l)}
        leaf_to_bucket = return_dict["leaf_to_bucket"]     # (N, 2)
        meta = return_dict.get("meta")

        if transform is not None and hasattr(transform, "reset"):
            transform.reset()

        result: Dict[str, Any] = {}
        for lvl, arr in input_by_level.items():
            # (T, N_l, C, H, W) -> (1, N_l, C, T, H, W)
            t_arr = np.stack([arr]).transpose(0, 2, 3, 1, 4, 5)
            if transform is not None:
                t_arr = transform(t_arr)
            result[f"input_level_{lvl}"] = t_arr
        for lvl, arr in target_by_level.items():
            t_arr = np.stack([arr]).transpose(0, 2, 3, 1, 4, 5)
            if transform is not None:
                t_arr = transform(t_arr)
            result[f"target_level_{lvl}"] = t_arr

        result["leaf_to_bucket"] = np.stack([leaf_to_bucket])

        if meta is not None:
            result["centers"] = np.stack([meta["centers"]])
            result["levels"] = np.stack([meta["levels"]])
            result["xmin"] = np.array([meta["domain"]["xmin"]])
            result["xmax"] = np.array([meta["domain"]["xmax"]])
            result["ymin"] = np.array([meta["domain"]["ymin"]])
            result["ymax"] = np.array([meta["domain"]["ymax"]])
            result["max_level_idx"] = np.array([meta["domain"]["max_level_idx"]])
            result["tile_width"] = np.array([meta["domain"]["tile_width"]])
            result["tile_height"] = np.array([meta["domain"]["tile_height"]])

        if return_dict.get("target_src") is not None:
            # target_src is (T, C, H, W) — no N dim, so no 6D transpose.
            targets_src_np = np.stack([return_dict["target_src"]])  # (1, T, C, H, W)
            if transform is not None:
                targets_src_np = apply_transform_src(transform, targets_src_np)
            result["target_src"] = targets_src_np
        return result


    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Input keys: 'input_paths', 'target_paths' (arrays of lists of strings)
        Output keys: 'input', 'target', 'meta'
        """
        
        batch_size = len(batch["input_paths"])

        # Per-sample augmentation state (e.g. PeriodicRoll). Freeze here so
        # every transform call below — input, target, target_src — shares one
        # shift. Covers the main regular and uniform-adaptive paths; the
        # native-adaptive branch resets inside its own assembler.
        if self.transform is not None and hasattr(self.transform, "reset"):
            self.transform.reset()

        # Peek at first sample to detect native-mode loader output.
        first_rd = self.loader(batch["input_paths"][0], batch["target_paths"][0])
        if "input_by_level" in first_rd:
            if batch_size != 1:
                raise ValueError(
                    "Native value_storage requires batch_size=1 (like uniform adaptive)."
                )
            result = self._assemble_native_adaptive_result(first_rd, self.transform)
            return _copy_transport_control_columns(result, batch)

        input_arr = []
        target_arr = []
        target_src_arr = []
        target_src_paths_arr = []
        centers_arr = []
        levels_arr = []
        leaf_support_arr = []
        root_support_arr = []

        for i in range(batch_size):
            curr_input_paths = batch["input_paths"][i]
            curr_target_paths = batch["target_paths"][i]

            return_dict = first_rd if i == 0 else self.loader(curr_input_paths, curr_target_paths)

            input_arr.append(return_dict["input"]) # each (T, C, H, W) or (T, N, C, H, W)
            target_arr.append(return_dict["target"])
            target_src_arr.append(return_dict["target_src"] if "target_src" in return_dict else []) # target_src only applicable if adaptive
            target_src_paths_arr.append(return_dict.get("target_src_paths"))
            meta = return_dict.get("meta")
            centers_arr.append(meta["centers"] if meta is not None else []) # each (N, 3) or []
            levels_arr.append(meta["levels"] if meta is not None else []) # each (N,) or []
            leaf_support_arr.append(
                meta.get("leaf_support_xy") if meta is not None else None
            )
            root_support_arr.append(
                meta.get("root_support_xy") if meta is not None else None
            )
            
        # 4. Stack batch dimension (B)
        
        # Regular (B, T, C, H, W) -> (B, C, T, H, W)
        # Adaptive (1, T, N, C, H, W) -> (1, N, C, T, H, W)
        centers_np = None
        levels_np = None
        targets_src_np = None
        if input_arr[0].ndim == 4:
            inputs_np = np.stack(input_arr).transpose(0, 2, 1, 3, 4) # (B, T, C, H, W) -> (B, C, T, H, W)
            targets_np = np.stack(target_arr).transpose(0, 2, 1, 3, 4)
        elif input_arr[0].ndim == 5:
            if batch_size != 1:
                raise ValueError(f"Adaptive shall have batch size 1, but got {batch_size}")
            inputs_np = np.stack(input_arr).transpose(0, 2, 3, 1, 4, 5)
            targets_np = np.stack(target_arr).transpose(0, 2, 3, 1, 4, 5)
            centers_np = np.stack(centers_arr) if centers_arr[0] is not None else None
            levels_np = np.stack(levels_arr) if levels_arr[0] is not None else None
            targets_src_np = np.stack(target_src_arr) if target_src_arr[0] is not None else None
            target_src_paths_np = None
            if target_src_paths_arr[0] is not None:
                target_src_paths_np = np.empty(1, dtype=object)
                target_src_paths_np[0] = target_src_paths_arr[0]  # list of strings
        else:
            raise ValueError(f"Unexpected input dimensions: {input_arr[0].shape}")
        
        if self.transform:
            inputs_np = self.transform(inputs_np)
            targets_np = self.transform(targets_np)
            if targets_src_np is not None:
                # target_src is (1, T, C, H, W) — different layout from input/target,
                # needs the T↔C transpose for NormalizeArray.
                targets_src_np = apply_transform_src(self.transform, targets_src_np)

        result = {
            "input": inputs_np,
            "target": targets_np,
        }
        if meta is not None:
            result["centers"] = centers_np
            result["levels"] = levels_np
            result["xmin"] = np.array([meta["domain"]["xmin"]])
            result["xmax"] = np.array([meta["domain"]["xmax"]])
            result["ymin"] = np.array([meta["domain"]["ymin"]])
            result["ymax"] = np.array([meta["domain"]["ymax"]])
            result["max_level_idx"] = np.array([meta["domain"]["max_level_idx"]])
            result["tile_width"] = np.array([meta["domain"]["tile_width"]])
            result["tile_height"] = np.array([meta["domain"]["tile_height"]])
            if leaf_support_arr[0] is not None:
                result["leaf_support_xy"] = np.stack(leaf_support_arr)  # (B, N, 2)
                result["root_support_xy"] = np.stack(root_support_arr)  # (B, 2)

        if input_arr[0].ndim == 5:
            if target_src_arr[0] is not None:
                result["target_src"] = targets_src_np
            if target_src_paths_np is not None:
                result["target_src_paths"] = target_src_paths_np

        return _copy_transport_control_columns(result, batch)


class SampleMapper:
    """Attach cached reference topology before existing fixed-window batch assembly.

    Used by: pipeline.apply_two_stage_topology_pipeline (Ray map_batches) and
    SampleMapperFactory (worker-local construction).
    """

    def __init__(self, mapper, options):
        self.mapper = mapper
        self.options = options
        self._store = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_store"] = None
        return state

    def __call__(self, batch):
        """Resolve one reference/window and delegate payload loading to the cached mapper."""
        from wamrvit.dataloader.topology_store import TopologyStore

        profiling = bool(
            getattr(getattr(self.mapper, "fill_mapper", None), "profile_data_transport", False)
        )
        started = time.perf_counter() if profiling else None
        batch = dict(batch)
        # The adaptive mapping contract is one reference topology per batch.
        if len(batch["input_paths"]) != 1:
            raise ValueError("Reference mapping requires map_batch_size=1.")
        if self._store is None:
            self._store = TopologyStore(self.options["store_path"])
        columns = self._store.get(str(batch["topology_key"][0]))
        for name, value in columns.items():
            batch[name] = value[np.newaxis]
        lookup_s = time.perf_counter() - started if profiling else None
        result = self.mapper(batch)
        if profiling:
            result["_transport_reference_lookup_s"] = np.asarray([lookup_s], dtype=np.float64)
            result["_transport_map_total_s"] = np.asarray(
                [time.perf_counter() - started], dtype=np.float64
            )
        return result

    def close(self):
        """Release mapper-owned readers after transport executors have joined."""
        if self._store is not None:
            self._store.close()
            self._store = None
        close = getattr(self.mapper, "close", None)
        if callable(close):
            close()


class SampleMapperFactory:
    """Wrap a bounded loader factory without losing limits or transport context.

    Used by: TrainingBatchSource._prepare_worker_stream.
    """

    def __init__(self, factory, options):
        self.factory = factory
        self.options = copy.copy(options)
        self.requires_transport_context = getattr(
            factory, "requires_transport_context", False
        )

    def __call__(self):
        """Construct a mapper inside the streaming loader's thread/process."""
        return SampleMapper(self.factory(), self.options)

    def with_output_limit(self, max_output_bytes):
        """Propagate StreamingPayloadIterator's pre-allocation bound to the inner factory."""
        return SampleMapperFactory(self.factory.with_output_limit(max_output_bytes), self.options)


class CachedSeq2SeqMapper:
    """
    Stage 2 Ray Data mapper for the --cache_topology pipeline.

    Reads TreeDiff columns from the materialized Stage 1 output, deserializes
    the topology, calls FillAndExportMapper, and assembles the final batch in
    the same format as Seq2SeqMapper.

    Used by: SampleMapper in the Ray pipeline and streaming mapper factories.
    """

    def __init__(
        self,
        fill_mapper: Callable,
        transform: Optional[Callable] = None,
    ):
        self.fill_mapper = fill_mapper
        self.transform = transform

    @staticmethod
    def _deserialize_diff(batch: Dict[str, np.ndarray]) -> "TreeDiff":
        return TreeDiff(
            xmin=float(batch["topo_xmin"][0]),
            xmax=float(batch["topo_xmax"][0]),
            ymin=float(batch["topo_ymin"][0]),
            ymax=float(batch["topo_ymax"][0]),
            nx_tiles=int(batch["topo_nx_tiles"][0]),
            ny_tiles=int(batch["topo_ny_tiles"][0]),
            max_level_idx=int(batch["topo_max_level_idx"][0]),
            channels=int(batch["topo_channels"][0]),
            ids=batch["topo_ids"][0],
            tuples=batch["topo_tuples"][0],
            values=None,
            ops=batch["topo_ops"][0] if "topo_ops" in batch else None,
        )

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        profiling = bool(
            getattr(self.fill_mapper, "profile_data_transport", False)
        )
        map_started = time.perf_counter() if profiling else None
        deserialize_started = time.perf_counter() if profiling else None
        diff = self._deserialize_diff(batch)
        if profiling:
            deserialize_s = time.perf_counter() - deserialize_started

        input_paths = batch["input_paths"][0]
        target_paths = batch["target_paths"][0]
        set_transport_context = getattr(
            self.fill_mapper, "set_transport_context", None
        )
        if callable(set_transport_context):
            context = {}
            for output_name, batch_name in (
                ("epoch", "_stream_epoch"),
                ("rank_sequence", "_stream_rank_sequence"),
                ("sample_index", "sample_index"),
                ("traj_idx", "traj_idx"),
                ("frame_idx", "frame_idx"),
            ):
                if batch_name in batch:
                    context[output_name] = int(
                        np.asarray(batch[batch_name]).reshape(-1)[0]
                    )
            set_transport_context(context)

        fill_started = time.perf_counter() if profiling else None
        return_dict = self.fill_mapper(input_paths, target_paths, diff)
        if profiling:
            fill_s = time.perf_counter() - fill_started
            profile = return_dict.pop("_transport_profile")
            assemble_started = time.perf_counter()
        result = Seq2SeqMapper._assemble_adaptive_result(return_dict, self.transform)
        result = _copy_transport_control_columns(result, batch)
        if profiling:
            profile.update(
                {
                    "deserialize_s": deserialize_s,
                    "fill_mapper_s": fill_s,
                    "assemble_transform_s": time.perf_counter()
                    - assemble_started,
                    "map_total_s": time.perf_counter() - map_started,
                    "payload_bytes": float(
                        sum(
                            value.nbytes
                            for value in result.values()
                            if isinstance(value, np.ndarray)
                        )
                    ),
                }
            )
            for name, value in profile.items():
                result[f"_transport_{name}"] = np.asarray(
                    [value], dtype=np.float64
                )
        return result

    def close(self) -> None:
        close = getattr(self.fill_mapper, "close", None)
        if callable(close):
            close()
