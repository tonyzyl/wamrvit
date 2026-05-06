import os
import datetime
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import warnings
from typing import Any, Optional, List, Callable, Dict, Union

from wamrvit.quad.quadtree import TreeDiff
from wamrvit.dataloader.transform import apply_transform_src


def generate_multi_trajectory_windows(
    file_path_list: List[List[str]],  # List of Trajectories
    input_seq_len: int,
    return_seq_len: int,
    interval_between_pred: int = 1,
    sampling_interval: int = 1
) -> List[Dict[str, Any]]:
    """
    Iterates over multiple trajectories, subsamples them, validates length,
    and generates training windows.

    Args:
        file_path_list: List of lists, ALREADY SORTED by trajectory and time.
        input_seq_len: Number of input frames.
        return_seq_len: Number of target frames.
        interval_between_pred: Step size between frames in a sequence.
        sampling_interval: Step size to subsample the raw file list.
    """
    all_windows = []
    
    # 1. Iterate over each trajectory
    for traj_idx, file_paths in enumerate(file_path_list):
        
        # A. Apply Downsampling (Outer Stride)
        if sampling_interval > 1:
            files_to_use = file_paths[::sampling_interval]
        else:
            files_to_use = file_paths

        # B. Calculate required span
        total_seq_frames = input_seq_len + return_seq_len
        # span = (frames - 1) * stride + 1
        seq_span = (total_seq_frames - 1) * interval_between_pred + 1

        # C. Validation: Is trajectory long enough?
        if len(files_to_use) < seq_span:
            # Skip this trajectory entirely
            warnings.warn(f"Trajectory {traj_idx}, starting with {files_to_use[0]}, "
                          f"is too short after subsampling. Required: {seq_span}, "
                          f"Found: {len(files_to_use)}. Skipping.")
            continue

        # D. Generate Windows for this specific trajectory
        # We can reuse the logic from the single-trajectory generator here
        # or inline it for clarity. Here is the inlined logic:
        
        num_valid_starts = len(files_to_use) - seq_span + 1
        
        for idx in range(num_valid_starts):
            # Calculate indices RELATIVE to this trajectory's file list
            input_end_idx = idx + (input_seq_len - 1) * interval_between_pred
            pred_start_idx = input_end_idx + interval_between_pred
            
            input_indices = range(idx, input_end_idx + 1, interval_between_pred)
            
            target_stop_idx = pred_start_idx + (return_seq_len - 1) * interval_between_pred + 1
            target_indices = range(pred_start_idx, target_stop_idx, interval_between_pred)
            
            # Create the Manifest Row
            row = {
                "input_paths": [files_to_use[i] for i in input_indices],
                "target_paths": [files_to_use[i] for i in target_indices],
                "traj_idx": traj_idx, 
                "frame_idx": idx
            }
            all_windows.append(row)

    return all_windows

def generate_ar_windows(
    file_path_list: List[str],
    input_seq_len: int,
    return_seq_len: int,
    interval_between_pred: int = 1,
    sampling_interval: int = 1
) -> List[Dict[str, Any]]:
    """
    Creates a list of dictionaries. Each dictionary defines ONE training sample
    (a specific set of input files and target files).

    Args:
        file_path_list ('List[str]'): 
            List of file paths to load from (sorted).
    """
    
    full_seq_len = (input_seq_len + return_seq_len - 1) * interval_between_pred + 1

    if sampling_interval > 1:
        files_to_use = file_path_list[::sampling_interval]
    else:
        files_to_use = file_path_list
        
    total_files = len(files_to_use)
    
    windows = []
    
    for idx in range(total_files - full_seq_len + 1):
        # Iterate through valid start indices
        input_end_idx = idx + (input_seq_len - 1) * interval_between_pred
        pred_start_idx = input_end_idx + interval_between_pred
        
        input_indices = range(idx, input_end_idx + 1, interval_between_pred)
        
        target_stop_idx = pred_start_idx + (return_seq_len - 1) * interval_between_pred + 1
        target_indices = range(pred_start_idx, target_stop_idx, interval_between_pred)
        
        # store the actual PATHS
        row = {
            "input_paths": [files_to_use[i] for i in input_indices],
            "target_paths": [files_to_use[i] for i in target_indices],
            "traj_idx": 0,
            "frame_idx": idx 
        }
        windows.append(row)
        
    return windows


class Seq2SeqMapper:
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

        input_data = return_dict["input"]   # (T_in, N, C, H, W)
        target_data = return_dict["target"] # (T_out, N, C, H, W)

        # (1, T, N, C, H, W) -> (1, N, C, T, H, W)
        inputs_np = np.stack([input_data]).transpose(0, 2, 3, 1, 4, 5)
        targets_np = np.stack([target_data]).transpose(0, 2, 3, 1, 4, 5)

        centers_np = np.stack([meta["centers"]]) if meta is not None else None
        levels_np = np.stack([meta["levels"]]) if meta is not None else None

        targets_src_np = None
        target_src_paths_np = None
        if return_dict.get("target_src") is not None:
            # target_src is (T, C, H, W) — no N dim, so no 6D transpose.
            targets_src_np = np.stack([return_dict["target_src"]])  # (1, T, C, H, W)
        if return_dict.get("target_src_paths") is not None:
            target_src_paths_np = np.empty(1, dtype=object)
            target_src_paths_np[0] = return_dict["target_src_paths"]

        if transform:
            inputs_np = transform(inputs_np)
            targets_np = transform(targets_np)
            if targets_src_np is not None:
                # target_src is (1, T, C, H, W) — different layout from input/target,
                # needs the T↔C transpose dance for NormalizeArray.
                targets_src_np = apply_transform_src(transform, targets_src_np)

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
            assert batch_size == 1, "Native value_storage requires batch_size=1 (like uniform adaptive)."
            return self._assemble_native_adaptive_result(first_rd, self.transform)

        input_arr = []
        target_arr = []
        target_src_arr = []
        target_src_paths_arr = []
        centers_arr = []
        levels_arr = []

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
            assert batch_size == 1, f"Adaptive shall have batch size 1, but got {batch_size}"
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

        if input_arr[0].ndim == 5:
            if target_src_arr[0] is not None:
                result["target_src"] = targets_src_np
            if target_src_paths_np is not None:
                result["target_src_paths"] = target_src_paths_np

        return result


class CachedSeq2SeqMapper:
    """
    Stage 2 Ray Data mapper for the --cache_topology pipeline.

    Reads TreeDiff columns from the materialized Stage 1 output, deserializes
    the topology, calls FillAndExportMapper, and assembles the final batch in
    the same format as Seq2SeqMapper.
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
        diff = self._deserialize_diff(batch)

        input_paths = batch["input_paths"][0]
        target_paths = batch["target_paths"][0]

        return_dict = self.fill_mapper(input_paths, target_paths, diff)

        return Seq2SeqMapper._assemble_adaptive_result(return_dict, self.transform)
