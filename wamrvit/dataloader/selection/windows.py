"""Fixed-cadence window construction inside caller-owned source regions.

Used by: training/offline startup and rollout. Gaps are snapshot-index increments,
not physical durations. selection/sources.py owns partitions and clocks; this
module selects complete windows inside those regions without reading payloads.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from wamrvit.dataloader.selection.sources import (
    parse_frame_indices,
    validate_source_partitions,
    validate_source_selection_config,
)


def generate_multi_trajectory_windows(
    file_path_list: List[List[str]],  # List of Trajectories
    input_seq_len: int,
    return_seq_len: int,
    interval_between_pred: int = 1,
    sampling_interval: int = 1,
    filename_pattern: Optional[str] = None,
    trajectory_start_fraction: float = 0.0,
    trajectory_stop_fraction: float = 1.0,
    trajectory_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Iterates over multiple trajectories, subsamples them, validates length,
    and generates training windows.

    Configured by: window_generator.target; called by prepare_window_splits and rollout.

    Args:
        file_path_list: List of lists, ALREADY SORTED by trajectory and time.
        input_seq_len: Number of input frames.
        return_seq_len: Number of target frames.
        interval_between_pred: Step size between frames in a sequence.
        sampling_interval: Step size to subsample the raw file list.
        filename_pattern: If set, parse frame indices from filenames and skip
            windows whose selected frames are not strided-contiguous. Use to
            reject windows that span mid-trajectory gaps (e.g., PLI dataset).
            Defaults to None, preserving the positional-index behavior for
            AMReX/HDF5/legacy callers.
        trajectory_start_fraction: Inclusive temporal fraction selected from
            every trajectory after subsampling.
        trajectory_stop_fraction: Exclusive temporal fraction selected from
            every trajectory after subsampling.
        trajectory_ids: Optional stable names aligned with ``file_path_list``.
            Names are copied into window descriptors so topology-cache keys
            cannot alias trajectories that share frame basenames.
    """
    if not 0.0 <= trajectory_start_fraction < trajectory_stop_fraction <= 1.0:
        raise ValueError(
            "trajectory fractions must satisfy "
            "0 <= trajectory_start_fraction < trajectory_stop_fraction <= 1."
        )
    if trajectory_ids is not None and len(trajectory_ids) != len(file_path_list):
        raise ValueError("trajectory_ids must align one-to-one with file_path_list.")

    all_windows = []
    expected_step = interval_between_pred * sampling_interval

    # 1. Iterate over each trajectory
    for traj_idx, file_paths in enumerate(file_path_list):

        # A. Apply Downsampling (Outer Stride), then the same temporal slice to
        # every trajectory. Slicing before windowing prevents targets crossing
        # the configured temporal boundary.
        if sampling_interval > 1:
            sampled_paths = file_paths[::sampling_interval]
        else:
            sampled_paths = file_paths
        slice_start = int(len(sampled_paths) * trajectory_start_fraction)
        slice_stop = int(len(sampled_paths) * trajectory_stop_fraction)
        if trajectory_stop_fraction == 1.0:
            slice_stop = len(sampled_paths)
        files_to_use = sampled_paths[slice_start:slice_stop]

        # B. Calculate required span
        total_seq_frames = input_seq_len + return_seq_len
        # span = (frames - 1) * stride + 1
        seq_span = (total_seq_frames - 1) * interval_between_pred + 1

        # C. Validation: Is trajectory long enough?
        if len(files_to_use) < seq_span:
            # Skip this trajectory entirely
            start_description = files_to_use[0] if files_to_use else "<empty slice>"
            warnings.warn(f"Trajectory {traj_idx}, starting with {start_description}, "
                          f"is too short after subsampling and slicing. Required: {seq_span}, "
                          f"Found: {len(files_to_use)}. Skipping.")
            continue

        # If gap detection is enabled, parse frame ids once per trajectory.
        frame_ids = parse_frame_indices(files_to_use, filename_pattern) if filename_pattern else None
        n_skipped_due_to_gap = 0

        # D. Generate Windows for this specific trajectory
        # We can reuse the logic from the single-trajectory generator here
        # or inline it for clarity. Here is the inlined logic:

        num_valid_starts = len(files_to_use) - seq_span + 1

        for idx in range(num_valid_starts):
            # Calculate indices RELATIVE to this trajectory's selected file list
            input_end_idx = idx + (input_seq_len - 1) * interval_between_pred
            pred_start_idx = input_end_idx + interval_between_pred

            input_indices = range(idx, input_end_idx + 1, interval_between_pred)

            target_stop_idx = pred_start_idx + (return_seq_len - 1) * interval_between_pred + 1
            target_indices = range(pred_start_idx, target_stop_idx, interval_between_pred)

            if frame_ids is not None:
                window_positions = list(input_indices) + list(target_indices)
                window_ids = frame_ids[window_positions]
                if not np.all(np.diff(window_ids) == expected_step):
                    n_skipped_due_to_gap += 1
                    continue

            # Create the Manifest Row
            row = {
                "input_paths": [files_to_use[i] for i in input_indices],
                "target_paths": [files_to_use[i] for i in target_indices],
                "traj_idx": traj_idx,
                "frame_idx": slice_start + idx,
            }
            if trajectory_ids is not None:
                row["trajectory_id"] = trajectory_ids[traj_idx]
            all_windows.append(row)

        if n_skipped_due_to_gap > 0:
            warnings.warn(f"Trajectory {traj_idx}: skipped {n_skipped_due_to_gap} "
                          f"window(s) that cross missing frames.")

    return all_windows

def generate_ar_windows(
    file_path_list: List[str],
    input_seq_len: int,
    return_seq_len: int,
    interval_between_pred: int = 1,
    sampling_interval: int = 1,
    filename_pattern: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Creates a list of dictionaries. Each dictionary defines ONE training sample
    (a specific set of input files and target files).

    Configured by: window_generator.target; called by prepare_window_splits and rollout.

    Args:
        file_path_list ('List[str]'):
            List of file paths to load from (sorted).
        filename_pattern: If set, parse frame indices from filenames and skip
            windows whose selected frames are not strided-contiguous. Defaults
            to None (positional-index behavior preserved).
    """

    full_seq_len = (input_seq_len + return_seq_len - 1) * interval_between_pred + 1
    expected_step = interval_between_pred * sampling_interval

    if sampling_interval > 1:
        files_to_use = file_path_list[::sampling_interval]
    else:
        files_to_use = file_path_list

    total_files = len(files_to_use)
    frame_ids = parse_frame_indices(files_to_use, filename_pattern) if filename_pattern else None
    n_skipped_due_to_gap = 0

    windows = []

    for idx in range(total_files - full_seq_len + 1):
        # Iterate through valid start indices
        input_end_idx = idx + (input_seq_len - 1) * interval_between_pred
        pred_start_idx = input_end_idx + interval_between_pred

        input_indices = range(idx, input_end_idx + 1, interval_between_pred)

        target_stop_idx = pred_start_idx + (return_seq_len - 1) * interval_between_pred + 1
        target_indices = range(pred_start_idx, target_stop_idx, interval_between_pred)

        if frame_ids is not None:
            window_positions = list(input_indices) + list(target_indices)
            window_ids = frame_ids[window_positions]
            if not np.all(np.diff(window_ids) == expected_step):
                n_skipped_due_to_gap += 1
                continue

        # store the actual PATHS
        row = {
            "input_paths": [files_to_use[i] for i in input_indices],
            "target_paths": [files_to_use[i] for i in target_indices],
            "traj_idx": 0,
            "frame_idx": idx
        }
        windows.append(row)

    if n_skipped_due_to_gap > 0:
        warnings.warn(f"AR trajectory: skipped {n_skipped_due_to_gap} window(s) "
                      f"that cross missing frames.")

    return windows


def fixed_windows_for_segment(window_config, segment, pf_depth):
    """Generate fixed windows inside an already-selected source segment.

    Used by: the manuscript reference partition builder and prepare_window_splits.
    Selection/downsampling is not repeated; original indices own gap checks.
    This helper never reads timestamps or field payloads.
    """
    from wamrvit.utils import instantiate_from_config

    config = copy.deepcopy(window_config)
    params = config.setdefault("params", {})
    params["return_seq_len"] *= pf_depth
    stride = params.pop("sampling_interval", 1)
    expected_gap = stride * params.get("interval_between_pred", 1)
    for name in (
        "trajectory_start_fraction",
        "trajectory_stop_fraction",
        "trajectory_ids",
        "filename_pattern",
    ):
        params.pop(name, None)
    flat = config["target"] == "wamrvit.dataloader.selection.windows.generate_ar_windows"
    params["file_path_list"] = list(segment.paths) if flat else [list(segment.paths)]
    indices = dict(zip(segment.paths, segment.indices))
    for row in instantiate_from_config(config):
        selected = row["input_paths"] + row["target_paths"]
        if not set(selected) <= indices.keys():
            raise ValueError("Window generator selected a frame outside its source segment.")
        if any(indices[b] - indices[a] != expected_gap for a, b in zip(selected, selected[1:])):
            continue
        row["frame_idx"] = indices[row["input_paths"][0]]
        if segment.trajectory_id is not None:
            row["trajectory_id"] = segment.trajectory_id
        yield row


def validate_temporal_config(config, *, topology_cached):
    """Check source and reference-store contracts before dataset construction.

    Used by: train.main.
    """
    from wamrvit.dataloader.topology_store import validate_topology_store_options

    validate_topology_store_options(config.get("topology_store", {}))
    validate_source_selection_config(config)
    if topology_cached and config["model"].get("adaptive", False):
        if config["train"].get("map_batch_size", 1) != 1:
            raise ValueError("The reference topology store requires map_batch_size=1.")


def prepare_window_splits(config, partitions):
    """Generate fixed windows inside caller-owned source partitions.

    Used by: train.main and tools.build_topology_cache.main.
    Returns: (None, windows_by_role), without loading clocks or field payloads.
    No ratio, trajectory selection, or statistical split decision belongs here.
    """
    validate_source_selection_config(config)
    partitions = validate_source_partitions(partitions)
    r = config["window_generator"]["params"]["return_seq_len"]
    if r != config["model"].get("return_seq_len", 1):
        raise ValueError("Window and model return_seq_len must match before PF expansion.")

    pf = config["train"].get("num_push_forward_steps", 1)
    result = {role: [] for role in partitions}
    segment_index = 0
    for role, segments in partitions.items():
        for segment in segments:
            for row in fixed_windows_for_segment(config["window_generator"], segment, pf):
                row["traj_idx"] = segment_index
                result[role].append(row)
            segment_index += 1
        if segments and not result[role]:
            raise ValueError(f"No feasible {role} windows inside its strict source partition.")
    return None, result
