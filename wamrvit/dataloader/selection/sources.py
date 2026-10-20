"""Source discovery, identity, permitted regions and dataset-owned clocks.

Used by: training/offline startup, fixed-window construction and source metadata.
Sources establish permitted frame regions and standalone dataset clocks; they do
not load field payloads or persist topology.
"""

from __future__ import annotations

import glob
import os
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, List, Tuple

import h5py
import numpy as np


def parse_hdf5_virtual_path(virtual_path: str) -> Tuple[str, int, int]:
    """Parse 'filepath:::traj_idx:::frame_idx' into (filepath, traj_idx, frame_idx)."""
    parts = virtual_path.rsplit(":::", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid HDF5 virtual path: {virtual_path!r}. Expected 'filepath:::traj_idx:::frame_idx'.")
    return parts[0], int(parts[1]), int(parts[2])


def parse_frame_indices(
    file_paths: List[str],
    filename_pattern: str = r".*id(\d+).*idx(\d+).*\.npz",
) -> np.ndarray:
    """Extract the frame index (2nd capture group) from each path in a single trajectory."""
    pattern = re.compile(filename_pattern)
    ids = []
    for p in file_paths:
        m = pattern.search(p)
        if m is None:
            raise ValueError(f"Path {p!r} does not match filename_pattern {filename_pattern!r}")
        ids.append(int(m.group(2)))
    return np.asarray(ids, dtype=np.int64)


def group_and_sort_files(
    file_list: List[str],
    filename_pattern: str = r".*id(\d+).*idx(\d+).*\.npz"
) -> List[List[str]]:
    """
    Parses a flat list of filenames, groups them by Simulation ID,
    and sorts them by Timestamp ID.

    Args:
        file_list: A flat list of file paths (strings).
        filename_pattern: Regex to extract (sim_id, timestamp_id).
                          Default matches: ...idXXXX...idxXXXX.npz

    Returns:
        A list of lists. Each inner list represents one distinct trajectory (sim_id),
        sorted chronologically by timestamp_id. The outer list is sorted by sim_id.
    """
    pattern = re.compile(filename_pattern)

    # Dictionary to group files: { sim_id: [(time_idx, filepath), ...] }
    grouped_files = defaultdict(list)

    print(f"Grouping {len(file_list)} files...")

    for filepath in file_list:
        match = pattern.search(filepath)
        if match:
            sim_id = int(match.group(1))
            time_idx = int(match.group(2))
            # Store tuple to avoid re-parsing during sort
            grouped_files[sim_id].append((time_idx, filepath))
        else:
            # Optional: Log warning for files that don't match pattern
            # print(f"Warning: Skipping file {filepath} (no regex match)")
            pass

    # Sort the Group IDs (Simulation IDs)
    sorted_sim_ids = sorted(grouped_files.keys())

    final_sorted_list = []

    for sim_id in sorted_sim_ids:
        # Get the list of (time, path) tuples for this simulation
        trajectories = grouped_files[sim_id]

        # Sort by time_idx (the first element of the tuple)
        trajectories.sort(key=lambda x: x[0])

        # Strip the time_idx back out, keeping only the filepath
        sorted_paths = [path for _, path in trajectories]

        final_sorted_list.append(sorted_paths)

    print(f"Found {len(final_sorted_list)} unique trajectories.")
    return final_sorted_list

class single_traj_parser:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, glob_pattern: str):
        return sorted(glob.glob(glob_pattern))

class multi_traj_under_single_dir_parser:
    def __init__(
        self,
        filename_pattern: str,
        trajectory_start: int = 0,
        trajectory_stop: int | None = None,
        *args,
        **kwargs,
    ):
        if trajectory_start < 0:
            raise ValueError("trajectory_start must be nonnegative.")
        if trajectory_stop is not None and trajectory_stop < trajectory_start:
            raise ValueError("trajectory_stop must not precede trajectory_start.")
        self.filename_pattern = filename_pattern
        self.trajectory_start = trajectory_start
        self.trajectory_stop = trajectory_stop

    def __call__(self, glob_pattern: str):
        trajectories = group_and_sort_files(
            glob.glob(glob_pattern), filename_pattern=self.filename_pattern
        )
        if self.trajectory_stop is not None and self.trajectory_stop > len(trajectories):
            raise ValueError(
                f"Requested trajectory_stop={self.trajectory_stop}, but only "
                f"{len(trajectories)} trajectories were found."
            )
        return trajectories[self.trajectory_start : self.trajectory_stop]


class multi_trajectory_glob_parser:
    """Resolve an explicit ordered list of trajectory globs under one root."""

    def __init__(self, trajectory_globs: Sequence[str], *args, **kwargs):
        if not trajectory_globs:
            raise ValueError("trajectory_globs must contain at least one pattern.")
        self.trajectory_globs = tuple(str(pattern) for pattern in trajectory_globs)

    def __call__(self, root_path: str):
        trajectories = []
        seen_paths = set()
        for pattern in self.trajectory_globs:
            resolved_pattern = (
                pattern if os.path.isabs(pattern) else os.path.join(root_path, pattern)
            )
            paths = sorted(glob.glob(resolved_pattern))
            if not paths:
                raise FileNotFoundError(
                    f"Trajectory glob {resolved_pattern!r} matched no paths."
                )
            duplicates = seen_paths.intersection(paths)
            if duplicates:
                raise ValueError(
                    f"Trajectory globs overlap at {sorted(duplicates)[:3]!r}."
                )
            seen_paths.update(paths)
            trajectories.append(paths)

        print(
            f"Found {len(trajectories)} trajectories with "
            f"{sum(len(paths) for paths in trajectories)} frames total."
        )
        return trajectories


class hdf5_multi_traj_parser:
    """
    File parser for HDF5 files that contain multiple trajectories and timesteps.

    Each HDF5 file stores data with shape (n_trajectories, n_timesteps, ...).
    This parser generates virtual paths of the form 'filepath:::traj_idx:::frame_idx'
    and groups them into List[List[str]] (one inner list per trajectory).
    """
    def __init__(self, field_group: str = "t0_fields", shape_field: str = "density", *args, **kwargs):
        self.field_group = field_group
        self.shape_field = shape_field

    def __call__(self, glob_pattern: str):
        hdf5_files = sorted(glob.glob(glob_pattern))
        all_trajectories = []

        for filepath in hdf5_files:
            with h5py.File(filepath, "r") as f:
                shape = f[self.field_group][self.shape_field].shape
                n_traj, n_timesteps = shape[0], shape[1]

            for traj_idx in range(n_traj):
                frames = [
                    f"{filepath}:::{traj_idx}:::{frame_idx}"
                    for frame_idx in range(n_timesteps)
                ]
                all_trajectories.append(frames)

        print(f"Found {len(hdf5_files)} HDF5 files, {len(all_trajectories)} trajectories total.")
        return all_trajectories


ROLES = ("train", "val", "test")


def canonical_source(path: str) -> str:
    """Canonical file/plotfile identity, including HDF5 virtual slices.

    Used by: validate_source_partitions and topology_store.source_key.
    """
    physical, *suffix = str(path).split(":::")
    return ":::".join([os.path.realpath(physical), *suffix])


@dataclass(frozen=True)
class SourceSegment:
    """Ordered trajectory region retaining original source indices.

    Used by: source_segments, explicit partition callers and window builders.
    """

    paths: tuple[str, ...]
    indices: tuple[int, ...]
    trajectory_id: str | None = None

    def __post_init__(self):
        if len(self.paths) != len(self.indices):
            raise ValueError("SourceSegment paths and indices must align one-to-one.")
        if any(isinstance(i, bool) or int(i) != i for i in self.indices):
            raise ValueError("SourceSegment indices must be integers.")
        if any(b <= a for a, b in zip(self.indices, self.indices[1:])):
            raise ValueError("SourceSegment indices must increase strictly.")

    def slice(self, start, stop):
        """Retain original indices when source selection or partitioning cuts a region."""
        return SourceSegment(self.paths[start:stop], self.indices[start:stop], self.trajectory_id)


def source_segments(paths, params=None):
    """Normalize either parsed layout, applying source selection once.

    Used by: prepare_source_partitions and explicit source-region callers.
    """
    params = params or {}
    if not paths:
        return []
    trajectories = [paths] if isinstance(paths[0], str) else paths
    stride = params.get("sampling_interval", 1)
    if isinstance(stride, bool) or int(stride) != stride or stride < 1:
        raise ValueError("sampling_interval must be a positive integer.")
    stride = int(stride)
    start = float(params.get("trajectory_start_fraction", 0))
    stop = float(params.get("trajectory_stop_fraction", 1))
    if not 0 <= start < stop <= 1:
        raise ValueError("Invalid trajectory selection fractions.")
    names = params.get("trajectory_ids")
    if names is not None and len(names) != len(trajectories):
        raise ValueError("trajectory_ids must align with the parsed trajectories.")
    segments = []
    for trajectory_index, trajectory in enumerate(trajectories):
        indices = (
            parse_frame_indices(trajectory, params["filename_pattern"])
            if params.get("filename_pattern")
            else np.arange(len(trajectory))
        )
        selected = tuple(map(str, trajectory[::stride]))
        indices = tuple(map(int, indices[::stride]))
        segment = SourceSegment(
            selected,
            indices,
            str(names[trajectory_index]) if names is not None else None,
        )
        lo, hi = int(len(selected) * start), int(len(selected) * stop)
        if hi > lo:
            segments.append(segment.slice(lo, hi))
    return segments


def validate_source_partitions(partitions):
    """Validate caller-owned boundaries without choosing or changing a split.

    Used by: source partition preparation and fixed-window builders.
    """
    unknown = set(partitions) - set(ROLES)
    if unknown:
        raise ValueError(f"Unknown source partition roles: {sorted(unknown)}.")
    result = {role: list(partitions.get(role, [])) for role in ROLES}
    seen = set()
    for role, segments in result.items():
        for segment in segments:
            if not isinstance(segment, SourceSegment):
                raise TypeError("Source partitions must contain SourceSegment instances.")
            identities = [canonical_source(path) for path in segment.paths]
            if len(set(identities)) != len(identities) or seen.intersection(identities):
                raise ValueError(
                    "Source frames overlap across segments or train/val/test partitions."
                )
            seen.update(identities)
    if not any(segment.paths for segment in result["train"]):
        raise ValueError("The training source partition is empty.")
    return result


def training_window_partitions(segments, config, train_fraction):
    """Preserve the fixed training-window prefix, then freeze disjoint source regions.

    Used by: prepare_source_partitions when no explicit role sources are supplied.
    The reference recipe uses the configured fixed intervals and training PF depth,
    never an evaluation horizon. Validation begins after each training frontier;
    overlapping validation windows are dropped, not training windows.
    """
    from wamrvit.dataloader.selection.windows import fixed_windows_for_segment

    if not 0 < train_fraction <= 1:
        raise ValueError("train.split_ratio must be in (0, 1].")
    if train_fraction == 1:
        # No percentage holdout: retain the entire declared training source region.
        return validate_source_partitions({"train": segments, "val": [], "test": []})
    pf = config["train"].get("num_push_forward_steps", 1)
    if isinstance(pf, bool) or not isinstance(pf, int) or pf < 1:
        raise ValueError("train.num_push_forward_steps must be a positive integer.")
    # Only window endpoints are needed for the prefix boundary. Do not retain a
    # second dataset-sized list of input/target descriptors during preparation.
    endpoints = []
    for segment in segments:
        positions = {path: i for i, path in enumerate(segment.paths)}
        endpoints.append(
            [
                positions[row["target_paths"][-1]]
                for row in fixed_windows_for_segment(config["window_generator"], segment, pf)
            ]
        )
    remaining = int(sum(map(len, endpoints)) * train_fraction)
    if not remaining:
        raise ValueError("The fixed training-window reference prefix is empty.")
    result = {role: [] for role in ROLES}
    for segment, ends in zip(segments, endpoints):
        take = min(remaining, len(ends))
        remaining -= take
        end = ends[take - 1] if take else -1
        if end >= 0:
            result["train"].append(segment.slice(0, end + 1))
        if end + 1 < len(segment.paths):
            result["val"].append(segment.slice(end + 1, len(segment.paths)))
    return validate_source_partitions(result)


def validate_source_selection_config(config):
    """Reject retired split controls at source and window preparation entrypoints."""
    if "split_basis" in config.get("train", {}):
        raise ValueError(
            "train.split_basis is removed; the manuscript training recipe is always used."
        )
    retired = {"test_fraction", "source_partitioner"}.intersection(config.get("data", {}))
    if retired:
        raise ValueError(
            f"Retired data options {sorted(retired)}; use explicit val/test sources or "
            "train.split_ratio for the train/validation fallback."
        )


def prepare_source_partitions(
    config: dict[str, Any],
    paths,
    *,
    val_paths=None,
    test_paths=None,
):
    """Select source segments before any fixed windows are constructed.

    Used by: train.main and tools.build_topology_cache.main.
    Explicit val/test sources override the train/validation fallback. Otherwise,
    the manuscript's fixed training-window prefix defines the source regions.
    Direct callers may instead supply partitions to prepare_window_splits.
    """
    from wamrvit.utils import instantiate_from_config

    validate_source_selection_config(config)
    data = config["data"]
    explicit = val_paths is not None or test_paths is not None
    explicit = explicit or bool(data.get("val_glob_pattern") or data.get("test_glob_pattern"))
    if explicit:
        parser = instantiate_from_config(config["file_parser"])
        if val_paths is None and data.get("val_glob_pattern"):
            val_paths = parser(data["val_glob_pattern"])
        if test_paths is None and data.get("test_glob_pattern"):
            test_paths = parser(data["test_glob_pattern"])
    params = config["window_generator"]["params"]
    selected = source_segments(paths, params)
    if explicit:
        result = validate_source_partitions(
            {
                "train": selected,
                "val": source_segments(val_paths or [], params),
                "test": source_segments(test_paths or [], params),
            }
        )
    else:
        ratio = float(config["train"].get("split_ratio", 0.8))
        result = training_window_partitions(selected, config, ratio)
    print(
        "Source partitions: "
        + ", ".join(
            f"{role}={sum(len(segment.paths) for segment in segments)} frames"
            for role, segments in result.items()
        )
    )
    return result


def index_times(paths, indices, *, step=1.0):
    """Convert original indices, not selected positions; used by source_time_reader."""
    return np.asarray(indices, dtype=np.float64) * step


def amrex_times(paths, indices=None):
    """Read native Header clocks; used by adaptive/regular loader time adapters."""
    from wamrvit.quad.amrex_to_qt import read_plotfile_header

    times = []
    for path in paths:
        header = read_plotfile_header(str(path))
        if "time" not in header:
            raise ValueError(f"Missing native timestamp in AMReX source {path}.")
        times.append(header["time"])
    return np.asarray(times, dtype=np.float64)


def source_time_reader(config):
    """Resolve and validate the dataset clock without reading source files.

    Used by: standalone source-metadata readers.
    """
    options = config.get("data", {}).get("time", {})
    source = options.get("source", "index")
    unknown = set(options) - {"source", "step"}
    if unknown:
        raise ValueError(f"Unsupported data.time options: {sorted(unknown)}.")
    if source == "index":
        step = options.get("step", 1.0)
        if (
            isinstance(step, bool)
            or not isinstance(step, (int, float))
            or not np.isfinite(step)
            or step <= 0
        ):
            raise ValueError("data.time.step must be finite and positive.")
        return partial(index_times, step=float(step))
    if source != "native":
        raise ValueError("data.time.source must be 'index' or 'native'.")
    if "step" in options:
        raise ValueError("data.time.step applies only to index time; remove it for native time.")
    from wamrvit.utils import get_obj_from_str

    loader_config = config.get("file_loader", {})
    target = loader_config.get("target")
    loader_class = get_obj_from_str(target) if target else None
    adapter = getattr(loader_class, "native_time_reader", None)
    if not callable(adapter):
        raise ValueError(f"Native timestamps are not supported by source loader {target!r}.")
    return adapter(dict(loader_config.get("params", {})))


def read_source_times(reader, paths, indices):
    """Materialize one clock per frame for standalone source metadata."""
    times = np.asarray(reader(paths, indices), dtype=np.float64)
    if times.shape != (len(paths),):
        raise ValueError("Source time adapter must return one timestamp per frame.")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Source timestamps must be finite and strictly increasing.")
    return times


def adjacent_time_gaps(times):
    """Compute adjacent durations from selected source timestamps."""
    gaps = np.diff(times).astype(np.float32)
    if not np.isfinite(gaps).all() or np.any(gaps <= 0):
        raise ValueError("Selected time gaps must remain positive and finite in FP32.")
    return gaps
