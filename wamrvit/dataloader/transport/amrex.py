"""Bounded, transient Ray object-store cache for decoded AMReX frames."""
from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from wamrvit.dataloader.transport.source_cache import (
    SourceItemAdapter,
    build_ray_source_cache,
    select_bounded_source_subset,
)
from wamrvit.quad.amrex_to_qt import read_plotfile_raw


def _cell_data_bytes(plotfile_path: str) -> int:
    return sum(
        path.stat().st_size
        for path in Path(plotfile_path).glob("Level_*/Cell_D_*")
        if path.is_file()
    )


def estimate_decoded_frame_bytes(
    paths: list[str],
    *,
    selected_fields: int,
    total_fields: int,
    estimate_padding_fraction: float = 0.02,
) -> int:
    """Conservatively estimate selected FP32 bytes from FP64 Cell_D files."""
    if selected_fields <= 0 or total_fields <= 0:
        raise ValueError("AMReX field counts must be positive.")
    if selected_fields > total_fields:
        raise ValueError("selected_fields cannot exceed total_fields.")
    if not math.isfinite(estimate_padding_fraction):
        raise ValueError("estimate_padding_fraction must be finite.")
    if estimate_padding_fraction < 0:
        raise ValueError("estimate_padding_fraction cannot be negative.")
    source_bytes = sum(_cell_data_bytes(path) for path in paths)
    # Cell_D payloads are FP64. The decoded cache stores selected FP32 fields;
    # Optionally pad for grid coordinates and source-layout variation. Runtime
    # loading still checks every measured item against its reservation.
    return int(
        source_bytes
        * selected_fields
        / total_fields
        / 2
        * (1 + estimate_padding_fraction)
    )


def select_bounded_frame_subset(
    paths: list[str],
    *,
    reference_counts: Mapping[str, int],
    selected_fields: int,
    total_fields: int,
    max_cache_bytes: int,
    usable_fraction: float = 0.95,
    estimate_padding_fraction: float = 0.02,
) -> tuple[list[str], dict[str, Any]]:
    """Greedily select a fixed decoded-frame subset by references per byte.

    This provides a dataset-length-bounded cache without a distributed LRU:
    selected frames use Plasma, while uncached frames transparently fall back
    to worker-local reads. The reserve covers per-frame estimate error.
    """
    unique_paths = sorted({os.path.abspath(path) for path in paths})
    normalized_counts = Counter()
    for path, count in reference_counts.items():
        normalized_counts[os.path.abspath(path)] += int(count)
    adapter = AMReXFrameAdapter(
        field_names=[str(index) for index in range(selected_fields)],
        patch_size=1,
        total_fields=total_fields,
        estimate_padding_fraction=estimate_padding_fraction,
    )
    try:
        selected, metrics = select_bounded_source_subset(
            unique_paths,
            reference_counts=normalized_counts,
            adapter=adapter,
            max_cache_bytes=max_cache_bytes,
            usable_fraction=usable_fraction,
        )
    except MemoryError as error:
        raise MemoryError(
            "Decoded AMReX bounded-subset cache cannot fit one estimated frame "
            f"within {max_cache_bytes * usable_fraction / 1024**3:.3f} GiB."
        ) from error
    metrics["candidate_frames"] = metrics.pop("candidate_items")
    metrics["selected_frames"] = metrics.pop("selected_items")
    return selected, metrics


def _decode_frame(
    path: str,
    field_names: list[str],
    patch_size: int | tuple[int, int],
    normalization_mean: np.ndarray | None = None,
    normalization_std: np.ndarray | None = None,
    normalization_eps: float = 0.0,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    started = time.perf_counter()
    grid_info, grid_data, header = read_plotfile_raw(
        path, field_names, patch_size
    )
    normalize_s = 0.0
    if normalization_mean is not None or normalization_std is not None:
        if normalization_mean is None or normalization_std is None:
            raise ValueError("AMReX frame normalization requires both mean and std.")
        if normalization_mean.shape != (grid_data.shape[1],) or normalization_std.shape != (
            grid_data.shape[1],
        ):
            raise ValueError("AMReX frame-normalization channel count mismatch.")
        normalize_started = time.perf_counter()
        stats_shape = (1, grid_data.shape[1], 1, 1)
        np.subtract(grid_data, normalization_mean.reshape(stats_shape), out=grid_data)
        np.divide(
            grid_data,
            normalization_std.reshape(stats_shape) + normalization_eps,
            out=grid_data,
        )
        normalize_s = time.perf_counter() - normalize_started
    decoded = (
        grid_info,
        grid_data,
        np.asarray(header["domain_lo"]),
        np.asarray(header["domain_hi"]),
    )
    decoded_bytes = sum(array.nbytes for array in decoded)
    return decoded, {
        "path": path,
        "decoded_bytes": decoded_bytes,
        "decode_s": time.perf_counter() - started,
        "normalize_s": normalize_s,
    }


class AMReXFrameAdapter(SourceItemAdapter[str]):
    """AMReX decoding behind the format-independent source-cache contract."""

    def __init__(
        self,
        *,
        field_names: list[str],
        patch_size: int | tuple[int, int],
        total_fields: int,
        normalization_mean: np.ndarray | None = None,
        normalization_std: np.ndarray | None = None,
        normalization_eps: float = 0.0,
        estimate_padding_fraction: float = 0.02,
    ) -> None:
        self.field_names = list(field_names)
        self.patch_size = patch_size
        self.total_fields = int(total_fields)
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.normalization_eps = float(normalization_eps)
        self.estimate_padding_fraction = float(estimate_padding_fraction)

    def estimate_size_bytes(self, path: str) -> int:
        return estimate_decoded_frame_bytes(
            [path],
            selected_fields=len(self.field_names),
            total_fields=self.total_fields,
            estimate_padding_fraction=self.estimate_padding_fraction,
        )

    def load(
        self, path: str
    ) -> tuple[Any, int, Mapping[str, float | int]]:
        decoded, metrics = _decode_frame(
            path,
            self.field_names,
            self.patch_size,
            self.normalization_mean,
            self.normalization_std,
            self.normalization_eps,
        )
        return decoded, int(metrics["decoded_bytes"]), {
            "decode_s": float(metrics["decode_s"]),
            "normalize_s": float(metrics["normalize_s"]),
        }


def build_ray_amrex_frame_cache(
    paths: list[str],
    *,
    field_names: list[str],
    patch_size: int | tuple[int, int],
    max_cache_bytes: int,
    concurrency: int = 16,
    normalization_mean: np.ndarray | None = None,
    normalization_std: np.ndarray | None = None,
    normalization_eps: float = 0.0,
    estimate_padding_fraction: float = 0.02,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decode unique frames once and retain their object refs in Plasma.

    The cache is transient and bounded before tasks launch. It stores source
    frames, not topology-specific or window-expanded training samples.
    """
    unique_paths = sorted({os.path.abspath(path) for path in paths})
    if not unique_paths:
        raise ValueError("AMReX frame cache requires at least one path.")
    if (normalization_mean is None) != (normalization_std is None):
        raise ValueError("AMReX frame normalization requires both mean and std.")
    if normalization_mean is not None:
        normalization_mean = np.asarray(normalization_mean, dtype=np.float32)
        normalization_std = np.asarray(normalization_std, dtype=np.float32)
        if normalization_mean.shape != (len(field_names),) or normalization_std.shape != (
            len(field_names),
        ):
            raise ValueError("AMReX frame-normalization channel count mismatch.")

    from wamrvit.quad.amrex_to_qt import read_plotfile_header

    adapter = AMReXFrameAdapter(
        field_names=field_names,
        patch_size=patch_size,
        total_fields=int(read_plotfile_header(unique_paths[0])["ncomp"]),
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        normalization_eps=normalization_eps,
        estimate_padding_fraction=estimate_padding_fraction,
    )
    frame_refs, generic_metrics = build_ray_source_cache(
        unique_paths,
        adapter=adapter,
        max_cache_bytes=max_cache_bytes,
        concurrency=concurrency,
    )
    item_metrics = generic_metrics["item_metrics"]
    decode_metrics = item_metrics.get("decode_s", {})
    normalize_metrics = item_metrics.get("normalize_s", {})
    metrics = {
        "frames": generic_metrics["items"],
        "estimated_bytes": generic_metrics["estimated_bytes"],
        "decoded_bytes": generic_metrics["resident_bytes"],
        "max_cache_bytes": generic_metrics["max_cache_bytes"],
        "reserved_bytes": generic_metrics["reserved_bytes"],
        "concurrency": generic_metrics["concurrency"],
        "decode_task_seconds_sum": float(decode_metrics.get("sum", 0.0)),
        "decode_task_seconds_max": float(decode_metrics.get("max", 0.0)),
        "normalized": normalization_mean is not None,
        "normalize_task_seconds_sum": float(normalize_metrics.get("sum", 0.0)),
        "normalize_task_seconds_max": float(normalize_metrics.get("max", 0.0)),
    }
    return frame_refs, metrics


class AMReXSourceProvider:
    """Prepare AMReX-specific state for generic worker-local streaming."""

    def __init__(self) -> None:
        pass

    def validate(self, config: dict[str, Any], *, topology_cached: bool) -> None:
        train_config = config["train"]
        loader_params = config.get("file_loader", {}).get("params", {})
        direct_export = bool(train_config.get("amrex_direct_export", False))
        direct_options = {
            "amrex_direct_pin_output": "train.amrex_direct_pin_output",
            "amrex_direct_topology_geometry": (
                "train.amrex_direct_topology_geometry"
            ),
            "amrex_direct_raster_sampler": "train.amrex_direct_raster_sampler",
        }
        for option, display_name in direct_options.items():
            if train_config.get(option, False) and not direct_export:
                raise ValueError(f"{display_name} requires train.amrex_direct_export.")

        if direct_export:
            if train_config.get("payload_transport") != "worker_stream" or not topology_cached:
                raise ValueError(
                    "train.amrex_direct_export requires worker_stream with "
                    "--cache_topology."
                )
            if loader_params.get("mode") != "amrex":
                raise ValueError("train.amrex_direct_export requires mode='amrex'.")
            if loader_params.get("value_storage", "uniform") != "uniform":
                raise ValueError(
                    "train.amrex_direct_export requires value_storage='uniform'."
                )

        cache_gib = float(train_config.get("amrex_frame_cache_gib", 0))
        partial = bool(train_config.get("amrex_frame_cache_partial", False))
        usable_fraction = float(
            train_config.get("amrex_frame_cache_usable_fraction", 0.95)
        )
        estimate_padding_fraction = float(
            train_config.get(
                "amrex_frame_cache_estimate_padding_fraction", 0.02
            )
        )
        normalize = bool(train_config.get("amrex_frame_cache_normalize", False))
        source_trace_dir = train_config.get("amrex_source_trace_dir")
        if source_trace_dir:
            if not direct_export:
                raise ValueError(
                    "train.amrex_source_trace_dir requires "
                    "train.amrex_direct_export."
                )
            if not train_config.get("profile_data_transport", False):
                raise ValueError(
                    "train.amrex_source_trace_dir requires "
                    "train.profile_data_transport=true."
                )
            if not os.path.isabs(str(source_trace_dir)):
                raise ValueError(
                    "train.amrex_source_trace_dir must be an absolute path."
                )
        if (direct_export or cache_gib) and train_config.get(
            "streaming_loader_backend", "thread"
        ) != "thread":
            raise ValueError(
                "AMReX direct export and decoded frame caching require "
                "train.streaming_loader_backend='thread'."
            )
        if partial and not cache_gib:
            raise ValueError(
                "train.amrex_frame_cache_partial requires a positive "
                "train.amrex_frame_cache_gib."
            )
        if not 0 < usable_fraction <= 1:
            raise ValueError(
                "train.amrex_frame_cache_usable_fraction must be in (0, 1]."
            )
        if not math.isfinite(estimate_padding_fraction):
            raise ValueError(
                "train.amrex_frame_cache_estimate_padding_fraction must be finite."
            )
        if estimate_padding_fraction < 0:
            raise ValueError(
                "train.amrex_frame_cache_estimate_padding_fraction cannot be "
                "negative."
            )
        if normalize and not cache_gib:
            raise ValueError(
                "train.amrex_frame_cache_normalize requires a positive "
                "train.amrex_frame_cache_gib."
            )
        if normalize and not direct_export:
            raise ValueError(
                "train.amrex_frame_cache_normalize requires "
                "train.amrex_direct_export."
            )
        if normalize and loader_params.get("return_src", False):
            raise ValueError(
                "Normalized AMReX frame caching does not support return_src."
            )
        if cache_gib:
            if train_config.get("payload_transport") != "worker_stream":
                raise ValueError(
                    "Decoded AMReX frame caching requires "
                    "train.payload_transport=worker_stream."
                )
            if not topology_cached:
                raise ValueError(
                    "Decoded AMReX frame caching requires --cache_topology."
                )
            object_store_gib = float(
                config["general"]["ray_object_store_memory"]
            )
            if cache_gib > object_store_gib * 0.85:
                raise MemoryError(
                    f"AMReX frame-cache cap {cache_gib:.3f} GiB exceeds 85% "
                    f"of the {object_store_gib:.3f}-GiB object store."
                )

    def prepare(
        self,
        config: dict[str, Any],
        windows: list[dict[str, Any]],
        *,
        topology_cached: bool,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        self.validate(config, topology_cached=topology_cached)
        train_config = config["train"]
        loader_params = config["file_loader"]["params"]
        fill_params: dict[str, Any] = {
            "direct_amrex_export": bool(
                train_config.get("amrex_direct_export", False)
            ),
            "direct_amrex_pin_output": bool(
                train_config.get("amrex_direct_pin_output", False)
            ),
            "direct_amrex_topology_geometry": bool(
                train_config.get("amrex_direct_topology_geometry", False)
            ),
            "direct_amrex_raster_sampler": bool(
                train_config.get("amrex_direct_raster_sampler", False)
            ),
        }
        cache_gib = float(train_config.get("amrex_frame_cache_gib", 0))
        if not cache_gib:
            return fill_params, None
        usable_fraction = float(
            train_config.get("amrex_frame_cache_usable_fraction", 0.95)
        )
        estimate_padding_fraction = float(
            train_config.get(
                "amrex_frame_cache_estimate_padding_fraction", 0.02
            )
        )

        reference_counts = Counter(
            os.path.abspath(path)
            for window in windows
            for path in window["input_paths"] + window["target_paths"]
        )
        frame_paths = sorted(reference_counts)
        field_names = list(loader_params["field_names"])
        cache_limit_bytes = int(cache_gib * 1024**3)
        normalize = bool(train_config.get("amrex_frame_cache_normalize", False))
        source_normalization = None
        if normalize:
            from wamrvit.dataloader.payload.transforms import NormalizeArray
            from wamrvit.utils import instantiate_from_config

            normalizer = instantiate_from_config(config["transform"])
            if not isinstance(normalizer, NormalizeArray):
                raise ValueError(
                    "train.amrex_frame_cache_normalize requires NormalizeArray."
                )
            source_normalization = {
                "mean": normalizer.mean.copy(),
                "std": normalizer.std.copy(),
                "eps": float(normalizer.eps),
            }

        if train_config.get("amrex_frame_cache_partial", False):
            from wamrvit.quad.amrex_to_qt import read_plotfile_header

            selected_paths, selection_metrics = select_bounded_frame_subset(
                frame_paths,
                reference_counts=reference_counts,
                selected_fields=len(field_names),
                total_fields=int(read_plotfile_header(frame_paths[0])["ncomp"]),
                max_cache_bytes=cache_limit_bytes,
                usable_fraction=usable_fraction,
                estimate_padding_fraction=estimate_padding_fraction,
            )
        else:
            selected_paths = frame_paths
            total_references = sum(reference_counts.values())
            selection_metrics = {
                "policy": "full",
                "candidate_frames": len(frame_paths),
                "selected_frames": len(frame_paths),
                "total_references": total_references,
                "selected_references": total_references,
                "reference_coverage": 1.0,
            }

        cache_started = time.perf_counter()
        frame_refs, metrics = build_ray_amrex_frame_cache(
            selected_paths,
            field_names=field_names,
            patch_size=loader_params.get("amrex_patch_size", 32),
            max_cache_bytes=cache_limit_bytes,
            concurrency=int(
                train_config.get("amrex_frame_cache_build_concurrency", 16)
            ),
            normalization_mean=(
                source_normalization["mean"] if source_normalization else None
            ),
            normalization_std=(
                source_normalization["std"] if source_normalization else None
            ),
            normalization_eps=(
                source_normalization["eps"] if source_normalization else 0.0
            ),
            estimate_padding_fraction=estimate_padding_fraction,
        )
        metrics.update(selection_metrics)
        metrics["build_wall_s"] = time.perf_counter() - cache_started
        train_config["amrex_frame_cache"] = metrics
        fill_params["amrex_frame_refs"] = frame_refs
        fill_params["amrex_source_normalization"] = source_normalization
        print(
            "Built transient decoded AMReX frame cache: "
            f"policy={metrics['policy']} frames={metrics['frames']} "
            f"bytes={metrics['decoded_bytes'] / 1024**3:.3f}GiB "
            f"wall={metrics['build_wall_s']:.2f}s."
        )
        print(
            "AMReX source cache metrics: "
            + json.dumps(metrics, sort_keys=True)
        )
        return fill_params, metrics

    def streaming_mapper_factory(
        self,
        config: dict[str, Any],
        fill_params: dict[str, Any],
        *,
        rank: int,
    ) -> Any:
        """Keep AMReX mapper construction behind the generic provider seam."""
        from wamrvit.dataloader.transport.streaming import StreamingMapperFactory

        mapper_fill_params = dict(fill_params)
        source_trace_dir = config["train"].get("amrex_source_trace_dir")
        if source_trace_dir:
            mapper_fill_params["amrex_source_trace_dir"] = str(
                source_trace_dir
            )
            mapper_fill_params["amrex_source_trace_rank"] = int(rank)
        return StreamingMapperFactory(
            mapper_fill_params,
            config["transform"],
            requires_transport_context=bool(source_trace_dir),
        )
