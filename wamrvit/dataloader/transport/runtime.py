"""Capacity, source-provider lifecycle, epoch iteration and batch delivery.

Used by: train.py startup and per-rank training loops. Sample
selection lives in selection/windows.py; mapper construction in payload/mappers.py.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import torch

VALID_PAYLOAD_TRANSPORTS = {
    "auto",
    "ray_resident",
    "ray_stream",
    "worker_stream",
}
SOURCE_FILL_PARAMS_KEY = "__source_fill_params__"


class BoundedStreamingMapperFactory(Protocol):
    """Factory for ordered, pre-allocation-bounded worker mappers.

    Mapped batches must preserve ``sample_index``, ``traj_idx``, and
    ``frame_idx`` control columns from their topology descriptor.
    """

    def __call__(self) -> Callable[[dict[str, np.ndarray]], dict[str, Any]]: ...

    def with_output_limit(
        self, max_output_bytes: int
    ) -> BoundedStreamingMapperFactory: ...


class SourceProvider(Protocol):
    """Dataset-specific setup for worker-local payload construction."""

    def validate(self, config: dict[str, Any], *, topology_cached: bool) -> None: ...

    def prepare(
        self,
        config: dict[str, Any],
        windows: list[dict[str, Any]],
        *,
        topology_cached: bool,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]: ...

    def streaming_mapper_factory(
        self,
        config: dict[str, Any],
        fill_params: dict[str, Any],
        *,
        rank: int,
    ) -> BoundedStreamingMapperFactory: ...


def has_stochastic_payload_mapping(config: dict[str, Any]) -> bool:
    """Detect known payload-time augmentation that residency would freeze."""
    loader_params = config.get("file_loader", {}).get("params", {})
    if float(loader_params.get("resample_coarsen_ratio", 0) or 0) > 0:
        return True
    if float(loader_params.get("resample_refine_ratio", 0) or 0) > 0:
        return True

    def contains_periodic_roll(value: Any) -> bool:
        if isinstance(value, dict):
            target = str(value.get("target", ""))
            return target.endswith("PeriodicRoll") or any(
                contains_periodic_roll(item) for item in value.values()
            )
        if isinstance(value, list):
            return any(contains_periodic_roll(item) for item in value)
        return False

    return contains_periodic_roll(config.get("transform", {}))


def allocated_node_memory_gib() -> float:
    """Return the smaller of physical RAM and the Slurm memory allocation."""
    gib = float(1024**3)
    physical_gib = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / gib
    slurm_mebibytes = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm_mebibytes:
        try:
            return min(physical_gib, float(slurm_mebibytes) / 1024)
        except ValueError:
            pass
    memory_per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    cpus = os.environ.get("SLURM_CPUS_ON_NODE")
    if memory_per_cpu and cpus:
        try:
            return min(physical_gib, float(memory_per_cpu) * float(cpus) / 1024)
        except ValueError:
            pass
    return physical_gib


def resolve_payload_transport(
    requested: str,
    *,
    estimated_payload_gib: float | None,
    object_store_gib: float,
    node_memory_gib: float,
    shared_memory_available_gib: float,
    object_store_usable_fraction: float = 0.85,
    node_headroom_gib: float = 256.0,
) -> tuple[str, dict[str, Any]]:
    """Resolve auto/resident before expansion and reject unsafe residency."""
    if requested not in VALID_PAYLOAD_TRANSPORTS:
        raise ValueError(
            f"payload transport must be one of {sorted(VALID_PAYLOAD_TRANSPORTS)}; "
            f"got {requested!r}."
        )
    if not 0 < object_store_usable_fraction <= 1:
        raise ValueError("object_store_usable_fraction must be in (0, 1].")
    if min(object_store_gib, node_memory_gib, shared_memory_available_gib) <= 0:
        raise ValueError("Resident capacity values must be positive.")
    if node_headroom_gib < 0:
        raise ValueError("node_headroom_gib must be non-negative.")

    usable_object_store_gib = object_store_gib * object_store_usable_fraction
    checks = {
        "estimate_available": estimated_payload_gib is not None,
        "fits_object_store": (
            estimated_payload_gib is not None
            and estimated_payload_gib <= usable_object_store_gib
        ),
        "object_store_fits_shared_memory": (
            object_store_gib <= shared_memory_available_gib
        ),
        "object_store_preserves_node_headroom": (
            object_store_gib + node_headroom_gib <= node_memory_gib
        ),
    }
    resident_safe = all(checks.values())
    decision: dict[str, Any] = {
        "requested": requested,
        "estimated_payload_gib": estimated_payload_gib,
        "object_store_gib": object_store_gib,
        "object_store_usable_fraction": object_store_usable_fraction,
        "usable_object_store_gib": usable_object_store_gib,
        "node_memory_gib": node_memory_gib,
        "shared_memory_available_gib": shared_memory_available_gib,
        "node_headroom_gib": node_headroom_gib,
        "checks": checks,
        "resident_safe": resident_safe,
    }

    if requested == "auto":
        resolved = "ray_resident" if resident_safe else "ray_stream"
    elif requested == "ray_resident":
        if not resident_safe:
            failed = [name for name, passed in checks.items() if not passed]
            raise MemoryError(
                "ray_resident preflight failed: "
                f"{', '.join(failed)}; decision={decision}."
            )
        resolved = requested
    else:
        resolved = requested
    decision["resolved"] = resolved
    return resolved, decision


def configure_payload_transport(
    config: dict[str, Any],
    *,
    load_in_ram: bool,
    topology_cached: bool,
) -> tuple[str, dict[str, Any]]:
    """Resolve transport settings and apply bounded runtime limits to config."""
    train_config = config["train"]
    requested = train_config.get("payload_transport", "auto")
    if requested not in VALID_PAYLOAD_TRANSPORTS:
        raise ValueError(
            "train.payload_transport must be one of "
            f"{sorted(VALID_PAYLOAD_TRANSPORTS)}; got {requested!r}."
        )
    if load_in_ram:
        if requested not in ("auto", "ray_resident"):
            raise ValueError(
                "--load_in_ram conflicts with "
                f"train.payload_transport={requested!r}."
            )
        requested = "ray_resident"

    gib = float(1024**3)
    node_memory_gib = allocated_node_memory_gib()
    shared_memory_available_gib = (
        shutil.disk_usage("/dev/shm").free / gib
        if os.path.isdir("/dev/shm")
        else node_memory_gib
    )
    object_store_gib = float(config["general"].get("ray_object_store_memory", 8))
    estimate = train_config.get("resident_payload_estimate_gib")
    resolved, decision = resolve_payload_transport(
        requested,
        estimated_payload_gib=float(estimate) if estimate is not None else None,
        object_store_gib=object_store_gib,
        node_memory_gib=node_memory_gib,
        shared_memory_available_gib=shared_memory_available_gib,
        object_store_usable_fraction=float(
            train_config.get("resident_object_store_usable_fraction", 0.85)
        ),
        node_headroom_gib=float(
            train_config.get("resident_node_headroom_gib", 256)
        ),
    )
    if resolved == "ray_resident" and has_stochastic_payload_mapping(config):
        raise ValueError(
            "ray_resident would freeze stochastic payload augmentation; use "
            "ray_stream or disable topology/transform augmentation."
        )

    train_config["payload_transport_requested"] = requested
    train_config["payload_transport"] = resolved
    train_config["payload_transport_capacity"] = decision
    if resolved in ("ray_stream", "worker_stream"):
        if resolved == "ray_stream":
            train_config["prefetch"] = int(
                train_config.get("ray_stream_prefetch_batches", 0)
            )
        stream_object_store_gib = int(
            train_config.get("ray_stream_object_store_memory_gib", 64)
        )
        stream_headroom_gib = float(
            train_config.get("ray_stream_node_headroom_gib", 128)
        )
        if stream_object_store_gib > shared_memory_available_gib:
            raise MemoryError(
                f"Ray stream object store requires {stream_object_store_gib} GiB, "
                f"but /dev/shm has {shared_memory_available_gib:.2f} GiB available."
            )
        if stream_object_store_gib + stream_headroom_gib > node_memory_gib:
            raise MemoryError(
                f"Ray stream requires {stream_object_store_gib} GiB object store + "
                f"{stream_headroom_gib} GiB headroom, but the node/Slurm "
                f"allocation is {node_memory_gib:.2f} GiB."
            )
        config["general"]["ray_object_store_memory"] = stream_object_store_gib
        decision["runtime_object_store_gib"] = stream_object_store_gib
        decision["stream_node_headroom_gib"] = stream_headroom_gib

    if resolved == "worker_stream":
        if not topology_cached:
            raise ValueError("worker_stream requires --cache_topology.")
        if int(train_config.get("batch_size", 1)) != 1:
            raise ValueError("worker_stream currently requires train.batch_size=1.")
        train_config.setdefault("streaming_prefetch_batches", 1)
        train_config.setdefault("streaming_memory_budget_gib_per_rank", 8)
        train_config.setdefault("streaming_loader_backend", "process")
        if int(train_config["streaming_prefetch_batches"]) <= 0:
            raise ValueError("train.streaming_prefetch_batches must be positive.")
        streaming_budget_gib = float(
            train_config["streaming_memory_budget_gib_per_rank"]
        )
        if streaming_budget_gib <= 0:
            raise ValueError(
                "train.streaming_memory_budget_gib_per_rank must be positive."
            )
        aggregate_streaming_gib = streaming_budget_gib * int(
            train_config["num_gpus"]
        )
        required_node_gib = (
            float(config["general"]["ray_object_store_memory"])
            + aggregate_streaming_gib
            + float(train_config.get("ray_stream_node_headroom_gib", 128))
        )
        if required_node_gib > node_memory_gib:
            raise MemoryError(
                "Worker stream requires object store + aggregate rank staging "
                f"+ headroom = {required_node_gib:.2f} GiB, but the node/Slurm "
                f"allocation is {node_memory_gib:.2f} GiB."
            )
        decision["aggregate_streaming_budget_gib"] = aggregate_streaming_gib
        decision["worker_stream_required_node_gib"] = required_node_gib

    print(
        f"Payload transport: requested={requested}, resolved={resolved}, "
        f"resident_safe={decision['resident_safe']}."
    )
    return resolved, decision


def _source_provider(config: dict[str, Any]) -> SourceProvider | None:
    provider_config = config["train"].get("source_provider")
    if not provider_config:
        return None
    from wamrvit.utils import instantiate_from_config

    return instantiate_from_config(provider_config)


def validate_source_provider(
    config: dict[str, Any], *, topology_cached: bool
) -> None:
    """Validate dataset-specific worker-stream options before Ray starts."""
    provider = _source_provider(config)
    if provider is not None:
        provider.validate(config, topology_cached=topology_cached)


def prepare_source_provider(
    config: dict[str, Any],
    windows: list[dict[str, Any]],
    *,
    topology_cached: bool,
) -> None:
    """Prepare source state and expose only mapper kwargs to the data pipeline."""
    provider = _source_provider(config)
    if provider is None:
        config.pop(SOURCE_FILL_PARAMS_KEY, None)
        return
    fill_params, metrics = provider.prepare(
        config, windows, topology_cached=topology_cached
    )
    config[SOURCE_FILL_PARAMS_KEY] = fill_params
    if metrics is not None:
        config["train"]["source_cache"] = metrics


def create_streaming_mapper_factory(
    config: dict[str, Any],
    fill_params: dict[str, Any],
    *,
    rank: int = 0,
) -> BoundedStreamingMapperFactory:
    """Let a source provider own worker mapping without coupling the transport."""
    provider = _source_provider(config)
    if provider is not None:
        factory_method = getattr(provider, "streaming_mapper_factory", None)
        if callable(factory_method):
            return factory_method(config, fill_params, rank=rank)

    from wamrvit.dataloader.transport.streaming import StreamingMapperFactory

    return StreamingMapperFactory(fill_params, config["transform"])


def materialize_with_capacity(
    datasets: list[Any], *, max_total_bytes: int
) -> tuple[list[Any], int]:
    """Materialize datasets and verify the measured resident footprint."""
    if max_total_bytes <= 0:
        raise ValueError("max_total_bytes must be positive.")
    materialized = []
    total_bytes = 0
    for dataset_index, dataset in enumerate(datasets):
        resident = dataset.materialize()
        size_bytes = resident.size_bytes()
        if size_bytes is None:
            raise RuntimeError(
                f"Resident dataset {dataset_index} has unknown size."
            )
        total_bytes += int(size_bytes)
        if total_bytes > max_total_bytes:
            raise MemoryError(
                f"Resident datasets reached {total_bytes / 1024**3:.3f} GiB after "
                f"dataset {dataset_index}; capacity is "
                f"{max_total_bytes / 1024**3:.3f} GiB."
            )
        materialized.append(resident)
    return materialized, total_bytes


def _profile_scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(np.asarray(value).reshape(-1)[0])


def _update_online_metric(
    metrics: dict[str, dict[str, Any]], name: str, value: float
) -> None:
    summary = metrics.setdefault(
        name, {"count": 0, "sum": 0.0, "max": value, "recent": []}
    )
    count = int(summary["count"])
    recent = summary["recent"]
    if len(recent) < 1024:
        recent.append(value)
    else:
        recent[count % len(recent)] = value
    summary["count"] = count + 1
    summary["sum"] = float(summary["sum"]) + value
    summary["max"] = max(float(summary["max"]), value)


class TrainingBatchSource:
    """Own one rank's transport setup and yield model-ready epoch batches.

    Used by: train.train_func.
    """

    def __init__(
        self,
        config: dict[str, Any],
        dataset: Any,
        *,
        rank: int,
    ) -> None:
        self.config = config
        self.train_config = config["train"]
        self.data_config = config["data"]
        self.dataset = dataset
        self.rank = rank
        world_size = int(self.train_config["num_gpus"])
        self.expected_count = int(self.data_config["num_train_samples"]) // world_size
        self.worker_stream = self.train_config.get("payload_transport") == "worker_stream"
        self.mapper_factory = None
        if self.worker_stream:
            self._prepare_worker_stream()

        self.pin_memory = bool(
            self.worker_stream or self.train_config.get("pin_memory", False)
        )
        self._iterator: Any = None
        self._streaming_iterator: Any = None
        self._active = False
        self._exhausted = False
        self._started = 0.0
        self._delivery_cursor = 0
        self._epoch_index = -1
        self._profile_values: dict[str, list[float]] = {}

    def __del__(self) -> None:
        try:
            if getattr(self, "_active", False):
                self.close_epoch()
        except Exception:
            pass

    def _prepare_worker_stream(self) -> None:
        pipeline_config = self.config.get("__cache_topology_pipeline__", {})
        fill_params = pipeline_config.get("fill_train_params")
        if not fill_params:
            raise RuntimeError("Worker streaming is missing topology fill parameters.")
        self.mapper_factory = create_streaming_mapper_factory(
            self.config, fill_params, rank=self.rank
        )
        sample_options = pipeline_config.get("sample_train_options")
        if sample_options is not None:
            from wamrvit.dataloader.payload.mappers import SampleMapperFactory

            self.mapper_factory = SampleMapperFactory(self.mapper_factory, sample_options)

    def start_epoch(
        self, epoch_index: int | None = None
    ) -> TrainingBatchSource:
        """Reset epoch iteration/accounting for the training loops."""
        if self._active:
            raise RuntimeError("The previous transport epoch is still active.")
        self._started = time.perf_counter()
        self._epoch_index = (
            self._epoch_index + 1 if epoch_index is None else int(epoch_index)
        )
        self._delivery_cursor = 0
        self._profile_values = {}
        self._streaming_iterator = None
        self._exhausted = False

        if self.worker_stream:
            from wamrvit.dataloader.transport.streaming import StreamingPayloadIterator

            topology_batches = self.dataset.iter_batches(
                batch_size=1,
                prefetch_batches=1,
                batch_format="numpy",
            )
            self._streaming_iterator = StreamingPayloadIterator(
                topology_batches,
                self.mapper_factory,
                prefetch_batches=int(
                    self.train_config.get("streaming_prefetch_batches", 1)
                ),
                memory_budget_bytes=int(
                    self.train_config.get(
                        "streaming_memory_budget_gib_per_rank", 8
                    )
                    * 1024**3
                ),
                pin_memory=True,
                loader_backend=self.train_config.get(
                    "streaming_loader_backend", "thread"
                ),
                overlap_staging=bool(
                    self.train_config.get("streaming_overlap_staging", False)
                ),
                epoch_index=self._epoch_index,
            )
            iterator = self._streaming_iterator
        else:
            # Native preparation validates/inverts topology on CPU before H2D.
            # Keep CPU prefetch/pinning; prepare_multi_scale_batch owns native transfers.
            native = self.config.get("model", {}).get("multi_scale_patch", False)
            iterator = self.dataset.iter_torch_batches(
                batch_size=self.train_config.get("batch_size", 1),
                prefetch_batches=self.train_config.get("prefetch", 2),
                pin_memory=self.pin_memory,
                device="cpu" if native else "auto",
            )

        self._iterator = iter(iterator)
        self._active = True
        return self

    def __iter__(self) -> TrainingBatchSource:
        if not self._active:
            raise RuntimeError("Call start_epoch() before iterating batches.")
        return self

    def __next__(self) -> dict[str, Any]:
        if not self._active:
            raise StopIteration
        wait_started = time.perf_counter()
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._exhausted = True
            raise
        except BaseException:
            self.close_epoch()
            raise

        if self.train_config.get("profile_data_transport", False):
            _update_online_metric(
                self._profile_values,
                "iterator_wait_s",
                time.perf_counter() - wait_started,
            )
            for name in list(batch):
                if not name.startswith("_transport_"):
                    continue
                value = batch.pop(name)
                metric_name = name.removeprefix("_transport_")
                _update_online_metric(
                    self._profile_values, metric_name, _profile_scalar(value)
                )
        if self.worker_stream:
            expected_identity = tuple(batch.pop("_stream_sample_identity"))
            actual_identity = tuple(
                int(np.asarray(batch[name]).reshape(-1)[0])
                for name in ("traj_idx", "frame_idx")
            )
            if actual_identity != expected_identity:
                self.close_epoch()
                raise RuntimeError(
                    "Worker-stream sample identity changed during mapping: "
                    f"expected={expected_identity}, actual={actual_identity}."
                )
            batch.pop("_stream_wait_s")

        sample_indices = batch.pop("sample_index", None)
        for control_name in ("traj_idx", "frame_idx"):
            batch.pop(control_name, None)
        # Only the count is needed; do not read GPU-resident sample IDs.
        if torch.is_tensor(sample_indices):
            delivered = sample_indices.numel()
        elif sample_indices is not None:
            delivered = np.asarray(sample_indices).size
        else:
            delivered = min(
                int(self.train_config.get("batch_size", 1)),
                self.expected_count - self._delivery_cursor,
            )
        if delivered <= 0:
            raise RuntimeError(
                f"Rank {self.rank} consumed more than {self.expected_count} samples."
            )
        self._delivery_cursor += delivered
        return batch

    def close_epoch(self) -> None:
        """Release worker-local executors when training exits an epoch early."""
        if self._streaming_iterator is not None:
            self._streaming_iterator.close()
        close = getattr(self._iterator, "close", None)
        if callable(close):
            close()
        self._iterator = None
        self._streaming_iterator = None
        self._active = False

    def finish_epoch(
        self, epoch: int
    ) -> tuple[dict[str, float | int], dict[str, float]]:
        """Validate delivery and return Ray-report and W&B transport metrics."""
        if not self._active:
            raise RuntimeError("No active transport epoch to finish.")
        if not self._exhausted:
            self.close_epoch()
            raise RuntimeError("Transport epoch ended before its iterator was exhausted.")

        profile_summary: dict[str, float] = {}
        for name, values in self._profile_values.items():
            count = int(values["count"])
            profile_summary[f"{name}_mean"] = (
                float(values["sum"]) / count if count else 0.0
            )
            profile_summary[f"{name}_p95"] = float(
                np.percentile(values["recent"], 95)
            )
            profile_summary[f"{name}_max"] = float(values["max"])
        if profile_summary:
            print(
                f"[Rank {self.rank}] data transport profile epoch {epoch}: "
                + json.dumps(profile_summary, sort_keys=True)
            )

        stream_summary: dict[str, float | int] = {}
        if self.worker_stream and self._streaming_iterator is not None:
            stream_summary = self._streaming_iterator.summary()
            print(
                f"[Rank {self.rank}] worker-stream epoch {epoch}: "
                f"batches={stream_summary['batches']} "
                f"fill_mean={stream_summary['fill_mean_s']:.4f}s "
                f"wait_mean={stream_summary['wait_mean_s']:.4f}s "
                f"stage_mean={stream_summary['stage_mean_s']:.4f}s "
                f"backend={stream_summary['loader_backend']} "
                f"overlap={stream_summary['overlap_staging']} "
                f"max_batch={stream_summary['max_batch_bytes'] / 1024**3:.3f}GiB"
            )

        if self._delivery_cursor != self.expected_count:
            self.close_epoch()
            raise RuntimeError(
                f"Rank {self.rank} consumed {self._delivery_cursor} samples; "
                f"expected {self.expected_count}."
            )
        elapsed_s = time.perf_counter() - self._started
        samples = self._delivery_cursor
        print(
            f"[Rank {self.rank}] transport epoch {epoch}: samples={samples} "
            f"elapsed={elapsed_s:.2f}s throughput={samples / elapsed_s:.3f} samples/s"
        )

        numeric_stream_summary = {
            name: value
            for name, value in stream_summary.items()
            if isinstance(value, (int, float, bool))
        }
        report_metrics: dict[str, float | int] = {
            "transport_epoch_s": elapsed_s,
            **{
                f"transport_{name}": value
                for name, value in profile_summary.items()
            },
            **{
                f"worker_stream_{name}": value
                for name, value in numeric_stream_summary.items()
            },
        }
        wandb_metrics = {
            **{
                f"transport/{name}": value
                for name, value in profile_summary.items()
            },
            **{
                f"transport/worker_stream_{name}": value
                for name, value in numeric_stream_summary.items()
            },
        }
        self.close_epoch()
        return report_metrics, wandb_metrics
