"""Format-independent bounded caching for decoded source items."""
from __future__ import annotations

import time
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Protocol


class SourceItemAdapter[SourceKey: Hashable](Protocol):
    """Estimate and load one format-specific source item.

    Implementations must be serializable by Ray. ``load`` returns the decoded
    item, its measured resident size, and optional numeric loader metrics.
    """

    def estimate_size_bytes(self, key: SourceKey) -> int: ...

    def load(
        self, key: SourceKey
    ) -> tuple[Any, int, Mapping[str, float | int]]: ...


def select_bounded_source_subset[SourceKey: Hashable](
    keys: Sequence[SourceKey],
    *,
    reference_counts: Mapping[SourceKey, int],
    adapter: SourceItemAdapter[SourceKey],
    max_cache_bytes: int,
    usable_fraction: float = 0.95,
) -> tuple[list[SourceKey], dict[str, Any]]:
    """Greedily select source items by references per estimated resident byte."""
    if max_cache_bytes <= 0:
        raise ValueError("max_cache_bytes must be positive.")
    if not 0 < usable_fraction <= 1:
        raise ValueError("usable_fraction must be in (0, 1].")

    unique_keys = sorted(set(keys), key=repr)
    estimates: dict[SourceKey, int] = {}
    for key in unique_keys:
        estimate = int(adapter.estimate_size_bytes(key))
        if estimate <= 0:
            raise ValueError(
                f"Source item {key!r} has non-positive size estimate {estimate}."
            )
        estimates[key] = estimate

    budget = int(max_cache_bytes * usable_fraction)
    ordered = sorted(
        unique_keys,
        key=lambda key: (
            -int(reference_counts.get(key, 0)) / estimates[key],
            -int(reference_counts.get(key, 0)),
            repr(key),
        ),
    )
    selected: list[SourceKey] = []
    estimated_bytes = 0
    for key in ordered:
        item_bytes = estimates[key]
        if estimated_bytes + item_bytes <= budget:
            selected.append(key)
            estimated_bytes += item_bytes

    if not selected:
        raise MemoryError(
            "Bounded source cache cannot fit one estimated source item within "
            f"{budget / 1024**3:.3f} GiB."
        )

    total_references = sum(
        int(reference_counts.get(key, 0)) for key in unique_keys
    )
    selected_references = sum(
        int(reference_counts.get(key, 0)) for key in selected
    )
    return selected, {
        "policy": "bounded_subset",
        "candidate_items": len(unique_keys),
        "selected_items": len(selected),
        "estimated_selected_bytes": estimated_bytes,
        "selection_budget_bytes": budget,
        "total_references": total_references,
        "selected_references": selected_references,
        "reference_coverage": (
            selected_references / total_references if total_references else 0.0
        ),
    }


def _load_source_item[SourceKey: Hashable](
    key: SourceKey,
    adapter: SourceItemAdapter[SourceKey],
    reserved_size_bytes: int,
) -> tuple[Any, dict[str, Any]]:
    """Load within a reservation before Ray publishes the value to Plasma."""
    started = time.perf_counter()
    value, size_bytes, item_metrics = adapter.load(key)
    size_bytes = int(size_bytes)
    if size_bytes <= 0:
        raise ValueError(
            f"Source item {key!r} has non-positive measured size {size_bytes}."
        )
    if size_bytes > reserved_size_bytes:
        raise MemoryError(
            f"Source item {key!r} decoded to {size_bytes / 1024**3:.3f} GiB, "
            f"exceeding its {reserved_size_bytes / 1024**3:.3f}-GiB "
            "preallocated cache reservation."
        )
    return value, {
        "key": key,
        "size_bytes": size_bytes,
        "reserved_size_bytes": reserved_size_bytes,
        "load_s": time.perf_counter() - started,
        "item_metrics": {
            name: float(metric) for name, metric in item_metrics.items()
        },
    }


def build_ray_source_cache[SourceKey: Hashable](
    keys: Sequence[SourceKey],
    *,
    adapter: SourceItemAdapter[SourceKey],
    max_cache_bytes: int,
    concurrency: int = 16,
) -> tuple[dict[SourceKey, Any], dict[str, Any]]:
    """Load source items under a logical payload-byte cap and retain object refs."""
    import ray

    unique_keys = sorted(set(keys), key=repr)
    if not unique_keys:
        raise ValueError("Source cache requires at least one key.")
    if max_cache_bytes <= 0 or concurrency <= 0:
        raise ValueError("Source cache limits must be positive.")

    estimates = {
        key: int(adapter.estimate_size_bytes(key)) for key in unique_keys
    }
    invalid = [key for key, size in estimates.items() if size <= 0]
    if invalid:
        raise ValueError(
            f"Source items have non-positive size estimates: {invalid[:3]!r}."
        )
    estimated_bytes = sum(estimates.values())
    if estimated_bytes > max_cache_bytes:
        raise MemoryError(
            "Source-cache preflight failed: estimated "
            f"{estimated_bytes / 1024**3:.3f} GiB exceeds "
            f"{max_cache_bytes / 1024**3:.3f} GiB."
        )

    # Distribute all remaining capacity proportionally. Each task verifies its
    # measured item against this reservation before returning, so concurrently
    # completed values cannot collectively publish more than the hard cap.
    reserve_headroom = max_cache_bytes - estimated_bytes
    reservations = {
        key: estimates[key]
        + reserve_headroom * estimates[key] // estimated_bytes
        for key in unique_keys
    }

    load_task = ray.remote(num_cpus=1, num_returns=2)(_load_source_item)
    adapter_ref = ray.put(adapter)
    pending: dict[Any, tuple[SourceKey, Any]] = {}
    item_refs: dict[SourceKey, Any] = {}
    resident_bytes = 0
    load_seconds: list[float] = []
    item_metric_totals: dict[str, dict[str, float]] = {}
    key_iterator = iter(unique_keys)

    def submit_one(key: SourceKey) -> None:
        data_ref, metadata_ref = load_task.remote(
            key, adapter_ref, reservations[key]
        )
        pending[metadata_ref] = (key, data_ref)

    for _ in range(min(concurrency, len(unique_keys))):
        submit_one(next(key_iterator))

    try:
        while pending:
            completed, _ = ray.wait(list(pending), num_returns=1)
            metadata_ref = completed[0]
            key, data_ref = pending.pop(metadata_ref)
            metadata = ray.get(metadata_ref)
            if metadata["key"] != key:
                raise RuntimeError("Source-cache task identity mismatch.")
            resident_bytes += int(metadata["size_bytes"])
            if resident_bytes > max_cache_bytes:
                raise MemoryError(
                    "Source cache exceeded its logical payload-byte cap: "
                    f"{resident_bytes / 1024**3:.3f} GiB > "
                    f"{max_cache_bytes / 1024**3:.3f} GiB."
                )
            load_seconds.append(float(metadata["load_s"]))
            for name, metric in metadata["item_metrics"].items():
                value = float(metric)
                summary = item_metric_totals.setdefault(
                    name, {"sum": 0.0, "max": value}
                )
                summary["sum"] += value
                summary["max"] = max(summary["max"], value)
            item_refs[key] = data_ref
            try:
                submit_one(next(key_iterator))
            except StopIteration:
                pass
    except BaseException:
        for remaining_metadata in pending:
            ray.cancel(remaining_metadata, force=True)
        pending.clear()
        item_refs.clear()
        raise

    return item_refs, {
        "items": len(item_refs),
        "estimated_bytes": estimated_bytes,
        "resident_bytes": resident_bytes,
        "max_cache_bytes": max_cache_bytes,
        "reserved_bytes": sum(reservations.values()),
        "concurrency": concurrency,
        "load_task_seconds_sum": float(sum(load_seconds)),
        "load_task_seconds_max": max(load_seconds, default=0.0),
        "item_metrics": item_metric_totals,
    }
