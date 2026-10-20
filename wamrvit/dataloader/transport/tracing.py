"""Opt-in, bounded-memory tracing for worker-local source accesses."""
from __future__ import annotations

import heapq
import json
import os
import threading
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


class SourceAccessTrace:
    """Write source-access events in bounded batches to one mapper-local file."""

    def __init__(
        self,
        trace_dir: str,
        *,
        rank: int,
        flush_events: int = 32,
    ) -> None:
        if flush_events <= 0:
            raise ValueError("flush_events must be positive.")
        self.trace_dir = os.path.abspath(trace_dir)
        self.rank = int(rank)
        self.flush_events = int(flush_events)
        self._events: list[dict[str, Any]] = []
        self._path: Path | None = None
        self._closed = False

    def record(
        self,
        key: str,
        *,
        cache_hit: bool,
        start_ns: int,
        source_ready_ns: int,
        normalize_ready_ns: int,
        end_ns: int,
        context: Mapping[str, int] | None = None,
        source_position: int,
    ) -> None:
        if self._closed:
            raise RuntimeError("Cannot record to a closed source-access trace.")
        if not start_ns <= source_ready_ns <= normalize_ready_ns <= end_ns:
            raise ValueError("Source-access trace timestamps are not monotonic.")
        event: dict[str, Any] = {
            "cache_hit": bool(cache_hit),
            "end_ns": int(end_ns),
            "key": str(key),
            "normalize_ready_ns": int(normalize_ready_ns),
            "pid": os.getpid(),
            "rank": self.rank,
            "source_position": int(source_position),
            "source_ready_ns": int(source_ready_ns),
            "start_ns": int(start_ns),
            "thread_id": threading.get_ident(),
        }
        if context is not None:
            collisions = sorted(event.keys() & context.keys())
            if collisions:
                raise ValueError(
                    "Source-access trace context cannot override reserved fields: "
                    + ", ".join(collisions)
                )
            event.update({name: int(value) for name, value in context.items()})
        self._events.append(event)
        if len(self._events) >= self.flush_events:
            self.flush()

    def flush(self) -> None:
        if not self._events:
            return
        if self._path is None:
            Path(self.trace_dir).mkdir(parents=True, exist_ok=True)
            self._path = Path(self.trace_dir) / (
                f"rank{self.rank}-pid{os.getpid()}-thread{threading.get_ident()}-"
                f"{uuid.uuid4().hex}.jsonl"
            )
        with self._path.open("a", encoding="utf-8") as handle:
            for event in self._events:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        self._events.clear()

    def close(self) -> None:
        if not self._closed:
            self.flush()
            self._closed = True


def _numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(max(values)),
    }


def _max_concurrency(
    events: Iterable[Mapping[str, Any]], *, end_name: str
) -> int:
    active_ends: list[int] = []
    maximum = 0
    for event in sorted(events, key=lambda item: int(item["start_ns"])):
        started = int(event["start_ns"])
        while active_ends and active_ends[0] <= started:
            heapq.heappop(active_ends)
        heapq.heappush(active_ends, int(event[end_name]))
        maximum = max(maximum, len(active_ends))
    return maximum


def _miss_contention(misses: list[dict[str, Any]]) -> dict[str, int]:
    active_reads_by_key: dict[str, list[int]] = defaultdict(list)
    duplicate_concurrent_reads = 0
    max_concurrent_reads_same_key = 0
    for event in sorted(misses, key=lambda item: int(item["start_ns"])):
        key = str(event["key"])
        started = int(event["start_ns"])
        active = active_reads_by_key[key]
        while active and active[0] <= started:
            heapq.heappop(active)
        if active:
            duplicate_concurrent_reads += 1
        heapq.heappush(active, int(event["source_ready_ns"]))
        max_concurrent_reads_same_key = max(
            max_concurrent_reads_same_key, len(active)
        )
    return {
        "duplicate_concurrent_reads": duplicate_concurrent_reads,
        "max_concurrent_miss_pipelines": _max_concurrency(
            misses, end_name="end_ns"
        ),
        "max_concurrent_reads": _max_concurrency(
            misses, end_name="source_ready_ns"
        ),
        "max_concurrent_reads_same_key": max_concurrent_reads_same_key,
    }


def summarize_source_access_events(
    events: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize cache coverage, contention, stages, and bounded reuse metrics."""
    records = [dict(event) for event in events]
    misses = [event for event in records if not bool(event["cache_hit"])]
    hits = len(records) - len(misses)
    contention = _miss_contention(misses)

    seconds = 1e-9
    miss_stage_seconds = {
        "read": [
            (int(event["source_ready_ns"]) - int(event["start_ns"])) * seconds
            for event in misses
        ],
        "normalize": [
            (
                int(event["normalize_ready_ns"])
                - int(event["source_ready_ns"])
            )
            * seconds
            for event in misses
        ],
        "sample": [
            (int(event["end_ns"]) - int(event["normalize_ready_ns"])) * seconds
            for event in misses
        ],
        "total": [
            (int(event["end_ns"]) - int(event["start_ns"])) * seconds
            for event in misses
        ],
    }

    rank_counts: dict[int, dict[str, Any]] = {}
    for rank in sorted({int(event["rank"]) for event in records}):
        rank_events = [event for event in records if int(event["rank"]) == rank]
        rank_misses = [event for event in rank_events if not event["cache_hit"]]
        rank_counts[rank] = {
            "accesses": len(rank_events),
            "hits": len(rank_events) - len(rank_misses),
            "misses": len(rank_misses),
            "unique_missed_items": len(
                {str(event["key"]) for event in rank_misses}
            ),
        }

    # Measure how far a missed key is reused along each actual rank lane without
    # retaining the full path sequence in the summary.
    last_sequence: dict[tuple[int, int, str], int] = {}
    reuse_distances: list[float] = []
    ordered_misses = sorted(
        misses,
        key=lambda event: (
            int(event.get("epoch", 0)),
            int(event["rank"]),
            int(event.get("rank_sequence", 0)),
            int(event["source_position"]),
        ),
    )
    for event in ordered_misses:
        lane_key = (
            int(event.get("epoch", 0)),
            int(event["rank"]),
            str(event["key"]),
        )
        sequence = int(event.get("rank_sequence", 0))
        previous = last_sequence.get(lane_key)
        if previous is not None:
            reuse_distances.append(float(sequence - previous))
        last_sequence[lane_key] = sequence

    epoch_counts = {}
    for epoch in sorted({int(event.get("epoch", 0)) for event in records}):
        epoch_events = [
            event for event in records if int(event.get("epoch", 0)) == epoch
        ]
        epoch_misses = [
            event for event in epoch_events if not bool(event["cache_hit"])
        ]
        epoch_counts[epoch] = {
            "accesses": len(epoch_events),
            "hits": len(epoch_events) - len(epoch_misses),
            "misses": len(epoch_misses),
            "unique_missed_items": len(
                {str(event["key"]) for event in epoch_misses}
            ),
            **_miss_contention(epoch_misses),
        }

    return {
        "accesses": len(records),
        "cache_hits": hits,
        "cache_misses": len(misses),
        "cache_reference_coverage": hits / len(records) if records else 0.0,
        "unique_missed_items": len({str(event["key"]) for event in misses}),
        **contention,
        "epoch_counts": epoch_counts,
        "miss_stage_seconds": {
            name: _numeric_summary(values)
            for name, values in miss_stage_seconds.items()
        },
        "rank_counts": rank_counts,
        "rank_lane_miss_reuse_distance_samples": _numeric_summary(
            reuse_distances
        ),
        "rank_lane_reuses": len(reuse_distances),
    }


def read_source_access_trace(trace_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in sorted(Path(trace_dir).glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                events.append(json.loads(line))
    return events
