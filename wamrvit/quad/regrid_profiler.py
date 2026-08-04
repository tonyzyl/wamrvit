from __future__ import annotations

import csv
import logging
import os
import time

logger = logging.getLogger(__name__)


def record_array_status(profiler, status, elapsed, *, label):
    backend_used = status["backend"]
    fallback_reason = (
        status["fallback_reason"]
        if backend_used == "object"
        else status.get("gpu_replay_fallback_reason") or status["fallback_reason"]
    )
    if backend_used == "object":
        if profiler is None or not getattr(profiler, "_fallback_warned", False):
            logger.warning("%s fell back to object (reason=%s)", label, fallback_reason)
            if profiler is not None:
                profiler._fallback_warned = True
    if profiler is not None:
        profiler.record(elapsed, backend_used, fallback_reason)

class RegridProfiler:
    """Accumulates per-regrid-call records across a rollout and writes them to disk.
    Resident on the (Ray) predictor actor; files are pid-keyed so data-parallel actor
    replicas do not clobber each other. ``write`` is idempotent-by-rewrite, so calling
    it once per batch leaves a complete cumulative file at the end."""

    def __init__(self):
        self.array_warmed = False
        self._fallback_warned = False
        self.records = []          # list[dict]
        self.sample_totals = []    # list[dict]: sample, total_s (rollout wall-time)
        self._sample = -1
        self._call = 0
        self._sample_t0 = None     # perf_counter at current sample start

    def _close_sample(self):
        """Record the elapsed wall-time of the in-progress sample, if any."""
        if self._sample_t0 is not None:
            self.sample_totals.append(
                {"sample": self._sample, "total_s": time.perf_counter() - self._sample_t0})
            self._sample_t0 = None

    def new_sample(self):
        """Call at the start of each rollout trajectory: closes the previous sample's
        wall-time (if open), advances the sample index, resets the per-sample call index,
        and stamps the new sample's start time (denominator for regrid-as-%-of-rollout)."""
        self._close_sample()
        self._sample += 1
        self._call = 0
        self._sample_t0 = time.perf_counter()

    def exclude_from_sample_total(self, seconds):
        """Exclude explicit benchmark warm-up work from the current rollout timer."""
        if self._sample_t0 is not None:
            self._sample_t0 += float(seconds)

    def record(self, seconds, backend_used, fallback_reason):
        self.records.append({
            "sample": self._sample if self._sample >= 0 else 0,
            "call": self._call,
            "seconds": float(seconds),
            "backend_used": backend_used,
            "fallback_reason": "" if fallback_reason is None else str(fallback_reason),
        })
        self._call += 1

    def write(self, out_dir, backend):
        self._close_sample()   # close the in-progress sample so its wall-time is recorded
        if not self.records:
            return
        os.makedirs(out_dir, exist_ok=True)
        tag = f"{backend}_{os.getpid()}"
        with open(os.path.join(out_dir, f"regrid_timing_{tag}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "sample", "call", "seconds", "backend_used", "fallback_reason"])
            w.writeheader()
            w.writerows(self.records)
        # Rollout wall-time per sample -> denominator for honest "regrid as % of rollout".
        with open(os.path.join(out_dir, f"regrid_rollout_total_{tag}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sample", "total_s"])
            w.writeheader()
            w.writerows(self.sample_totals)
