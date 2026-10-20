"""Bounded worker-local payload streaming for topology-cached training."""
from __future__ import annotations

import copy
import multiprocessing
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

import numpy as np
import torch

from wamrvit.dataloader.payload.loaders import FillAndExportMapper
from wamrvit.dataloader.payload.mappers import CachedSeq2SeqMapper
from wamrvit.utils import instantiate_from_config

_EXCLUDED_COLUMNS = {"target_src", "target_src_paths"}


class StreamingMapperFactory:
    """Create one stateful fill/transform mapper per loader thread."""

    def __init__(
        self,
        fill_params: dict[str, Any],
        transform_config: dict[str, Any],
        *,
        requires_transport_context: bool = False,
    ):
        self.fill_params = copy.deepcopy(fill_params)
        self.transform_config = copy.deepcopy(transform_config)
        self.requires_transport_context = bool(requires_transport_context)

    def __call__(self) -> CachedSeq2SeqMapper:
        return CachedSeq2SeqMapper(
            fill_mapper=FillAndExportMapper(**copy.deepcopy(self.fill_params)),
            transform=instantiate_from_config(copy.deepcopy(self.transform_config)),
        )

    def with_output_limit(self, max_output_bytes: int) -> StreamingMapperFactory:
        if not self.fill_params.get("direct_amrex_export", False):
            raise ValueError(
                "The standard streaming mapper can enforce a pre-allocation "
                "limit only for direct AMReX export. Other large-source "
                "providers must supply their own bounded mapper factory."
            )
        fill_params = copy.deepcopy(self.fill_params)
        fill_params["max_output_bytes"] = int(max_output_bytes)
        return StreamingMapperFactory(
            fill_params,
            self.transform_config,
            requires_transport_context=self.requires_transport_context,
        )


def _to_training_batch(
    mapped: dict[str, Any],
    *,
    pin_memory: bool,
    max_bytes: int,
    inflight_slots: int,
) -> tuple[dict[str, torch.Tensor], int]:
    arrays: dict[str, np.ndarray] = {}
    total_bytes = 0
    for name, value in mapped.items():
        if name in _EXCLUDED_COLUMNS:
            continue
        array = np.asarray(value)
        if array.dtype == object:
            raise TypeError(
                f"Worker-stream output {name!r} has unsupported object dtype. "
                "Providers must omit non-training metadata explicitly."
            )
        if not array.flags.c_contiguous or not array.flags.writeable:
            array = np.array(array, copy=True, order="C")
        arrays[name] = array
        total_bytes += int(array.nbytes)
    if total_bytes > max_bytes:
        raise MemoryError(
            f"Streaming batch requires {total_bytes / 1024**3:.2f} GiB; "
            f"the {inflight_slots}-slot budget allows "
            f"{max_bytes / 1024**3:.2f} GiB per batch."
        )
    output = {}
    for name, array in arrays.items():
        tensor = torch.from_numpy(array)
        if pin_memory:
            output[name] = tensor if tensor.is_pinned() else tensor.pin_memory()
        else:
            output[name] = tensor.clone()
    return output, total_bytes


_PROCESS_MAPPER: CachedSeq2SeqMapper | None = None


def _sample_identity(topology_batch: dict[str, np.ndarray]) -> tuple[int, int]:
    return (
        int(np.asarray(topology_batch["traj_idx"]).reshape(-1)[0]),
        int(np.asarray(topology_batch["frame_idx"]).reshape(-1)[0]),
    )


def _initialize_process_mapper(mapper_factory: StreamingMapperFactory) -> None:
    global _PROCESS_MAPPER
    _PROCESS_MAPPER = mapper_factory()


def _map_in_process(
    topology_batch: dict[str, np.ndarray],
) -> tuple[tuple[int, int], dict[str, Any], float]:
    if _PROCESS_MAPPER is None:
        raise RuntimeError("Streaming process mapper was not initialized.")
    started = time.perf_counter()
    mapped = _PROCESS_MAPPER(topology_batch)
    return _sample_identity(topology_batch), mapped, time.perf_counter() - started


class StreamingPayloadIterator:
    """Fill topology rows in worker threads with a bounded in-flight queue.

    Ray carries shuffled topology rows only. Worker-local threads read source
    data, reconstruct the adaptive payload, and pin only a fixed number of
    in-flight batches. No expanded payload is persisted to disk.
    """

    def __init__(
        self,
        topology_batches: Iterator[dict[str, np.ndarray]],
        mapper_factory: Callable[[], CachedSeq2SeqMapper],
        *,
        prefetch_batches: int = 1,
        memory_budget_bytes: int,
        pin_memory: bool = True,
        loader_backend: str = "thread",
        overlap_staging: bool = False,
        epoch_index: int = 0,
    ):
        if prefetch_batches < 1:
            raise ValueError("prefetch_batches must be at least 1.")
        if memory_budget_bytes <= 0:
            raise ValueError("memory_budget_bytes must be positive.")
        if loader_backend not in ("thread", "process"):
            raise ValueError("loader_backend must be 'thread' or 'process'.")
        self.topology_batches = iter(topology_batches)
        self.mapper_factory = mapper_factory
        self.prefetch_batches = prefetch_batches
        self.overlap_staging = overlap_staging
        # Include the payload retained by the training loop, all mapper futures,
        # the batch being delivered, and (with overlap) its mapped source form.
        self.inflight_slots = prefetch_batches + (4 if overlap_staging else 2)
        self.per_batch_budget_bytes = memory_budget_bytes // self.inflight_slots
        self.memory_budget_bytes = memory_budget_bytes
        self.pin_memory = pin_memory
        self.loader_backend = loader_backend
        self.epoch_index = int(epoch_index)
        self._submission_sequence = 0
        output_limiter = getattr(mapper_factory, "with_output_limit", None)
        if not callable(output_limiter):
            raise TypeError(
                "Worker-stream mapper factories must implement "
                "with_output_limit(max_output_bytes) and enforce the limit "
                "before allocating their final payload."
            )
        self.mapper_factory = output_limiter(self.per_batch_budget_bytes)
        self.requires_transport_context = bool(
            getattr(self.mapper_factory, "requires_transport_context", False)
        )
        if loader_backend == "process":
            self.executor = ProcessPoolExecutor(
                max_workers=prefetch_batches,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_process_mapper,
                initargs=(self.mapper_factory,),
            )
        else:
            self.executor = ThreadPoolExecutor(
                max_workers=prefetch_batches,
                thread_name_prefix="source-payload-loader",
            )
        self.thread_local = threading.local()
        self.futures: deque[
            Future[tuple[tuple[int, int], dict[str, Any], float]]
        ] = deque()
        self.stage_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="payload-stager")
            if overlap_staging
            else None
        )
        self.delivery_future: Future[
            tuple[tuple[int, int], dict[str, torch.Tensor], float, float, int]
        ] | None = None
        self.exhausted = False
        self.closed = False
        self._summary_count = 0
        self._fill_total_s = 0.0
        self._wait_total_s = 0.0
        self._stage_total_s = 0.0
        self._max_batch_bytes = 0
        self._thread_mappers: list[CachedSeq2SeqMapper] = []
        self._thread_mappers_lock = threading.Lock()
        for _ in range(prefetch_batches):
            if not self._submit_next():
                break
        if self.overlap_staging and self.futures:
            self.delivery_future = self.stage_executor.submit(
                self._resolve_and_stage_next
            )

    def _mapper(self) -> CachedSeq2SeqMapper:
        mapper = getattr(self.thread_local, "mapper", None)
        if mapper is None:
            mapper = self.mapper_factory()
            self.thread_local.mapper = mapper
            with self._thread_mappers_lock:
                self._thread_mappers.append(mapper)
        return mapper

    def _map_in_thread(
        self, topology_batch: dict[str, np.ndarray]
    ) -> tuple[tuple[int, int], dict[str, Any], float]:
        started = time.perf_counter()
        mapped = self._mapper()(topology_batch)
        return (
            _sample_identity(topology_batch),
            mapped,
            time.perf_counter() - started,
        )

    def _submit_next(self) -> bool:
        if self.exhausted:
            return False
        try:
            topology_batch = next(self.topology_batches)
        except StopIteration:
            self.exhausted = True
            return False
        if self.requires_transport_context:
            topology_batch["_stream_epoch"] = np.asarray(
                [self.epoch_index], dtype=np.int64
            )
            topology_batch["_stream_rank_sequence"] = np.asarray(
                [self._submission_sequence], dtype=np.int64
            )
        self._submission_sequence += 1
        mapper = (
            _map_in_process
            if self.loader_backend == "process"
            else self._map_in_thread
        )
        self.futures.append(self.executor.submit(mapper, topology_batch))
        return True

    def _resolve_and_stage_next(
        self,
    ) -> tuple[tuple[int, int], dict[str, torch.Tensor], float, float, int]:
        future = self.futures.popleft()
        sample_identity, mapped, fill_s = future.result()
        # Replenish mapping before the pinned copy so both stages overlap.
        self._submit_next()
        stage_started = time.perf_counter()
        batch, batch_bytes = _to_training_batch(
            mapped,
            pin_memory=self.pin_memory,
            max_bytes=self.per_batch_budget_bytes,
            inflight_slots=self.inflight_slots,
        )
        stage_s = time.perf_counter() - stage_started
        return sample_identity, batch, fill_s, stage_s, batch_bytes

    def close(self) -> None:
        if not self.closed:
            if self.stage_executor is not None:
                self.stage_executor.shutdown(wait=True, cancel_futures=True)
            self.executor.shutdown(wait=True, cancel_futures=True)
            for mapper in self._thread_mappers:
                close = getattr(mapper, "close", None)
                if callable(close):
                    close()
            self._thread_mappers.clear()
            self.closed = True

    def __del__(self) -> None:
        try:
            if not getattr(self, "closed", True):
                self.close()
        except Exception:
            pass

    def __iter__(self) -> StreamingPayloadIterator:
        return self

    def __next__(self) -> dict[str, Any]:
        wait_started = time.perf_counter()
        if self.overlap_staging:
            if self.delivery_future is None:
                self.close()
                raise StopIteration
            try:
                sample_identity, batch, fill_s, stage_s, batch_bytes = (
                    self.delivery_future.result()
                )
            except BaseException:
                self.close()
                raise
            wait_s = time.perf_counter() - wait_started
            self.delivery_future = (
                self.stage_executor.submit(self._resolve_and_stage_next)
                if self.futures
                else None
            )
        else:
            if not self.futures:
                self.close()
                raise StopIteration
            future = self.futures.popleft()
            try:
                sample_identity, mapped, fill_s = future.result()
            except BaseException:
                self.close()
                raise
            wait_s = time.perf_counter() - wait_started
            stage_started = time.perf_counter()
            try:
                batch, batch_bytes = _to_training_batch(
                    mapped,
                    pin_memory=self.pin_memory,
                    max_bytes=self.per_batch_budget_bytes,
                    inflight_slots=self.inflight_slots,
                )
                del mapped
                # Replenish only after staging. This avoids retaining a newly
                # mapped output alongside both the mapped and pinned forms.
                self._submit_next()
            except BaseException:
                self.close()
                raise
            stage_s = time.perf_counter() - stage_started
        batch["_stream_sample_identity"] = sample_identity
        self._summary_count += 1
        self._fill_total_s += fill_s
        self._wait_total_s += wait_s
        self._stage_total_s += stage_s
        self._max_batch_bytes = max(self._max_batch_bytes, batch_bytes)
        batch["_stream_wait_s"] = wait_s + stage_s
        return batch

    def summary(self) -> dict[str, float | int]:
        count = self._summary_count
        return {
            "batches": count,
            "fill_mean_s": self._fill_total_s / count if count else 0.0,
            "wait_mean_s": self._wait_total_s / count if count else 0.0,
            "stage_mean_s": self._stage_total_s / count if count else 0.0,
            "max_batch_bytes": self._max_batch_bytes,
            "inflight_slots": self.inflight_slots,
            "loader_backend": self.loader_backend,
            "overlap_staging": self.overlap_staging,
            "memory_budget_bytes": self.memory_budget_bytes,
        }
