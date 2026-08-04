from __future__ import annotations

import hashlib
import logging
import resource
import time

import numpy as np
import torch

from wamrvit.quad.array_regrid_uniform import (
    array_regrid_from_sequence, array_regrid_topology_from_detectors,
)
from wamrvit.quad.gpu_payload_replay import replay_payload
from wamrvit.quad.regrid_lineage import DetectorSpec, extract_latest_detectors

logger = logging.getLogger(__name__)


def _latest_physical_detector_channels(
    channel,
    *,
    channels: int,
    timesteps: int,
) -> tuple[int, ...]:
    """Map CPU packed-channel selection to latest-frame physical channels.

    Invalid packed indices are dropped exactly as in the CPU array path. Selections
    involving earlier timesteps or duplicate physical channels cannot be represented by
    detector-only replay and therefore trigger its existing visible CPU fallback.
    """
    packed_count = int(channels) * int(timesteps)
    if channel is None:
        selected = list(range(packed_count))
    elif isinstance(channel, (int, np.integer)):
        selected = [int(channel)]
    else:
        selected = [int(index) for index in channel]
    valid = [index for index in selected if 0 <= index < packed_count]
    latest_offset = (int(timesteps) - 1) * int(channels)
    if not valid:
        raise ValueError("GPU replay requires at least one valid detector channel.")
    if any(index < latest_offset for index in valid):
        raise ValueError("GPU replay detectors must select only the latest timestep.")
    physical = tuple(index - latest_offset for index in valid)
    if len(set(physical)) != len(physical):
        raise ValueError("GPU replay detector channels must be unique.")
    return physical


def _uniform_topology_hash(meta):
    keys = np.column_stack((np.asarray(meta["tiles"]), np.asarray(meta["xy_idx"])[:, :3]))
    order = np.lexsort(tuple(keys[:, column] for column in reversed(range(keys.shape[1]))))
    return hashlib.sha256(np.ascontiguousarray(keys[order]).tobytes()).hexdigest()

def _run_gpu_replay_uniform(
    data, meta, *, detector_channels, detector_fields, cell_scale_mode,
    tol_frac, max_passes, adapt_nearby, capacity,
):

    if not isinstance(data, torch.Tensor) or not data.is_cuda:
        raise ValueError("gpu_replay requires a CUDA sequence tensor.")
    spec = DetectorSpec(tuple(detector_channels), tuple(detector_fields))
    start = time.perf_counter()
    torch.cuda.synchronize(data.device)
    d2h_start = time.perf_counter()
    detector_gpu = extract_latest_detectors(data, spec)
    detector_cpu = np.ascontiguousarray(detector_gpu.detach().cpu().numpy())
    torch.cuda.synchronize(data.device)
    detector_d2h = time.perf_counter() - d2h_start
    topology_start = time.perf_counter()
    _detector_out, out_meta, plan, topology_status = (
        array_regrid_topology_from_detectors(
            detector_cpu,
            meta,
            detector_spec=spec,
            cell_scale_mode=cell_scale_mode,
            tol_frac=tol_frac,
            max_passes=max_passes,
            adapt_nearby=adapt_nearby,
            capacity=capacity,
        )
    )
    cpu_topology = time.perf_counter() - topology_start
    output, replay_status = replay_payload(data, plan, return_status=True)
    complete_regrid = time.perf_counter() - start
    replay_timings = replay_status["timings"]
    status = {
        "backend": "array",
        "fallback_reason": None,
        "array_regrid_payload_backend": "gpu_replay",
        "gpu_replay_fallback_reason": None,
        "detector_shape": list(detector_cpu.shape),
        "detector_bytes": int(detector_cpu.nbytes),
        "detector_sha256": hashlib.sha256(detector_cpu.tobytes()).hexdigest(),
        "full_payload_d2h_bytes": 0,
        "avoided_full_payload_d2h_bytes": int(data.numel() * data.element_size()),
        "lineage_bytes": replay_status["lineage_bytes"],
        "operation_nodes": replay_status["operation_nodes"],
        "final_slots": replay_status["final_slots"],
        "topology_sha256": _uniform_topology_hash(out_meta),
        "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(data.device)),
        "cpu_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "timings": {
            "detector_d2h": detector_d2h,
            "cpu_topology": cpu_topology,
            "lineage_h2d": replay_timings["lineage_h2d"],
            "gpu_refine": replay_timings["gpu_refine"],
            "gpu_coarsen": replay_timings["gpu_coarsen"],
            "gpu_gather": replay_timings["gpu_gather"],
            "gpu_replay": replay_timings["gpu_replay"],
            "complete_regrid": complete_regrid,
        },
        "topology_status": topology_status,
    }
    return output, out_meta, status


def run_gpu_replay_uniform_with_fallback(
    data: torch.Tensor,
    meta: dict,
    *,
    detector_channels: tuple[int, ...] | None,
    detector_fields: tuple[str, ...] | None,
    cell_scale_mode: str | None,
    tol_frac,
    channel,
    max_passes: int,
    adapt_nearby: int,
    capacity: int,
    value_storage: str,
):
    try:
        resolved_channels = (
            _latest_physical_detector_channels(
                channel,
                channels=int(data.shape[1]),
                timesteps=int(data.shape[2]),
            )
            if detector_channels is None
            else tuple(int(index) for index in detector_channels)
        )
        resolved_fields = (
            tuple(f"channel_{index}" for index in resolved_channels)
            if detector_fields is None
            else tuple(detector_fields)
        )
        return _run_gpu_replay_uniform(
            data,
            meta,
            detector_channels=resolved_channels,
            detector_fields=resolved_fields,
            cell_scale_mode=cell_scale_mode,
            tol_frac=tol_frac,
            max_passes=max_passes,
            adapt_nearby=adapt_nearby,
            capacity=capacity,
        )
    except Exception as exc:
        was_cuda_oom = isinstance(exc, torch.cuda.OutOfMemoryError)
        reason = "cuda_oom" if was_cuda_oom else f"{type(exc).__name__}: {exc}"
        logger.warning(
            "GPU payload replay failed; using cpu_eager for this call (reason=%s)",
            reason,
        )

    # Run recovery after leaving the exception scope so its traceback no longer retains
    # failed replay intermediates. Emptying the allocator cache gives a genuine OOM
    # fallback the best chance to upload the smaller CPU-eager result successfully.
    if was_cuda_oom:
        torch.cuda.empty_cache()
    sequence = np.ascontiguousarray(data.detach().cpu().numpy())
    cpu_out, cpu_meta, cpu_status = array_regrid_from_sequence(
        sequence,
        meta,
        cell_scale_mode=cell_scale_mode,
        tol_frac=tol_frac,
        channel=channel,
        max_passes=max_passes,
        adapt_nearby=adapt_nearby,
        capacity=capacity,
        disable_warnings=True,
        output_layout="sequence",
        value_storage=value_storage,
    )
    output = torch.from_numpy(cpu_out).to(device=data.device, dtype=data.dtype)
    status = dict(cpu_status)
    status.update(
        {
            "array_regrid_payload_backend": "cpu_eager",
            "gpu_replay_fallback_reason": reason,
            "detector_shape": None,
            "detector_bytes": 0,
            "full_payload_d2h_bytes": int(sequence.nbytes),
            "avoided_full_payload_d2h_bytes": 0,
        }
    )
    return output, cpu_meta, status
