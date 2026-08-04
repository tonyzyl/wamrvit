# wamrvit/quad/regrid_dispatch.py
"""Backend dispatch for adaptive regrid: object vs array, with timing and fallback
visibility. The single seam through which rollout/animation choose a regrid backend
for the array-vs-object comparison harness. ``regrid_native_dispatch`` covers the
multi-scale native path; ``regrid_uniform_dispatch`` covers the uniform-patch path."""
import logging
import time

import numpy as np

from wamrvit.quad.adapt_wavelet import regrid, regrid_native
from wamrvit.quad.array_regrid_native import (
    array_regrid_native_from_sequence,
    warm_array_regrid_native_kernels,
)
from wamrvit.quad.array_regrid_uniform import (
    array_regrid_from_sequence,
    array_regrid_from_tensor,
    warm_array_regrid_kernels,
)
from wamrvit.quad.gpu_regrid_backend import run_gpu_replay_uniform_with_fallback
from wamrvit.quad.quad_utils import quadtree_to_tensor, tensor_to_quadtree
from wamrvit.quad.regrid_profiler import RegridProfiler, record_array_status

logger = logging.getLogger(__name__)

VALID_BACKENDS = ("object", "array")
VALID_ARRAY_PAYLOAD_BACKENDS = ("cpu_eager", "gpu_replay")


def _warm_array_from_inputs(by_level, meta, *, C, T):
    """Prime the native numba kernels using shapes derived from the live inputs, so the
    first measured array regrid is not charged JIT-compile time."""
    L = int(meta["domain"]["max_level_idx"])
    finest = by_level[L]
    base_h, base_w = int(finest.shape[-2]), int(finest.shape[-1])  # finest = base patch
    warm_array_regrid_native_kernels(
        base_patch_h=base_h, base_patch_w=base_w, max_level_idx=L,
        channels=int(C), timesteps=int(T))


def regrid_native_dispatch(by_level, leaf_to_bucket, meta, *,
                           backend, profiler=None, max_passes=10, **regrid_kwargs):
    """Run native regrid via the chosen backend; return the object backend's 3-tuple
    ``(new_by_level, new_l2b, new_meta)``. ``regrid_kwargs`` are forwarded unchanged so
    both backends see identical arguments (C, T, tol_frac, cell_scale_mode,
    adapt_on_channels, adapt_nearby, allow_coarsening). ``max_passes`` is forwarded
    EXPLICITLY to both backends so a future change to one backend's default cannot make
    the two diverge silently."""
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"Unknown regrid_backend={backend!r}; expected one of {VALID_BACKENDS}.")

    if backend == "object":
        t0 = time.perf_counter()
        new_by_level, new_l2b, new_meta = regrid_native(
            by_level, leaf_to_bucket, meta, max_passes=max_passes, **regrid_kwargs)
        dt = time.perf_counter() - t0
        if profiler is not None:
            profiler.record(dt, "object", None)
        return new_by_level, new_l2b, new_meta

    # backend == "array"
    if profiler is not None and not getattr(profiler, "array_warmed", False):
        _warm_array_from_inputs(by_level, meta, C=regrid_kwargs["C"], T=regrid_kwargs["T"])
        profiler.array_warmed = True
    t0 = time.perf_counter()
    new_by_level, new_l2b, new_meta, status = array_regrid_native_from_sequence(
        by_level, leaf_to_bucket, meta, max_passes=max_passes, **regrid_kwargs)
    dt = time.perf_counter() - t0
    backend_used = status["backend"]
    fallback_reason = status["fallback_reason"]
    if backend_used == "object":
        # Rate-limit: warn once per rollout (profiler-scoped). A forced-fallback config
        # (e.g. allow_coarsening=False) would otherwise emit one WARNING per regrid call.
        # The per-call fact is still in the profiler record and surfaced as the fallback
        # count in the report, so nothing is hidden.
        if profiler is None or not getattr(profiler, "_fallback_warned", False):
            logger.warning(
                "regrid array backend fell back to object (reason=%s)", fallback_reason)
            if profiler is not None:
                profiler._fallback_warned = True
    if profiler is not None:
        profiler.record(dt, backend_used, fallback_reason)
    return new_by_level, new_l2b, new_meta


def _object_uniform_regrid(data, meta, *, cell_scale_mode, tol_frac, channel,
                           max_passes, adapt_nearby, allow_coarsening):
    """Object uniform regrid: build the quadtree, regrid in place, export back to a
    packed ``(N, T*C, H, W)`` tensor. This is exactly the inline block ``_call_uniform``
    used before the dispatch existed (incl. ``allow_coarsening``), so the default
    object rollout is byte-for-byte unchanged."""
    qt = tensor_to_quadtree(data, meta, cell_scale_mode=cell_scale_mode)
    regrid(qt, tol_frac=tol_frac, channel=channel, max_passes=max_passes,
           adapt_nearby=adapt_nearby, allow_coarsening=allow_coarsening,
           disable_warnings=True)
    return quadtree_to_tensor(qt, return_tensor=False, cell_scale_mode=cell_scale_mode)


def _warm_array_uniform_from_inputs(data):
    """Prime the uniform copy-backed kernels from a packed ``(N,T*C,H,W)`` input."""
    warm_array_regrid_kernels(
        flat_channels=int(data.shape[1]),
        patch_size=(int(data.shape[-2]), int(data.shape[-1])))


def _warm_array_uniform_sequence_from_inputs(data, *, value_storage):
    """Prime the sequence-layout kernels used by the source/copy array entry."""
    warm_array_regrid_kernels(
        flat_channels=int(data.shape[1]) * int(data.shape[2]),
        patch_size=(int(data.shape[-2]), int(data.shape[-1])),
        warm_source=value_storage == "source",
        warm_copy=value_storage == "copy",
        output_layout="sequence",
    )


def regrid_uniform_sequence_dispatch(
    data, meta, *, backend, profiler=None, max_passes=10, cell_scale_mode,
    tol_frac, channel, adapt_nearby=0, allow_coarsening=True,
    value_storage="source", capacity=8192, status_collector=None,
    payload_backend="cpu_eager", detector_channels=None,
    detector_fields=None,
):
    """Run the optimized uniform array backend on ``(N,C,T,H,W)`` sequence data.

    This entry deliberately accepts only ``backend='array'``. The object rollout stays on
    :func:`regrid_uniform_dispatch` so its established flatten/build/export path remains
    unchanged. ``value_storage='source'`` keeps initial AMReX leaves in the input sequence
    instead of copying their ~GiB payload into the fixed-capacity mutable workspace.
    Output is always sequence layout ``(new_N,C,T,H,W)``.
    """
    if backend != "array":
        raise ValueError(
            "regrid_uniform_sequence_dispatch only supports backend='array'; "
            f"got {backend!r}.")
    if value_storage not in {"source", "copy"}:
        raise ValueError(
            f"Unknown array_regrid_value_storage={value_storage!r}; expected 'source' or 'copy'.")
    if payload_backend not in VALID_ARRAY_PAYLOAD_BACKENDS:
        raise ValueError(
            f"Unknown array_regrid_payload_backend={payload_backend!r}; expected one of "
            f"{VALID_ARRAY_PAYLOAD_BACKENDS}.")
    if len(data.shape) != 5:
        raise ValueError(f"Expected sequence shape (N,C,T,H,W), got {data.shape}.")

    if not allow_coarsening:
        fallback_t0 = time.perf_counter()
        # The array engine has no refinement-only mode. Preserve the existing explicit
        # fallback contract and convert the object result back to sequence layout.
        sequence = data.detach().cpu().numpy() if hasattr(data, "detach") else data
        sequence = np.asarray(sequence)
        n, c, t, h, w = sequence.shape
        flat = np.ascontiguousarray(
            sequence.transpose(0, 2, 1, 3, 4).reshape(n, t * c, h, w))
        out, out_meta = regrid_uniform_dispatch(
            flat, meta, backend="array", profiler=profiler, max_passes=max_passes,
            cell_scale_mode=cell_scale_mode, tol_frac=tol_frac, channel=channel,
            adapt_nearby=adapt_nearby, allow_coarsening=False)
        out_sequence = out.reshape(out.shape[0], t, c, h, w).transpose(0, 2, 1, 3, 4)
        if status_collector is not None:
            status_collector.append({
                "kind": "measured",
                "complete_regrid_s": time.perf_counter() - fallback_t0,
                "status": {
                    "backend": "object",
                    "fallback_reason": "allow_coarsening_false",
                    "array_regrid_payload_backend": "cpu_eager",
                    "gpu_replay_fallback_reason": None,
                },
            })
        return np.ascontiguousarray(out_sequence), out_meta

    entry_kwargs = {
        "cell_scale_mode": cell_scale_mode,
        "tol_frac": tol_frac,
        "channel": channel,
        "max_passes": max_passes,
        "adapt_nearby": adapt_nearby,
        "capacity": capacity,
        "disable_warnings": True,
        "output_layout": "sequence",
        "value_storage": value_storage,
    }
    if payload_backend == "gpu_replay":
        physical_channels = (
            None if detector_channels is None else tuple(detector_channels)
        )
        field_names = None if detector_fields is None else tuple(detector_fields)

        def _gpu_entry():
            return run_gpu_replay_uniform_with_fallback(
                data,
                meta,
                detector_channels=physical_channels,
                detector_fields=field_names,
                cell_scale_mode=cell_scale_mode,
                tol_frac=tol_frac,
                channel=channel,
                max_passes=max_passes,
                adapt_nearby=adapt_nearby,
                capacity=capacity,
                value_storage=value_storage,
            )

        if profiler is not None and not getattr(profiler, "array_warmed", False):
            warm_t0 = time.perf_counter()
            _, _, warm_status = _gpu_entry()
            warm_dt = time.perf_counter() - warm_t0
            profiler.exclude_from_sample_total(warm_dt)
            profiler.array_warmed = True
            if status_collector is not None:
                status_collector.append({
                    "kind": "warmup",
                    "real_input_warmup_s": warm_dt,
                    "status": warm_status,
                })
        t0 = time.perf_counter()
        out, out_meta, status = _gpu_entry()
        dt = time.perf_counter() - t0
        record_array_status(
            profiler, status, dt, label="regrid array(gpu replay) backend"
        )
        if status_collector is not None:
            status_collector.append({
                "kind": "measured", "complete_regrid_s": dt, "status": status
            })
        return out, out_meta

    if profiler is not None and not getattr(profiler, "array_warmed", False):
        synthetic_t0 = time.perf_counter()
        _warm_array_uniform_sequence_from_inputs(data, value_storage=value_storage)
        synthetic_dt = time.perf_counter() - synthetic_t0
        # Synthetic buffers compile common kernels, but production topology can execute
        # additional lazy Numba branches. Exercise this exact input once and discard the
        # result so neither JIT nor warm-up execution contaminates measured regrid or
        # rollout time. The public array entry imports topology into fresh workspaces and
        # does not mutate ``data`` or ``meta``, so the following measured call sees the
        # same inputs.
        warm_t0 = time.perf_counter()
        _, _, warm_status = array_regrid_from_sequence(data, meta, **entry_kwargs)
        warm_dt = time.perf_counter() - warm_t0
        profiler.exclude_from_sample_total(warm_dt)
        profiler.array_warmed = True
        if status_collector is not None:
            status_collector.append({
                "kind": "warmup",
                "synthetic_warmup_s": synthetic_dt,
                "real_input_warmup_s": warm_dt,
                "status": warm_status,
            })
        logger.info(
            "Untimed real-input array regrid warm-up completed in %.3fs "
            "(backend=%s, fallback_reason=%s)",
            warm_dt, warm_status["backend"], warm_status["fallback_reason"])
    t0 = time.perf_counter()
    out, out_meta, status = array_regrid_from_sequence(data, meta, **entry_kwargs)
    dt = time.perf_counter() - t0
    record_array_status(
        profiler, status, dt, label="regrid array(uniform sequence) backend")
    if status_collector is not None:
        status_collector.append({"kind": "measured", "complete_regrid_s": dt, "status": status})
    return out, out_meta


def regrid_uniform_dispatch(data, meta, *, backend, profiler=None, max_passes=10,
                            cell_scale_mode, tol_frac, channel, adapt_nearby=0,
                            allow_coarsening=True):
    """Run uniform-patch adaptive regrid via the chosen backend; return the object
    path's ``(out, out_meta)``. ``data`` is the packed ``(N, T*C, H, W)`` cell tensor and
    ``channel`` indexes that packed axis -- the exact inputs ``_call_uniform`` already
    builds. ``coarsen_ratio`` is left at the shared 0.25 default on both sides
    (parity-required). The array path (`array_regrid_from_tensor`) self-reports object
    fallback via ``status['backend']``; the array engine has no allow_coarsening knob, so
    ``allow_coarsening=False`` defers to object rather than run an unfair comparison
    (mirrors the native dispatch's allow_coarsening=False -> object contract)."""
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"Unknown regrid_backend={backend!r}; expected one of {VALID_BACKENDS}.")

    if backend == "object" or (backend == "array" and not allow_coarsening):
        t0 = time.perf_counter()
        out, out_meta = _object_uniform_regrid(
            data, meta, cell_scale_mode=cell_scale_mode, tol_frac=tol_frac,
            channel=channel, max_passes=max_passes, adapt_nearby=adapt_nearby,
            allow_coarsening=allow_coarsening)
        dt = time.perf_counter() - t0
        if backend == "array":
            # allow_coarsening=False forces the object path; record it as a fallback so
            # the report never silently treats an object run as an array measurement.
            if profiler is None or not getattr(profiler, "_fallback_warned", False):
                logger.warning("regrid array(uniform) deferred to object "
                               "(allow_coarsening=False has no array equivalent)")
                if profiler is not None:
                    profiler._fallback_warned = True
            if profiler is not None:
                profiler.record(dt, "object", "allow_coarsening_false")
        elif profiler is not None:
            profiler.record(dt, "object", None)
        return out, out_meta

    # backend == "array"
    if profiler is not None and not getattr(profiler, "array_warmed", False):
        _warm_array_uniform_from_inputs(data)
        profiler.array_warmed = True
    t0 = time.perf_counter()
    out, out_meta, status = array_regrid_from_tensor(
        data, meta, cell_scale_mode=cell_scale_mode, tol_frac=tol_frac, channel=channel,
        max_passes=max_passes, adapt_nearby=adapt_nearby, disable_warnings=True)
    dt = time.perf_counter() - t0
    backend_used = status["backend"]
    fallback_reason = status["fallback_reason"]
    if backend_used == "object":
        if profiler is None or not getattr(profiler, "_fallback_warned", False):
            logger.warning(
                "regrid array(uniform) backend fell back to object (reason=%s)",
                fallback_reason)
            if profiler is not None:
                profiler._fallback_warned = True
    if profiler is not None:
        profiler.record(dt, backend_used, fallback_reason)
    return out, out_meta
