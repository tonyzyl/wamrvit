"""Inference-time benchmark across model variants × datasets.

For one (model_variant × dataset) combo, samples N random test trajectories
(seeded RNG, t_start=0 each), materializes one starting window at a time on the
GPU, warms up the inference path (njit + cuDNN autotune), then runs
`predict_steps` autoregressive iterations per trajectory while timing the forward
pass and regrid block separately. Writes a JSON with per-trajectory timing,
PyTorch CUDA-memory measurements, and aggregated means/maxima.

No metric computation, no GT loading beyond what the loader emits incidentally,
no Ray, no DDP. Single-GPU, single-process.

Usage:
    uv run wamrvit/rollout/benchmark_inference.py \\
        +experiment=benchmark_inference \\
        dataset=pli_adaptive \\
        inference.checkpoint_path="/path/to/some_run/model" \\
        benchmark.output_path="results/benchmark_inference/pli_adaptive_uniform.json" \\
        benchmark.variant_tag=adaptive_uniform
"""

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import time
import warnings
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from numba import get_num_threads

from wamrvit.dataloader.trajectory_loader import Seq2SeqMapper
from wamrvit.native_train_utils import unpack_native_batch
from wamrvit.quad.adapt_wavelet import regrid
from wamrvit.quad.array_regrid import configure_array_regrid_num_threads
from wamrvit.quad.quad_utils import quadtree_to_tensor, tensor_to_quadtree
from wamrvit.quad.regrid_dispatch import (
    regrid_native_dispatch,
    regrid_uniform_sequence_dispatch,
)
from wamrvit.quad.yt_utils import make_regular_centers
from wamrvit.quadtree_transformer import QuadTreeTransformer
from wamrvit.utils import instantiate_from_config, load_config


def _git_metadata() -> dict[str, Any]:
    """Return reproducibility metadata without requiring GitPython."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    return {"git_commit": commit, "git_dirty": dirty}


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _canonical_topology_keys(meta: dict[str, Any]) -> np.ndarray:
    """Return sorted ``(tile_x,tile_y,level,x,y)`` keys for loader or export metadata."""
    tiles, xy_idx = _canonical_topology_arrays(meta)
    keys = np.column_stack((tiles[:, 0], tiles[:, 1], xy_idx[:, 0:3])).astype(np.int64)
    if keys.size == 0:
        return keys.reshape(0, 5)
    order = np.lexsort(tuple(keys[:, i] for i in range(4, -1, -1)))
    return keys[order]


def _topology_fingerprint(meta: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_topology_keys(meta).tobytes()).hexdigest()


def _thread_metadata(requested: int | None) -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0))
    return {
        "array_regrid_num_threads_requested": requested,
        "array_regrid_num_threads_actual": int(get_num_threads()),
        "affinity_cpu_count": len(affinity),
        "affinity_cpu_list": affinity,
    }


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_regrid_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            value.copy() if isinstance(value, np.ndarray)
            else dict(value) if isinstance(value, dict)
            else value
        )
        for key, value in meta.items()
    }


def _canonical_topology_arrays(meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "tiles" in meta and "xy_idx" in meta:
        return np.asarray(meta["tiles"]), np.asarray(meta["xy_idx"])
    # Loader metadata predates the array export fields. Use the same importer as
    # the regrid engine so fixture identity cannot drift from runtime geometry.
    from wamrvit.quad.array_regrid_geometry import _import_topology_metadata

    top = _import_topology_metadata(
        len(meta["levels"]), meta, meta.get("cell_scale_mode", "level_idx")
    )
    tiles = np.column_stack((top.tile_ix, top.tile_iy)).astype(np.int32)
    xy_idx = np.column_stack((top.level_idx, top.x_idx, top.y_idx)).astype(np.int32)
    return tiles, xy_idx


def _write_regrid_artifact(
    path: str, values: np.ndarray, meta: dict[str, Any], status: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Write a comparator-compatible NPZ and return its small manifest record."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    tiles, xy_idx = _canonical_topology_arrays(meta)
    artifact_meta = dict(meta)
    artifact_meta["tiles"], artifact_meta["xy_idx"] = tiles, xy_idx
    np.savez(
        path,
        values=np.asarray(values),
        tiles=tiles,
        xy_idx=xy_idx,
        centers=np.asarray(meta["centers"]),
        levels=np.asarray(meta["levels"]),
        domain_json=np.asarray(json.dumps(meta["domain"], sort_keys=True)),
        status_json=np.asarray(json.dumps(status or {}, sort_keys=True)),
    )
    return {
        "path": os.path.abspath(path),
        "sha256": _sha256_file(path),
        "shape": list(np.asarray(values).shape),
        "dtype": str(np.asarray(values).dtype),
        "topology_sha256": _topology_fingerprint(artifact_meta),
    }


@contextmanager
def _timed(out_list: list[float], sync_cuda: bool = True):
    """Append elapsed seconds to out_list. CUDA-syncs at the boundaries
    so async kernel launches don't under-count GPU work."""
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        out_list.append(time.perf_counter() - t0)


def _detect_variant(model, ckpt_path: str) -> str:
    """One of: adaptive_uniform, adaptive_native, regular, swin."""
    cls_name = type(model).__name__
    if cls_name == "SwinV2Transformer":
        return "swin"
    if not getattr(model.config, "adaptive", False):
        return "regular"
    return (
        "adaptive_native"
        if getattr(model.config, "multi_scale_patch", False)
        else "adaptive_uniform"
    )


def _load_model(inf_cfg: dict, model_cfg: dict):
    """Auto-detect SwinV2 vs QuadTreeTransformer from the checkpoint's config.json,
    mirroring rollout_regular.py:28-44."""
    ckpt_path = inf_cfg["checkpoint_path"]
    is_diffusers = inf_cfg.get("is_diffusers", True)
    cls_name = model_cfg.get("_class_name", "QuadTreeTransformer")
    if is_diffusers:
        ckpt_config_path = os.path.join(ckpt_path, "config.json")
        if os.path.exists(ckpt_config_path):
            with open(ckpt_config_path) as f:
                cls_name = json.load(f).get("_class_name", cls_name)
    if cls_name == "SwinV2Transformer":
        from wamrvit.swin_transformer import SwinV2Transformer

        model_cls = SwinV2Transformer
    else:
        model_cls = QuadTreeTransformer
    if is_diffusers:
        print(f"Loading {cls_name} via diffusers from: {ckpt_path}")
        model = model_cls.from_pretrained(ckpt_path)
    else:
        print(f"Loading {cls_name} state dict from: {ckpt_path}")
        model = model_cls(**model_cfg)
        sd = torch.load(ckpt_path, map_location="cpu")
        if "module." in list(sd.keys())[0]:
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
    return model


class BenchmarkRunner:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        inf_cfg = config["inference"]
        self.model = _load_model(inf_cfg, config["model"])
        self.model.to(self.device).eval()
        self.variant = _detect_variant(self.model, inf_cfg["checkpoint_path"])
        self.transform = instantiate_from_config(config["transform"])

        # Inference-loop knobs (mirrors AutoregressivePredictor* constructors).
        self.predict_steps = int(inf_cfg.get("predict_steps", 20))
        self.model_return_seq_len = int(self.model.config.return_seq_len)
        self.num_forward_calls = math.ceil(self.predict_steps / self.model_return_seq_len)
        self.pred_mode = inf_cfg.get("pred_mode", "target")
        self.regrid_interval = inf_cfg.get("regrid_interval", None)
        self.regrid_adapt_nearby = int(inf_cfg.get("regrid_adapt_nearby", 0))
        self.allow_coarsening = inf_cfg.get("allow_coarsening", True)
        self.regrid_backend = inf_cfg.get("regrid_backend", "object")
        if self.regrid_backend not in ("object", "array"):
            raise ValueError(
                f"inference.regrid_backend={self.regrid_backend!r}; expected 'object' or 'array'."
            )
        self.array_regrid_value_storage = inf_cfg.get(
            "array_regrid_value_storage", "copy")
        self.array_regrid_payload_backend = inf_cfg.get(
            "array_regrid_payload_backend", "cpu_eager"
        )
        self.array_regrid_capacity = int(inf_cfg.get("array_regrid_capacity", 8192))
        self.array_regrid_num_threads = inf_cfg.get("array_regrid_num_threads")
        if self.regrid_backend == "array":
            configure_array_regrid_num_threads(self.array_regrid_num_threads)
            print(
                "Array regrid runtime: "
                f"value_storage={self.array_regrid_value_storage}, "
                f"payload_backend={self.array_regrid_payload_backend}, "
                f"capacity={self.array_regrid_capacity}, "
                f"numba_threads={get_num_threads()}, "
                f"affinity_cpus={len(os.sched_getaffinity(0))}"
            )

        loader_params = config.get("file_loader", {}).get("params", {}) or {}
        self.cell_scale_mode = loader_params.get("cell_scale_mode", "area")
        self.adapt_on_channels = loader_params.get("adapt_on_channels", None)
        self.tol_frac = loader_params.get("tol_frac", 0.01)
        self.regrid_tol_frac = inf_cfg.get("regrid_tol_frac", self.tol_frac)
        field_names = list(loader_params.get("field_names", []))
        detector_channels = self.adapt_on_channels or list(range(len(field_names)))
        self.regrid_detector_fields = tuple(
            field_names[index] if index < len(field_names) else f"channel_{index}"
            for index in detector_channels
        )

        if self.variant in ("adaptive_uniform", "adaptive_native"):
            if self.regrid_interval is None:
                warnings.warn(
                    "regrid_interval not set on an adaptive variant; benchmark will skip regrid."
                )
            elif self.regrid_interval % self.model_return_seq_len != 0:
                warnings.warn(
                    f"regrid_interval ({self.regrid_interval}) not divisible by return_seq_len "
                    f"({self.model_return_seq_len}); regrids snap to forward-call boundaries."
                )

        self.num_levels = (
            int(getattr(self.model.config, "max_level_idx", 0)) + 1
            if self.variant == "adaptive_native"
            else 0
        )

        # Benchmark knobs.
        b_cfg = config["benchmark"]
        self.num_trajectories = int(b_cfg.get("num_trajectories", 10))
        self.seed = int(b_cfg.get("seed", 42))
        self.warmup_steps = int(b_cfg.get("warmup_steps", 1))
        self.collect_memory = bool(b_cfg.get("collect_memory", True))
        self.variant_tag = b_cfg.get("variant_tag", self.variant)
        self.dataset_tag = b_cfg.get("dataset_tag", "unknown")
        self.artifact_dir = b_cfg.get("artifact_dir")
        configured_artifact_calls = b_cfg.get("artifact_call_indices")
        self.artifact_call_indices = (
            {int(call_idx) for call_idx in configured_artifact_calls}
            if configured_artifact_calls is not None else None
        )
        self.artifact_manifest: list[dict[str, Any]] = []
        self._artifact_fixture_id: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    # Sampling: N random windows from the test split, seeded so the same
    # set is reused across variants on a given dataset. Window-level
    # (not trajectory-level) sampling: PLI/TRL test splits span many
    # trajectories with traj_idx varying; AMReX is single-trajectory with
    # all windows on one traj — both are handled the same way.
    # ------------------------------------------------------------------

    def _build_test_windows(self) -> list[dict[str, Any]]:
        """Replicate rollout_adaptive.py:874-888 windowing + split."""
        cfg = self.config
        file_parser = instantiate_from_config(cfg["file_parser"])
        all_paths = file_parser(cfg["data"]["glob_pattern"])

        # We don't need predict_steps GT frames here — the benchmark drops the
        # target — so leave window_generator at its default return_seq_len.
        wg_cfg = dict(cfg["window_generator"])
        wg_cfg["params"] = dict(wg_cfg.get("params", {}))
        wg_cfg["params"]["file_path_list"] = all_paths
        windows = instantiate_from_config(wg_cfg)
        split_ratio = float(cfg["inference"].get("split_ratio", 0.8))
        split_idx = int(len(windows) * split_ratio)
        return windows[split_idx:]

    def _pick_test_windows(self, test_windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(test_windows) < self.num_trajectories:
            raise RuntimeError(
                f"Test split has only {len(test_windows)} windows; "
                f"asked for {self.num_trajectories}."
            )
        rng = np.random.default_rng(self.seed)
        chosen_idx = rng.choice(len(test_windows), size=self.num_trajectories, replace=False)
        return [test_windows[i] for i in sorted(chosen_idx.tolist())]

    def _build_mapper(self) -> Seq2SeqMapper:
        """Same loader+transform construction as production rollout, with
        augmentation knobs forced off (we're inferring, not training)."""
        loader_cfg = dict(self.config["file_loader"])
        loader_cfg["params"] = dict(loader_cfg.get("params", {}) or {})
        for k in ("resample_coarsen_ratio", "resample_refine_ratio"):
            if loader_cfg["params"].get(k, 0.0) > 0.0:
                loader_cfg["params"][k] = 0.0
        loader = instantiate_from_config(loader_cfg)
        transform = instantiate_from_config(self.config["transform"])
        return Seq2SeqMapper(loader=loader, transform=transform)

    def _materialize_window(
        self, window: dict[str, Any], mapper: Seq2SeqMapper
    ) -> dict[str, Any]:
        """Load one selected window onto the benchmark device.

        Mapping is deliberately outside the timed rollout. ``run`` calls this
        one window at a time so peak memory represents batch-size-one inference
        rather than ten simultaneously resident benchmark fixtures.
        """
        batch = {
            "input_paths": [window["input_paths"]],
            "target_paths": [window["target_paths"]],
        }
        mapped = mapper(batch)
        sp = self._materialize_starting_point(mapped)
        sp["traj_idx"] = int(window.get("traj_idx", 0))
        sp["frame_idx"] = int(window.get("frame_idx", 0))
        sp["input_paths"] = list(window["input_paths"])
        return sp

    def sample_starting_points(self) -> list[dict[str, Any]]:
        """Materialize all selected fixtures for diagnostic callers.

        The production timing path in ``run`` streams fixtures through
        ``_materialize_window`` instead. This compatibility method remains for
        topology diagnostics that intentionally inspect several fixtures.
        """
        windows = self._build_test_windows()
        chosen = self._pick_test_windows(windows)
        mapper = self._build_mapper()
        return [self._materialize_window(window, mapper) for window in chosen]

    def _materialize_starting_point(self, mapped: dict[str, np.ndarray]) -> dict[str, Any]:
        """Convert mapper output into ready-to-use torch tensors on device.
        Layout differs by variant; the per-step kernels read these fields."""

        def _unpack(arr):
            return arr[0] if (isinstance(arr, np.ndarray) and arr.dtype == object) else arr

        if self.variant == "regular" or self.variant == "swin":
            inp = torch.from_numpy(_unpack(mapped["input"])).to(
                self.device, dtype=torch.float32
            )  # (B, C, T_in, H, W)
            return {"input": inp}

        # Adaptive paths: meta dict + per-variant input layout.
        domain = {
            "xmin": _unpack(mapped["xmin"]).item(),
            "xmax": _unpack(mapped["xmax"]).item(),
            "ymin": _unpack(mapped["ymin"]).item(),
            "ymax": _unpack(mapped["ymax"]).item(),
            "max_level_idx": _unpack(mapped["max_level_idx"]).item(),
            "tile_width": _unpack(mapped["tile_width"]).item(),
            "tile_height": _unpack(mapped["tile_height"]).item(),
        }
        ref_centers_np = _unpack(mapped["centers"])
        ref_levels_np = _unpack(mapped["levels"])
        if ref_centers_np.ndim == 3:
            ref_centers_np = ref_centers_np.squeeze(0)
        if ref_levels_np.ndim == 2:
            ref_levels_np = ref_levels_np.squeeze(0)
        meta = {"centers": ref_centers_np, "levels": ref_levels_np, "domain": domain}

        if self.variant == "adaptive_uniform":
            inp = torch.from_numpy(_unpack(mapped["input"])).to(self.device, dtype=torch.float32)
            if inp.ndim == 6:
                inp = inp.squeeze(0)  # (N, C, T_in, Ph, Pw)
            centers = torch.from_numpy(ref_centers_np).to(self.device, dtype=torch.float32)
            return {"input": inp, "centers": centers, "meta": meta}

        # adaptive_native: per-level columns + leaf_to_bucket + centers.
        inputs_by_level, _, leaf_to_bucket, centers = unpack_native_batch(
            mapped,
            self.num_levels,
            self.device,
            non_blocking=False,
        )
        return {
            "input_by_level": inputs_by_level,
            "leaf_to_bucket": leaf_to_bucket,
            "centers": centers,
            "meta": meta,
        }

    # ------------------------------------------------------------------
    # Warmup: trigger numba jit compile + cuDNN autotune. Output discarded.
    # ------------------------------------------------------------------

    def warmup(self, sp: dict[str, Any]):
        self._synthetic_warmup_durations_s = []
        if self.variant == "adaptive_uniform" and self.regrid_backend == "array":
            from wamrvit.quad.array_regrid import warm_array_regrid_kernels

            data = sp["input"]
            start = time.perf_counter()
            warm_array_regrid_kernels(
                flat_channels=int(data.shape[1]) * int(data.shape[2]),
                patch_size=(int(data.shape[-2]), int(data.shape[-1])),
                warm_source=self.array_regrid_value_storage == "source",
                warm_copy=self.array_regrid_value_storage == "copy",
                output_layout="sequence",
            )
            self._synthetic_warmup_durations_s.append(time.perf_counter() - start)
        self._warmup_durations_s = []
        for _ in range(self.warmup_steps):
            start = time.perf_counter()
            self._timed_rollout(sp, _is_warmup=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._warmup_durations_s.append(time.perf_counter() - start)

    # ------------------------------------------------------------------
    # Per-trajectory timed rollout.
    # ------------------------------------------------------------------

    def _timed_rollout(self, sp: dict[str, Any], _is_warmup: bool = False) -> dict[str, Any]:
        """Run one trajectory's autoregressive rollout, returning per-call
        forward and regrid times. _is_warmup just discards the timings."""
        forward_times: list[float] = []
        regrid_times: list[float] = []  # 0.0 for non-regrid calls; positive on regrid calls.
        n_cells_per_call: list[int] = []  # token count fed to each forward (= centers.shape[0]).
        # Per-event sub-step lists: list[float] of length == num_regrid_events.
        # Empty for regular/swin variants. Sum of substeps ≈ regrid_times sum
        # (modulo nanoseconds in trivial Python ops between the with-blocks).
        rg_substeps: dict[str, list[float]] = {
            "d2h": [], "t2q": [], "core": [], "q2t": [], "h2d": [],
        }
        array_regrid_events: list[dict[str, Any]] = []

        if self.variant == "adaptive_uniform":
            fixture_id = (int(sp.get("traj_idx", -1)), int(sp.get("frame_idx", -1)))
            capture_artifacts = bool(
                not _is_warmup
                and self.artifact_dir
                and (self._artifact_fixture_id is None or self._artifact_fixture_id == fixture_id)
            )
            if capture_artifacts and self._artifact_fixture_id is None:
                self._artifact_fixture_id = fixture_id
            self._rollout_uniform(
                sp, forward_times, regrid_times, n_cells_per_call, rg_substeps,
                array_regrid_events, capture_artifacts=capture_artifacts,
            )
        elif self.variant == "adaptive_native":
            self._rollout_native(sp, forward_times, regrid_times, n_cells_per_call, rg_substeps)
        else:  # regular or swin
            self._rollout_regular(sp, forward_times, regrid_times, n_cells_per_call)

        return {
            "traj_idx": sp.get("traj_idx", -1),
            "frame_idx": sp.get("frame_idx", -1),
            "input_paths": sp.get("input_paths", []),
            "per_call_forward_s": forward_times,
            "per_call_regrid_s": regrid_times,
            "per_call_num_cells": n_cells_per_call,
            "forward_s": float(sum(forward_times)),
            "regrid_s": float(sum(regrid_times)),
            "total_s": float(sum(forward_times) + sum(regrid_times)),
            "mean_num_cells": float(np.mean(n_cells_per_call)) if n_cells_per_call else 0.0,
            "regrid_substeps": rg_substeps,
            "array_regrid_events": array_regrid_events,
        }

    # ----------------- variant kernels -----------------

    def _rollout_uniform(
        self,
        sp: dict[str, Any],
        fwd_t: list[float],
        rg_t: list[float],
        n_cells_t: list[int],
        rg_substeps: dict[str, list[float]],
        array_regrid_events: list[dict[str, Any]] | None = None,
        *,
        capture_artifacts: bool = False,
    ):
        if array_regrid_events is None:
            array_regrid_events = []
        inputs = sp["input"].clone()
        centers = sp["centers"].clone()
        meta = {
            k: (v.copy() if isinstance(v, np.ndarray) else dict(v) if isinstance(v, dict) else v)
            for k, v in sp["meta"].items()
        }
        R = self.model_return_seq_len
        T_in = inputs.shape[2]

        with torch.no_grad():
            for call_idx in range(self.num_forward_calls):
                timestep_idx = call_idx * R
                # --- Regrid block (timed iff a regrid actually happens) ---
                regrid_block = []
                artifact_payload = None
                if (
                    self.regrid_interval is not None
                    and timestep_idx > 0
                    and timestep_idx % self.regrid_interval == 0
                ):
                    with _timed(regrid_block):
                        with _timed(rg_substeps["d2h"], sync_cuda=True):
                            N, c_in, t_in, h, w = inputs.shape
                            if self.regrid_backend == "array":
                                sequence_input = (
                                    inputs
                                    if getattr(
                                        self, "array_regrid_payload_backend", "cpu_eager"
                                    ) == "gpu_replay"
                                    else inputs.detach().cpu().numpy()
                                )
                            else:
                                flat_input = (
                                    inputs.transpose(1, 2)
                                    .contiguous()
                                    .view(N, t_in * c_in, h, w)
                                    .cpu()
                                    .numpy()
                                )
                        ch_offset = (t_in - 1) * c_in
                        use_channels = (
                            [ch + ch_offset for ch in self.adapt_on_channels]
                            if self.adapt_on_channels is not None
                            else list(range(ch_offset, t_in * c_in))
                        )
                        if self.regrid_backend == "array":
                            input_meta = _copy_regrid_meta(meta)
                            rg_substeps["t2q"].append(0.0)
                            with _timed(rg_substeps["core"], sync_cuda=False):
                                new_sequence_np, meta = regrid_uniform_sequence_dispatch(
                                    sequence_input, meta, backend="array", profiler=None,
                                    max_passes=10, cell_scale_mode=self.cell_scale_mode,
                                    tol_frac=self.regrid_tol_frac, channel=use_channels,
                                    adapt_nearby=self.regrid_adapt_nearby,
                                    allow_coarsening=self.allow_coarsening,
                                    value_storage=self.array_regrid_value_storage,
                                    capacity=self.array_regrid_capacity,
                                    status_collector=array_regrid_events,
                                    payload_backend=getattr(
                                        self, "array_regrid_payload_backend", "cpu_eager"
                                    ),
                                    detector_channels=self.adapt_on_channels,
                                    detector_fields=getattr(
                                        self, "regrid_detector_fields", None
                                    ),
                                )
                            measured_events = [
                                event for event in array_regrid_events
                                if event.get("kind") == "measured"
                            ]
                            if measured_events:
                                levels, counts = np.unique(
                                    np.asarray(meta["levels"]), return_counts=True
                                )
                                measured_events[-1]["leaf_count"] = int(len(meta["levels"]))
                                measured_events[-1]["per_level_counts"] = {
                                    str(int(level)): int(count)
                                    for level, count in zip(levels, counts)
                                }
                            if (
                                capture_artifacts
                                and self.artifact_dir
                                and (
                                    self.artifact_call_indices is None
                                    or call_idx in self.artifact_call_indices
                                )
                            ):
                                stem = (
                                    f"traj{int(sp.get('traj_idx', -1))}_"
                                    f"frame{int(sp.get('frame_idx', -1))}_call{call_idx}"
                                )
                                status = measured_events[-1].get("status", {}) if measured_events else {}
                                artifact_payload = (
                                    stem, call_idx, sequence_input, input_meta,
                                    new_sequence_np, meta, status,
                                )
                            rg_substeps["q2t"].append(0.0)
                        else:
                            with _timed(rg_substeps["t2q"], sync_cuda=False):
                                qt = tensor_to_quadtree(
                                    flat_input, meta, cell_scale_mode=self.cell_scale_mode
                                )
                            with _timed(rg_substeps["core"], sync_cuda=False):
                                regrid(
                                    qt,
                                    tol_frac=self.regrid_tol_frac,
                                    channel=use_channels,
                                    max_passes=10,
                                    adapt_nearby=self.regrid_adapt_nearby,
                                    allow_coarsening=self.allow_coarsening,
                                    disable_warnings=True,
                                )
                            with _timed(rg_substeps["q2t"], sync_cuda=False):
                                new_input_np, meta = quadtree_to_tensor(
                                    qt, return_tensor=False, cell_scale_mode=self.cell_scale_mode
                                )
                                new_N = new_input_np.shape[0]
                                new_sequence_np = np.ascontiguousarray(
                                    new_input_np.reshape(new_N, t_in, c_in, h, w)
                                    .transpose(0, 2, 1, 3, 4))
                        with _timed(rg_substeps["h2d"], sync_cuda=True):
                            centers = torch.from_numpy(meta["centers"]).to(
                                self.device, dtype=torch.float32
                            )
                            inputs = (
                                new_sequence_np.to(self.device, dtype=torch.float32)
                                if isinstance(new_sequence_np, torch.Tensor)
                                else torch.from_numpy(new_sequence_np).to(
                                    self.device, dtype=torch.float32
                                )
                            )
                rg_t.append(regrid_block[0] if regrid_block else 0.0)
                # Artifact I/O is deliberately outside every benchmark timing scope.
                if artifact_payload is not None:
                    stem, artifact_call, before, before_meta, after, after_meta, status = artifact_payload
                    if not self.artifact_manifest:
                        record = _write_regrid_artifact(
                            os.path.join(self.artifact_dir, f"{stem}_input.npz"),
                            before, before_meta,
                        )
                        record.update({"kind": "input", "call_idx": artifact_call})
                        self.artifact_manifest.append(record)
                    record = _write_regrid_artifact(
                        os.path.join(self.artifact_dir, f"{stem}_output.npz"),
                        after, after_meta, status,
                    )
                    record.update({"kind": "output", "call_idx": artifact_call})
                    self.artifact_manifest.append(record)

                # --- Forward block ---
                n_cells_t.append(int(centers.shape[0]))
                fwd_block = []
                with _timed(fwd_block):
                    pred_full = self.model(inputs, centers)  # (N, C, R, Ph, Pw)
                    if self.pred_mode == "residual":
                        pred_full = pred_full + inputs[:, :, -1].unsqueeze(2)
                    if call_idx < self.num_forward_calls - 1:
                        num_from_input = max(T_in - R, 0)
                        if num_from_input > 0:
                            inputs = torch.cat((inputs[:, :, -num_from_input:], pred_full), dim=2)
                        else:
                            inputs = pred_full[:, :, -T_in:]
                fwd_t.append(fwd_block[0])

    def _rollout_native(
        self,
        sp: dict[str, Any],
        fwd_t: list[float],
        rg_t: list[float],
        n_cells_t: list[int],
        rg_substeps: dict[str, list[float]],
    ):
        curr = {lvl: v.clone() for lvl, v in sp["input_by_level"].items()}
        leaf_to_bucket = sp["leaf_to_bucket"].clone()
        centers = sp["centers"].clone()
        meta = {
            k: (v.copy() if isinstance(v, np.ndarray) else dict(v) if isinstance(v, dict) else v)
            for k, v in sp["meta"].items()
        }
        R = self.model_return_seq_len
        T_in = next(v.shape[2] for v in curr.values() if v.shape[0] > 0)

        with torch.no_grad():
            for call_idx in range(self.num_forward_calls):
                timestep_idx = call_idx * R

                regrid_block = []
                if (
                    self.regrid_interval is not None
                    and timestep_idx > 0
                    and timestep_idx % self.regrid_interval == 0
                ):
                    with _timed(regrid_block):
                        curr, leaf_to_bucket, centers, meta = self._regrid_native_inplace(
                            curr, leaf_to_bucket, meta, rg_substeps
                        )
                rg_t.append(regrid_block[0] if regrid_block else 0.0)

                n_cells_t.append(int(centers.shape[0]))
                fwd_block = []
                with _timed(fwd_block):
                    pred_by_level = self.model.forward_multi_scale(curr, leaf_to_bucket, centers)
                    if self.pred_mode == "residual":
                        for lvl in pred_by_level:
                            pred_by_level[lvl] = pred_by_level[lvl] + curr[lvl][:, :, -1:]
                    if call_idx < self.num_forward_calls - 1:
                        num_from_input = max(T_in - R, 0)
                        for lvl in curr:
                            if num_from_input > 0:
                                curr[lvl] = torch.cat(
                                    (curr[lvl][:, :, -num_from_input:], pred_by_level[lvl]),
                                    dim=2,
                                )
                            else:
                                curr[lvl] = pred_by_level[lvl][:, :, -T_in:]
                fwd_t.append(fwd_block[0])

    def _regrid_native_inplace(self, curr_by_level, leaf_to_bucket, meta, rg_substeps):
        """Mirror of AutoregressivePredictorAdaptive._regrid_native (no GT path).
        Inlines regrid_native() body so each sub-step gets its own timer."""
        from wamrvit.quad.quad_utils import quadtree_to_tensor_native, tensor_to_quadtree_native

        C_orig = T_in_local = None
        for arr in curr_by_level.values():
            if arr.shape[0] > 0:
                C_orig, T_in_local = arr.shape[1], arr.shape[2]
                break

        with _timed(rg_substeps["d2h"], sync_cuda=True):
            np_by_level = {lvl: arr.cpu().numpy() for lvl, arr in curr_by_level.items()}
            l2b_np = leaf_to_bucket.cpu().numpy()

        if self.regrid_backend == "array":
            # Array engine adapts the per-level packed buckets directly -- no quadtree
            # build/teardown -- so t2q/q2t are definitionally zero. The pack + array
            # regrid + per-level unpack are all timed as "core".
            rg_substeps["t2q"].append(0.0)
            with _timed(rg_substeps["core"], sync_cuda=False):
                # The array engine consumes and returns the 5-D native buckets
                # (n, C, T, H, W) directly -- array_regrid_native_from_sequence already
                # unfolds to 5-D -- so the object branch's pack-to-4-D / unpack scaffolding
                # does NOT apply here. Mirrors the working rollout usage
                # (rollout_adaptive.py, regrid_native_dispatch call ~:626): pass np_by_level
                # straight in, use new_by_level_np straight out.
                new_by_level_np, new_l2b_np, new_meta = regrid_native_dispatch(
                    np_by_level, l2b_np, meta, backend="array", profiler=None,
                    max_passes=10, C=C_orig, T=T_in_local, tol_frac=self.regrid_tol_frac,
                    cell_scale_mode=self.cell_scale_mode,
                    adapt_on_channels=self.adapt_on_channels,
                    adapt_nearby=self.regrid_adapt_nearby,
                    allow_coarsening=self.allow_coarsening,
                )
            rg_substeps["q2t"].append(0.0)
        else:
            with _timed(rg_substeps["t2q"], sync_cuda=False):
                flat_by_level: dict[int, np.ndarray] = {}
                for lvl, arr in np_by_level.items():
                    if arr.shape[0] == 0:
                        flat_by_level[lvl] = arr[:, :0]
                        continue
                    n, c, t, h, w = arr.shape
                    flat_by_level[lvl] = arr.transpose(0, 2, 1, 3, 4).reshape(n, t * c, h, w)
                qt = tensor_to_quadtree_native(
                    flat_by_level, l2b_np, meta, cell_scale_mode=self.cell_scale_mode
                )

            with _timed(rg_substeps["core"], sync_cuda=False):
                ch_offset = (T_in_local - 1) * C_orig
                use_channels = (
                    [ch + ch_offset for ch in self.adapt_on_channels]
                    if self.adapt_on_channels is not None
                    else list(range(ch_offset, T_in_local * C_orig))
                )
                regrid(
                    qt,
                    tol_frac=self.regrid_tol_frac,
                    channel=use_channels,
                    max_passes=10,
                    adapt_nearby=self.regrid_adapt_nearby,
                    allow_coarsening=self.allow_coarsening,
                    disable_warnings=True,
                )

            with _timed(rg_substeps["q2t"], sync_cuda=False):
                new_buckets, new_l2b_np, new_meta = quadtree_to_tensor_native(
                    qt, cell_scale_mode=self.cell_scale_mode
                )
                new_by_level_np: dict[int, np.ndarray] = {}
                for lvl, flat_arr in new_buckets.items():
                    n_l = flat_arr.shape[0]
                    if n_l == 0:
                        h_l = flat_arr.shape[-2] if flat_arr.ndim >= 3 else 0
                        w_l = flat_arr.shape[-1] if flat_arr.ndim >= 3 else 0
                        new_by_level_np[lvl] = np.zeros(
                            (0, C_orig, T_in_local, h_l, w_l), dtype=flat_arr.dtype
                        )
                        continue
                    _, tc, h_l, w_l = flat_arr.shape
                    new_by_level_np[lvl] = np.ascontiguousarray(
                        flat_arr.reshape(n_l, T_in_local, C_orig, h_l, w_l).transpose(0, 2, 1, 3, 4)
                    )

        with _timed(rg_substeps["h2d"], sync_cuda=True):
            new_by_level = {
                lvl: torch.from_numpy(arr).to(self.device, dtype=torch.float32)
                for lvl, arr in new_by_level_np.items()
            }
            new_l2b_t = torch.from_numpy(new_l2b_np).to(self.device).long()
            new_centers_t = torch.from_numpy(new_meta["centers"]).to(
                self.device, dtype=torch.float32
            )
        return new_by_level, new_l2b_t, new_centers_t, new_meta

    def _rollout_regular(
        self, sp: dict[str, Any], fwd_t: list[float], rg_t: list[float], n_cells_t: list[int]
    ):
        inputs = sp["input"].clone()
        B, C, T_in, H, W = inputs.shape
        centers = make_regular_centers(H, W, p=self.model.config.patch_size, device=self.device)
        R = self.model_return_seq_len

        with torch.no_grad():
            for call_idx in range(self.num_forward_calls):
                rg_t.append(0.0)  # No regrid on regular path; keep length aligned.
                n_cells_t.append(int(centers.shape[0]))
                fwd_block = []
                with _timed(fwd_block):
                    pred_full = self.model(inputs, centers)
                    if self.pred_mode == "residual":
                        pred_full = pred_full + inputs[:, :, -1].unsqueeze(2)
                    if call_idx < self.num_forward_calls - 1:
                        num_from_input = max(T_in - R, 0)
                        if num_from_input > 0:
                            inputs = torch.cat((inputs[:, :, -num_from_input:], pred_full), dim=2)
                        else:
                            inputs = pred_full[:, :, -T_in:]
                fwd_t.append(fwd_block[0])

    # ------------------------------------------------------------------
    # Aggregate + JSON output
    # ------------------------------------------------------------------

    def _aggregate(self, per_traj: list[dict[str, Any]]) -> dict[str, Any]:
        totals = np.array([t["total_s"] for t in per_traj])
        forwards = np.array([t["forward_s"] for t in per_traj])
        regrids = np.array([t["regrid_s"] for t in per_traj])

        # mean per-call forward (averaged across steps and trajectories).
        per_call_fwd = np.array([t["per_call_forward_s"] for t in per_traj])  # (N_traj, num_calls)
        per_call_rg = np.array([t["per_call_regrid_s"] for t in per_traj])

        # Mean only over the regrid-event slots; averaging zeros from non-regrid
        # calls would dilute the regrid cost.
        mask = per_call_rg > 0
        if mask.any():
            mean_per_call_regrid_only = float(per_call_rg[mask].mean())
            num_regrid_events_per_traj = int(mask[0].sum())
        else:
            mean_per_call_regrid_only = 0.0
            num_regrid_events_per_traj = 0

        q1, median, q3 = np.percentile(totals, [25, 50, 75])
        agg = {
            "mean_total_s": float(totals.mean()),
            "std_total_s": float(totals.std(ddof=0)),
            "median_total_s": float(median),
            "iqr_total_s": float(q3 - q1),
            "min_total_s": float(totals.min()),
            "max_total_s": float(totals.max()),
            "mean_forward_s": float(forwards.mean()),
            "std_forward_s": float(forwards.std(ddof=0)),
            "mean_regrid_s": float(regrids.mean()),
            "std_regrid_s": float(regrids.std(ddof=0)),
            "mean_per_call_forward_s": float(per_call_fwd.mean()),
            "mean_per_call_regrid_only_s": mean_per_call_regrid_only,
            "frac_regrid": float(regrids.sum() / max(totals.sum(), 1e-12)),
            "num_regrid_events_per_traj": num_regrid_events_per_traj,
        }

        # Token-count sanity (= centers.shape[0] at each forward call).
        if all("per_call_num_cells" in t for t in per_traj):
            per_call_cells = np.array([t["per_call_num_cells"] for t in per_traj], dtype=float)
            agg["mean_num_cells_per_call"] = float(per_call_cells.mean())
            agg["std_num_cells_per_call"] = float(per_call_cells.std(ddof=0))
            agg["min_num_cells"] = int(per_call_cells.min())
            agg["max_num_cells"] = int(per_call_cells.max())

        # PyTorch CUDA allocator peaks. Each trajectory resets peak statistics
        # after warmup with exactly one starting window resident. Keep values in
        # bytes in the JSON; table consumers can choose their display unit.
        memory_keys = (
            "torch_cuda_baseline_allocated_bytes",
            "torch_cuda_peak_allocated_bytes",
            "torch_cuda_incremental_peak_allocated_bytes",
            "torch_cuda_baseline_reserved_bytes",
            "torch_cuda_peak_reserved_bytes",
        )
        if all(all(key in t for key in memory_keys) for t in per_traj):
            for key in memory_keys:
                values = np.asarray([t[key] for t in per_traj], dtype=np.float64)
                suffix = key.removeprefix("torch_cuda_")
                agg[f"mean_torch_cuda_{suffix}"] = float(values.mean())
                agg[f"max_torch_cuda_{suffix}"] = int(values.max())

        # Sub-step breakdown (adaptive variants only). Per-event mean across
        # all regrid events from all trajectories — directly comparable to
        # mean_per_call_regrid_only_s. Sum of substeps ≈ envelope.
        substep_keys = ["d2h", "t2q", "core", "q2t", "h2d"]
        all_vals: dict[str, list[float]] = {k: [] for k in substep_keys}
        for t in per_traj:
            rs = t.get("regrid_substeps", {})
            for k in substep_keys:
                all_vals[k].extend(rs.get(k, []))
        if any(all_vals[k] for k in substep_keys):
            for k in substep_keys:
                v = all_vals[k]
                agg[f"mean_per_event_regrid_{k}_s"] = float(np.mean(v)) if v else 0.0

        array_events = [
            event for trajectory in per_traj
            for event in trajectory.get("array_regrid_events", [])
            if event.get("kind") == "measured"
        ]
        if array_events:
            agg["array_regrid_fallback_count"] = sum(
                event.get("status", {}).get("backend") != "array" for event in array_events
            )
            agg["gpu_replay_fallback_count"] = sum(
                bool(event.get("status", {}).get("gpu_replay_fallback_reason"))
                for event in array_events
            )
            agg["mean_array_regrid_leaf_count"] = float(np.mean([
                event.get("leaf_count", 0) for event in array_events
            ]))
        return agg

    def run(self) -> dict[str, Any]:
        windows = self._pick_test_windows(self._build_test_windows())
        if len(windows) == 0:
            raise RuntimeError("No starting points sampled.")
        mapper = self._build_mapper()
        cuda_memory_enabled = bool(
            self.collect_memory
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )

        print(f"Sampled {len(windows)} test trajectories. Warming up...")
        per_traj: list[dict[str, Any]] = []
        selected_inputs: list[dict[str, Any]] = []
        for i, window in enumerate(windows):
            # Keep only one starting window resident. Mapping and H2D transfer
            # remain outside the forward/regrid timing contract.
            sp = self._materialize_window(window, mapper)
            if i == 0:
                self.warmup(sp)
                print(
                    f"Warmup done. Running {len(windows)} timed rollouts × "
                    f"{self.predict_steps} steps..."
                )

            if cuda_memory_enabled:
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
                baseline_allocated = int(torch.cuda.memory_allocated(self.device))
                baseline_reserved = int(torch.cuda.memory_reserved(self.device))

            res = self._timed_rollout(sp)

            if cuda_memory_enabled:
                torch.cuda.synchronize(self.device)
                peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
                res.update({
                    "torch_cuda_baseline_allocated_bytes": baseline_allocated,
                    "torch_cuda_peak_allocated_bytes": peak_allocated,
                    "torch_cuda_incremental_peak_allocated_bytes": max(
                        peak_allocated - baseline_allocated, 0
                    ),
                    "torch_cuda_baseline_reserved_bytes": baseline_reserved,
                    "torch_cuda_peak_reserved_bytes": int(
                        torch.cuda.max_memory_reserved(self.device)
                    ),
                })

            selected_inputs.append({
                "traj_idx": int(sp.get("traj_idx", -1)),
                "frame_idx": int(sp.get("frame_idx", -1)),
                "input_paths": sp.get("input_paths", []),
                "topology_sha256": (
                    _topology_fingerprint(sp["meta"]) if "meta" in sp else None
                ),
            })
            per_traj.append(res)
            memory_text = (
                f"  peak_cuda={res['torch_cuda_peak_allocated_bytes'] / 2**30:.2f}GiB"
                if cuda_memory_enabled else ""
            )
            print(
                f"  [{i + 1}/{len(windows)}] traj_idx={res['traj_idx']:>3}  "
                f"total={res['total_s']:.3f}s  fwd={res['forward_s']:.3f}s  "
                f"regrid={res['regrid_s']:.3f}s  "
                f"mean_cells={res['mean_num_cells']:.1f}{memory_text}"
            )
            del sp

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        result = {
            "metadata": {
                "model_variant": self.variant,
                "variant_tag": self.variant_tag,
                "dataset_tag": self.dataset_tag,
                "checkpoint_path": self.config["inference"]["checkpoint_path"],
                "predict_steps": self.predict_steps,
                "regrid_interval": self.regrid_interval,
                "num_trajectories": self.num_trajectories,
                "seed": self.seed,
                "warmup_steps": self.warmup_steps,
                "collect_memory": self.collect_memory,
                "cuda_memory_collected": cuda_memory_enabled,
                "swin_patch_embedding": (
                    "composed_conv3d" if self.variant == "swin" else None
                ),
                "memory_protocol": (
                    "one starting window resident; torch.cuda peaks reset after warmup "
                    "before each measured rollout; non-PyTorch CUDA allocations excluded"
                    if cuda_memory_enabled
                    else "disabled (collect_memory=false or CUDA unavailable)"
                ),
                "device": str(self.device),
                "gpu_name": gpu_name,
                "torch_version": torch.__version__,
                "model_return_seq_len": self.model_return_seq_len,
                "num_forward_calls": self.num_forward_calls,
                "pred_mode": self.pred_mode,
                "regrid_tol_frac": self.regrid_tol_frac,
                "regrid_backend": self.regrid_backend,
                "array_regrid_value_storage": self.array_regrid_value_storage,
                "array_regrid_payload_backend": getattr(
                    self, "array_regrid_payload_backend", "cpu_eager"
                ),
                "array_regrid_capacity": self.array_regrid_capacity,
                **_thread_metadata(self.array_regrid_num_threads),
                "hostname": platform.node(),
                "cpu_model": _cpu_model(),
                "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "warmup": {
                    "synthetic_s": getattr(self, "_synthetic_warmup_durations_s", []),
                    "real_input_s": getattr(self, "_warmup_durations_s", []),
                    "measured_s": [float(t["total_s"]) for t in per_traj],
                },
                "selected_inputs": selected_inputs,
                "artifact_manifest": self.artifact_manifest,
                **_git_metadata(),
            },
            "per_trajectory": per_traj,
            "aggregate": self._aggregate(per_traj),
        }
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Standalone YAML (legacy). Omit to use Hydra config groups.",
    )
    args, unknown = parser.parse_known_args()
    config = load_config(args, unknown)

    runner = BenchmarkRunner(config)
    result = runner.run()

    out_path = config["benchmark"].get("output_path")
    if not out_path:
        # Auto-derive from variant + dataset tag.
        out_dir = config["benchmark"].get("output_dir", "results/benchmark_inference")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{runner.dataset_tag}_{runner.variant_tag}.json")
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote: {out_path}")
    agg = result["aggregate"]
    print(f"  mean_total_s = {agg['mean_total_s']:.3f} ± {agg['std_total_s']:.3f}")
    if "max_torch_cuda_peak_allocated_bytes" in agg:
        print(
            "  peak_cuda_allocated = "
            f"{agg['max_torch_cuda_peak_allocated_bytes'] / 2**30:.2f} GiB max, "
            f"{agg['mean_torch_cuda_peak_allocated_bytes'] / 2**30:.2f} GiB mean"
        )
    if runner.variant in ("adaptive_uniform", "adaptive_native"):
        print(f"  mean_forward_s = {agg['mean_forward_s']:.3f}")
        print(f"  mean_regrid_s = {agg['mean_regrid_s']:.3f}  (frac={agg['frac_regrid']:.1%})")
        if "mean_per_event_regrid_core_s" in agg:
            envelope = agg.get("mean_per_call_regrid_only_s", 0.0)
            substep_sum = sum(
                agg.get(f"mean_per_event_regrid_{k}_s", 0.0)
                for k in ("d2h", "t2q", "core", "q2t", "h2d")
            )
            print(f"  per-event regrid breakdown (envelope={envelope:.4f}s):")
            print(f"    d2h  = {agg['mean_per_event_regrid_d2h_s']:.4f}s")
            print(f"    t2q  = {agg['mean_per_event_regrid_t2q_s']:.4f}s  (tensor_to_quadtree)")
            print(f"    core = {agg['mean_per_event_regrid_core_s']:.4f}s  (regrid() function)")
            print(f"    q2t  = {agg['mean_per_event_regrid_q2t_s']:.4f}s  (quadtree_to_tensor)")
            print(f"    h2d  = {agg['mean_per_event_regrid_h2d_s']:.4f}s")
            print(f"    sum  = {substep_sum:.4f}s")


if __name__ == "__main__":
    main()
