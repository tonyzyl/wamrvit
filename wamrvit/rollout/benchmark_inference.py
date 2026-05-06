"""Inference-time benchmark across model variants × datasets.

For one (model_variant × dataset) combo, samples N random test trajectories
(seeded RNG, t_start=0 each), preloads their starting windows into GPU memory,
warms up the inference path (njit + cuDNN autotune), then runs `predict_steps`
autoregressive iterations per trajectory while timing the forward pass and
regrid block separately. Writes a JSON with per-trajectory and aggregated means.

No metric computation, no GT loading beyond what the loader emits incidentally,
no Ray, no DDP. Single-GPU, single-process.

See docs/zany-petting-cookie or the plan file for design details.

Usage:
    uv run wamrvit/rollout/benchmark_inference.py \\
        +experiment=benchmark_inference \\
        dataset=pli_adaptive \\
        inference.checkpoint_path="/path/to/some_run/model" \\
        benchmark.output_path="results/benchmark_inference/pli_adaptive_uniform.json" \\
        benchmark.variant_tag=adaptive_uniform
"""

import argparse
import json
import math
import os
import time
import warnings
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from wamrvit.dataloader.trajectory_loader import Seq2SeqMapper
from wamrvit.native_train_utils import unpack_native_batch
from wamrvit.quad.adapt_wavelet import regrid, regrid_native
from wamrvit.quad.quad_utils import quadtree_to_tensor, tensor_to_quadtree
from wamrvit.quad.yt_utils import make_regular_centers
from wamrvit.quadtree_transformer import QuadTreeTransformer
from wamrvit.utils import instantiate_from_config, load_config


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

        loader_params = config.get("file_loader", {}).get("params", {}) or {}
        self.cell_scale_mode = loader_params.get("cell_scale_mode", "area")
        self.adapt_on_channels = loader_params.get("adapt_on_channels", None)
        self.tol_frac = loader_params.get("tol_frac", 0.01)
        self.regrid_tol_frac = inf_cfg.get("regrid_tol_frac", self.tol_frac)

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
        self.variant_tag = b_cfg.get("variant_tag", self.variant)
        self.dataset_tag = b_cfg.get("dataset_tag", "unknown")

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

    def sample_starting_points(self) -> list[dict[str, Any]]:
        windows = self._build_test_windows()
        chosen = self._pick_test_windows(windows)
        mapper = self._build_mapper()

        starting_points: list[dict[str, Any]] = []
        for w in chosen:
            batch = {
                "input_paths": [w["input_paths"]],
                "target_paths": [w["target_paths"]],
            }
            mapped = mapper(batch)
            sp = self._materialize_starting_point(mapped)
            sp["traj_idx"] = int(w.get("traj_idx", 0))
            sp["frame_idx"] = int(w.get("frame_idx", 0))
            sp["input_paths"] = list(w["input_paths"])
            starting_points.append(sp)
        return starting_points

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
        for _ in range(self.warmup_steps):
            self._timed_rollout(sp, _is_warmup=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Per-trajectory timed rollout.
    # ------------------------------------------------------------------

    def _timed_rollout(self, sp: dict[str, Any], _is_warmup: bool = False) -> dict[str, Any]:
        """Run one trajectory's autoregressive rollout, returning per-call
        forward and regrid times. _is_warmup just discards the timings."""
        forward_times: list[float] = []
        regrid_times: list[float] = []  # 0.0 for non-regrid calls; positive on regrid calls.
        n_cells_per_call: list[int] = []  # token count fed to each forward (= centers.shape[0]).

        if self.variant == "adaptive_uniform":
            self._rollout_uniform(sp, forward_times, regrid_times, n_cells_per_call)
        elif self.variant == "adaptive_native":
            self._rollout_native(sp, forward_times, regrid_times, n_cells_per_call)
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
        }

    # ----------------- variant kernels -----------------

    def _rollout_uniform(
        self, sp: dict[str, Any], fwd_t: list[float], rg_t: list[float], n_cells_t: list[int]
    ):
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
                if (
                    self.regrid_interval is not None
                    and timestep_idx > 0
                    and timestep_idx % self.regrid_interval == 0
                ):
                    with _timed(regrid_block):
                        N, c_in, t_in, h, w = inputs.shape
                        flat_input = (
                            inputs.transpose(1, 2)
                            .contiguous()
                            .view(N, t_in * c_in, h, w)
                            .cpu()
                            .numpy()
                        )
                        qt = tensor_to_quadtree(
                            flat_input, meta, cell_scale_mode=self.cell_scale_mode
                        )
                        ch_offset = (t_in - 1) * c_in
                        use_channels = (
                            [ch + ch_offset for ch in self.adapt_on_channels]
                            if self.adapt_on_channels is not None
                            else list(range(ch_offset, t_in * c_in))
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
                        new_input_np, meta = quadtree_to_tensor(
                            qt, return_tensor=False, cell_scale_mode=self.cell_scale_mode
                        )
                        centers = torch.from_numpy(meta["centers"]).to(
                            self.device, dtype=torch.float32
                        )
                        new_N = new_input_np.shape[0]
                        inputs = torch.from_numpy(
                            np.ascontiguousarray(new_input_np)
                            .reshape(new_N, t_in, c_in, h, w)
                            .transpose(0, 2, 1, 3, 4)
                        ).to(self.device, dtype=torch.float32)
                rg_t.append(regrid_block[0] if regrid_block else 0.0)

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
        self, sp: dict[str, Any], fwd_t: list[float], rg_t: list[float], n_cells_t: list[int]
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
                            curr, leaf_to_bucket, meta
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

    def _regrid_native_inplace(self, curr_by_level, leaf_to_bucket, meta):
        """Mirror of AutoregressivePredictorAdaptive._regrid_native (no GT path)."""
        C_orig = T_in = None
        for arr in curr_by_level.values():
            if arr.shape[0] > 0:
                C_orig, T_in = arr.shape[1], arr.shape[2]
                break
        np_by_level = {lvl: arr.cpu().numpy() for lvl, arr in curr_by_level.items()}
        l2b_np = leaf_to_bucket.cpu().numpy()
        new_by_level_np, new_l2b, new_meta = regrid_native(
            np_by_level,
            l2b_np,
            meta,
            C=C_orig,
            T=T_in,
            tol_frac=self.regrid_tol_frac,
            cell_scale_mode=self.cell_scale_mode,
            adapt_on_channels=self.adapt_on_channels,
            adapt_nearby=self.regrid_adapt_nearby,
            allow_coarsening=self.allow_coarsening,
        )
        new_by_level = {
            lvl: torch.from_numpy(arr).to(self.device, dtype=torch.float32)
            for lvl, arr in new_by_level_np.items()
        }
        new_l2b_t = torch.from_numpy(new_l2b).to(self.device).long()
        new_centers_t = torch.from_numpy(new_meta["centers"]).to(self.device, dtype=torch.float32)
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

        agg = {
            "mean_total_s": float(totals.mean()),
            "std_total_s": float(totals.std(ddof=0)),
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
        return agg

    def run(self) -> dict[str, Any]:
        starting_points = self.sample_starting_points()
        if len(starting_points) == 0:
            raise RuntimeError("No starting points sampled.")
        print(f"Sampled {len(starting_points)} test trajectories. Warming up...")
        self.warmup(starting_points[0])
        print(
            f"Warmup done. Running {len(starting_points)} timed rollouts × "
            f"{self.predict_steps} steps..."
        )
        per_traj = []
        for i, sp in enumerate(starting_points):
            res = self._timed_rollout(sp)
            per_traj.append(res)
            print(
                f"  [{i + 1}/{len(starting_points)}] traj_idx={res['traj_idx']:>3}  "
                f"total={res['total_s']:.3f}s  fwd={res['forward_s']:.3f}s  "
                f"regrid={res['regrid_s']:.3f}s  mean_cells={res['mean_num_cells']:.1f}"
            )

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
                "device": str(self.device),
                "gpu_name": gpu_name,
                "torch_version": torch.__version__,
                "model_return_seq_len": self.model_return_seq_len,
                "num_forward_calls": self.num_forward_calls,
                "pred_mode": self.pred_mode,
                "regrid_tol_frac": self.regrid_tol_frac,
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
    if runner.variant in ("adaptive_uniform", "adaptive_native"):
        print(f"  mean_forward_s = {agg['mean_forward_s']:.3f}")
        print(f"  mean_regrid_s = {agg['mean_regrid_s']:.3f}  (frac={agg['frac_regrid']:.1%})")


if __name__ == "__main__":
    main()
