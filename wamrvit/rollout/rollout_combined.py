"""
Combined rollout script that runs both adaptive and regular models side-by-side.
Computes per-refinement-level metrics for both models using the adaptive model's
quadtree structure, plus standard uniform-grid metrics.
"""

import argparse
import math
import os
import time
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import ray
import torch

from wamrvit.checkpoint import load_model
from wamrvit.dataloader.selection.splits import prepare_evaluation_windows
from wamrvit.dataloader.payload.mappers import Seq2SeqMapper, _copy_transport_control_columns
from wamrvit.dataloader.payload.transforms import apply_transform_src
from wamrvit.quad.quad_utils import (
    make_regular_centers,
    quadtree_to_tensor,
    tensor_to_uniform,
)
from wamrvit.rollout.regular_metrics import (
    METRIC_REGISTRY,
    eval_per_level_metric,
    mean_numeric_result_columns,
    normalize_numeric_result_columns,
)
from wamrvit.rollout.rollout_adaptive import (
    AutoregressivePredictorAdaptive,
    build_native_gt_buckets,
    build_uniform_gt,
    eval_native_uniform_metrics,
    eval_per_level_metrics,
    leaf_export_order,
)
from wamrvit.utils import instantiate_from_config, load_config

# ---------------------------------------------------------------------------
# Profiling helper
# ---------------------------------------------------------------------------
#
# Each predictor / evaluator maintains a per-call timings dict that we emit
# as ``time_<phase>`` columns in the batch output. ``sync_cuda=True`` forces a
# CUDA sync before and after the block so GPU op timings aren't underestimated
# (GPU kernels are launched async otherwise). Use only for blocks that
# actually touch the GPU; syncs are expensive on fully-CPU paths.


@contextmanager
def _timer(store: dict[str, float], key: str, sync_cuda: bool = False):
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        store[key] = store.get(key, 0.0) + (time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Data Mapper — extends Seq2SeqMapper to also forward input_src
# ---------------------------------------------------------------------------


class CombinedMapper(Seq2SeqMapper):
    """Extends Seq2SeqMapper to also pass through ``input_src`` for the regular model.

    For AMReX mode, AdaptiveLoader emits ``input_src_paths`` (disk paths)
    rather than an eager ``input_src`` tensor. The mapper materializes the
    tensor here via a lazily-instantiated ``YTAmReXRegularLoader`` so that
    the downstream RegularPredictor stays mode-agnostic — it always sees
    ``input_src`` as a transform-normalized tensor.
    """

    _amrex_src_loader = None  # lazy YTAmReXRegularLoader, built on first AMReX batch

    def __init__(self, loader, transform=None, regular_target_level: int = 0):
        super().__init__(loader=loader, transform=transform)
        # AMReX projection level for the *regular* path's input materialization.
        # Must match the regular checkpoint's training projection: the paper
        # p32 baselines use level 2 (1024×6144). Mismatch on Swin raises an
        # AssertionError in timm's PatchEmbed; on the ViT path it runs silently
        # OOD (no shape check).
        self.regular_target_level = int(regular_target_level)

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        timings: dict[str, float] = {}
        with _timer(timings, "time_mapper_total"):
            result = _copy_transport_control_columns(self._call_inner(batch, timings), batch)
        for k, v in timings.items():
            result[k] = np.array([v], dtype=np.float32)
        return result

    def _materialize_amrex_input_src(self, input_src_paths: list[str]) -> np.ndarray:
        """Load AMReX input frames via yt and return (T_in, C, H, W) numpy array."""
        if self._amrex_src_loader is None:
            from wamrvit.dataloader.payload.loaders import YTAmReXRegularLoader

            self._amrex_src_loader = YTAmReXRegularLoader(
                field_names=self.loader.field_names,
                domain_from="domain",
                target_level=self.regular_target_level,
            )
        # The YT loader expects non-empty target_paths; reuse the last input
        # path as a dummy target. The returned "target" is discarded here —
        # GT delivery for AMReX flows through ``target_src_paths`` separately.
        out = self._amrex_src_loader(list(input_src_paths), list(input_src_paths[-1:]))
        return out["input"]  # (T_in, C, H, W)

    def _apply_transform_src(self, arr_5d: np.ndarray) -> np.ndarray:
        """Thin wrapper around :func:`apply_transform_src` for src-layout tensors."""
        return apply_transform_src(self.transform, arr_5d)

    def _call_inner(
        self, batch: dict[str, np.ndarray], timings: dict[str, float]
    ) -> dict[str, np.ndarray]:
        batch_size = len(batch["input_paths"])
        if batch_size != 1:
            raise ValueError(f"Adaptive shall have batch size 1, but got {batch_size}")

        with _timer(timings, "time_mapper_loader"):
            return_dict = self.loader(batch["input_paths"][0], batch["target_paths"][0])

        # --- Native (multi-scale) path ---
        if "input_by_level" in return_dict:
            result = self._assemble_native_adaptive_result(return_dict, self.transform)
            # Add input_src for the regular model. (1, T_in, C, H, W) src layout —
            # needs the T↔C transpose for NormalizeArray.
            if return_dict.get("input_src") is not None:
                input_src_np = np.stack([return_dict["input_src"]])
                if self.transform:
                    input_src_np = self._apply_transform_src(input_src_np)
                result["input_src"] = input_src_np
            for key in ("idx", "traj_idx", "frame_idx"):
                if key in batch:
                    result[key] = batch[key]
            return result

        # --- Uniform path (original) ---
        input_arr = [return_dict["input"]]  # [(T_in, N, C, Ph, Pw)]
        target_arr = [return_dict["target"]]  # [(T_out, N, C, Ph, Pw)]
        input_src_arr = [return_dict.get("input_src")]  # [(T_in, C, H, W)] or [None]
        target_src_arr = [return_dict.get("target_src")]  # [(T_out, C, H, W)] or [None]
        target_src_paths_arr = [return_dict.get("target_src_paths")]
        input_src_paths_arr = [return_dict.get("input_src_paths")]
        meta = return_dict.get("meta")
        centers_arr = [meta["centers"] if meta is not None else []]  # [(N, 3)]
        levels_arr = [meta["levels"] if meta is not None else []]  # [(N,)]

        if input_arr[0].ndim != 5:
            raise ValueError("Combined rollout requires AdaptiveLoader (5D input)")

        # AMReX path-based delivery: loader emitted input_src_paths instead of
        # an eager input_src tensor. Materialize the tensor here via yt so the
        # downstream regular predictor sees the same (1, T_in, C, H, W) shape
        # as npz/hdf5 modes. Transform is applied below alongside PLI/TRL.
        if input_src_arr[0] is None and input_src_paths_arr[0] is not None:
            with _timer(timings, "time_mapper_amrex_yt_load"):
                input_src_arr[0] = self._materialize_amrex_input_src(input_src_paths_arr[0])

        # Permute (T, N, C, H, W) → (B=1, N, C, T, H, W) so predictor sees time along axis 3.
        inputs_np = np.stack(input_arr).transpose(0, 2, 3, 1, 4, 5)  # (1, N, C, T_in, Ph, Pw)
        targets_np = np.stack(target_arr).transpose(0, 2, 3, 1, 4, 5)  # (1, N, C, T_out, Ph, Pw)
        centers_np = np.stack(centers_arr) if centers_arr[0] is not None else None  # (1, N, 3)
        levels_np = np.stack(levels_arr) if levels_arr[0] is not None else None  # (1, N)
        targets_src_np = (
            np.stack(target_src_arr) if target_src_arr[0] is not None else None
        )  # (1, T_out, C, H, W)
        inputs_src_np = (
            np.stack(input_src_arr) if input_src_arr[0] is not None else None
        )  # (1, T_in,  C, H, W)
        target_src_paths_np = None
        if target_src_paths_arr[0] is not None:
            target_src_paths_np = np.empty(1, dtype=object)
            target_src_paths_np[0] = target_src_paths_arr[0]

        if self.transform:
            inputs_np = self.transform(inputs_np)
            targets_np = self.transform(targets_np)
            if targets_src_np is not None:
                targets_src_np = self._apply_transform_src(targets_src_np)
            if inputs_src_np is not None:
                inputs_src_np = self._apply_transform_src(inputs_src_np)

        result = {
            "input": inputs_np,
            "target": targets_np,
        }

        if meta is not None:
            result["centers"] = centers_np
            result["levels"] = levels_np
            result["xmin"] = np.array([meta["domain"]["xmin"]])
            result["xmax"] = np.array([meta["domain"]["xmax"]])
            result["ymin"] = np.array([meta["domain"]["ymin"]])
            result["ymax"] = np.array([meta["domain"]["ymax"]])
            result["max_level_idx"] = np.array([meta["domain"]["max_level_idx"]])
            result["tile_width"] = np.array([meta["domain"]["tile_width"]])
            result["tile_height"] = np.array([meta["domain"]["tile_height"]])

        if targets_src_np is not None:
            result["target_src"] = targets_src_np
        if target_src_paths_np is not None:
            result["target_src_paths"] = target_src_paths_np
        if inputs_src_np is not None:
            result["input_src"] = inputs_src_np

        for key in ("idx", "traj_idx", "frame_idx"):
            if key in batch:
                result[key] = batch[key]

        return result


# ---------------------------------------------------------------------------
# Adaptive Model Predictor
# ---------------------------------------------------------------------------


class AdaptivePredictor(AutoregressivePredictorAdaptive):
    """Shared adaptive rollout with the combined pipeline's timing and output schema."""

    def __init__(self, config: dict):
        super().__init__(config, checkpoint_key="checkpoint_path_adaptive")

    def _time_phase(self, name: str):
        return _timer(self._timings, name, sync_cuda=True)

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self._timings: dict[str, float] = {}
        with _timer(self._timings, "time_adaptive_total", sync_cuda=True):
            out = super().__call__(batch)
        # Normalization helpers: token count and step count for per-cell timing math.
        if "ref_centers" in out:
            ref = out["ref_centers"]
            while hasattr(ref, "ndim") and ref.ndim > 2 and ref.shape[0] == 1:
                ref = ref.squeeze(0)
            self._timings["adaptive_n_tokens"] = float(ref.shape[0]) if ref.ndim >= 1 else 0.0
        self._timings["adaptive_steps"] = float(self.predict_steps)
        for k, v in self._timings.items():
            out[k] = np.array([v], dtype=np.float32)
        return out

    def _format_output(self, out: dict, batch: dict, *, leaf_to_bucket=None) -> dict:
        for key in (
            "preds_list", "preds_by_level_list", "leaf_to_bucket_list",
            "centers_list", "levels_list",
        ):
            if key in out:
                out[f"adaptive_{key}"] = out.pop(key)
        # Only combined rollout forwards the regular model's dense input. Do not
        # add multi-GiB AMReX payloads to adaptive-only Ray output, or copy them here.
        if "input_src" in batch:
            source = batch["input_src"]
            source = source[0] if source.dtype == object else source
            if source is not None:
                out["input_src"] = source
        return out


# ---------------------------------------------------------------------------
# Regular Model Predictor
# ---------------------------------------------------------------------------


class RegularPredictor:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        inf_cfg = config["inference"]
        ckpt = inf_cfg["checkpoint_path_regular"]

        self.model = load_model(ckpt, config.get("model_regular", config["model"]))

        self.model.to(self.device)
        self.model.eval()

        self.transform = instantiate_from_config(config["transform"])
        self.predict_steps = inf_cfg.get("predict_steps", 1)
        self.model_return_seq_len = self.model.config.return_seq_len
        self.num_forward_calls = math.ceil(self.predict_steps / self.model_return_seq_len)
        self.pred_mode = inf_cfg.get("pred_mode", "target")

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        timings: dict[str, float] = {}
        with _timer(timings, "time_regular_total", sync_cuda=True):

            def _unpack(arr):
                return arr[0] if arr.dtype == object else arr

            if "input_src" not in batch:
                raise ValueError(
                    "RegularPredictor requires 'input_src'. "
                    "Set return_src=True in file_loader config."
                )
            input_src_np = _unpack(batch["input_src"])  # (1, T_in, C, H, W) energy-collapsed

            # Squeeze leading singleton from Ray batching → (T_in, C, H, W).
            while input_src_np.ndim > 4 and input_src_np.shape[0] == 1:
                input_src_np = input_src_np.squeeze(0)
            # (T_in, C, H, W) → (C, T_in, H, W) → add batch axis → (1, C, T_in, H, W).
            input_reg = torch.from_numpy(input_src_np.transpose(1, 0, 2, 3)[np.newaxis]).to(
                self.device, dtype=torch.float32
            )

            B, C, T_in, H, W = input_reg.shape
            R = self.model_return_seq_len
            centers = make_regular_centers(
                H, W, p=self.model.config.patch_size, device=self.device
            )  # (N_tokens, 3)

            curr_input_seq = input_reg.clone()  # (1, C, T_in, H, W)
            all_preds = []  # list[(1, C, 1, H, W)]

            with torch.no_grad():
                for call_idx in range(self.num_forward_calls):
                    timestep_idx = call_idx * R
                    steps_this_call = min(R, self.predict_steps - timestep_idx)

                    with _timer(timings, "time_regular_forward", sync_cuda=True):
                        pred_full = self.model(curr_input_seq, centers)  # (1, C, R, H, W)

                    with _timer(timings, "time_regular_post", sync_cuda=True):
                        if self.pred_mode == "residual":
                            pred_full = pred_full + curr_input_seq[:, :, -1].unsqueeze(2)

                        pred_to_store = pred_full[:, :, :steps_this_call]
                        pred_np = pred_to_store.cpu().numpy()
                        if hasattr(self.transform, "inverse_transform") and callable(
                            self.transform.inverse_transform
                        ):
                            pred_phys = self.transform.inverse_transform(pred_np)
                        else:
                            pred_phys = pred_np

                        # Store per-timestep
                        for t in range(steps_this_call):
                            all_preds.append(pred_phys[:, :, t : t + 1])

                        if call_idx < self.num_forward_calls - 1:
                            num_from_input = max(T_in - R, 0)
                            if num_from_input > 0:
                                curr_input_seq = torch.cat(
                                    (curr_input_seq[:, :, -num_from_input:], pred_full), dim=2
                                )
                            else:
                                curr_input_seq = pred_full[:, :, -T_in:]

            # Pack regular predictions as object array (per-timestep list)
            out = dict(batch)  # pass through all adaptive outputs
            out["regular_preds_list"] = np.empty(1, dtype=object)
            out["regular_preds_list"][0] = all_preds  # list of (1, C, 1, H, W)

            # Remove input_src — no longer needed downstream
            out.pop("input_src", None)

        for k, v in timings.items():
            out[k] = np.array([v], dtype=np.float32)
        return out


# ---------------------------------------------------------------------------
# Combined Metric Evaluator
# ---------------------------------------------------------------------------


class CombinedMetricEvaluator:
    """
    Compute per-level and uniform-grid metrics for both adaptive and regular models.
    All metrics are driven by METRIC_REGISTRY via config ``metrics``.

    Requires source GT (return_src=True):
      1. target_src (numpy array) — uniform or native mode
      2. target_src_paths (file paths) — uniform-patch AMReX mode only
    """

    def __init__(self, config: dict):
        self.predict_steps = config["inference"].get("predict_steps", 1)
        self.fields = config["data"]["field_names"]
        loader_params = config.get("file_loader", {}).get("params", {})
        self.cell_scale_mode = loader_params.get("cell_scale_mode", "area")
        self.compute_uniform_metrics = config["inference"].get("compute_uniform_metrics", True)
        self.mode = loader_params.get("mode", "npz")
        self.amrex_patch_size = loader_params.get("amrex_patch_size", 32)
        self.amrex_field_names = loader_params.get("field_names", self.fields)
        self.metric_names = config["inference"].get("metrics", ["RMSE", "VRMSE"])
        for m in self.metric_names:
            if m not in METRIC_REGISTRY:
                raise ValueError(f"Unknown metric: {m}. Available: {list(METRIC_REGISTRY.keys())}")

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self._timings: dict[str, float] = {}
        with _timer(self._timings, "time_eval_total"):
            if "native_mode" in batch:
                out = self._eval_native(batch)
            else:
                out = self._eval_uniform(batch)
        # Forward any upstream timing columns so everything lands in one dataframe.
        for k, v in batch.items():
            if isinstance(k, str) and k.startswith("time_"):
                val = v[0] if hasattr(v, "shape") and v.ndim >= 1 and v.shape[0] == 1 else v
                out[k] = np.array([float(val)], dtype=np.float32)
        for k, v in batch.items():
            if isinstance(k, str) and k in ("adaptive_n_tokens", "adaptive_steps"):
                val = v[0] if hasattr(v, "shape") and v.ndim >= 1 and v.shape[0] == 1 else v
                out[k] = np.array([float(val)], dtype=np.float32)
        for k, v in self._timings.items():
            out[k] = np.array([v], dtype=np.float32)
        return out

    def _eval_uniform(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        def _unpack(arr):
            return arr[0] if arr.dtype == object else arr

        adaptive_preds_list = _unpack(batch["adaptive_preds_list"])
        adaptive_centers_list = _unpack(batch["adaptive_centers_list"])
        adaptive_levels_list = _unpack(batch["adaptive_levels_list"])
        regular_preds_list = _unpack(batch["regular_preds_list"])

        target_src = _unpack(batch["target_src"]) if "target_src" in batch else None
        target_src_paths = (
            _unpack(batch["target_src_paths"]) if "target_src_paths" in batch else None
        )
        if target_src is None and target_src_paths is None:
            raise ValueError(
                "Combined evaluation requires target_src or target_src_paths. "
                "Set return_src=True in file_loader config."
            )
        if target_src is not None:
            while target_src.ndim > 4 and target_src.shape[0] == 1:
                target_src = target_src.squeeze(0)

        domain = {
            k: _unpack(batch[f"domain_{k}"]).item()
            for k in ["xmin", "xmax", "ymin", "ymax", "max_level_idx", "tile_width", "tile_height"]
        }
        max_level_idx = int(domain["max_level_idx"])

        results = {}
        B = 1

        for s in range(self.predict_steps):
            for lvl in range(max_level_idx + 1):
                results[f"step_{s}_level_{lvl}_count"] = np.zeros(B, dtype=np.float32)
                for m in self.metric_names:
                    for c_idx in range(len(self.fields)):
                        adap_key = f"step_{s}_level_{lvl}_{m}_adaptive_{self.fields[c_idx]}"
                        reg_key = f"step_{s}_level_{lvl}_{m}_regular_{self.fields[c_idx]}"
                        results[adap_key] = np.full(B, np.nan, dtype=np.float32)
                        results[reg_key] = np.full(B, np.nan, dtype=np.float32)

            if self.compute_uniform_metrics:
                for m in self.metric_names:
                    for c_idx in range(len(self.fields)):
                        results[f"step_{s}_uniform_{m}_adaptive_{self.fields[c_idx]}"] = np.zeros(
                            B, dtype=np.float32
                        )
                        results[f"step_{s}_uniform_{m}_regular_{self.fields[c_idx]}"] = np.zeros(
                            B, dtype=np.float32
                        )

        for step in range(self.predict_steps):
            pred_adaptive = adaptive_preds_list[step]  # (N_step, C, 1, Ph, Pw)
            step_centers = adaptive_centers_list[step]  # (N_step, 3)
            step_levels = adaptive_levels_list[step]  # (N_step,)
            pred_regular = regular_preds_list[step]  # nested to (C, H, W) below

            # Unwrap any leading singletons and drop time axis → (C, H, W).
            while pred_regular.ndim > 3 and pred_regular.shape[0] == 1:
                pred_regular = pred_regular.squeeze(0)
            if pred_regular.ndim == 4:
                pred_regular = pred_regular[:, 0]

            step_meta = {"centers": step_centers, "levels": step_levels, "domain": domain}

            reg_pred_frame = pred_regular  # (C, H, W)
            adaptive_frame = pred_adaptive[:, :, 0, :, :]  # (N_step, C, Ph, Pw)

            with _timer(self._timings, "time_eval_gt_assembly"):
                gt_src_frame = target_src[step] if target_src is not None else None
                gt_on_qt, qt_struct = build_uniform_gt(
                    adaptive_frame,
                    step_meta,
                    gt_src_frame=gt_src_frame,
                    gt_src_path=(
                        target_src_paths[step]
                        if target_src is None and target_src_paths is not None else None
                    ),
                    amrex_field_names=self.amrex_field_names,
                    amrex_patch_size=self.amrex_patch_size,
                    cell_scale_mode=self.cell_scale_mode,
                )

                # Project regular-model frame onto adaptive quadtree → (N_step, C, Ph, Pw).
                qt_struct.assign_from_array(reg_pred_frame)
                reg_on_qt, reg_meta = quadtree_to_tensor(
                    qt_struct, return_tensor=False, cell_scale_mode=self.cell_scale_mode
                )
                order = leaf_export_order(
                    qt_struct, reg_meta, step_meta, cell_scale_mode=self.cell_scale_mode
                )
                if order is not None:
                    reg_on_qt = reg_on_qt[order]

            with _timer(self._timings, "time_eval_perlevel"):
                unique_levels = np.unique(step_levels)
                for lvl in unique_levels:
                    lvl = int(lvl)
                    mask = step_levels == lvl

                    adap_l = adaptive_frame[mask]  # (N_l, C, Ph, Pw)
                    gt_l = gt_on_qt[mask]  # (N_l, C, Ph, Pw)
                    reg_l = reg_on_qt[mask]  # (N_l, C, Ph, Pw)

                    # Metric expects channel-last: (N_l, Ph, Pw, C).
                    adap_t = torch.from_numpy(adap_l).permute(0, 2, 3, 1).float()
                    gt_t = torch.from_numpy(gt_l).permute(0, 2, 3, 1).float()
                    reg_t = torch.from_numpy(reg_l).permute(0, 2, 3, 1).float()

                    results[f"step_{step}_level_{lvl}_count"][0] = float(np.sum(mask))

                    for metric_name in self.metric_names:
                        val_adap = eval_per_level_metric(metric_name, adap_t, gt_t)  # (C,)
                        val_reg = eval_per_level_metric(metric_name, reg_t, gt_t)  # (C,)
                        for c_idx in range(len(self.fields)):
                            results[
                                f"step_{step}_level_{lvl}_{metric_name}_adaptive_{self.fields[c_idx]}"
                            ][0] = val_adap[c_idx].item()
                            results[
                                f"step_{step}_level_{lvl}_{metric_name}_regular_{self.fields[c_idx]}"
                            ][0] = val_reg[c_idx].item()

            if self.compute_uniform_metrics:
                with _timer(self._timings, "time_eval_uniform"):
                    adaptive_uniform = tensor_to_uniform(
                        adaptive_frame, step_meta, cell_scale_mode=self.cell_scale_mode
                    )  # (C, H, W)

                    if gt_src_frame is not None:
                        gt_uniform = gt_src_frame  # (C, H, W)
                    else:
                        gt_uniform = tensor_to_uniform(
                            gt_on_qt, step_meta, cell_scale_mode=self.cell_scale_mode
                        )  # (C, H, W)

                    # Metric expects (B=1, H, W, C).
                    adap_u_t = (
                        torch.from_numpy(adaptive_uniform).permute(1, 2, 0).unsqueeze(0).float()
                    )
                    reg_u_t = torch.from_numpy(reg_pred_frame).permute(1, 2, 0).unsqueeze(0).float()
                    gt_u_t = torch.from_numpy(gt_uniform).permute(1, 2, 0).unsqueeze(0).float()

                    for metric_name in self.metric_names:
                        metric_cls = METRIC_REGISTRY[metric_name]
                        val_adap = metric_cls.eval(adap_u_t, gt_u_t, n_spatial_dims=2)  # (1, C)
                        val_reg = metric_cls.eval(reg_u_t, gt_u_t, n_spatial_dims=2)  # (1, C)
                        for c_idx in range(len(self.fields)):
                            results[
                                f"step_{step}_uniform_{metric_name}_adaptive_{self.fields[c_idx]}"
                            ][0] = val_adap[0, c_idx].item()
                            results[
                                f"step_{step}_uniform_{metric_name}_regular_{self.fields[c_idx]}"
                            ][0] = val_reg[0, c_idx].item()

        return results

    def _eval_native(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Native-mode combined metrics: adaptive (per-level) + regular (uniform)."""

        def _unpack(arr):
            return arr[0] if arr.dtype == object else arr

        adaptive_preds_by_level_list = _unpack(batch["adaptive_preds_by_level_list"])
        adaptive_l2b_list = _unpack(batch["adaptive_leaf_to_bucket_list"])
        adaptive_centers_list = _unpack(batch["adaptive_centers_list"])
        adaptive_levels_list = _unpack(batch["adaptive_levels_list"])
        regular_preds_list = _unpack(batch["regular_preds_list"])

        target_src = _unpack(batch["target_src"]) if "target_src" in batch else None
        if target_src is None:
            raise ValueError(
                "Native combined evaluation requires dense target_src. "
                "Set return_src=True in file_loader config."
            )
        while target_src.ndim > 4 and target_src.shape[0] == 1:
            target_src = target_src.squeeze(0)

        domain = {
            k: _unpack(batch[f"domain_{k}"]).item()
            for k in ["xmin", "xmax", "ymin", "ymax", "max_level_idx", "tile_width", "tile_height"]
        }
        max_level_idx = int(domain["max_level_idx"])

        first_pred = adaptive_preds_by_level_list[0]
        C = next(v.shape[1] for v in first_pred.values() if v.shape[0] > 0)

        results = {}
        B = 1

        for s in range(self.predict_steps):
            for c_idx in range(C):
                results[f"step_{s}_c_adaptive_{self.fields[c_idx]}"] = np.zeros(B, dtype=np.float32)
                results[f"step_{s}_c_regular_{self.fields[c_idx]}"] = np.zeros(B, dtype=np.float32)
            for lvl in range(max_level_idx + 1):
                results[f"step_{s}_level_{lvl}_count"] = np.zeros(B, dtype=np.float32)
                for m in self.metric_names:
                    for c_idx in range(len(self.fields)):
                        adap_key = f"step_{s}_level_{lvl}_{m}_adaptive_{self.fields[c_idx]}"
                        reg_key = f"step_{s}_level_{lvl}_{m}_regular_{self.fields[c_idx]}"
                        results[adap_key] = np.full(B, np.nan, dtype=np.float32)
                        results[reg_key] = np.full(B, np.nan, dtype=np.float32)
            if self.compute_uniform_metrics:
                for m in self.metric_names:
                    for c_idx in range(len(self.fields)):
                        results[f"step_{s}_uniform_{m}_adaptive_{self.fields[c_idx]}"] = np.zeros(
                            B, dtype=np.float32
                        )
                        results[f"step_{s}_uniform_{m}_regular_{self.fields[c_idx]}"] = np.zeros(
                            B, dtype=np.float32
                        )

        for step in range(self.predict_steps):
            pred_by_level = adaptive_preds_by_level_list[step]  # {lvl: (N_l, C, 1, H_l, W_l)}
            step_l2b = adaptive_l2b_list[step]  # (N, 2)
            step_centers = adaptive_centers_list[step]  # (N, 3)
            step_levels = adaptive_levels_list[step]  # (N,)
            step_meta = {"centers": step_centers, "levels": step_levels, "domain": domain}

            # Unwrap regular prediction to (C, H, W).
            pred_regular = regular_preds_list[step]
            while pred_regular.ndim > 3 and pred_regular.shape[0] == 1:
                pred_regular = pred_regular.squeeze(0)
            if pred_regular.ndim == 4:
                pred_regular = pred_regular[:, 0]
            reg_pred_frame = pred_regular  # (C, H, W)

            gt_src_frame = target_src[step]  # (C, H, W)

            # Build GT per-level and project regular predictions onto native quadtree.
            with _timer(self._timings, "time_eval_gt_assembly"):
                gt_by_level = build_native_gt_buckets(
                    pred_by_level,
                    step_l2b,
                    step_meta,
                    gt_src_frame,
                    cell_scale_mode=self.cell_scale_mode,
                )
                reg_by_level = build_native_gt_buckets(
                    pred_by_level,
                    step_l2b,
                    step_meta,
                    reg_pred_frame,
                    cell_scale_mode=self.cell_scale_mode,
                )

            with _timer(self._timings, "time_eval_perlevel"):
                # Per-level metrics for adaptive model.
                eval_per_level_metrics(
                    pred_by_level,
                    gt_by_level,
                    max_level_idx,
                    self.metric_names,
                    self.fields,
                    step,
                    results,
                    key_prefix="adaptive_",
                )
                # Per-level metrics for regular model.
                eval_per_level_metrics(
                    reg_by_level,
                    gt_by_level,
                    max_level_idx,
                    self.metric_names,
                    self.fields,
                    step,
                    results,
                    key_prefix="regular_",
                )

            # Uniform-grid metrics.
            if self.compute_uniform_metrics:
                with _timer(self._timings, "time_eval_uniform"):
                    eval_native_uniform_metrics(
                        pred_by_level,
                        step_l2b,
                        step_meta,
                        gt_src_frame,
                        self.metric_names,
                        self.fields,
                        step,
                        results,
                        key_prefix="adaptive_",
                    )
                    # Regular model predictions are already on uniform grid.
                    pred_reg_t = (
                        torch.from_numpy(reg_pred_frame).permute(1, 2, 0).unsqueeze(0).float()
                    )
                    gt_t = torch.from_numpy(gt_src_frame).permute(1, 2, 0).unsqueeze(0).float()
                    for metric_name in self.metric_names:
                        metric_cls = METRIC_REGISTRY[metric_name]
                        val = metric_cls.eval(pred_reg_t, gt_t, n_spatial_dims=2)
                        for c_idx in range(len(self.fields)):
                            results[
                                f"step_{step}_uniform_{metric_name}_regular_{self.fields[c_idx]}"
                            ][0] = val[0, c_idx].item()

        return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to standalone yaml file (legacy). Omit to use Hydra config groups.",
    )
    parser.add_argument("--debug", action="store_true", help="Run with fewer samples.")
    parser.add_argument("--eval_full", action="store_true", help="Evaluate every time step.")
    args, unknown = parser.parse_known_args()

    config = load_config(args, unknown)
    base_dir = os.getcwd()

    steps = config["inference"].get("predict_steps", 1)
    adaptive_name = config["inference"]["checkpoint_path_adaptive"].split("/")[-2]
    regular_name = config["inference"]["checkpoint_path_regular"].split("/")[-2]
    config["inference"]["model_name_adaptive"] = adaptive_name
    config["inference"]["model_name_regular"] = regular_name

    default_csv = f"rmse_combined_{adaptive_name}_vs_{regular_name}_steps{steps}.csv"
    if config["inference"].get("split_policy", "internal") != "prepartitioned":
        default_csv = default_csv.removesuffix(".csv") + "_heldout.csv"
    csv_path = config["inference"].get("output_csv") or os.path.join("results", default_csv)
    if not os.path.isabs(csv_path):
        csv_path = os.path.abspath(os.path.join(base_dir, csv_path))
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    config["inference"]["output_csv"] = csv_path

    # --- Build windows ---
    data_config = config["data"]
    file_parser = instantiate_from_config(config["file_parser"])
    all_paths = file_parser(data_config["glob_pattern"])

    windows = prepare_evaluation_windows(
        all_paths, config, eval_full=args.eval_full, debug=args.debug,
    )

    if not ray.is_initialized():
        ray.init(
            object_store_memory=config["inference"].get("ray_object_store_memory", 10) * 1024**3
        )
    ds = ray.data.from_items(windows)

    # --- Data loading ---
    loader_cfg = config["file_loader"]
    loader_params = loader_cfg.get("params", {})
    for key in ("resample_coarsen_ratio", "resample_refine_ratio"):
        if loader_params.get(key, 0.0) > 0.0:
            warnings.warn(
                f"Rollout: overriding {key}={loader_params[key]} → 0.0 "
                "(augmentation disabled during inference)"
            )
            loader_params[key] = 0.0

    file_loader = instantiate_from_config(loader_cfg)
    transform = instantiate_from_config(config["transform"])

    mapper = CombinedMapper(
        loader=file_loader,
        transform=transform,
        regular_target_level=config["inference"].get("regular_target_level", 0),
    )

    # Per-stage RAM budgets (GB). Combined runs both adaptive + regular predictors,
    # so the mapper output carries both feature sets — defaults are between the
    # adaptive-only and regular-only rollouts. Override in inference config.
    GB = 1024**3
    mapper_mem_gb = config["inference"].get("mapper_memory_gb", 10)
    adaptive_pred_mem_gb = config["inference"].get(
        "adaptive_predictor_memory_gb", config["inference"].get("predictor_memory_gb", 15)
    )
    regular_pred_mem_gb = config["inference"].get("regular_predictor_memory_gb", 8)
    eval_mem_gb = config["inference"].get("eval_memory_gb", 4)

    mapper_concurrency = config["inference"].get(
        "mapper_concurrency", config["inference"].get("num_cpus", 10)
    )
    eval_concurrency = config["inference"].get(
        "eval_concurrency", config["inference"].get("num_cpus", 10)
    )

    ds = ds.map_batches(
        mapper,
        batch_size=1,
        batch_format="numpy",
        memory=int(mapper_mem_gb * GB),
        compute=ray.data.TaskPoolStrategy(size=mapper_concurrency),
    )

    # --- Adaptive inference ---
    print("Running combined distributed inference pipeline...")
    ds = ds.map_batches(
        AdaptivePredictor,
        fn_constructor_args=(config,),
        batch_size=1,
        num_gpus=1,
        compute=ray.data.ActorPoolStrategy(size=config["inference"].get("num_gpus_adaptive", 1)),
        batch_format="numpy",
        memory=int(adaptive_pred_mem_gb * GB),
    )

    # --- Regular inference ---
    ds = ds.map_batches(
        RegularPredictor,
        fn_constructor_args=(config,),
        batch_size=1,
        num_gpus=1,
        compute=ray.data.ActorPoolStrategy(size=config["inference"].get("num_gpus_regular", 1)),
        batch_format="numpy",
        memory=int(regular_pred_mem_gb * GB),
    )

    # --- Metric evaluation ---
    results_ds = ds.map_batches(
        CombinedMetricEvaluator,
        fn_constructor_args=(config,),
        batch_size=1,
        batch_format="numpy",
        compute=ray.data.ActorPoolStrategy(size=eval_concurrency),
        memory=int(eval_mem_gb * GB),
    )

    df = results_ds.to_pandas()
    if df.empty:
        raise ValueError(
            "The evaluation dataset is empty! Check your glob pattern and split logic."
        )

    # --- Aggregate into final CSV ---
    metric_names = config["inference"].get("metrics", ["RMSE", "VRMSE"])
    eval_fields = data_config["field_names"]
    max_level_idx = int(config["file_loader"]["params"].get("num_levels", 3)) - 1

    mean_vals = mean_numeric_result_columns(df)

    results_dict = {"Step": []}

    # Column headers: uniform metrics
    has_uniform = any(col.startswith("step_0_uniform_") for col in df.columns)
    if has_uniform:
        for model_tag in ("adaptive", "regular"):
            for m in metric_names:
                for field in eval_fields:
                    results_dict[f"uniform_{m}_{model_tag}_{field}"] = []

    # Column headers: per-level metrics
    for lvl in range(max_level_idx + 1):
        results_dict[f"level_{lvl}_count"] = []
        for model_tag in ("adaptive", "regular"):
            for m in metric_names:
                for field in eval_fields:
                    results_dict[f"level_{lvl}_{m}_{model_tag}_{field}"] = []

    for s in range(steps):
        results_dict["Step"].append(s + 1)

        # Uniform metrics
        if has_uniform:
            for model_tag in ("adaptive", "regular"):
                for m in metric_names:
                    for field in eval_fields:
                        col = f"step_{s}_uniform_{m}_{model_tag}_{field}"
                        results_dict[f"uniform_{m}_{model_tag}_{field}"].append(
                            mean_vals.get(col, np.nan)
                        )

        # Per-level metrics
        for lvl in range(max_level_idx + 1):
            count_col = f"step_{s}_level_{lvl}_count"
            results_dict[f"level_{lvl}_count"].append(mean_vals.get(count_col, 0.0))
            for model_tag in ("adaptive", "regular"):
                for m in metric_names:
                    for field in eval_fields:
                        col = f"step_{s}_level_{lvl}_{m}_{model_tag}_{field}"
                        results_dict[f"level_{lvl}_{m}_{model_tag}_{field}"].append(
                            mean_vals.get(col, np.nan)
                        )

    final_df = pd.DataFrame(results_dict)
    final_df.to_csv(csv_path, index=False)

    print("\n=== Combined Inference Complete ===")
    print(f"Adaptive model: {adaptive_name}")
    print(f"Regular model:  {regular_name}")
    print(f"Results saved to: {csv_path}")

    # Print a compact summary — uniform metrics only
    if has_uniform:
        uniform_cols = ["Step"] + [c for c in final_df.columns if c.startswith("uniform_")]
        print("\n--- Uniform Metrics ---")
        print(final_df[uniform_cols].to_string(index=False))

    # Print per-level summary (counts + first metric)
    if metric_names:
        first_metric = metric_names[0]
        level_summary_cols = ["Step"]
        for lvl in range(max_level_idx + 1):
            level_summary_cols.append(f"level_{lvl}_count")
            for model_tag in ("adaptive", "regular"):
                col = f"level_{lvl}_{first_metric}_{model_tag}_{eval_fields[0]}"
                if col in final_df.columns:
                    level_summary_cols.append(col)
        print(f"\n--- Per-Level Summary ({first_metric}, {eval_fields[0]}) ---")
        print(final_df[level_summary_cols].to_string(index=False))

    # ------------------------------------------------------------------
    # Per-stage wall-clock profile (mean per sample across the full run)
    # ------------------------------------------------------------------
    # Each ``time_*`` column holds seconds spent in that block for one
    # input sample. Means answer "what consumes time per sample end-to-end";
    # sums give total wall-clock per stage over the evaluated subset. Note
    # that stages run through separate Ray pools (mapper = TaskPool,
    # predictors = ActorPool, evaluator = ActorPool), so summed stage times
    # overlap — interpret as "CPU-seconds in stage", not wall-clock.
    time_cols = sorted(c for c in df.columns if c.startswith("time_"))
    if time_cols:
        n = len(df)
        means = mean_vals[time_cols]
        sums = normalize_numeric_result_columns(df[time_cols]).sum(skipna=True, min_count=1)
        n_tok_col = "adaptive_n_tokens" if "adaptive_n_tokens" in df.columns else None
        n_tok_mean = float(mean_vals[n_tok_col]) if n_tok_col else float("nan")

        print(
            f"\n--- Timing Profile (n={n} samples"
            + (f", avg N_tokens={n_tok_mean:.0f}" if n_tok_col else "")
            + ") ---"
        )
        print(f"{'stage':38s} {'mean_s':>10s} {'sum_s':>10s} {'share':>8s}")
        # Buckets for computing "share" against the headline stage total.
        # Each top-level stage has its own total; sub-stages share against it.
        bucket_totals = {
            "mapper": float(means.get("time_mapper_total", 0.0)),
            "adaptive": float(means.get("time_adaptive_total", 0.0)),
            "regular": float(means.get("time_regular_total", 0.0)),
            "eval": float(means.get("time_eval_total", 0.0)),
        }

        def _bucket_for(col: str) -> str:
            for k in ("mapper", "adaptive", "regular", "eval"):
                if col.startswith(f"time_{k}"):
                    return k
            return ""

        for col in time_cols:
            bkt = _bucket_for(col)
            tot = bucket_totals.get(bkt, 0.0)
            mean_s = float(means[col])
            sum_s = float(sums[col])
            share = (
                f"{(100.0 * mean_s / tot):6.1f}%"
                if tot > 0 and not col.endswith("_total")
                else "   --"
            )
            print(f"{col:38s} {mean_s:10.4f} {sum_s:10.2f} {share:>8s}")

        # Diagnostic summary: highest-cost sub-stage within each bucket.
        print("\n--- Where to attack first ---")
        for bkt, tot in bucket_totals.items():
            if not tot > 0:
                continue
            sub_cols = [
                c for c in time_cols
                if _bucket_for(c) == bkt and not c.endswith("_total") and pd.notna(means[c])
            ]
            if not sub_cols:
                continue
            top = max(sub_cols, key=lambda c: float(means[c]))
            top_mean = float(means[top])
            unaccounted = max(0.0, tot - sum(float(means[c]) for c in sub_cols))
            print(
                f"  {bkt:9s} total {tot:7.3f}s  | top sub-stage: {top} ({top_mean:.3f}s, "
                f"{100.0 * top_mean / tot:.1f}%)  | unaccounted (bookkeeping/data transfer): "
                f"{unaccounted:.3f}s ({100.0 * unaccounted / tot:.1f}%)"
            )


if __name__ == "__main__":
    main()
