import argparse
import math
import os
import time
import warnings
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import ray
import torch
from numba import get_num_threads

from wamrvit.dataloader.trajectory_loader import Seq2SeqMapper
from wamrvit.dataloader.transform import inverse_transform_src
from wamrvit.native_train_utils import unpack_native_batch
from wamrvit.quad.adapt_wavelet import regrid_native
from wamrvit.quad.array_regrid import configure_array_regrid_num_threads
from wamrvit.quad.regrid_dispatch import (
    regrid_native_dispatch,
    regrid_uniform_dispatch,
    regrid_uniform_sequence_dispatch,
    RegridProfiler,
)
from wamrvit.quad.quad_utils import (
    quadtree_to_tensor,
    quadtree_to_tensor_native,
    tensor_to_quadtree,
    tensor_to_quadtree_native,
    tensor_to_uniform,
    tensor_to_uniform_native,
)
from wamrvit.quadtree_transformer import QuadTreeTransformer
from wamrvit.rollout.regular_metrics import METRIC_REGISTRY, eval_per_level_metric
from wamrvit.utils import instantiate_from_config, load_config

# ---------------------------------------------------------------------------
# Shared helpers for native-mode metric evaluation
# (used by MetricEvaluator and CombinedMetricEvaluator in rollout_combined)
# ---------------------------------------------------------------------------


def build_native_gt_buckets(
    pred_by_level: dict[int, np.ndarray],
    step_l2b: np.ndarray,
    step_meta: dict[str, Any],
    gt_src_frame: np.ndarray,
    cell_scale_mode: str = "area",
) -> dict[int, np.ndarray]:
    """Build per-level GT buckets by projecting a dense GT frame onto a native quadtree.

    Args:
        pred_by_level: Prediction buckets (used as shape template).
            ``{lvl: (N_l, C, [1,] H_l, W_l)}``.
        step_l2b: ``(N, 2)`` leaf-to-bucket mapping.
        step_meta: Quadtree metadata dict.
        gt_src_frame: Dense GT array ``(C, H, W)``.
        cell_scale_mode: Scale mode for center/level encoding.

    Returns:
        ``{lvl: (N_l, C, H_l, W_l)}`` GT per-level arrays.
    """
    zero_buckets = {
        lvl: np.zeros_like(arr[:, :, 0]) if arr.ndim == 5 else np.zeros_like(arr)
        for lvl, arr in pred_by_level.items()
    }
    qt = tensor_to_quadtree_native(
        zero_buckets, step_l2b, step_meta, cell_scale_mode=cell_scale_mode
    )
    qt.assign_from_array(gt_src_frame)
    gt_buckets, _, _ = quadtree_to_tensor_native(qt, cell_scale_mode=cell_scale_mode)
    return gt_buckets


def eval_per_level_metrics(
    pred_by_level: dict[int, np.ndarray],
    gt_by_level: dict[int, np.ndarray],
    max_level_idx: int,
    metric_names: list,
    fields: list,
    step: int,
    results: dict[str, np.ndarray],
    key_prefix: str = "",
) -> None:
    """Compute per-level metrics using METRIC_REGISTRY and write into *results* in-place.

    Also computes per-channel cell MSE (``step_{s}_c_{field}``).

    Args:
        pred_by_level: ``{lvl: (N_l, C, [1,] H_l, W_l)}``.
        gt_by_level: ``{lvl: (N_l, C, H_l, W_l)}``.
        max_level_idx: Maximum level index.
        metric_names: List of metric names (keys into ``METRIC_REGISTRY``).
        fields: Channel/field names.
        step: Current prediction step index.
        results: Mutable results dict to write into.
        key_prefix: Optional prefix for result keys (e.g. ``"adaptive_"``).
    """
    C = len(fields)
    for lvl in range(max_level_idx + 1):
        pred_lvl = pred_by_level.get(lvl)
        gt_lvl = gt_by_level.get(lvl)
        if pred_lvl is None or gt_lvl is None or pred_lvl.shape[0] == 0:
            continue

        pred_frame = pred_lvl[:, :, 0] if pred_lvl.ndim == 5 else pred_lvl
        gt_frame = gt_lvl

        n_l = pred_frame.shape[0]
        results[f"step_{step}_level_{lvl}_count"][0] = float(n_l)

        pred_t = torch.from_numpy(pred_frame).permute(0, 2, 3, 1).float()
        gt_t = torch.from_numpy(gt_frame).permute(0, 2, 3, 1).float()

        for metric_name in metric_names:
            val = eval_per_level_metric(metric_name, pred_t, gt_t)
            for c in range(C):
                results[f"step_{step}_level_{lvl}_{metric_name}_{key_prefix}{fields[c]}"][0] = val[
                    c
                ].item()

    # Per-channel cell MSE.
    for c in range(C):
        ch_mse = 0.0
        ch_count = 0
        for lvl in range(max_level_idx + 1):
            pred_lvl = pred_by_level.get(lvl)
            gt_lvl = gt_by_level.get(lvl)
            if pred_lvl is None or gt_lvl is None or pred_lvl.shape[0] == 0:
                continue
            pf = pred_lvl[:, c, 0] if pred_lvl.ndim == 5 else pred_lvl[:, c]
            gf = gt_lvl[:, c]
            ch_mse += float(((pf - gf) ** 2).sum())
            ch_count += pf.size
        if ch_count > 0:
            results[f"step_{step}_c_{key_prefix}{fields[c]}"][0] = ch_mse / ch_count


def eval_native_uniform_metrics(
    pred_by_level: dict[int, np.ndarray],
    step_l2b: np.ndarray,
    step_meta: dict[str, Any],
    gt_src_frame: np.ndarray,
    metric_names: list,
    fields: list,
    step: int,
    results: dict[str, np.ndarray],
    key_prefix: str = "",
) -> None:
    """Compute uniform-grid metrics for native-mode predictions.

    Projects per-level predictions onto the finest uniform grid via
    ``tensor_to_uniform_native``, then evaluates against ``gt_src_frame``.
    Writes into *results* in-place.
    """
    pred_no_time = {
        lvl: arr[:, :, 0] if arr.ndim == 5 else arr for lvl, arr in pred_by_level.items()
    }
    pred_uniform = tensor_to_uniform_native(pred_no_time, step_l2b, step_meta)

    pred_t = torch.from_numpy(pred_uniform).permute(1, 2, 0).unsqueeze(0).float()
    gt_t = torch.from_numpy(gt_src_frame).permute(1, 2, 0).unsqueeze(0).float()

    C = len(fields)
    for metric_name in metric_names:
        metric_cls = METRIC_REGISTRY[metric_name]
        val = metric_cls.eval(pred_t, gt_t, n_spatial_dims=2)
        for c in range(C):
            results[f"step_{step}_uniform_{metric_name}_{key_prefix}{fields[c]}"][0] = val[
                0, c
            ].item()


class AutoregressivePredictorAdaptive:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        inf_cfg = config["inference"]
        if inf_cfg.get("is_diffusers", True):
            print("Loading model via diffusers from:", inf_cfg["checkpoint_path"])
            self.model = QuadTreeTransformer.from_pretrained(inf_cfg["checkpoint_path"])
        else:
            print("Loading model state dict from:", inf_cfg["checkpoint_path"])
            self.model = QuadTreeTransformer(**config["model"])
            state_dict = torch.load(inf_cfg["checkpoint_path"], map_location="cpu")
            if "module." in list(state_dict.keys())[0]:
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)

        assert self.model.config.adaptive, (
            "Loaded model must be an adaptive QuadTreeTransformer for this rollout script."
        )
        self.multi_scale = getattr(self.model.config, "multi_scale_patch", False)
        self.num_levels = int(self.model.config.max_level_idx) + 1 if self.multi_scale else 0

        self.model.to(self.device)
        self.model.eval()

        self.transform = instantiate_from_config(config["transform"])

        self.predict_steps = inf_cfg.get("predict_steps", 1)
        self.model_return_seq_len = self.model.config.return_seq_len
        self.num_forward_calls = math.ceil(self.predict_steps / self.model_return_seq_len)
        self.pred_mode = inf_cfg.get("pred_mode", "target")
        self.regrid_interval = inf_cfg.get("regrid_interval", None)
        self.regrid_adapt_nearby = inf_cfg.get("regrid_adapt_nearby", 0)
        loader_params = config.get("file_loader", {}).get("params", {})
        self.cell_scale_mode = loader_params.get("cell_scale_mode", "area")
        self.adapt_on_channels = loader_params.get("adapt_on_channels", None)
        self.tol_frac = loader_params.get("tol_frac", 0.01)
        if inf_cfg.get("regrid_tol_frac") is not None:
            self.regrid_tol_frac = inf_cfg["regrid_tol_frac"]
            self.allow_coarsening = inf_cfg.get("allow_coarsening", True)
        else:
            warnings.warn(
                "regrid_tol_frac not set in inference config, "
                f"defaulting to file_loader's tol_frac={self.tol_frac}"
            )
            self.regrid_tol_frac = self.tol_frac

        self.regrid_backend = inf_cfg.get("regrid_backend", "object")
        self.array_regrid_value_storage = inf_cfg.get(
            "array_regrid_value_storage", "copy")
        self.array_regrid_payload_backend = inf_cfg.get(
            "array_regrid_payload_backend", "cpu_eager"
        )
        if (
            self.regrid_backend == "array"
            and self.multi_scale
            and self.array_regrid_payload_backend == "gpu_replay"
        ):
            raise ValueError(
                "array_regrid_payload_backend=gpu_replay is supported only for "
                "uniform-patch rollout."
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
        field_names = list(loader_params.get("field_names", []))
        detector_channels = self.adapt_on_channels or list(range(len(field_names)))
        self.regrid_detector_fields = tuple(
            field_names[index] if index < len(field_names) else f"channel_{index}"
            for index in detector_channels
        )
        # Profiling is opt-in (regrid_profile): default object rollouts stay
        # side-effect-free -- no timing CSVs written. Active only for a comparison run.
        self.regrid_profiler = RegridProfiler() if inf_cfg.get("regrid_profile", False) else None
        # Output dir for profiler files: beside the metric CSV (set in main before actors spawn).
        _out_csv = inf_cfg.get("output_csv")
        self.regrid_out_dir = os.path.dirname(_out_csv) if _out_csv else os.getcwd()

        if (
            self.regrid_interval is not None
            and self.regrid_interval % self.model_return_seq_len != 0
        ):
            warnings.warn(
                f"regrid_interval ({self.regrid_interval}) is not divisible by "
                f"model return_seq_len ({self.model_return_seq_len}). Regridding "
                f"will only occur at forward-call boundaries (every "
                f"{self.model_return_seq_len} timesteps)."
            )

        if self.model.config.cell_scale_mode != self.cell_scale_mode:
            warnings.warn(
                f"Model cell_scale_mode ({self.model.config.cell_scale_mode}) does not match \
                data loader's cell_scale_mode ({self.cell_scale_mode}), overriding loader's mode"
            )
            self.cell_scale_mode = self.model.config.cell_scale_mode

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.multi_scale:
            return self._call_native(batch)
        return self._call_uniform(batch)

    def _call_uniform(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        def _unpack(arr):
            return arr[0] if arr.dtype == object else arr

        input_np = _unpack(batch["input"])  # (1, N, C, T_in, Ph, Pw)
        target_np = _unpack(batch["target"])  # (1, N, C, T_out, Ph, Pw)
        target_src_np = (
            _unpack(batch["target_src"]) if "target_src" in batch else None
        )  # (1, T_out, C, H, W)
        target_src_paths = (
            _unpack(batch["target_src_paths"]) if "target_src_paths" in batch else None
        )

        inputs = torch.from_numpy(input_np).to(
            self.device, dtype=torch.float32
        )  # (1, N, C, T_in, Ph, Pw)
        targets = torch.from_numpy(target_np).to(self.device, dtype=torch.float32)

        domain = {
            "xmin": _unpack(batch["xmin"]).item(),
            "xmax": _unpack(batch["xmax"]).item(),
            "ymin": _unpack(batch["ymin"]).item(),
            "ymax": _unpack(batch["ymax"]).item(),
            "max_level_idx": _unpack(batch["max_level_idx"]).item(),
            "tile_width": _unpack(batch["tile_width"]).item(),
            "tile_height": _unpack(batch["tile_height"]).item(),
        }
        ref_centers_np = _unpack(batch["centers"])  # (1, N, 3) or (N, 3)
        ref_levels_np = _unpack(batch["levels"])  # (1, N)    or (N,)

        ref_meta = {"centers": ref_centers_np, "levels": ref_levels_np, "domain": domain}

        meta = ref_meta
        centers = torch.from_numpy(ref_centers_np).to(self.device, dtype=torch.float32)

        if inputs.ndim == 6:
            # Strip leading ray-batch axis → (N, C, T_in, Ph, Pw).
            inputs = inputs.squeeze(0)
            targets = targets.squeeze(0)
            centers = centers.squeeze(0)  # (N, 3)
            ref_centers_np = ref_centers_np.squeeze(0)
            ref_levels_np = ref_levels_np.squeeze(0)
            ref_meta["centers"] = ref_centers_np
            ref_meta["levels"] = ref_levels_np

        N_grids, C, T_in, H, W = inputs.shape  # H == Ph, W == Pw
        R = self.model_return_seq_len

        curr_input_seq = inputs.clone()  # (N, C, T_in, Ph, Pw)

        all_preds = []
        all_centers = []
        all_levels = []

        if self.regrid_profiler is not None:
            self.regrid_profiler.new_sample()
        with torch.no_grad():
            for call_idx in range(self.num_forward_calls):
                timestep_idx = call_idx * R
                steps_this_call = min(R, self.predict_steps - timestep_idx)

                # --- Regrid current input sequence ---
                if (
                    self.regrid_interval is not None
                    and timestep_idx > 0
                    and timestep_idx % self.regrid_interval == 0
                ):
                    N, c_in, t_in, h, w = curr_input_seq.shape  # (N, C, T_in, Ph, Pw)
                    # Adapt on the last frame's channels (offset into packed T*C axis).
                    ch_offset = (t_in - 1) * c_in
                    use_channels = (
                        [ch + ch_offset for ch in self.adapt_on_channels]
                        if self.adapt_on_channels is not None
                        else list(range(ch_offset, t_in * c_in))
                    )

                    if self.regrid_backend == "array":
                        # Keep the model's sequence layout and let source storage retain
                        # initial leaves there instead of copying the full AMReX payload
                        # into the fixed-capacity mutable workspace.
                        sequence_input = (
                            curr_input_seq
                            if self.array_regrid_payload_backend == "gpu_replay"
                            else curr_input_seq.detach().cpu().numpy()
                        )
                        new_sequence_np, meta = regrid_uniform_sequence_dispatch(
                            sequence_input,
                            meta,
                            backend="array",
                            profiler=self.regrid_profiler,
                            max_passes=10,
                            cell_scale_mode=self.cell_scale_mode,
                            tol_frac=self.regrid_tol_frac,
                            channel=use_channels,
                            adapt_nearby=self.regrid_adapt_nearby,
                            allow_coarsening=self.allow_coarsening,
                            value_storage=self.array_regrid_value_storage,
                            capacity=self.array_regrid_capacity,
                            payload_backend=self.array_regrid_payload_backend,
                            detector_channels=self.adapt_on_channels,
                            detector_fields=self.regrid_detector_fields,
                        )
                    else:
                        # Preserve the established object path exactly: pack time×channel
                        # into the axis expected by tensor_to_quadtree.
                        flat_input = (
                            curr_input_seq.transpose(1, 2)
                            .contiguous()
                            .view(N, t_in * c_in, h, w)
                            .cpu()
                            .numpy()
                        )
                        new_input_np, meta = regrid_uniform_dispatch(
                            flat_input,
                            meta,
                            backend="object",
                            profiler=self.regrid_profiler,
                            max_passes=10,
                            cell_scale_mode=self.cell_scale_mode,
                            tol_frac=self.regrid_tol_frac,
                            channel=use_channels,
                            adapt_nearby=self.regrid_adapt_nearby,
                            allow_coarsening=self.allow_coarsening,
                        )
                        new_N = new_input_np.shape[0]
                        new_sequence_np = np.ascontiguousarray(
                            new_input_np.reshape(new_N, t_in, c_in, h, w)
                            .transpose(0, 2, 1, 3, 4))

                    centers = torch.from_numpy(meta["centers"]).to(
                        self.device, dtype=torch.float32
                    )
                    curr_input_seq = (
                        new_sequence_np.to(self.device, dtype=torch.float32)
                        if isinstance(new_sequence_np, torch.Tensor)
                        else torch.from_numpy(new_sequence_np).to(
                            self.device, dtype=torch.float32
                        )
                    )
                # -------------------------------------

                pred_full = self.model(curr_input_seq, centers)  # (N, C, R, Ph, Pw)

                if self.pred_mode == "residual":
                    # Add back the last input frame → absolute prediction.
                    pred_full = pred_full + curr_input_seq[:, :, -1].unsqueeze(
                        2
                    )  # (N, C, R, Ph, Pw)

                pred_to_store = pred_full[:, :, :steps_this_call]  # (N, C, steps_this_call, Ph, Pw)
                pred_np = pred_to_store.cpu().numpy()

                if hasattr(self.transform, "inverse_transform") and callable(
                    self.transform.inverse_transform
                ):
                    pred_phys = self.transform.inverse_transform(pred_np)
                else:
                    pred_phys = pred_np

                # Store per-timestep predictions and current grid metadata.
                for t in range(steps_this_call):
                    all_preds.append(pred_phys[:, :, t : t + 1])  # (N, C, 1, Ph, Pw)
                    all_centers.append(meta["centers"])  # (N, 3)
                    all_levels.append(meta["levels"])  # (N,)

                if call_idx < self.num_forward_calls - 1:
                    # Slide window: keep last (T_in - R) input frames + append R-frame pred.
                    num_from_input = max(T_in - R, 0)
                    if num_from_input > 0:
                        curr_input_seq = torch.cat(
                            (curr_input_seq[:, :, -num_from_input:], pred_full), dim=2
                        )  # (N, C, T_in, Ph, Pw)
                    else:
                        curr_input_seq = pred_full[:, :, -T_in:]  # (N, C, T_in, Ph, Pw)

        gt_np = targets.cpu().numpy()
        if hasattr(self.transform, "inverse_transform") and callable(
            self.transform.inverse_transform
        ):
            gt_phys = self.transform.inverse_transform(gt_np)
        else:
            gt_phys = gt_np

        # Pack lists into object arrays so Ray Data handles them properly as a single row element
        out_dict = {
            "preds_list": np.empty(1, dtype=object),
            "centers_list": np.empty(1, dtype=object),
            "levels_list": np.empty(1, dtype=object),
            "target": np.expand_dims(gt_phys, 0),
            "ref_centers": np.expand_dims(ref_meta["centers"], 0),
            "ref_levels": np.expand_dims(ref_meta["levels"], 0),
        }

        if target_src_np is not None:
            # target_src is (..., T, C, H, W) — different layout from pred/gt,
            # needs the T↔C transpose dance for NormalizeArray.
            target_src_np = inverse_transform_src(self.transform, target_src_np)
            out_dict["target_src"] = target_src_np

        if target_src_paths is not None:
            out_dict["target_src_paths"] = np.empty(1, dtype=object)
            out_dict["target_src_paths"][0] = target_src_paths

        out_dict["preds_list"][0] = all_preds
        out_dict["centers_list"][0] = all_centers
        out_dict["levels_list"][0] = all_levels

        for k, v in ref_meta["domain"].items():
            out_dict[f"domain_{k}"] = np.expand_dims(np.array([v]), 0)

        if "idx" in batch:
            out_dict["idx"] = batch["idx"]
        if "traj_idx" in batch:
            out_dict["traj_idx"] = batch["traj_idx"]
        if "frame_idx" in batch:
            out_dict["frame_idx"] = batch["frame_idx"]

        if self.regrid_profiler is not None:
            self.regrid_profiler.write(self.regrid_out_dir, self.regrid_backend)
        return out_dict

    # ------------------------------------------------------------------
    # Native (multi-scale) rollout path
    # ------------------------------------------------------------------

    def _call_native(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Autoregressive rollout for value_storage='native' (multi-scale patchify)."""

        def _unpack(arr):
            return arr[0] if arr.dtype == object else arr

        target_src_np = (
            _unpack(batch["target_src"]) if "target_src" in batch else None
        )  # (T_out, C, H, W) after squeeze

        # Unpack per-level input/target columns.
        # inputs_by_level / targets_by_level: {lvl: (N_l, C, T, H_l, W_l)};
        # leaf_to_bucket: (N, 2); centers: (N, 3).
        inputs_by_level, targets_by_level, leaf_to_bucket, centers = unpack_native_batch(
            batch,
            self.num_levels,
            self.device,
            non_blocking=False,
        )

        domain = {
            k: _unpack(batch[k]).item()
            for k in ["xmin", "xmax", "ymin", "ymax", "max_level_idx", "tile_width", "tile_height"]
        }
        ref_centers_np = _unpack(batch["centers"])  # (1, N, 3) or (N, 3)
        ref_levels_np = _unpack(batch["levels"])  # (1, N)    or (N,)
        if ref_centers_np.ndim == 3:
            ref_centers_np = ref_centers_np.squeeze(0)
        if ref_levels_np.ndim == 2:
            ref_levels_np = ref_levels_np.squeeze(0)

        ref_meta = {"centers": ref_centers_np, "levels": ref_levels_np, "domain": domain}
        meta = ref_meta

        R = self.model_return_seq_len
        # Infer T_in from any non-empty level.
        T_in = next(v.shape[2] for v in inputs_by_level.values() if v.shape[0] > 0)

        # Current input sequence per level: {lvl: (N_l, C, T_in, H_l, W_l)}
        # with H_l=Ph*s, W_l=Pw*s, s=2^(L-lvl).
        curr_by_level = {lvl: v.clone() for lvl, v in inputs_by_level.items()}

        all_preds_by_level = []  # list[{lvl: (N_l, C, 1, H_l, W_l)}] per timestep
        all_leaf_to_bucket = []  # list[(N, 2)]
        all_centers = []  # list[(N, 3)]
        all_levels = []  # list[(N,)]

        if self.regrid_profiler is not None:
            self.regrid_profiler.new_sample()
        with torch.no_grad():
            for call_idx in range(self.num_forward_calls):
                timestep_idx = call_idx * R
                steps_this_call = min(R, self.predict_steps - timestep_idx)

                # --- Regrid ---
                if (
                    self.regrid_interval is not None
                    and timestep_idx > 0
                    and timestep_idx % self.regrid_interval == 0
                ):
                    curr_by_level, leaf_to_bucket, centers, meta = self._regrid_native(
                        curr_by_level,
                        leaf_to_bucket,
                        meta,
                    )

                pred_by_level = self.model.forward_multi_scale(
                    curr_by_level,
                    leaf_to_bucket,
                    centers,
                )  # {lvl: (N_l, C, R, H_l, W_l)}

                if self.pred_mode == "residual":
                    # Add last input frame per level (broadcast across R output frames).
                    for lvl in pred_by_level:
                        pred_by_level[lvl] = pred_by_level[lvl] + curr_by_level[lvl][:, :, -1:]

                # Inverse transform & store per-timestep.
                for t in range(steps_this_call):
                    step_pred = {}
                    for lvl, arr in pred_by_level.items():
                        slice_np = arr[:, :, t : t + 1].cpu().numpy()  # (N_l, C, 1, H_l, W_l)
                        if hasattr(self.transform, "inverse_transform") and callable(
                            self.transform.inverse_transform
                        ):
                            slice_np = self.transform.inverse_transform(slice_np)
                        step_pred[lvl] = slice_np
                    all_preds_by_level.append(step_pred)
                    all_leaf_to_bucket.append(leaf_to_bucket.cpu().numpy())
                    all_centers.append(meta["centers"])
                    all_levels.append(meta["levels"])

                # Autoregressive feeding per level: slide T_in window, append R-frame prediction.
                if call_idx < self.num_forward_calls - 1:
                    num_from_input = max(T_in - R, 0)
                    for lvl in curr_by_level:
                        if num_from_input > 0:
                            curr_by_level[lvl] = torch.cat(
                                (curr_by_level[lvl][:, :, -num_from_input:], pred_by_level[lvl]),
                                dim=2,
                            )  # (N_l, C, T_in, H_l, W_l)
                        else:
                            curr_by_level[lvl] = pred_by_level[lvl][:, :, -T_in:]

        # Inverse-transform GT per level.
        gt_by_level = {}
        for lvl, arr in targets_by_level.items():
            gt_np = arr.cpu().numpy()
            if hasattr(self.transform, "inverse_transform") and callable(
                self.transform.inverse_transform
            ):
                gt_np = self.transform.inverse_transform(gt_np)
            gt_by_level[lvl] = gt_np

        # --- Pack output ---
        out_dict: dict[str, Any] = {
            "native_mode": np.array([1], dtype=np.int32),
            "preds_by_level_list": np.empty(1, dtype=object),
            "leaf_to_bucket_list": np.empty(1, dtype=object),
            "centers_list": np.empty(1, dtype=object),
            "levels_list": np.empty(1, dtype=object),
            "ref_centers": np.expand_dims(ref_meta["centers"], 0),
            "ref_levels": np.expand_dims(ref_meta["levels"], 0),
            "ref_leaf_to_bucket": np.expand_dims(leaf_to_bucket.cpu().numpy(), 0),
        }

        # Pack GT per-level as object array.
        out_dict["gt_by_level"] = np.empty(1, dtype=object)
        out_dict["gt_by_level"][0] = gt_by_level

        if target_src_np is not None:
            # target_src is (..., T, C, H, W) — different layout from pred/gt,
            # needs the T↔C transpose dance for NormalizeArray.
            target_src_np = inverse_transform_src(self.transform, target_src_np)
            out_dict["target_src"] = target_src_np

        out_dict["preds_by_level_list"][0] = all_preds_by_level
        out_dict["leaf_to_bucket_list"][0] = all_leaf_to_bucket
        out_dict["centers_list"][0] = all_centers
        out_dict["levels_list"][0] = all_levels

        for k, v in ref_meta["domain"].items():
            out_dict[f"domain_{k}"] = np.expand_dims(np.array([v]), 0)

        if "idx" in batch:
            out_dict["idx"] = batch["idx"]
        if "traj_idx" in batch:
            out_dict["traj_idx"] = batch["traj_idx"]
        if "frame_idx" in batch:
            out_dict["frame_idx"] = batch["frame_idx"]

        if self.regrid_profiler is not None:
            self.regrid_profiler.write(self.regrid_out_dir, self.regrid_backend)
        return out_dict

    def _regrid_native(self, curr_by_level, leaf_to_bucket, meta):
        """Regrid native-mode predictions: exact refine/coarsen via native quadtree.

        Args:
            curr_by_level:    {lvl: (N_l, C, T_in, H_l, W_l)} torch tensors on device.
            leaf_to_bucket:   (N, 2) long torch tensor on device.
            meta:             dict with "centers" (N, 3), "levels" (N,), "domain" (dict).
        Returns:
            new_by_level:     {lvl: (N_l_new, C, T_in, H_l, W_l)} torch tensors on device.
            new_l2b_t:        (N_new, 2) long torch tensor.
            new_centers_t:    (N_new, 3) float torch tensor.
            new_meta:         updated meta dict.
        """
        # Infer C, T from any non-empty level.
        C_orig = T_in = None
        for arr in curr_by_level.values():
            if arr.shape[0] > 0:
                C_orig, T_in = arr.shape[1], arr.shape[2]
                break

        # Move to numpy for the shared regrid function.
        np_by_level = {lvl: arr.cpu().numpy() for lvl, arr in curr_by_level.items()}
        l2b_np = leaf_to_bucket.cpu().numpy()

        new_by_level_np, new_l2b, new_meta = regrid_native_dispatch(
            np_by_level,
            l2b_np,
            meta,
            backend=self.regrid_backend,
            profiler=self.regrid_profiler,
            C=C_orig,
            T=T_in,
            tol_frac=self.regrid_tol_frac,
            cell_scale_mode=self.cell_scale_mode,
            adapt_on_channels=self.adapt_on_channels,
            adapt_nearby=self.regrid_adapt_nearby,
            allow_coarsening=self.allow_coarsening,
        )

        # Move back to torch.
        new_by_level = {
            lvl: torch.from_numpy(arr).to(self.device, dtype=torch.float32)
            for lvl, arr in new_by_level_np.items()
        }
        new_l2b_t = torch.from_numpy(new_l2b).to(self.device).long()  # (N_new, 2)
        new_centers_t = torch.from_numpy(new_meta["centers"]).to(
            self.device, dtype=torch.float32
        )  # (N_new, 3)

        return new_by_level, new_l2b_t, new_centers_t, new_meta


class MetricEvaluator:
    """
    Evaluator to compute metrics on the rollout predictions.
    Offloads the CPU-heavy Quadtree projection mapping from the GPU actor.

    Supports three GT sources (in priority order):
      1. target_src (numpy array) — NPZ mode with return_src=True
      2. target_src_paths (file paths) — AMReX mode with return_src=True
      3. Fallback: reference grid target from the predictor output
    """

    def __init__(self, config: dict):
        self.predict_steps = config["inference"].get("predict_steps", 1)
        self.fields = config["data"]["field_names"]
        loader_params = config.get("file_loader", {}).get("params", {})
        self.cell_scale_mode = loader_params.get("cell_scale_mode", "area")
        self.metrics = config["inference"].get("metrics", ["RMSE", "VRMSE"])
        self.compute_uniform_metrics = config["inference"].get("compute_uniform_metrics", True)
        self.mode = loader_params.get("mode", "npz")
        self.amrex_patch_size = loader_params.get("amrex_patch_size", 32)
        self.amrex_field_names = loader_params.get("field_names", self.fields)
        self.verbose = config["inference"].get("eval_verbose", False)
        self._pid = os.getpid()
        self._call_count = 0
        for m in self.metrics:
            if m not in METRIC_REGISTRY:
                raise ValueError(
                    f"Unknown uniform metric: {m}. Available: {list(METRIC_REGISTRY.keys())}"
                )

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # Dispatch between native and uniform mode based on predictor output.
        if "native_mode" in batch:
            return self._eval_native(batch)
        return self._eval_uniform(batch)

    # ------------------------------------------------------------------
    # Uniform-mode evaluation (original path)
    # ------------------------------------------------------------------

    def _eval_uniform(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        def _unpack(arr):
            return arr[0] if arr.dtype == object else arr

        preds_list = _unpack(
            batch["preds_list"]
        )  # list[(N_step, C, 1, Ph, Pw)], len = predict_steps
        centers_list = _unpack(batch["centers_list"])  # list[(N_step, 3)]
        levels_list = _unpack(batch["levels_list"])  # list[(N_step,)]

        targets = _unpack(batch["target"])  # (N_ref, C, T_out, Ph, Pw) on ref grid
        targets_src = (
            _unpack(batch["target_src"]) if "target_src" in batch else None
        )  # (T_out, C, H, W) after squeeze
        target_src_paths = (
            _unpack(batch["target_src_paths"]) if "target_src_paths" in batch else None
        )
        if targets_src is not None:
            while targets_src.ndim > 4 and targets_src.shape[0] == 1:
                targets_src = targets_src.squeeze(0)
        if targets.ndim == 5:
            targets = np.expand_dims(targets, 0)  # → (B=1, N_ref, C, T_out, Ph, Pw)

        ref_centers = _unpack(batch["ref_centers"])  # (1, N_ref, 3) or (N_ref, 3)
        ref_levels = _unpack(batch["ref_levels"])  # (1, N_ref)    or (N_ref,)

        if ref_centers.ndim == 3:
            ref_centers = ref_centers.squeeze(0)
        if ref_levels.ndim == 2:
            ref_levels = ref_levels.squeeze(0)

        domain = {
            k: _unpack(batch[f"domain_{k}"]).item()
            for k in ["xmin", "xmax", "ymin", "ymax", "max_level_idx", "tile_width", "tile_height"]
        }
        ref_meta = {"centers": ref_centers, "levels": ref_levels, "domain": domain}
        max_level_idx = int(domain["max_level_idx"])

        B = targets.shape[0]
        C = targets.shape[2]

        results = {
            f"step_{s}_c_{self.fields[c]}": np.zeros(B, dtype=np.float32)
            for s in range(self.predict_steps)
            for c in range(C)
        }

        has_any_src = targets_src is not None or target_src_paths is not None
        if has_any_src and self.metrics and self.compute_uniform_metrics:
            for s in range(self.predict_steps):
                for m in self.metrics:
                    for c in range(C):
                        results[f"step_{s}_uniform_{m}_{self.fields[c]}"] = np.zeros(
                            B, dtype=np.float32
                        )

        for s in range(self.predict_steps):
            for lvl in range(max_level_idx + 1):
                results[f"step_{s}_level_{lvl}_count"] = np.zeros(B, dtype=np.float32)
                for m in self.metrics:
                    for c in range(C):
                        results[f"step_{s}_level_{lvl}_{m}_{self.fields[c]}"] = np.full(
                            B, np.nan, dtype=np.float32
                        )

        self._call_count += 1
        t_block = time.perf_counter()

        for b in range(B):
            for step in range(self.predict_steps):
                pred_phys = preds_list[step]  # (N_step, C, 1, Ph, Pw)
                step_centers = centers_list[step]  # (N_step, 3)
                step_levels = levels_list[step]  # (N_step,)

                step_meta = {"centers": step_centers, "levels": step_levels, "domain": domain}
                N_step = pred_phys.shape[0]
                c_out, _, ph, pw = pred_phys.shape[1:]

                # --- GT assignment: pick best available source ---
                t_gt = time.perf_counter()
                gt_arr = None  # will be set if quadtree-based GT is built
                if targets_src is not None:
                    gt_src_frame = targets_src[step]  # (C, H, W) dense
                    step_qt = tensor_to_quadtree(
                        np.zeros((N_step, c_out, ph, pw), dtype=np.float32),
                        step_meta,
                        cell_scale_mode=self.cell_scale_mode,
                    )
                    step_qt.assign_from_array(gt_src_frame)
                    gt_arr, _ = quadtree_to_tensor(
                        step_qt, return_tensor=False, cell_scale_mode=self.cell_scale_mode
                    )  # (N_step, C, Ph, Pw)
                    target_eval = gt_arr[
                        :, :, np.newaxis
                    ]  # (N_step, C, 1, Ph, Pw) — match pred shape
                elif target_src_paths is not None:
                    from wamrvit.quad.amrex_to_qt import assign_from_amrex

                    step_qt = tensor_to_quadtree(
                        np.zeros((N_step, c_out, ph, pw), dtype=np.float32),
                        step_meta,
                        cell_scale_mode=self.cell_scale_mode,
                    )
                    assign_from_amrex(
                        step_qt,
                        target_src_paths[step],
                        self.amrex_field_names,
                        patch_size=self.amrex_patch_size,
                    )
                    gt_arr, _ = quadtree_to_tensor(
                        step_qt, return_tensor=False, cell_scale_mode=self.cell_scale_mode
                    )  # (N_step, C, Ph, Pw)
                    target_eval = gt_arr[:, :, np.newaxis]
                else:
                    target_ref = targets[b, :, :, step : step + 1]  # (N_ref, C, 1, Ph, Pw)
                    if target_ref.shape[0] == N_step:
                        # Prediction grid == reference grid — use directly.
                        target_eval = target_ref
                    else:
                        # Grids differ: project ref → uniform → step-quadtree.
                        gt_ref_frame = target_ref[:, :, 0]  # (N_ref, C, Ph, Pw)
                        gt_uniform = tensor_to_uniform(
                            gt_ref_frame, ref_meta, cell_scale_mode=self.cell_scale_mode
                        )  # (C, H, W)
                        step_qt = tensor_to_quadtree(
                            np.zeros((N_step, c_out, ph, pw), dtype=np.float32),
                            step_meta,
                            cell_scale_mode=self.cell_scale_mode,
                        )
                        step_qt.assign_from_array(gt_uniform)
                        gt_arr, _ = quadtree_to_tensor(
                            step_qt, return_tensor=False, cell_scale_mode=self.cell_scale_mode
                        )  # (N_step, C, Ph, Pw)
                        target_eval = gt_arr[:, :, np.newaxis]  # (N_step, C, 1, Ph, Pw)
                # ------------------------------------------
                t_gt_done = time.perf_counter()

                for c in range(C):
                    mse = np.mean((pred_phys[:, c] - target_eval[:, c]) ** 2)
                    results[f"step_{step}_c_{self.fields[c]}"][b] = float(mse)

                # --- Per-level metrics ---
                t_metrics = time.perf_counter()
                unique_levels = np.unique(step_levels)
                for lvl in unique_levels:
                    lvl = int(lvl)
                    mask = step_levels == lvl

                    pred_l = pred_phys[mask][:, :, 0, :, :]  # (N_l, C, Ph, Pw)
                    gt_l = target_eval[mask][:, :, 0, :, :]  # (N_l, C, Ph, Pw)

                    # METRIC_REGISTRY expects channel-last: (N, H, W, C).
                    pred_t = torch.from_numpy(pred_l).permute(0, 2, 3, 1).float()
                    gt_t = torch.from_numpy(gt_l).permute(0, 2, 3, 1).float()

                    results[f"step_{step}_level_{lvl}_count"][b] = float(np.sum(mask))

                    for metric_name in self.metrics:
                        val = eval_per_level_metric(metric_name, pred_t, gt_t)  # (C,)
                        for c in range(C):
                            key = f"step_{step}_level_{lvl}_{metric_name}_{self.fields[c]}"
                            results[key][b] = val[c].item()
                t_metrics_done = time.perf_counter()

                # --- Uniform grid metrics via regular_metrics ---
                t_uniform = time.perf_counter()
                if self.compute_uniform_metrics and self.metrics:
                    if targets_src is not None:
                        frame = pred_phys[:, :, 0, :, :]  # (N_step, C, Ph, Pw)
                        pred_uniform = tensor_to_uniform(
                            frame, step_meta, cell_scale_mode=self.cell_scale_mode
                        )  # (C, H, W)
                        gt_uniform_eval = targets_src[step]  # (C, H, W)

                        # Metric expects (B=1, H, W, C).
                        pred_t = (
                            torch.from_numpy(pred_uniform).permute(1, 2, 0).unsqueeze(0).float()
                        )
                        gt_t = (
                            torch.from_numpy(gt_uniform_eval).permute(1, 2, 0).unsqueeze(0).float()
                        )

                        for metric_name in self.metrics:
                            metric_cls = METRIC_REGISTRY[metric_name]
                            val = metric_cls.eval(pred_t, gt_t, n_spatial_dims=2)  # (1, C)
                            for c in range(C):
                                results[f"step_{step}_uniform_{metric_name}_{self.fields[c]}"][
                                    b
                                ] = val[0, c].item()

                    elif target_src_paths is not None and gt_arr is not None:
                        frame = pred_phys[:, :, 0, :, :]  # (N_step, C, Ph, Pw)
                        pred_uniform = tensor_to_uniform(
                            frame, step_meta, cell_scale_mode=self.cell_scale_mode
                        )  # (C, H, W)
                        gt_uniform_eval = tensor_to_uniform(
                            gt_arr, step_meta, cell_scale_mode=self.cell_scale_mode
                        )  # (C, H, W)

                        pred_t = (
                            torch.from_numpy(pred_uniform).permute(1, 2, 0).unsqueeze(0).float()
                        )
                        gt_t = (
                            torch.from_numpy(gt_uniform_eval).permute(1, 2, 0).unsqueeze(0).float()
                        )

                        for metric_name in self.metrics:
                            metric_cls = METRIC_REGISTRY[metric_name]
                            val = metric_cls.eval(pred_t, gt_t, n_spatial_dims=2)  # (1, C)
                            for c in range(C):
                                results[f"step_{step}_uniform_{metric_name}_{self.fields[c]}"][
                                    b
                                ] = val[0, c].item()
                t_uniform_done = time.perf_counter()

                if self.verbose:
                    print(
                        f"[eval pid={self._pid} call={self._call_count}] "
                        f"step {step}: N_step={N_step}, "
                        f"gt_build={t_gt_done - t_gt:.3f}s, "
                        f"metrics={t_metrics_done - t_metrics:.3f}s, "
                        f"uniform={t_uniform_done - t_uniform:.3f}s",
                        flush=True,
                    )

        t_block_done = time.perf_counter()
        if self.verbose:
            print(
                f"[eval pid={self._pid} call={self._call_count}] "
                f"block done in {t_block_done - t_block:.2f}s "
                f"(B={B}, steps={self.predict_steps})",
                flush=True,
            )

        return results

    # ------------------------------------------------------------------
    # Native-mode evaluation
    # ------------------------------------------------------------------

    def _eval_native(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Metrics for native (multi-scale) rollout predictions."""

        def _unpack(arr):
            return arr[0] if arr.dtype == object else arr

        preds_by_level_list = _unpack(
            batch["preds_by_level_list"]
        )  # list[{lvl: (N_l, C, 1, H_l, W_l)}]
        leaf_to_bucket_list = _unpack(batch["leaf_to_bucket_list"])  # list[(N, 2)]
        centers_list = _unpack(batch["centers_list"])  # list[(N, 3)]
        levels_list = _unpack(batch["levels_list"])  # list[(N,)]

        targets_src = (
            _unpack(batch["target_src"]) if "target_src" in batch else None
        )  # (T_out, C, H, W)
        if targets_src is not None:
            while targets_src.ndim > 4 and targets_src.shape[0] == 1:
                targets_src = targets_src.squeeze(0)

        domain = {
            k: _unpack(batch[f"domain_{k}"]).item()
            for k in ["xmin", "xmax", "ymin", "ymax", "max_level_idx", "tile_width", "tile_height"]
        }
        max_level_idx = int(domain["max_level_idx"])

        first_pred = preds_by_level_list[0]
        C = next(v.shape[1] for v in first_pred.values() if v.shape[0] > 0)

        B = 1
        results: dict[str, np.ndarray] = {}

        has_src = targets_src is not None
        for s in range(self.predict_steps):
            for c in range(C):
                results[f"step_{s}_c_{self.fields[c]}"] = np.zeros(B, dtype=np.float32)
            for lvl in range(max_level_idx + 1):
                results[f"step_{s}_level_{lvl}_count"] = np.zeros(B, dtype=np.float32)
                for m in self.metrics:
                    for c in range(C):
                        results[f"step_{s}_level_{lvl}_{m}_{self.fields[c]}"] = np.full(
                            B, np.nan, dtype=np.float32
                        )
            if has_src and self.compute_uniform_metrics:
                for m in self.metrics:
                    for c in range(C):
                        results[f"step_{s}_uniform_{m}_{self.fields[c]}"] = np.zeros(
                            B, dtype=np.float32
                        )

        for step in range(self.predict_steps):
            pred_by_level = preds_by_level_list[step]  # {lvl: (N_l, C, 1, H_l, W_l)}
            step_l2b = leaf_to_bucket_list[step]  # (N, 2)
            step_centers = centers_list[step]  # (N, 3)
            step_levels = levels_list[step]  # (N,)
            step_meta = {"centers": step_centers, "levels": step_levels, "domain": domain}

            # Build GT per-level buckets: either project dense src or use per-level targets.
            gt_by_level = None
            gt_src_frame = None
            if targets_src is not None:
                gt_src_frame = targets_src[step]  # (C, H, W)
                gt_by_level = build_native_gt_buckets(
                    pred_by_level,
                    step_l2b,
                    step_meta,
                    gt_src_frame,
                    cell_scale_mode=self.cell_scale_mode,
                )  # {lvl: (N_l, C, H_l, W_l)}

            if gt_by_level is None:
                gt_dict = _unpack(
                    batch["gt_by_level"]
                )  # {lvl: (N_l, C, T_out, H_l, W_l)} or (N_l, C, H_l, W_l)
                gt_by_level = {}
                for lvl, arr in gt_dict.items():
                    gt_by_level[lvl] = (
                        arr[:, :, step] if arr.ndim == 5 else arr
                    )  # (N_l, C, H_l, W_l)

            # Per-level + cell MSE metrics.
            eval_per_level_metrics(
                pred_by_level,
                gt_by_level,
                max_level_idx,
                self.metrics,
                self.fields,
                step,
                results,
            )

            # Uniform grid metrics.
            if (
                has_src
                and self.compute_uniform_metrics
                and self.metrics
                and gt_src_frame is not None
            ):
                eval_native_uniform_metrics(
                    pred_by_level,
                    step_l2b,
                    step_meta,
                    gt_src_frame,
                    self.metrics,
                    self.fields,
                    step,
                    results,
                )

        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to standalone yaml file (legacy). Omit to use Hydra config groups.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode with fewer samples."
    )
    parser.add_argument(
        "--eval_full", action="store_true", help="Whether to run the evaluation on every time step."
    )
    args, unknown = parser.parse_known_args()

    config = load_config(args, unknown)

    base_dir = os.getcwd()

    steps = config["inference"].get("predict_steps", 1)
    model_name = config["inference"]["checkpoint_path"].split("/")[-2]
    config["inference"]["model_name"] = model_name

    default_csv = f"rmse_adaptive_{model_name}_steps{steps}.csv"

    csv_path = config["inference"].get("output_csv") or os.path.join("results", default_csv)
    if not os.path.isabs(csv_path):
        csv_path = os.path.abspath(os.path.join(base_dir, csv_path))
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    config["inference"]["output_csv"] = csv_path

    if not ray.is_initialized():
        ray.init(
            object_store_memory=config["inference"].get("ray_object_store_memory", 10)
            * 1024
            * 1024
            * 1024
        )

    data_config = config["data"]
    file_parser = instantiate_from_config(config["file_parser"])
    all_paths = file_parser(data_config["glob_pattern"])

    # Make the window generator generalizable
    window_config = dict(config["window_generator"])
    window_config["params"]["return_seq_len"] = config["inference"].get("predict_steps", 1)
    params = dict(window_config.get("params", {}))
    params["file_path_list"] = all_paths
    window_config["params"] = params

    windows = instantiate_from_config(window_config)

    split_idx = int(len(windows) * config["inference"].get("split_ratio", 0.8))
    windows = windows[split_idx:]

    if not args.eval_full:
        # eval every "steps" steps to get a snapshot of performance across the rollout
        windows = windows[::steps]
        print(f"Evaluating every {steps} steps. Total evaluation samples: {len(windows)}")

    if args.debug:
        windows = windows[:100]

    for i, w in enumerate(windows):
        if isinstance(w, dict):
            w["idx"] = i

    total_samples = len(windows)

    # 1. Group windows by trajectory ID
    traj_to_indices = defaultdict(list)
    for i, w in enumerate(windows):
        if isinstance(w, dict):
            w["idx"] = i  # Assign global index
            t_id = w.get("traj_idx", 0)
            traj_to_indices[t_id].append(i)

    print(
        f"Total samples in validation split: {total_samples}, "
        f"Trajectories: {len(traj_to_indices)}\n"
    )

    ds = ray.data.from_items(windows)

    # Instantiate file loader strictly from the config context
    # Disable resample augmentation during rollout — it's training-only
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

    class MapperWithID(Seq2SeqMapper):
        def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            out = super().__call__(batch)
            if "idx" in batch:
                out["idx"] = batch["idx"]
            if "traj_idx" in batch:
                out["traj_idx"] = batch["traj_idx"]
            if "frame_idx" in batch:
                out["frame_idx"] = batch["frame_idx"]
            return out

    mapper = MapperWithID(loader=file_loader, transform=transform)

    # Per-stage RAM budgets (GB). Defaults are tuned for AMReX rollout; override
    # in inference config (e.g. inference.mapper_memory_gb: 10) if needed.
    GB = 1024**3
    mapper_mem_gb = config["inference"].get("mapper_memory_gb", 10)
    predictor_mem_gb = config["inference"].get("predictor_memory_gb", 15)
    eval_mem_gb = config["inference"].get("eval_memory_gb", 4)

    mapper_concurrency = config["inference"].get("mapper_concurrency", 4)
    eval_concurrency = config["inference"].get("eval_concurrency", 4)

    ds = ds.map_batches(
        mapper,
        batch_size=1,
        batch_format="numpy",
        compute=ray.data.TaskPoolStrategy(size=mapper_concurrency),
        memory=int(mapper_mem_gb * GB),
    )

    print("Running adaptive distributed inference pipeline...")
    pipeline_ds = ds.map_batches(
        AutoregressivePredictorAdaptive,
        fn_constructor_args=(config,),
        batch_size=1,
        num_gpus=1,
        compute=ray.data.ActorPoolStrategy(size=config["inference"].get("num_gpus", 1)),
        batch_format="numpy",
        memory=int(predictor_mem_gb * GB),
    )

    results_ds = pipeline_ds.map_batches(
        MetricEvaluator,
        fn_constructor_args=(config,),
        batch_size=1,
        batch_format="numpy",
        compute=ray.data.ActorPoolStrategy(size=eval_concurrency),
        memory=int(eval_mem_gb * GB),
    )

    df = results_ds.to_pandas()

    if df.empty:
        raise ValueError(
            "The evaluation dataset is empty! Check your glob pattern and debug split logic."
        )

    mean_vals = df.mean(numeric_only=True)
    rmse_series = np.sqrt(mean_vals)

    results_dict = {"Step": []}
    steps = config["inference"].get("predict_steps", 1)
    metrics = config["inference"].get("metrics", ["RMSE", "VRMSE"])
    max_level_idx = int(config["file_loader"]["params"].get("num_levels", 3)) - 1

    for field in data_config["field_names"]:
        results_dict[f"cell_RMSE_{field}"] = []

    # Check if uniform metric columns are present in the results
    has_uniform = any(col.startswith("step_0_uniform_") for col in df.columns)
    if has_uniform:
        for m in metrics:
            for field in data_config["field_names"]:
                results_dict[f"uniform_{m}_{field}"] = []

    # Check if per-level metric columns are present
    has_levels = any(col.startswith("step_0_level_") for col in df.columns)
    if has_levels:
        for lvl in range(max_level_idx + 1):
            results_dict[f"level_{lvl}_count"] = []
            for m in metrics:
                for field in data_config["field_names"]:
                    results_dict[f"level_{lvl}_{m}_{field}"] = []

    for s in range(steps):
        results_dict["Step"].append(s + 1)
        for field in data_config["field_names"]:
            col_name = f"step_{s}_c_{field}"
            results_dict[f"cell_RMSE_{field}"].append(rmse_series[col_name])

        if has_uniform:
            for m in metrics:
                for field in data_config["field_names"]:
                    col_name = f"step_{s}_uniform_{m}_{field}"
                    results_dict[f"uniform_{m}_{field}"].append(mean_vals.get(col_name, np.nan))

        if has_levels:
            for lvl in range(max_level_idx + 1):
                count_col = f"step_{s}_level_{lvl}_count"
                results_dict[f"level_{lvl}_count"].append(mean_vals.get(count_col, 0.0))
                for m in metrics:
                    for field in data_config["field_names"]:
                        col = f"step_{s}_level_{lvl}_{m}_{field}"
                        results_dict[f"level_{lvl}_{m}_{field}"].append(mean_vals.get(col, np.nan))

    final_df = pd.DataFrame(results_dict)

    final_df.to_csv(csv_path, index=False)

    print("\n=== Inference Complete ===")
    print(final_df.to_string(index=False))

    if has_levels and metrics:
        first_metric = metrics[0]
        level_summary_cols = ["Step"]
        for lvl in range(max_level_idx + 1):
            level_summary_cols.append(f"level_{lvl}_count")
            col = f"level_{lvl}_{first_metric}_{data_config['field_names'][0]}"
            if col in final_df.columns:
                level_summary_cols.append(col)
        print(f"\n--- Per-Level Summary ({first_metric}, {data_config['field_names'][0]}) ---")
        print(final_df[level_summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
