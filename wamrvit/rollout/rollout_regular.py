import argparse
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import ray
import torch

from wamrvit.dataloader.loader import YTAmReXRegularLoader
from wamrvit.dataloader.trajectory_loader import Seq2SeqMapper
from wamrvit.quad.yt_utils import make_regular_centers
from wamrvit.quadtree_transformer import QuadTreeTransformer
from wamrvit.rollout.regular_metrics import METRIC_REGISTRY
from wamrvit.utils import instantiate_from_config, load_config


class AutoregressivePredictor:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        inf_cfg = config["inference"]

        model_class_name = config["model"].get("_class_name", "QuadTreeTransformer")
        if inf_cfg.get("is_diffusers", False):
            import json

            ckpt_config_path = os.path.join(inf_cfg["checkpoint_path"], "config.json")
            if os.path.exists(ckpt_config_path):
                with open(ckpt_config_path) as f:
                    model_class_name = json.load(f).get("_class_name", model_class_name)

        if model_class_name == "SwinV2Transformer":
            from wamrvit.swin_transformer import SwinV2Transformer

            model_cls = SwinV2Transformer
        else:
            model_cls = QuadTreeTransformer

        if inf_cfg.get("is_diffusers", False):
            print("Loading model via diffusers from:", inf_cfg["checkpoint_path"])
            self.model = model_cls.from_pretrained(inf_cfg["checkpoint_path"])
        else:
            print("Loading model state dict from:", inf_cfg["checkpoint_path"])
            self.model = model_cls(**config["model"])
            state_dict = torch.load(inf_cfg["checkpoint_path"], map_location="cpu")
            if "module." in list(state_dict.keys())[0]:
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        # Use the same config-driven transform pattern as training/adaptive rollout.
        self.transform = instantiate_from_config(config["transform"])

        self.predict_steps = inf_cfg.get("predict_steps", 1)
        self.model_return_seq_len = self.model.config.return_seq_len
        self.num_forward_calls = math.ceil(self.predict_steps / self.model_return_seq_len)
        self.pred_mode = inf_cfg.get("pred_mode", "target")

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        inputs = torch.from_numpy(batch["input"]).to(
            self.device, dtype=torch.float32
        )  # (B, C, T_in, H, W)
        targets = torch.from_numpy(batch["target"]).to(
            self.device, dtype=torch.float32
        )  # (B, C, T_out, H, W)

        B, C, T_in, H, W = inputs.shape
        centers = make_regular_centers(
            H, W, p=self.model.config.patch_size, device=self.device
        )  # (N_tokens, 3)
        R = self.model_return_seq_len

        curr_input_seq = inputs.clone()  # (B, C, T_in, H, W)

        all_preds = []  # list[(B, C, steps_this_call, H, W)]
        all_gts = []

        with torch.no_grad():
            for call_idx in range(self.num_forward_calls):
                timestep_idx = call_idx * R
                steps_this_call = min(R, self.predict_steps - timestep_idx)

                pred_full = self.model(curr_input_seq, centers)  # (B, C, R, H, W)

                if self.pred_mode == "residual":
                    pred_full = pred_full + curr_input_seq[:, :, -1].unsqueeze(2)  # (B, C, R, H, W)

                pred_to_store = pred_full[:, :, :steps_this_call]  # (B, C, steps_this_call, H, W)
                gt = targets[
                    :, :, timestep_idx : timestep_idx + steps_this_call
                ]  # (B, C, steps_this_call, H, W)
                pred_np = pred_to_store.cpu().numpy()
                gt_np = gt.cpu().numpy()

                # Inverse transform outputs back to physical space
                if hasattr(self.transform, "inverse_transform") and callable(
                    self.transform.inverse_transform
                ):
                    pred_phys = self.transform.inverse_transform(pred_np)
                    gt_phys = self.transform.inverse_transform(gt_np)
                else:
                    pred_phys = pred_np
                    gt_phys = gt_np

                all_preds.append(pred_phys)
                all_gts.append(gt_phys)

                if call_idx < self.num_forward_calls - 1:
                    # Slide window: keep last (T_in - R) input frames, append the full R-frame pred.
                    num_from_input = max(T_in - R, 0)
                    if num_from_input > 0:
                        curr_input_seq = torch.cat(
                            (curr_input_seq[:, :, -num_from_input:], pred_full), dim=2
                        )  # (B, C, T_in, H, W)
                    else:
                        curr_input_seq = pred_full[:, :, -T_in:]  # (B, C, T_in, H, W)

        pred_full = np.concatenate(all_preds, axis=2)  # (B, C, predict_steps, H, W)
        gt_full = np.concatenate(all_gts, axis=2)  # (B, C, predict_steps, H, W)

        out_dict = {"pred": pred_full, "target": gt_full}
        if "idx" in batch:
            out_dict["idx"] = batch["idx"]
        if "traj_idx" in batch:
            out_dict["traj_idx"] = batch["traj_idx"]
        if "frame_idx" in batch:
            out_dict["frame_idx"] = batch["frame_idx"]

        return out_dict


class MetricEvaluator:
    """
    Evaluator to compute metrics on regular-grid rollout predictions.
    Takes heavy arrays, computes numbers, and returns ONLY numbers.
    """

    def __init__(self, config: dict):
        self.predict_steps = config["inference"].get("predict_steps", 1)
        # Support both legacy `data.fields` and newer `data.field_names`.
        self.fields = config["data"].get("field_names", config["data"].get("fields", []))
        if not self.fields:
            raise ValueError(
                "Expected `data.field_names` (or legacy `data.fields`) in rollout config."
            )
        self.metrics = config["inference"].get("metrics", ["RMSE", "VRMSE"])
        for m in self.metrics:
            if m not in METRIC_REGISTRY:
                raise ValueError(
                    f"Unknown uniform metric: {m}. Available: {list(METRIC_REGISTRY.keys())}"
                )

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        preds = batch["pred"]
        targets = batch["target"]
        B, C, T_total, H, W = preds.shape

        results = {
            f"step_{s}_c_{self.fields[c]}": np.zeros(B, dtype=np.float32)
            for s in range(self.predict_steps)
            for c in range(C)
        }

        # Uniform metric columns
        if self.metrics:
            for s in range(self.predict_steps):
                for m in self.metrics:
                    for c in range(C):
                        results[f"step_{s}_uniform_{m}_{self.fields[c]}"] = np.zeros(
                            B, dtype=np.float32
                        )

        for b in range(B):
            for step in range(self.predict_steps):
                # Cell MSE (legacy metric)
                for c in range(C):
                    mse = np.mean((preds[b, c, step] - targets[b, c, step]) ** 2)
                    results[f"step_{step}_c_{self.fields[c]}"][b] = float(mse)

                # Uniform metrics via METRIC_REGISTRY
                if self.metrics:
                    # pred/target slices: (C, H, W) -> permute (H, W, C); add batch dim
                    pred_slice = (
                        torch.from_numpy(preds[b, :, step]).permute(1, 2, 0).unsqueeze(0).float()
                    )
                    gt_slice = (
                        torch.from_numpy(targets[b, :, step]).permute(1, 2, 0).unsqueeze(0).float()
                    )

                    for metric_name in self.metrics:
                        metric_cls = METRIC_REGISTRY[metric_name]
                        val = metric_cls.eval(pred_slice, gt_slice, n_spatial_dims=2)  # (1, C)
                        for c in range(C):
                            results[f"step_{step}_uniform_{metric_name}_{self.fields[c]}"][b] = val[
                                0, c
                            ].item()

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

    base_dir = os.getcwd()  # Get the terminal's current working directory

    steps = config["inference"].get("predict_steps", 1)
    model_name = config["inference"]["checkpoint_path"].split("/")[-2]
    config["inference"]["model_name"] = model_name

    default_csv = f"rmse_{model_name}_steps{steps}.csv"

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

    # Build windows from config so we can support both single-trajectory PLT and
    # multi-trajectory NPZ parsers with one rollout script.
    window_config = dict(config["window_generator"])
    window_config["params"]["return_seq_len"] = config["inference"].get("predict_steps", 1)
    params = dict(window_config.get("params", {}))
    params["file_path_list"] = all_paths
    window_config["params"] = params
    windows = instantiate_from_config(window_config)

    # Keep train/val split behavior configurable and consistent with adaptive rollout.
    split_idx = int(len(windows) * config["inference"].get("split_ratio", 0.8))
    windows = windows[split_idx:]

    if not args.eval_full:
        windows = windows[::steps]
        print(f"Evaluating every {steps} steps. Total evaluation samples: {len(windows)}")

    if args.debug:
        windows = windows[:100]

    for i, w in enumerate(windows):
        if isinstance(w, dict):
            w["idx"] = i

    total_samples = len(windows)

    # Group windows by trajectory ID for logging
    traj_to_indices = defaultdict(list)
    for i, w in enumerate(windows):
        if isinstance(w, dict):
            t_id = w.get("traj_idx", 0)
            traj_to_indices[t_id].append(i)

    print(
        f"Total samples in validation split: {total_samples}, "
        f"Trajectories: {len(traj_to_indices)}\n"
    )

    ds = ray.data.from_items(windows)

    # Use explicit loader from config when present (e.g., NPZ regular path).
    # Fallback keeps old PLT behavior without forcing config churn.
    if "file_loader" in config:
        file_loader = instantiate_from_config(config["file_loader"])
    else:
        fallback_fields = data_config.get("field_names", data_config.get("fields"))
        file_loader = YTAmReXRegularLoader(
            field_names=fallback_fields,
            domain_from=data_config.get("domain_from", "domain"),
        )
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

    # Per-stage RAM budgets (GB). Regular-grid blocks are smaller than adaptive,
    # so defaults here are lower. Override in inference config if needed.
    GB = 1024**3
    mapper_mem_gb = config["inference"].get("mapper_memory_gb", 3)
    predictor_mem_gb = config["inference"].get("predictor_memory_gb", 8)
    eval_mem_gb = config["inference"].get("eval_memory_gb", 2)

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

    # 1. Distributed Inference (Predicting)
    print("Running distributed inference pipeline...")
    pipeline_ds = ds.map_batches(
        AutoregressivePredictor,
        fn_constructor_args=(config,),
        batch_size=1,
        num_gpus=1,
        compute=ray.data.ActorPoolStrategy(size=config["inference"].get("num_gpus", 1)),
        batch_format="numpy",
        memory=int(predictor_mem_gb * GB),
    )

    # 2. Metric Evaluation
    results_ds = pipeline_ds.map_batches(
        MetricEvaluator,
        fn_constructor_args=(config,),
        batch_size=1,
        batch_format="numpy",
        compute=ray.data.ActorPoolStrategy(size=eval_concurrency),
        memory=int(eval_mem_gb * GB),
    )

    # 3. Trigger execution and aggregate
    df = results_ds.to_pandas()

    if df.empty:
        raise ValueError(
            "The evaluation dataset is empty! Check your glob pattern and debug split logic."
        )

    mean_mse = df.mean(numeric_only=True)
    rmse_series = np.sqrt(mean_mse)

    results_dict = {"Step": []}
    steps = config["inference"].get("predict_steps", 1)
    metrics = config["inference"].get("metrics", ["RMSE", "VRMSE"])

    # Support both field naming conventions in existing configs.
    eval_fields = data_config.get("field_names", data_config.get("fields", []))
    if not eval_fields:
        raise ValueError("Expected `data.field_names` (or legacy `data.fields`) in rollout config.")

    for field in eval_fields:
        results_dict[f"cell_RMSE_{field}"] = []

    # Check if uniform metric columns are present in the results
    has_uniform = any(col.startswith("step_0_uniform_") for col in df.columns)
    mean_uniform = df.mean(numeric_only=True) if has_uniform else None
    if has_uniform:
        for m in metrics:
            for field in eval_fields:
                results_dict[f"uniform_{m}_{field}"] = []

    for s in range(steps):
        results_dict["Step"].append(s + 1)
        for field in eval_fields:
            col_name = f"step_{s}_c_{field}"
            results_dict[f"cell_RMSE_{field}"].append(rmse_series[col_name])

        if has_uniform:
            for m in metrics:
                for field in eval_fields:
                    col_name = f"step_{s}_uniform_{m}_{field}"
                    results_dict[f"uniform_{m}_{field}"].append(mean_uniform[col_name])

    final_df = pd.DataFrame(results_dict)

    final_df.to_csv(csv_path, index=False)

    print("\n=== Inference Complete ===")
    print(final_df.to_string(index=False))


if __name__ == "__main__":
    main()
