from __future__ import annotations

import argparse
import os
import math
import random
import warnings
import tempfile
import numpy as np
import torch
from typing import Dict

import ray
from ray import train
from ray.train import Checkpoint, ScalingConfig, RunConfig, CheckpointConfig, FailureConfig
from ray.train.torch import TorchConfig, TorchTrainer, prepare_model

from diffusers.utils import is_wandb_available
from diffusers.training_utils import EMAModel

from wamrvit.quadtree_transformer import QuadTreeTransformer
from wamrvit.utils import is_main_process, instantiate_from_config, get_scheduler_with_min_lr, UniformLossMaskScheduler, load_config, apply_two_stage_topology_pipeline, resolve_effective_config, get_git_info

from wamrvit.dataloader.trajectory_loader import Seq2SeqMapper
from wamrvit.quad.yt_utils import make_regular_centers
from wamrvit.native_train_utils import (
    unpack_native_batch, multi_scale_loss, residual_targets,
    UniformGeometry, full_field_loss, advance_window,
)


if is_wandb_available():
    import wandb


def train_func(config_dict: Dict):
    # Unpack configurations
    general_config = config_dict["general"]
    train_config = config_dict["train"]
    loss_fn_config = config_dict["loss_fn"]
    data_config = config_dict["data"]
    model_config = config_dict["model"]
    optimizer_config = config_dict["optimizer"]
    lr_scheduler_config = config_dict["lr_scheduler"]
    ema_config = config_dict.get("ema", {"use_ema": False})
    loss_mask_scheduler_config = config_dict.get("loss_mask_scheduler", None)

    # 1. Setup Device & Seeds
    device = train.torch.get_device()
    rank = train.get_context().get_world_rank()
    worker_seed = train_config["seed"] + rank

    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    # 3. Model Init
    model_class_name = model_config.pop("_class_name", "QuadTreeTransformer")
    if model_class_name == "SwinV2Transformer":
        from wamrvit.swin_transformer import SwinV2Transformer
        model_cls = SwinV2Transformer
    else:
        model_cls = QuadTreeTransformer
    model = model_cls(**model_config)

    # Native batches can leave per-level branches unused on individual ranks.
    parallel_strategy_kwargs = (
        {"broadcast_buffers": False} if model_class_name == "SwinV2Transformer" else {}
    )
    if model_config.get("multi_scale_patch"):
        parallel_strategy_kwargs["find_unused_parameters"] = True
    model = prepare_model(model, parallel_strategy_kwargs=parallel_strategy_kwargs)

    if rank == 0:
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
    loss_fn = instantiate_from_config(loss_fn_config)

    # 4. Resolve Epoch vs Step based training
    global_batch_size = train_config["batch_size"] * train_config["num_gpus"]
    grad_accum_steps = train_config.get("grad_accum_steps", 1)
    num_update_steps_per_epoch = math.ceil(data_config["num_train_samples"] / global_batch_size / grad_accum_steps)

    is_step_based = "num_train_steps" in train_config
    if is_step_based:
        assert "num_train_epochs" not in train_config, "Cannot specify both 'num_train_steps' and 'num_train_epochs' in the config."
        max_train_steps = train_config["num_train_steps"]
        max_epochs = float('inf')
        save_freq = train_config.get("save_every_steps")
    else:
        assert "num_train_epochs" in train_config, "Must specify either 'num_train_steps' or 'num_train_epochs'."
        max_epochs = train_config["num_train_epochs"]
        max_train_steps = max_epochs * num_update_steps_per_epoch
        save_freq = train_config.get("save_every_epochs")

    # LR Scheduler
    lr_scheduler = None
    if lr_scheduler_config and lr_scheduler_config.get("name"):
        lr_scheduler = get_scheduler_with_min_lr(
            name=lr_scheduler_config["name"],
            optimizer=optimizer,
            base_lr=optimizer_config["lr"],
            min_lr=lr_scheduler_config.get("min_lr", 0.0),
            num_warmup_steps=lr_scheduler_config["num_warmup_steps"],
            num_training_steps=max_train_steps,
            num_cycles=lr_scheduler_config.get("num_cycles", 1),
            power=lr_scheduler_config.get("power", 1.0),
        )

    if loss_mask_scheduler_config is not None:
        assert model_config["adaptive"], "Loss Mask Scheduler is only applicable for adaptive models."
        loss_mask_scheduler = UniformLossMaskScheduler(
            min_keep_ratio=loss_mask_scheduler_config.get("min_keep_ratio", 0.1),
            max_step_ratio=loss_mask_scheduler_config.get("max_step_ratio", 0.8),
            schedule_type=loss_mask_scheduler_config.get("schedule_type", "cosine"),
            min_keep_patch=loss_mask_scheduler_config.get("min_keep_patch", 100)
        )
    else:
        loss_mask_scheduler = None

    # 5. EMA Setup
    ema_model = None
    if ema_config.get("use_ema", False):
        ema_model = EMAModel(
            model.module.parameters() if hasattr(model, "module") else model.parameters(),
            decay=ema_config["ema_max_decay"],
            use_ema_warmup=True,
            update_after_step=ema_config["ema_update_after_step"],
            inv_gamma=ema_config["ema_inv_gamma"],
            power=ema_config["ema_power"],
            model_cls=model_cls,
            model_config=model.module.config if hasattr(model, "module") else model.config,
            foreach=ema_config.get("foreach", False),
        )
        ema_model.to(device)

    # 6. Resume / Checkpoint Loading
    start_epoch = 0
    global_step = 0
    checkpoint = train.get_checkpoint()

    mixed_precision = train_config.get("mixed_precision", "no")
    amp_dtype = None

    if mixed_precision == "bf16":
        amp_dtype = torch.bfloat16
    elif mixed_precision == "fp16":
        raise NotImplementedError("FP16 mixed precision is not supported in this script due to potential instability.")
    else:
        warnings.warn(f"Mixed precision not set or unrecognized ({mixed_precision}), defaulting to fp32.")
        amp_dtype = torch.float32

    print(f"[Rank {rank}] Training with precision: {mixed_precision}")

    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            if rank == 0:
                print(f"[Rank {rank}] Loading checkpoint from: {checkpoint_dir}")

            # Load Model
            ckpt_path = os.path.join(checkpoint_dir, "model.pt")
            if os.path.exists(ckpt_path):
                state = torch.load(ckpt_path, map_location=device)
                if hasattr(model, "module"):
                    model.module.load_state_dict(state)
                else:
                    model.load_state_dict(state)

            # Load Optimizer
            opt_path = os.path.join(checkpoint_dir, "optimizer.pt")
            if os.path.exists(opt_path):
                optimizer.load_state_dict(torch.load(opt_path, map_location=device))

            # Load Scheduler
            sched_path = os.path.join(checkpoint_dir, "lr_scheduler.pt")
            if lr_scheduler is not None and os.path.exists(sched_path):
                lr_scheduler.load_state_dict(torch.load(sched_path, map_location=device))

            # Load EMA
            ema_path = os.path.join(checkpoint_dir, "ema.pt")
            if ema_model is not None and os.path.exists(ema_path):
                ema_model.load_state_dict(torch.load(ema_path, map_location=device))

            # Load Training State
            state_path = os.path.join(checkpoint_dir, "training_state.pt")
            if os.path.exists(state_path):
                train_state = torch.load(state_path, map_location="cpu")
                # When resuming step-based, the epoch might just be a rough counter
                start_epoch = train_state["epoch"] + (0 if is_step_based else 1)
                global_step = train_state["global_step"]

    # 7. Logging Setup
    wandb_enabled = config_dict["args"]["wandb"]
    if wandb_enabled and is_main_process() and is_wandb_available():
        effective_config = resolve_effective_config(config_dict)
        effective_config["git"] = get_git_info()
        wandb.init(
            project="wamrvit",
            config=effective_config,
            name=general_config.get("run_name", "amrex_run"),
            reinit=True,
        )

    # 8. Data Shards
    ds_train = train.get_dataset_shard("train")

    num_push_forward_steps = train_config.get("num_push_forward_steps", 1)
    pin_memory = train_config.get("pin_memory", False)

    # Multi-scale native-storage path
    multi_scale = bool(model_config.get("multi_scale_patch", False))
    if multi_scale:
        assert model_config.get("adaptive", True), "multi_scale_patch requires adaptive=True."
        file_loader_params = config_dict.get("file_loader", {}).get("params", {})
        assert file_loader_params.get("value_storage", "uniform") == "native", \
               "multi_scale_patch requires file_loader.value_storage='native'."
    num_levels_native = int(model_config.get("max_level_idx", 2)) + 1 if multi_scale else 0

    # Native-mode loss mode. 
    # "per_leaf" (default) keeps the multi_scale_loss behavior; 
    # "full_field" scatters per-level predictions onto the uniform finest-resolution grid 
    # and calls loss_fn on that aligning the training objective with the uniform-grid eval metric.
    # Only active when multi_scale=True.
    loss_mode = train_config.get("loss_mode", "per_leaf")
    if loss_mode not in ("per_leaf", "full_field"):
        raise ValueError(f"train.loss_mode must be 'per_leaf' or 'full_field'; got {loss_mode!r}.")
    if loss_mode == "full_field" and not multi_scale:
        raise ValueError("loss_mode='full_field' requires multi_scale_patch=True.")
    # Geometry is dataset-level-constant but domain fields (xmin/xmax/...) come
    # from the batch. Cache on first encounter; reused for all subsequent batches.
    _geom_cache: Dict[str, UniformGeometry] = {}
    def _ensure_geom(batch) -> UniformGeometry:
        if "geom" not in _geom_cache:
            _geom_cache["geom"] = UniformGeometry.from_batch(
                batch,
                base_patch_h=int(model_config["patch_size"][0]),
                base_patch_w=int(model_config["patch_size"][1]),
            )
        return _geom_cache["geom"]

    def run_checkpointing(current_epoch: int, current_step: int, current_metrics: dict):
        checkpoint_obj = None
        if is_main_process():
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save(state_dict, os.path.join(temp_checkpoint_dir, "model.pt"))

                torch.save(optimizer.state_dict(), os.path.join(temp_checkpoint_dir, "optimizer.pt"))
                if lr_scheduler:
                    torch.save(lr_scheduler.state_dict(), os.path.join(temp_checkpoint_dir, "lr_scheduler.pt"))

                if ema_model is not None:
                    torch.save(ema_model.state_dict(), os.path.join(temp_checkpoint_dir, "ema.pt"))

                train_state = {
                    "epoch": current_epoch,
                    "global_step": current_step,
                }
                torch.save(train_state, os.path.join(temp_checkpoint_dir, "training_state.pt"))

                checkpoint_obj = Checkpoint.from_directory(temp_checkpoint_dir)
                train.report(current_metrics, checkpoint=checkpoint_obj)
        else:
            train.report(current_metrics)

    # -------------------------------------------------------------------------
    #  Main Training Loop
    # -------------------------------------------------------------------------
    epoch = start_epoch
    last_save_step = 0

    while epoch < max_epochs and global_step < max_train_steps:
        model.train()

        batch_iterator = ds_train.iter_torch_batches(
            batch_size=train_config.get("batch_size", 1),
            prefetch_batches=train_config.get("prefetch", 2),
            pin_memory=pin_memory,
        )

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(batch_iterator):
            if os.path.exists(".stop_signal"):
                if rank == 0:
                    print(f"[Epoch {epoch}] Graceful stop signal detected. Exiting batch loop cleanly.")
                break

            if multi_scale:
                inputs_by_level, targets_by_level, leaf_to_bucket, centers = unpack_native_batch(
                    batch, num_levels_native, device, non_blocking=pin_memory
                )
                mask_1d = None  # per-bucket masking not yet wired for multi-scale
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None and mixed_precision != "no")):
                    curr_inputs = inputs_by_level
                    input_seq_len = next(iter(inputs_by_level.values())).shape[2]
                    T_out = next(iter(targets_by_level.values())).shape[2]
                    assert T_out % num_push_forward_steps == 0, (
                        f"native target T_out ({T_out}) not divisible by "
                        f"num_push_forward_steps ({num_push_forward_steps})"
                    )
                    return_seq_len = T_out // num_push_forward_steps

                    step_total_loss = 0.0

                    for push_forward_step_idx in range(num_push_forward_steps):
                        target_start_idx = push_forward_step_idx * return_seq_len
                        target_end_idx = (push_forward_step_idx + 1) * return_seq_len
                        curr_target = {lvl: t[:, :, target_start_idx:target_end_idx] for lvl, t in targets_by_level.items()}
                        if train_config["pred_mode"] == "residual":
                            targets_learning = residual_targets(curr_target, curr_inputs)
                        else:
                            targets_learning = curr_target
                        pred_by_level = model(
                            curr_inputs, centers, leaf_to_bucket=leaf_to_bucket,
                        )
                        if loss_mode == "full_field":
                            step_loss = full_field_loss(
                                loss_fn, pred_by_level, targets_learning,
                                leaf_to_bucket, centers, _ensure_geom(batch),
                            )
                        else:
                            step_loss = multi_scale_loss(loss_fn, pred_by_level, targets_learning)
                        step_total_loss += step_loss
                        if push_forward_step_idx < num_push_forward_steps - 1:
                            curr_inputs = advance_window(
                                curr_inputs, pred_by_level, input_seq_len,
                                return_seq_len, train_config["pred_mode"],
                            )
                    loss = step_total_loss / num_push_forward_steps
                loss = loss / grad_accum_steps
                loss.backward()
                if (batch_idx + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    if ema_model is not None:
                        ema_model.step(model.parameters())
                    global_step += 1
                    loss_val = float(loss.item() * grad_accum_steps)
                    if wandb_enabled and is_main_process() and is_wandb_available():
                        logs = {"train/loss": loss_val, "epoch": epoch, "step": global_step}
                        if lr_scheduler:
                            logs["train/lr"] = lr_scheduler.get_last_lr()[0]
                        wandb.log(logs, step=global_step)
                continue

            seq_in = batch["input"].to(device, non_blocking=pin_memory)
            target = batch["target"].to(device, non_blocking=pin_memory)

            mask_1d = None
            if model_config["adaptive"]:
                seq_in = seq_in.squeeze(0) # -> (N_grids, C, T_in, H, W)
                target = target.squeeze(0)
                centers = batch["centers"].squeeze(0).to(device, non_blocking=pin_memory)
                if loss_mask_scheduler is not None:
                    mask_1d = loss_mask_scheduler.get_mask(seq_in.shape[0], global_step, max_train_steps, device=device) # (N_grids,) boolean mask
                    num_active_grids = mask_1d.sum().item()
            else:
                centers = make_regular_centers(seq_in.shape[-2], seq_in.shape[-1], p=model_config["patch_size"], device=device)

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None and mixed_precision != "no")):
                curr_input_seq = seq_in.clone()
                input_seq_len = curr_input_seq.shape[2]
                return_seq_len = target.shape[2] // num_push_forward_steps

                step_total_loss = 0.0

                for push_forward_step_idx in range(num_push_forward_steps):
                    target_start_idx = push_forward_step_idx * return_seq_len
                    target_end_idx = (push_forward_step_idx + 1) * return_seq_len
                    curr_target = target[:, :, target_start_idx:target_end_idx]

                    if train_config["pred_mode"] == "residual":
                        curr_target_learning = curr_target - curr_input_seq[:, :, -1].unsqueeze(2)
                    else:
                        curr_target_learning = curr_target

                    pred = model(curr_input_seq, centers)
                    step_loss = loss_fn(pred, curr_target_learning, mask=mask_1d)
                    step_total_loss += step_loss

                    if push_forward_step_idx < num_push_forward_steps - 1:
                        curr_input_seq = advance_window(
                            curr_input_seq, pred, input_seq_len,
                            return_seq_len, train_config["pred_mode"],
                        )

                loss = step_total_loss / num_push_forward_steps

            # Scale the loss by the number of accumulation steps
            loss = loss / grad_accum_steps

            # Backward pass happens every batch
            loss.backward()

            # Optimizer step and logging happen only every `grad_accum_steps`
            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                if lr_scheduler is not None:
                    lr_scheduler.step()

                if ema_model is not None:
                    ema_model.step(model.parameters())

                global_step += 1

                # Logging
                loss_val = float(loss.item() * grad_accum_steps) # Unscale for logging

                if wandb_enabled and is_main_process() and is_wandb_available():
                    logs = {"train/loss": loss_val, "epoch": epoch, "step": global_step}
                    if lr_scheduler:
                        logs["train/lr"] = lr_scheduler.get_last_lr()[0]
                    if loss_mask_scheduler is not None and model_config["adaptive"]:
                        logs["train/active_grid_ratio"] = num_active_grids / seq_in.shape[0]
                    wandb.log(logs, step=global_step)

        if os.path.exists(".stop_signal"):
            if rank == 0:
                print("Exiting epoch loop due to stop signal.")
            break

        current_metrics = {"epoch": epoch, "step": global_step}

        if save_freq is None:
            should_save = False
        elif is_step_based:
            is_final = global_step >= max_train_steps
            should_save = ((global_step - last_save_step) >= save_freq) or is_final
        else:
            is_final = (epoch + 1) >= max_epochs
            should_save = ((epoch + 1) % save_freq == 0) or is_final

        if should_save:
            run_checkpointing(current_epoch=epoch, current_step=global_step, current_metrics=current_metrics)
            last_save_step = global_step
        else:
            train.report(current_metrics)

        epoch += 1

    # Save model in diffusers safetensor format
    if is_main_process():
        if ema_model is not None:
            print("Swapping EMA weights into model for upload...")
            ema_model.store(model.parameters())
            ema_model.copy_to(model.parameters())

        final_save_path = os.path.join(general_config["save_dir"], general_config.get("run_name", "amrex_run"), "model")
        os.makedirs(final_save_path, exist_ok=True)

        print(f"Saving final model to {final_save_path}...")
        if train_config["num_gpus"] > 1:
            model.module.save_pretrained(final_save_path)
        else:
            model.save_pretrained(final_save_path)

        if ema_model is not None:
            ema_model.restore(model.parameters())

    # Finish
    if wandb_enabled and is_wandb_available() and is_main_process():
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to standalone yaml file (legacy). Omit to use Hydra config groups.")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument("--load_in_ram", action="store_true", help="Whether to load entire dataset into RAM. Not recommended for large datasets.")
    parser.add_argument("--cache_topology", action="store_true", help="Materialize quadtree topology in a first stage. Augmentation stays lazy and varies per epoch.")
    parser.add_argument("--topology_cache_dir", type=str, default=None, help="Directory to save/load cached Ray datasets for quadtree topology.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with fewer epochs and smaller dataset.")
    args, unknown = parser.parse_known_args()

    config = load_config(args, unknown)

    # 1. Initialize Ray
    if not ray.is_initialized():
        ray.init(object_store_memory=config["general"].get("ray_object_store_memory", 8) * 1024 * 1024 * 1024,
                 object_spilling_directory=config["general"].get("ray_spill_dir", None))

    # 2. Data Discovery
    data_config = config["data"]

    file_parser = instantiate_from_config(config["file_parser"])
    all_paths = file_parser(data_config["glob_pattern"])
    if args.debug:
        all_paths = all_paths[:10]

    print(f"Found {len(all_paths)} snapshots.")

    window_config = dict(config["window_generator"])
    window_config["params"]["return_seq_len"] *= config["train"].get("num_push_forward_steps", 1)
    params = dict(window_config.get("params", {}))
    params["file_path_list"] = all_paths # avoid flooding logs
    window_config["params"] = params
    assert window_config["params"]["return_seq_len"] == int(config["model"].get("return_seq_len", 1)*config["train"].get("num_push_forward_steps", 1)), \
        f"Return sequence length in window generator ({window_config['params']['return_seq_len']}) \
            must match model config ({config['model'].get('return_seq_len', 1)}) * num_push_forward_steps ({config['train'].get('num_push_forward_steps', 1)})."

    windows = instantiate_from_config(window_config)

    split_idx = int(len(windows) * config["train"]["split_ratio"])
    train_windows = windows[:split_idx]
    print(f"Using first {split_idx} of {len(windows)} windows for training (split_ratio {config['train']['split_ratio']}).")

    data_config["num_train_samples"] = len(train_windows)

    # 5. Create Ray Datasets
    # Turn the list of dicts into a distributed dataset
    ds_train = ray.data.from_items(train_windows)

    # 6. Instantiate Loader & Mapper
    file_loader = instantiate_from_config(config["file_loader"])
    transform = instantiate_from_config(config["transform"])

    # 7. Apply Map Batches
    mem_per_task = 10 * 1024 * 1024 * 1024  # 10 GiB
    map_batch_size = config["train"].get("map_batch_size", 1)
    actor_pool_size = config["train"].get("actor_pool_size", None)

    is_adaptive = config["model"].get("adaptive", False)

    if args.cache_topology and is_adaptive:
        # Two-stage pipeline: cached topology + lazy augmentation ---
        ds_train = apply_two_stage_topology_pipeline(
            ds_train, train_windows, "train", config, args,
            map_batch_size, actor_pool_size, mem_per_task, transform
        )

    else:
        # single-stage pipeline ---
        ds_train = ds_train.random_shuffle(seed=config["train"]["seed"])
        train_mapper = Seq2SeqMapper(loader=file_loader, transform=transform)

        ds_train = ds_train.map_batches(
            train_mapper,
            batch_size=map_batch_size,
            batch_format="numpy",
            compute=ray.data.TaskPoolStrategy(size=actor_pool_size),
            memory=mem_per_task,
        )

        if args.load_in_ram:
            print("Materializing Training Data into RAM (this may take a while)...")
            ds_train = ds_train.materialize()
            print("Materialization Complete. Data is now pinned in memory.")

    # 8. Trainer Configuration
    run_config = RunConfig(
        storage_path=os.path.abspath(config["general"]["save_dir"]),
        name=config["general"].get("run_name", "amrex_run"),
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
        ),
        failure_config=FailureConfig(max_failures=0)
    )

    scaling_config = ScalingConfig(
        num_workers=config["train"]["num_gpus"],
        use_gpu=True,
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=config,
        datasets={"train": ds_train},
        scaling_config=scaling_config,
        run_config=run_config,
    )

    print("Starting Training...")
    result = trainer.fit()
    print(f"Training finished. Best Checkpoint: {result.checkpoint}")

if __name__ == "__main__":
    main()
