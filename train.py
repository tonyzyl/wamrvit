from __future__ import annotations

import argparse
import math
import os
import random
import tempfile
import warnings
from typing import Dict

import numpy as np
import ray
import torch
from diffusers.training_utils import EMAModel
from diffusers.utils import is_wandb_available
from ray import train
from ray.train import (
    Checkpoint,
    CheckpointConfig,
    FailureConfig,
    RunConfig,
    ScalingConfig,
)
from ray.train.torch import TorchTrainer, prepare_model

from wamrvit.dataloader.transport.runtime import (
    TrainingBatchSource,
    configure_payload_transport,
    materialize_with_capacity,
    prepare_source_provider,
    validate_source_provider,
)
from wamrvit.dataloader.sequence import advance_window
from wamrvit.dataloader.payload.mappers import Seq2SeqMapper
from wamrvit.dataloader.pipeline import apply_two_stage_topology_pipeline
from wamrvit.loss import UniformLossMaskScheduler, full_field_loss, multi_scale_loss
from wamrvit.multi_scale import UniformGeometry, prepare_multi_scale_batch, residual_targets
from wamrvit.optim import (
    GradientAccumulator,
    build_lr_scheduler,
    build_optimizer,
    describe_optimizer,
    optimizer_updates_per_epoch,
    positive_int,
)
from wamrvit.quad.quad_utils import make_regular_centers
from wamrvit.quadtree_transformer import QuadTreeTransformer
from wamrvit.utils import (
    get_git_info,
    instantiate_from_config,
    is_main_process,
    load_config,
    resolve_effective_config,
)

if is_wandb_available():
    import wandb


def train_func(config_dict: Dict):
    # Unpack configurations
    general_config = config_dict["general"]
    train_config = config_dict["train"]
    loss_fn_config = config_dict["loss_fn"]
    data_config = config_dict["data"]
    model_config = dict(config_dict["model"])
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
    # Prepare model for DDP. Swin disables buffer sync; native level buckets
    # may leave some encoder/output branches unused.
    parallel_strategy_kwargs = (
        {"broadcast_buffers": False} if model_class_name == "SwinV2Transformer" else {}
    )
    if model_config.get("multi_scale_patch"):
        parallel_strategy_kwargs["find_unused_parameters"] = True
    model = prepare_model(model, parallel_strategy_kwargs=parallel_strategy_kwargs)

    if rank == 0:
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = build_optimizer(model, optimizer_config)
    if rank == 0:
        print(describe_optimizer(optimizer))
    loss_fn = instantiate_from_config(loss_fn_config)

    # 4. Resolve epoch vs. optimizer-step based training. Ray's equal shard
    # split drops the global remainder, so derive the scheduler horizon from
    # the batches each rank actually consumes rather than the global ceiling.
    batch_size = positive_int(train_config, "batch_size", 1)
    world_size = positive_int(train_config, "num_gpus", 1)
    grad_accum_steps = positive_int(train_config, "grad_accum_steps", 1)
    num_update_steps_per_epoch = optimizer_updates_per_epoch(
        num_samples=int(data_config["num_train_samples"]),
        world_size=world_size,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
    )
    if rank == 0:
        print(
            "Gradient accumulation: "
            f"steps={grad_accum_steps}, optimizer_updates_per_epoch="
            f"{num_update_steps_per_epoch}"
        )

    is_step_based = "num_train_steps" in train_config
    if is_step_based:
        if "num_train_epochs" in train_config:
            raise ValueError(
                "Cannot specify both 'num_train_steps' and 'num_train_epochs' in the config."
            )
        max_train_steps = train_config["num_train_steps"]
        max_epochs = float('inf')
        save_freq = train_config.get("save_every_steps")
    else:
        if "num_train_epochs" not in train_config:
            raise ValueError("Must specify either 'num_train_steps' or 'num_train_epochs'.")
        max_epochs = train_config["num_train_epochs"]
        max_train_steps = max_epochs * num_update_steps_per_epoch
        save_freq = train_config.get("save_every_epochs")

    # LR Scheduler
    lr_scheduler = build_lr_scheduler(
        optimizer,
        lr_scheduler_config,
        num_training_steps=max_train_steps,
    )

    if loss_mask_scheduler_config is not None:
        if not model_config["adaptive"]:
            raise ValueError("Loss Mask Scheduler is only applicable for adaptive models.")
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
        effective_config.pop("__source_fill_params__", None)
        effective_config.pop("__cache_topology_pipeline__", None)
        effective_config["git"] = get_git_info()
        wandb.init(
            project="wamrvit",
            config=effective_config,
            name=general_config.get("run_name", "train_run"),
            reinit=True,
        )

    # 8. Rank-owned data shard. Ray owns shuffle and equal splitting.
    ds_train = train.get_dataset_shard("train")
    if ds_train is None:
        raise RuntimeError(f"Rank {rank} did not receive the training dataset.")
    batch_source = TrainingBatchSource(config_dict, ds_train, rank=rank)

    num_push_forward_steps = train_config.get("num_push_forward_steps", 1)
    pin_memory = batch_source.pin_memory

    # Multi-scale native-storage path
    multi_scale = bool(model_config.get("multi_scale_patch", False))
    if multi_scale:
        if not model_config.get("adaptive", True):
            raise ValueError("multi_scale_patch requires adaptive=True.")
        file_loader_params = config_dict.get("file_loader", {}).get("params", {})
        if file_loader_params.get("value_storage", "uniform") != "native":
            raise ValueError("multi_scale_patch requires file_loader.value_storage='native'.")
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

    accumulator = GradientAccumulator(
        model,
        optimizer,
        steps=grad_accum_steps,
        lr_scheduler=lr_scheduler,
        ema_model=ema_model,
    )
    latest_train_loss: torch.Tensor | None = None

    def record_optimizer_update(
        mean_loss: torch.Tensor,
        *,
        active_grid_ratio: float | None = None,
    ) -> None:
        nonlocal global_step, latest_train_loss
        global_step += 1
        latest_train_loss = mean_loss
        if wandb_enabled and is_main_process() and is_wandb_available():
            logs = {
                "train/loss": float(mean_loss.item()),
                "epoch": epoch,
                "step": global_step,
            }
            if lr_scheduler:
                logs["train/lr"] = lr_scheduler.get_last_lr()[0]
            if active_grid_ratio is not None:
                logs["train/active_grid_ratio"] = active_grid_ratio
            wandb.log(logs, step=global_step)

    # -------------------------------------------------------------------------
    #  Main Training Loop
    # -------------------------------------------------------------------------
    epoch = start_epoch
    last_save_step = 0

    while epoch < max_epochs and global_step < max_train_steps:
        model.train()
        batch_iterator = batch_source.start_epoch(epoch)
        epoch_start_step = global_step
        latest_train_loss = None
        accumulator.start_epoch()

        for batch_idx, batch in enumerate(batch_iterator):
            if os.path.exists(".stop_signal"):
                if rank == 0:
                    print(
                        f"[Epoch {epoch}] Graceful stop signal detected. "
                        "Exiting batch loop cleanly."
                    )
                break

            if multi_scale:
                inputs_by_level, targets_by_level, bucket_layout = prepare_multi_scale_batch(
                    batch, num_levels_native, device, non_blocking=pin_memory,
                    geometry=_ensure_geom(batch) if loss_mode == "full_field" else None,
                )
                leaf_to_bucket, centers = bucket_layout.leaf_to_bucket, bucket_layout.centers
                mask_1d = None  # per-bucket masking not yet wired for multi-scale
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None and mixed_precision != "no")):
                    curr_inputs = inputs_by_level
                    input_seq_len = next(iter(inputs_by_level.values())).shape[2]
                    T_out = next(iter(targets_by_level.values())).shape[2]
                    if T_out % num_push_forward_steps != 0:
                        raise ValueError(
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
                            curr_inputs,
                            centers,
                            leaf_to_bucket=leaf_to_bucket,
                            bucket_layout=bucket_layout,
                        )
                        if loss_mode == "full_field":
                            step_loss = full_field_loss(
                                loss_fn, pred_by_level, targets_learning,
                                leaf_to_bucket, centers, _ensure_geom(batch),
                                bucket_layout=bucket_layout,
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
                update_loss = accumulator.backward(loss)
                if update_loss is not None:
                    record_optimizer_update(update_loss)
                if is_step_based and global_step >= max_train_steps:
                    break
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
                curr_input_seq = seq_in  # Read-only; advance_window replaces the window.
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

                    pred = model(curr_input_seq, centers)  # (N, C, R, Ph, Pw)
                    step_loss = loss_fn(pred, curr_target_learning, mask=mask_1d)
                    step_total_loss += step_loss

                    if push_forward_step_idx < num_push_forward_steps - 1:
                        curr_input_seq = advance_window(
                            curr_input_seq, pred, input_seq_len,
                            return_seq_len, train_config["pred_mode"],
                        )

                loss = step_total_loss / num_push_forward_steps

            update_loss = accumulator.backward(loss)
            if update_loss is not None:
                active_grid_ratio = (
                    num_active_grids / seq_in.shape[0]
                    if loss_mask_scheduler is not None and model_config["adaptive"]
                    else None
                )
                record_optimizer_update(
                    update_loss,
                    active_grid_ratio=active_grid_ratio,
                )

            if is_step_based and global_step >= max_train_steps:
                break
        else:
            update_loss = accumulator.flush()
            if update_loss is not None:
                record_optimizer_update(update_loss)

        if os.path.exists(".stop_signal"):
            batch_source.close_epoch()
            if rank == 0:
                print("Exiting epoch loop due to stop signal.")
            break

        if is_step_based and global_step >= max_train_steps:
            batch_source.close_epoch()
            train.report({"epoch": epoch, "step": global_step})
            break

        batch = None
        transport_metrics, transport_wandb_metrics = batch_source.finish_epoch(epoch)
        epoch_updates = global_step - epoch_start_step
        if epoch_updates != num_update_steps_per_epoch:
            raise RuntimeError(
                f"Epoch {epoch} made {epoch_updates} optimizer updates; "
                f"expected {num_update_steps_per_epoch}."
            )
        if latest_train_loss is None:
            raise RuntimeError(f"Epoch {epoch} completed without an optimizer update.")
        epoch_train_loss = float(latest_train_loss.item())
        if not math.isfinite(epoch_train_loss):
            raise FloatingPointError(
                f"Epoch {epoch} ended with non-finite training loss {epoch_train_loss}."
            )
        if rank == 0:
            print(
                f"[Epoch {epoch}] optimizer updates: actual={epoch_updates} "
                f"expected={num_update_steps_per_epoch} global_step={global_step} "
                f"last_group_mean_loss={epoch_train_loss:.8g}"
            )
        if (
            transport_wandb_metrics
            and wandb_enabled
            and is_main_process()
            and is_wandb_available()
        ):
            wandb.log(transport_wandb_metrics, step=global_step)

        current_metrics = {
            "epoch": epoch,
            "step": global_step,
            "optimizer_updates_this_epoch": epoch_updates,
            "optimizer_updates_per_epoch": num_update_steps_per_epoch,
            "train_loss": epoch_train_loss,
            **transport_metrics,
        }

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

        final_save_path = os.path.join(
            general_config["save_dir"],
            general_config.get("run_name", "train_run"),
            "model",
        )
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
    parser.add_argument(
        "--load_in_ram",
        action="store_true",
        help=(
            "Deprecated alias for ray_resident; requires a validated "
            "train.resident_payload_estimate_gib capacity preflight."
        ),
    )
    parser.add_argument("--cache_topology", action="store_true", help="Materialize quadtree topology in a first stage. Augmentation stays lazy and varies per epoch.")
    parser.add_argument("--topology_cache_dir", type=str, default=None, help="Directory to save/load cached Ray datasets for quadtree topology.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with fewer epochs and smaller dataset.")
    args, unknown = parser.parse_known_args()

    config = load_config(args, unknown)
    train_config = config["train"]
    resolved_transport, capacity_decision = configure_payload_transport(
        config,
        load_in_ram=args.load_in_ram,
        topology_cached=args.cache_topology,
    )

    validate_source_provider(config, topology_cached=args.cache_topology)
    from wamrvit.dataloader.selection.windows import prepare_window_splits, validate_temporal_config

    validate_temporal_config(config, topology_cached=args.cache_topology)

    # 1. Initialize Ray
    if not ray.is_initialized():
        ray.init(
            object_store_memory=int(
                config["general"].get("ray_object_store_memory", 8) * 1024**3
            ),
            object_spilling_directory=config["general"].get("ray_spill_dir"),
        )

    # 2. Data Discovery
    data_config = config["data"]

    file_parser = instantiate_from_config(config["file_parser"])
    all_paths = file_parser(data_config["glob_pattern"])
    if args.debug:
        all_paths = all_paths[:10]

    print(f"Found {len(all_paths)} snapshots.")

    from wamrvit.dataloader.selection.sources import prepare_source_partitions
    from wamrvit.dataloader.selection.splits import record_source_partitions

    source_partitions = prepare_source_partitions(config, all_paths)
    record_source_partitions(all_paths, source_partitions, config)
    _, windows_by_role = prepare_window_splits(config, source_partitions)
    train_windows = windows_by_role["train"]
    print(f"Using {len(train_windows)} training windows.")
    data_config["num_train_samples"] = len(train_windows)
    world_size = positive_int(train_config, "num_gpus", 1)
    if len(train_windows) < world_size:
        raise ValueError(
            f"Training has {len(train_windows)} windows for {world_size} ranks."
        )
    print(
        "Ray scheduling: random shuffle and equal splitting; exact "
        "cross-session rank-lane replay is not enforced."
    )

    prepare_source_provider(
        config,
        train_windows,
        topology_cached=args.cache_topology,
    )

    # Adaptive topology construction remains seed-independent; each pipeline
    # creates its own descriptors, and Ray owns shuffle and equal splitting.
    is_adaptive = config["model"].get("adaptive", False)

    # 6. Instantiate Loader & Mapper
    file_loader = instantiate_from_config(config["file_loader"])
    transform = instantiate_from_config(config["transform"])

    # 7. Apply Map Batches
    mem_per_task = 10 * 1024 * 1024 * 1024  # 10 GiB
    map_batch_size = train_config.get("map_batch_size", 1)
    actor_pool_size = train_config.get("actor_pool_size")

    if args.cache_topology and is_adaptive:
        # Two-stage pipeline: cached topology + lazy augmentation.
        ds_train = apply_two_stage_topology_pipeline(
            train_windows,
            "train",
            config,
            args,
            map_batch_size,
            actor_pool_size,
            mem_per_task,
            transform,
        )
    else:
        ds_train = ray.data.from_items(train_windows)
        train_mapper = Seq2SeqMapper(loader=file_loader, transform=transform)
        stage2_pool_size = train_config.get(
            "ray_stream_map_concurrency", actor_pool_size
        )
        ds_train = ds_train.random_shuffle(seed=train_config["seed"])
        ds_train = ds_train.map_batches(
            train_mapper,
            batch_size=map_batch_size,
            batch_format="numpy",
            compute=ray.data.TaskPoolStrategy(size=stage2_pool_size),
            memory=mem_per_task,
        )

    if resolved_transport == "ray_resident":
        print("Materializing training data into Ray's object store...")
        resident_limit = int(capacity_decision["usable_object_store_gib"] * 1024**3)
        materialized, resident_bytes = materialize_with_capacity(
            [ds_train], max_total_bytes=resident_limit
        )
        ds_train = materialized[0]
        train_config["resident_materialized_bytes"] = resident_bytes
        print(f"Resident materialization complete: {resident_bytes / 1024**3:.3f} GiB.")

    # 8. Trainer Configuration
    run_config = RunConfig(
        storage_path=os.path.abspath(config["general"]["save_dir"]),
        name=config["general"].get("run_name", "train_run"),
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
