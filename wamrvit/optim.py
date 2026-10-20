from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from diffusers.optimization import get_scheduler
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


def positive_int(config: Mapping[str, Any], name: str, default: int) -> int:
    """Read a strictly positive integer training option without coercing it."""
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"train.{name} must be a positive integer; got {value!r}.")
    return value


def optimizer_updates_per_epoch(
    *,
    num_samples: int,
    world_size: int,
    batch_size: int,
    grad_accum_steps: int,
) -> int:
    """Return updates made by one equal Ray shard, including its short group.

    Ray's equal split gives every rank ``num_samples // world_size`` samples and
    drops the global remainder. The worker batches that rank-local count, then
    flushes a final incomplete accumulation group at the epoch boundary.
    """
    for name, value, allow_zero in (
        ("num_samples", num_samples, True),
        ("world_size", world_size, False),
        ("batch_size", batch_size, False),
        ("grad_accum_steps", grad_accum_steps, False),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or (not allow_zero and value == 0)
        ):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be a {qualifier} integer; got {value!r}.")
    samples_per_rank = num_samples // world_size
    microbatches_per_rank = math.ceil(samples_per_rank / batch_size)
    return math.ceil(microbatches_per_rank / grad_accum_steps)


class GradientAccumulator:
    """Own gradient averaging and optimizer-update boundaries for one worker."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Any,
        *,
        steps: int,
        lr_scheduler: Any = None,
        ema_model: Any = None,
        max_grad_norm: float = 1.0,
    ) -> None:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError(f"Gradient accumulation steps must be positive; got {steps!r}.")
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.lr_scheduler = lr_scheduler
        self.ema_model = ema_model
        self.max_grad_norm = float(max_grad_norm)
        self.microbatches = 0
        self._loss_sum: torch.Tensor | None = None

    def start_epoch(self) -> None:
        if self.microbatches:
            raise RuntimeError("Cannot start an epoch with unflushed accumulated gradients.")
        self.optimizer.zero_grad()

    def backward(self, loss: torch.Tensor) -> torch.Tensor | None:
        """Accumulate one scalar loss and update when a full group is ready."""
        if loss.ndim != 0:
            raise ValueError(f"Accumulated loss must be scalar; got shape {tuple(loss.shape)}.")
        self._loss_sum = (
            loss.detach().float()
            if self._loss_sum is None
            else self._loss_sum + loss.detach().float()
        )
        backward_loss = loss if self.steps == 1 else loss / self.steps
        backward_loss.backward()
        self.microbatches += 1
        return self._update() if self.microbatches == self.steps else None

    def flush(self) -> torch.Tensor | None:
        """Update from a final short group, averaging by its actual size."""
        return self._update() if self.microbatches else None

    def _update(self) -> torch.Tensor:
        group_size = self.microbatches
        if not group_size or self._loss_sum is None:
            raise RuntimeError("No accumulated gradients are ready for an update.")

        # Full groups were divided by self.steps during backward, preserving the
        # historical numerical path. Rescale only a short final group so it is
        # averaged by the number of microbatches that actually contributed.
        if group_size != self.steps:
            correction = self.steps / group_size
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        if self.ema_model is not None:
            self.ema_model.step(self.model.parameters())

        mean_loss = self._loss_sum / group_size
        self.microbatches = 0
        self._loss_sum = None
        return mean_loss


def get_scheduler_with_min_lr(
    name: str,
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    min_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    power: float = 1.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Wrapper around diffusers get_scheduler that enforces a minimum learning rate.

    Args:
        base_lr: The starting learning rate (must match optimizer's initial lr).
        min_lr: The target minimum learning rate.
    """
    min_lr_ratio = min_lr / base_lr
    if name == "polynomial":
        from diffusers.optimization import get_polynomial_decay_schedule_with_warmup

        return get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            lr_end=min_lr,  # Native support
            power=power,
            last_epoch=last_epoch,
        )

    if name == "cosine":

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            if current_step > num_training_steps:
                return min_lr_ratio

            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))  # Standard 1->0

            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    if name == "cosine_with_restarts":

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            if current_step > num_training_steps:
                return min_lr_ratio

            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )

            if progress >= 1.0:
                return min_lr_ratio

            return min_lr_ratio + (1.0 - min_lr_ratio) * (
                0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0)))
            )

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    # 4. Fallback for others (Constant, etc. don't need min_lr usually)
    return get_scheduler(
        name,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        power=power,
    )


def build_optimizer(
    model: nn.Module,
    optimizer_config: Mapping[str, Any],
) -> Optimizer:
    """Build the AdamW optimizer used by the training entrypoints."""
    config = dict(optimizer_config)
    optimizer_name = str(config.pop("name", "adamw")).lower()
    if optimizer_name != "adamw":
        raise ValueError(
            f"Unsupported optimizer.name={optimizer_name!r}; expected 'adamw'."
        )
    return torch.optim.AdamW(model.parameters(), **config)


def build_lr_scheduler(
    optimizer: Optimizer,
    scheduler_config: Mapping[str, Any] | None,
    num_training_steps: int,
) -> LRScheduler | None:
    if not scheduler_config or not scheduler_config.get("name"):
        return None
    return get_scheduler_with_min_lr(
        name=scheduler_config["name"],
        optimizer=optimizer,
        base_lr=float(optimizer.param_groups[0]["lr"]),
        min_lr=float(scheduler_config.get("min_lr", 0.0)),
        num_warmup_steps=scheduler_config["num_warmup_steps"],
        num_training_steps=num_training_steps,
        num_cycles=scheduler_config.get("num_cycles", 1),
        power=scheduler_config.get("power", 1.0),
    )


def describe_optimizer(optimizer: Optimizer) -> str:
    return f"Optimizer: {optimizer.__class__.__name__}"
