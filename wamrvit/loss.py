# LpLoss is adapted from NeuralOperator.
# Copyright (c) 2023 NeuralOperator developers
# SPDX-License-Identifier: MIT
# See the top-level LICENSE for the MIT permission notice and disclaimer.
import math
from collections.abc import Callable

import torch
import torch.nn.functional as F

from wamrvit.multi_scale import MultiScaleBucketLayout, UniformGeometry, geometry_key
from wamrvit.quad.quad_utils import scatter_native_to_uniform_torch


class UniformLossMaskScheduler:
    def __init__(
        self,
        min_keep_ratio: float,
        max_step_ratio: float,
        schedule_type: str = "cosine",
        min_keep_patch: int = 100,
    ):
        """
        Generates a uniform random mask for loss calculation, gradually increasing
        the percentage of patches kept over time.

        Args:
            min_keep_ratio: Starting probability of keeping a patch (e.g., 0.15).
            max_step_ratio: The training progress ratio (0 to 1) at which the mask reaches 1.0.
            schedule_type: "linear" or "cosine"
            min_keep_patch: Absolute minimum number of patches to keep in any forward pass.
        """
        self.min_keep_ratio = min_keep_ratio
        self.max_step_ratio = max_step_ratio
        self.schedule_type = schedule_type
        self.min_keep_patch = min_keep_patch

        if not (0.0 <= min_keep_ratio <= 1.0):
            raise ValueError(f"min_keep_ratio should be in [0, 1], got {min_keep_ratio}")
        if not (0.0 < max_step_ratio <= 1.0):
            raise ValueError(f"max_step_ratio should be in (0, 1], got {max_step_ratio}")
        if schedule_type not in ["linear", "cosine"]:
            raise ValueError("schedule_type must be 'linear' or 'cosine'")
        if not (min_keep_patch >= 1):
            raise ValueError(f"min_keep_patch must be at least 1, got {min_keep_patch}")

    def get_ratio(self, current_step: int, total_steps: int) -> float:
        # 1. Calculate how far along we are in the training process
        current_step_ratio = current_step / total_steps

        # 2. Calculate progress specifically relative to our masking schedule, clamping at 1.0
        progress = min(1.0, current_step_ratio / self.max_step_ratio)

        if progress == 1.0:
            return 1.0

        if self.schedule_type == "linear":
            return self.min_keep_ratio + (1.0 - self.min_keep_ratio) * progress
        elif self.schedule_type == "cosine":
            # Cosine annealing: starts slow, accelerates in the middle, slows down near 1.0
            return self.min_keep_ratio + (1.0 - self.min_keep_ratio) * 0.5 * (
                1.0 - math.cos(math.pi * progress)
            )

    def get_mask(
        self, num_patches: int, current_step: int, total_steps: int, device: torch.device
    ) -> torch.Tensor:
        """
        Generates the 1D boolean/float mask tensor for the current step.
        """
        current_ratio = self.get_ratio(current_step, total_steps)

        # Fast path for full grid
        if current_ratio >= 1.0:
            return torch.ones(num_patches, device=device)

        # Create binary mask (1 = keep, 0 = drop)
        mask = (torch.rand(num_patches, device=device) < current_ratio).float()

        # Safety fallback: Ensure at least `min_keep_patch` patches are selected
        current_keep_count = mask.sum().item()

        if current_keep_count < self.min_keep_patch:
            # Prevent requesting more patches than actually exist
            target_count = min(self.min_keep_patch, num_patches)
            shortfall = int(target_count - current_keep_count)

            if shortfall > 0:
                # Find indices that are currently 0 (dropped)
                zero_indices = (mask == 0).nonzero(as_tuple=True)[0]
                # Randomly select exactly `shortfall` number of these indices and flip them to 1
                chosen_zeros = zero_indices[
                    torch.randperm(len(zero_indices), device=device)[:shortfall]
                ]
                mask[chosen_zeros] = 1.0

        return mask


class LpLoss:
    # loss function with rel/abs Lp loss, modified from neuralop:
    # https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/losses/data_losses.py
    """
    LpLoss: Lp loss function, return the relative loss by default
    Args:
        d: int, start dimension of the field. E,g., for shape like (b, c, h, w), d=2 (default 1)
        p: int, p in Lp norm, default 2
        reduce_dims: int or list of int, dimensions to reduce
        reductions: str or list of str, 'sum' or 'mean'
        eps: float, clamping the denominator in relative loss.
        method: 'rel' or 'abs', return relative or absolute loss

    Call: (y_pred, y, weight=None, mask=None)
    """

    def __init__(self, d=1, p=2, reduce_dims=0, reductions="mean", method="rel", eps=1e-6):
        super().__init__()

        self.d = d
        self.p = p
        self.eps = eps
        if method == "rel":
            self.method = self.rel
        elif method == "abs":
            self.method = self.abs
        else:
            raise ValueError("method should be 'rel' or 'abs'")

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims

        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                if not (reductions == "sum" or reductions == "mean"):
                    raise ValueError(f"reductions must be 'sum' or 'mean'; got {reductions!r}.")
                self.reductions = [reductions] * len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    if not (reductions[j] == "sum" or reductions[j] == "mean"):
                        raise ValueError(
                            f"reductions must be 'sum' or 'mean'; got {reductions[j]!r}."
                        )
                self.reductions = reductions

    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == "sum":
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)
        return x

    def abs(self, x, y, weight: torch.Tensor | None = None):
        if self.p == 2:
            # Numerically stable L2 norm: sqrt(sum(x^2) + eps)
            if weight is None:
                diff_sq = torch.sum(
                    (torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d))
                    ** 2,
                    dim=-1,
                    keepdim=False,
                )
            else:
                diff_sq = torch.sum(
                    (torch.flatten(weight * (x - y), start_dim=-self.d)) ** 2,
                    dim=-1,
                    keepdim=False,
                )
            return torch.sqrt(diff_sq + self.eps)
        else:
            # Fallback for other p-norms (e.g., p=1)
            if weight is None:
                diff = torch.norm(
                    torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d),
                    p=self.p,
                    dim=-1,
                    keepdim=False,
                )
            else:
                diff = torch.norm(
                    torch.flatten(weight * (x - y), start_dim=-self.d),
                    p=self.p,
                    dim=-1,
                    keepdim=False,
                )
            return diff

    def rel(self, x, y, weight: torch.Tensor | None = None):
        if self.p == 2:
            # Numerically stable L2 norm: sqrt(sum(x^2) + eps)
            if weight is None:
                diff_sq = torch.sum(
                    (torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d))
                    ** 2,
                    dim=-1,
                    keepdim=False,
                )
                ynorm_sq = torch.sum(
                    torch.flatten(y, start_dim=-self.d) ** 2, dim=-1, keepdim=False
                )
            else:
                diff_sq = torch.sum(
                    (torch.flatten(weight * (x - y), start_dim=-self.d)) ** 2,
                    dim=-1,
                    keepdim=False,
                )
                ynorm_sq = torch.sum(
                    (torch.flatten(weight * y, start_dim=-self.d)) ** 2, dim=-1, keepdim=False
                )

            diff = torch.sqrt(diff_sq + self.eps)
            ynorm = torch.sqrt(ynorm_sq + self.eps)
        else:
            # Fallback for other p-norms (e.g., p=1)
            if weight is None:
                diff = torch.norm(
                    torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d),
                    p=self.p,
                    dim=-1,
                    keepdim=False,
                )
                ynorm = torch.norm(
                    torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False
                )
            else:
                diff = torch.norm(
                    torch.flatten(weight * (x - y), start_dim=-self.d),
                    p=self.p,
                    dim=-1,
                    keepdim=False,
                )
                ynorm = torch.norm(
                    torch.flatten(weight * y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False
                )

        ynorm = torch.clamp(ynorm, min=self.eps)  # Extra safety against division by zero
        diff = diff / ynorm  # (B, C)
        return diff

    def __call__(
        self, y_pred, y, weight: torch.Tensor | None = None, mask: torch.Tensor | None = None
    ):
        y = y.float()
        y_pred = y_pred.float()
        diff = self.method(y_pred, y, weight=weight)

        if mask is not None:
            # Broadcast mask (N,) to match diff shape (N, C, ...)
            mask_view_shape = [mask.shape[0]] + [1] * (diff.dim() - 1)
            mask_view = mask.view(*mask_view_shape)
            diff = diff * mask_view

        if self.reduce_dims is not None:
            # If masked, a standard mean over dim=0 divides by N instead of mask.sum()
            # We override the reduction logic for dim=0 if a mask is present
            if mask is not None and 0 in self.reduce_dims:
                for j in range(len(self.reduce_dims)):
                    dim = self.reduce_dims[j]
                    if self.reductions[j] == "sum":
                        diff = torch.sum(diff, dim=dim, keepdim=True)
                    else:  # mean
                        if dim == 0:
                            # Use sum / unmasked_count
                            diff = torch.sum(diff, dim=dim, keepdim=True) / torch.clamp(
                                mask.sum(), min=1e-8
                            )
                        else:
                            diff = torch.mean(diff, dim=dim, keepdim=True)
            else:
                diff = self.reduce_all(diff)

            diff = diff.squeeze()

        return diff

    @torch.no_grad()
    def get_loss_per_var(
        self,
        y_pred,
        y,
        weight: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        """Assuming input of order [atm_vars, sur_vars]"""
        diff = self.method(y_pred, y, weight=weight)
        y = y.float()
        y_pred = y_pred.float()

        if mask is not None:
            mask_view_shape = [mask.shape[0]] + [1] * (diff.dim() - 1)
            mask_view = mask.view(*mask_view_shape)
            diff = diff * mask_view

        batch_mean = torch.empty(
            y_pred.shape[1],
            device=y_pred.device,
            dtype=y_pred.dtype,
        )
        for i in range(0, y_pred.shape[1]):
            if mask is not None:
                # Divide by unmasked count for the mean
                batch_mean[i] = diff[:, i].sum() / torch.clamp(mask.sum(), min=1e-8)
            else:
                batch_mean[i] = diff[:, i].mean()

            if batch_mean[i] < 0:
                print(f"Negative mean for var {i}: {batch_mean[i]}")

        return batch_mean


class MSELoss:
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, y_pred, y, mask: torch.Tensor | None = None):
        # Always compute unreduced loss first so we can apply the mask
        diff = F.mse_loss(y_pred.float(), y.float(), reduction="none")

        if mask is not None:
            # Broadcast mask (N,) to match diff shape (N, C, ...)
            mask_view_shape = [mask.shape[0]] + [1] * (diff.dim() - 1)
            mask_view = mask.view(*mask_view_shape)
            diff = diff * mask_view

            if self.reduction == "mean":
                # Divide by (number of active patches * elements per patch)
                elements_per_patch = diff.numel() // diff.shape[0]
                return diff.sum() / (mask.sum() * elements_per_patch + 1e-8)
            elif self.reduction == "sum":
                return diff.sum()
            else:
                return diff
        else:
            # Standard PyTorch behavior if no mask is provided
            if self.reduction == "mean":
                return diff.mean()
            elif self.reduction == "sum":
                return diff.sum()
            else:
                return diff

    @torch.no_grad()
    def get_loss_per_var(
        self,
        y_pred,
        y,
        weight: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        """Assuming input of order [atm_vars, sur_vars]"""
        if weight is None:
            diff = (y_pred - y) ** 2
        else:
            diff = weight * ((y_pred - y) ** 2)

        if mask is not None:
            mask_view_shape = [mask.shape[0]] + [1] * (diff.dim() - 1)
            mask_view = mask.view(*mask_view_shape)
            diff = diff * mask_view

        batch_mean = torch.empty(
            y_pred.shape[1],
            device=y_pred.device,
            dtype=y_pred.dtype,
        )

        for i in range(0, y_pred.shape[1]):
            if mask is not None:
                # diff[:, i] has shape (N, T, H, W) or (N, H, W) depending on sequence logic
                # Calculate elements per patch for this specific variable
                elements_per_var_per_patch = diff[:, i].numel() // diff.shape[0]

                # Mean over only the active patches
                batch_mean[i] = diff[:, i].sum() / (mask.sum() * elements_per_var_per_patch + 1e-8)
            else:
                batch_mean[i] = diff[:, i].mean()

            if batch_mean[i] < 0:
                print(f"Negative mean for var {i}: {batch_mean[i]}")

        return batch_mean


def multi_scale_loss(
    loss_fn: Callable,
    pred_by_level: dict[int, torch.Tensor],
    target_by_level: dict[int, torch.Tensor],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-scale loss analogous to a single-tensor call in uniform mode.

    Two reduction paths, auto-selected by inspecting loss_fn:

    **Norm-based losses** (LpLoss — detected by having a `.method` attribute):
    Compute per-leaf unreduced norms at each level via ``loss_fn.method``,
    concatenate across levels, then take a single mean. Note: under the
    sum-based L_p norm (``method='abs'``), a level-ℓ leaf's norm magnitude
    scales with ``block_size^(1/p)``; larger leaves therefore contribute
    larger absolute norms for the same per-pixel error.

    **Element-wise mean losses** (MSELoss, etc.):
    Pixel-count-weighted sum, equivalent to ``loss_fn(cat_all_pixels)``.

    ``mask`` (optional per-bucket mask) is not yet supported.
    Used by the training entrypoints when ``train.loss_mode='per_leaf'``.
    """
    if mask is not None:
        raise NotImplementedError("Per-bucket masking is not yet wired for multi-scale loss.")

    # --- Norm-based path (LpLoss) -------------------------------------------
    if hasattr(loss_fn, "method") and hasattr(loss_fn, "d"):
        all_leaf_losses: list[torch.Tensor] = []
        for lvl, pred in pred_by_level.items():
            tgt = target_by_level[lvl]
            if pred.shape[0] == 0:
                continue
            # .method returns per-leaf, per-channel values: (N_l, C)
            all_leaf_losses.append(loss_fn.method(pred, tgt))
        if not all_leaf_losses:
            any_pred = next(iter(pred_by_level.values()))
            return any_pred.sum() * 0.0
        return torch.cat(all_leaf_losses, dim=0).mean()

    # --- Element-wise mean path (MSELoss, etc.) ------------------------------
    total_num = 0.0
    weighted_sum = None
    for lvl, pred in pred_by_level.items():
        tgt = target_by_level[lvl]
        if pred.shape[0] == 0:
            continue
        count = float(pred.numel())
        lvl_loss = loss_fn(pred, tgt)
        term = lvl_loss * count
        weighted_sum = term if weighted_sum is None else weighted_sum + term
        total_num += count
    if weighted_sum is None or total_num == 0:
        any_pred = next(iter(pred_by_level.values()))
        return any_pred.sum() * 0.0
    return weighted_sum / total_num


def full_field_loss(
    loss_fn: Callable,
    pred_by_level: dict[int, torch.Tensor],
    target_by_level: dict[int, torch.Tensor],
    leaf_to_bucket: torch.Tensor,
    centers: torch.Tensor,
    geom: UniformGeometry,
    *,
    bucket_layout: MultiScaleBucketLayout | None = None,
) -> torch.Tensor:
    """Multi-scale full-field objective with a scatter-free absolute L2 fast path.

    Used by the training entrypoints when ``train.loss_mode='full_field'``.
    The fast path requires a batch-owned, coverage-validated layout and the exact
    built-in LpLoss(d=3, p=2, method='abs'). Other losses, subclasses, and callers
    without a validated layout retain scatter followed by the supplied loss.

    Works transparently with ``pred_mode="residual"`` because ``target_by_level``
    is already the residual target (``multi_scale.residual_targets`` subtracts
    the last input frame per level *before* this function is called).

    The optimized reduction sums squared errors globally *before* sqrt, adds
    epsilon once per channel, and retains LpLoss's configured reductions. The
    generic fallback remains differentiable through its scatter assignments.
    """
    kwargs = geom.as_kwargs()
    if bucket_layout is not None:
        bucket_layout.validate(
            leaf_to_bucket,
            [pred_by_level[level].shape[0] for level in range(len(bucket_layout.bucket_sizes))],
            centers=centers if bucket_layout.centers is not None else None,
        )
    if (
        bucket_layout is not None
        and bucket_layout.field_geometry == geometry_key(kwargs)
        and type(loss_fn) is LpLoss
        and loss_fn.d == 3
        and loss_fn.p == 2
        and loss_fn.method == loss_fn.abs
    ):
        if (
            pred_by_level.keys() != target_by_level.keys()
            or set(pred_by_level) != set(range(len(bucket_layout.bucket_sizes)))
        ):
            raise ValueError("Native full-field loss requires matching level buckets.")
        squared_errors = None
        channel_time_shape = None
        for level, pred in pred_by_level.items():
            target = target_by_level[level]
            spatial_shape = (
                geom.base_patch_h << (geom.max_level_idx - level),
                geom.base_patch_w << (geom.max_level_idx - level),
            )
            if pred.ndim != 5 or pred.shape != target.shape or pred.shape[-2:] != spatial_shape:
                raise ValueError(
                    "Native prediction/target shapes do not match full-field geometry."
                )
            if channel_time_shape is None:
                channel_time_shape = pred.shape[1:3]
            elif channel_time_shape != pred.shape[1:3]:
                raise ValueError(
                    "Native full-field channels and output times must agree across levels."
                )
            if pred.shape[0] == 0:
                continue
            term = (pred.float() - target.float()).square().sum(dim=(0, 2, 3, 4))
            squared_errors = term if squared_errors is None else squared_errors + term
        # Coverage validation guarantees at least one leaf for a nonempty domain.
        diff = torch.sqrt(squared_errors.unsqueeze(0) + loss_fn.eps)  # (1, C)
        return loss_fn.reduce_all(diff).squeeze() if loss_fn.reduce_dims is not None else diff
    pred_u = scatter_native_to_uniform_torch(pred_by_level, leaf_to_bucket, centers, **kwargs)
    tgt_u = scatter_native_to_uniform_torch(target_by_level, leaf_to_bucket, centers, **kwargs)
    return loss_fn(pred_u, tgt_u)
