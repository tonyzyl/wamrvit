"""Helpers for training/inference with value_storage='native' (multi-scale patchify).

Input batches from Seq2SeqMapper._assemble_native_adaptive_result contain
per-level columns `input_level_{l}` and `target_level_{l}`, plus `leaf_to_bucket`.
This module centralises:
  - unpacking those columns into dicts of tensors on device,
  - computing a multi-scale loss that is analogous to a single-tensor call in
    uniform mode (each leaf contributes one value to the final mean, regardless
    of its native spatial size),
  - a full-field loss that scatters per-level predictions onto the uniform
    finest-resolution grid and hands (pred, target) to the eval-metric-aligned
    loss_fn — training/eval objective alignment.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from wamrvit.quad.quad_utils import scatter_native_to_uniform_torch


def unpack_native_batch(
    batch: dict[str, np.ndarray | torch.Tensor],
    num_levels: int,
    device: torch.device,
    non_blocking: bool = False,
):
    """Pull per-level input/target columns out of a Ray batch dict.

    Accepts three encodings of the leading batch=1 dimension:
      - torch.Tensor of shape (1, N_l, ...) — Ray Train's torch DataLoader.
      - np.ndarray (numeric dtype) of shape (1, N_l, ...) — fixed-shape rows.
      - np.ndarray (dtype=object) of shape (1,) wrapping the inner ndarray of
        shape (N_l, ...) — Ray Data's storage for variable-shape per-row arrays
        (the typical case for native multi-scale rollout, where N_l varies
        across samples).

    Returns:
        values_by_level: {lvl: (N_l, C, T_in, H_l, W_l)}.
        targets_by_level: {lvl: (N_l, C, T_out, H_l, W_l)}.
        leaf_to_bucket:   (N, 2) long.
        centers:          (N, 3) float.
    """

    def _strip_batch(arr):
        # Strip the leading batch=1 dim regardless of how it's encoded.
        if isinstance(arr, torch.Tensor):
            return arr.squeeze(0).to(device, non_blocking=non_blocking)
        if isinstance(arr, np.ndarray):
            if arr.dtype == object:
                # Object wrap: arr[0] is already (N_l, ...).
                return torch.from_numpy(arr[0]).to(device, non_blocking=non_blocking)
            return torch.from_numpy(arr).squeeze(0).to(device, non_blocking=non_blocking)
        raise TypeError(f"Unsupported batch entry type {type(arr)}")

    values_by_level: dict[int, torch.Tensor] = {}
    targets_by_level: dict[int, torch.Tensor] = {}
    for lvl in range(num_levels):
        values_by_level[lvl] = _strip_batch(batch[f"input_level_{lvl}"])
        targets_by_level[lvl] = _strip_batch(batch[f"target_level_{lvl}"])

    leaf_to_bucket = _strip_batch(batch["leaf_to_bucket"]).long()
    centers = _strip_batch(batch["centers"])

    return values_by_level, targets_by_level, leaf_to_bucket, centers


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
    scales with ``block_size^(1/p)`` — see docs/multi_scale_patchify.md for
    the per-level budget-share analysis.

    **Element-wise mean losses** (MSELoss, etc.):
    Pixel-count-weighted sum, equivalent to ``loss_fn(cat_all_pixels)``.

    ``mask`` (optional per-bucket mask) is not yet supported.
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


def residual_targets(
    targets_by_level: dict[int, torch.Tensor],
    inputs_by_level: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    """pred_mode='residual': return targets - last_input_frame per level."""
    return {
        lvl: tgt - inputs_by_level[lvl][:, :, -1:, :, :] for lvl, tgt in targets_by_level.items()
    }


@dataclass(frozen=True)
class UniformGeometry:
    """Scatter geometry for :func:`full_field_loss` / :func:`scatter_native_to_uniform_torch`.

    All fields are dataset-level constants and should be constructed once per
    training run. Use :func:`UniformGeometry.from_batch` to extract them from
    the domain-scalar columns that ``Seq2SeqMapper`` already emits.
    """

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    tile_h: float
    tile_w: float
    max_level_idx: int
    base_patch_h: int
    base_patch_w: int

    def as_kwargs(self) -> dict[str, int | float]:
        return {
            "xmin": self.xmin,
            "xmax": self.xmax,
            "ymin": self.ymin,
            "ymax": self.ymax,
            "tile_h": self.tile_h,
            "tile_w": self.tile_w,
            "max_level_idx": self.max_level_idx,
            "base_patch_h": self.base_patch_h,
            "base_patch_w": self.base_patch_w,
        }

    @staticmethod
    def from_batch(
        batch: dict[str, np.ndarray | torch.Tensor],
        base_patch_h: int,
        base_patch_w: int,
    ) -> "UniformGeometry":
        """Read domain scalars from a Ray-Data batch dict.

        Seq2SeqMapper emits ``xmin/xmax/ymin/ymax/tile_width/tile_height/max_level_idx``
        as length-1 arrays per batch (see trajectory_loader.py:246-252).
        """

        def _scalar(name, cast):
            v = batch[name]
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            return cast(np.asarray(v).reshape(-1)[0])

        return UniformGeometry(
            xmin=_scalar("xmin", float),
            xmax=_scalar("xmax", float),
            ymin=_scalar("ymin", float),
            ymax=_scalar("ymax", float),
            tile_h=_scalar("tile_height", float),
            tile_w=_scalar("tile_width", float),
            max_level_idx=_scalar("max_level_idx", int),
            base_patch_h=int(base_patch_h),
            base_patch_w=int(base_patch_w),
        )


def full_field_loss(
    loss_fn: Callable,
    pred_by_level: dict[int, torch.Tensor],
    target_by_level: dict[int, torch.Tensor],
    leaf_to_bucket: torch.Tensor,
    centers: torch.Tensor,
    geom: UniformGeometry,
) -> torch.Tensor:
    """Scatter per-level (pred, target) to the uniform finest-resolution grid and
    apply ``loss_fn`` — same call-shape as the regular-finest model's loss.

    Works transparently with ``pred_mode="residual"`` because ``target_by_level``
    is already the residual target (``residual_targets`` subtracts the last
    input frame per level *before* this function is called).

    Gradients flow through the scatter's advanced-index assignment.
    """
    kwargs = geom.as_kwargs()
    pred_u = scatter_native_to_uniform_torch(pred_by_level, leaf_to_bucket, centers, **kwargs)
    tgt_u = scatter_native_to_uniform_torch(target_by_level, leaf_to_bucket, centers, **kwargs)
    return loss_fn(pred_u, tgt_u)
