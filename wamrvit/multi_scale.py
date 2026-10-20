"""Batch-owned layouts, geometry, and payload preparation for multi-scale models.

Used by encoders, the backbone, losses, training, and rollout. This module has no
model, loss, or training-loop dependencies. The underlying storage contract stays
``value_storage='native'``: per-level ``input_level_{l}`` / ``target_level_{l}``
columns and a canonical ``leaf_to_bucket`` mapping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

_GEOMETRY_KEYS = (
    "xmin",
    "xmax",
    "ymin",
    "ymax",
    "tile_h",
    "tile_w",
    "max_level_idx",
    "base_patch_h",
    "base_patch_w",
)


def geometry_key(geometry: Mapping[str, int | float]) -> tuple:
    return tuple(geometry[key] for key in _GEOMETRY_KEYS)


def _version(tensor: torch.Tensor | None) -> int | None:
    # Inference tensors have no version counter; their topology must remain read-only.
    return None if tensor is None or tensor.is_inference() else tensor._version


def _validate_coverage(mapping, centers, geometry):
    """Check rectangular quadtree coverage on the base-patch lattice, not pixels."""
    xmin, xmax, ymin, ymax, tile_h, tile_w, maximum, ph, pw = geometry_key(geometry)
    if maximum < 0 or ph < 1 or pw < 1 or tile_h <= 0 or tile_w <= 0:
        raise ValueError("Invalid native full-field geometry.")
    factor = 1 << maximum
    nx, ny = (xmax - xmin) / tile_w, (ymax - ymin) / tile_h
    if nx <= 0 or ny <= 0 or not np.isclose(nx, round(nx)) or not np.isclose(ny, round(ny)):
        raise ValueError("Native full-field domains must contain whole root tiles.")
    width, height = round(nx) * factor, round(ny) * factor
    coordinates = centers.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(coordinates[:, :2]).all():
        raise ValueError("Native centers must be finite.")
    levels = mapping[:, 0]
    extent = 1 << (maximum - levels)
    # Match the native scatter's rounded pixel starts. Valid native leaves align
    # to base patches; validating on that lattice avoids a dense C*T*H*W field.
    px = np.rint((coordinates[:, 0] * (xmax - xmin) / tile_w * factor - extent / 2) * pw)
    py = np.rint((coordinates[:, 1] * (ymax - ymin) / tile_h * factor - extent / 2) * ph)
    if np.any(px % pw) or np.any(py % ph):
        raise ValueError("Native leaf starts must align to base patches.")
    x, y = (px / pw).astype(np.int64), (py / ph).astype(np.int64)
    if np.any(x < 0) or np.any(y < 0) or np.any(x + extent > width) or np.any(y + extent > height):
        raise ValueError("Native leaves extend outside the full-field domain.")
    delta = np.zeros((height + 1, width + 1), dtype=np.int32)
    np.add.at(delta, (y, x), 1)
    np.add.at(delta, (y + extent, x), -1)
    np.add.at(delta, (y, x + extent), -1)
    np.add.at(delta, (y + extent, x + extent), 1)
    coverage = delta[:-1, :-1].cumsum(0).cumsum(1)
    if not np.all(coverage == 1):
        raise ValueError("Native full-field leaves must cover the domain exactly once.")


@dataclass(frozen=True, eq=False)
class MultiScaleBucketLayout:
    """Read-only metadata for one multi-scale sample; construct with ``from_mapping``.

    ``indices[level][bucket_position]`` is the canonical leaf index. The same
    permutation serves encoder scatter, decoder gather, and per-leaf conditions.
    Carry this object with its batch, never cache by shapes or store on a model.
    Rebuild after topology changes. Inference-mode tensors must not be mutated;
    ordinary tensors additionally have their version counters checked.

    ``field_geometry`` is populated only after CPU coverage validation. It permits
    the global L2 fast path; a permutation alone does not certify coverage.
    No parameters, payload values, or pixel-index arrays are stored here.
    """

    leaf_to_bucket: torch.Tensor
    packed_indices: torch.Tensor
    bucket_sizes: tuple[int, ...]
    centers: torch.Tensor | None = None
    field_geometry: tuple | None = None
    indices: tuple[torch.Tensor, ...] = field(init=False, repr=False)
    _versions: tuple = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "indices", self.packed_indices.split(self.bucket_sizes))
        object.__setattr__(
            self,
            "_versions",
            tuple(_version(t) for t in (self.leaf_to_bucket, self.packed_indices, self.centers)),
        )

    @classmethod
    def from_mapping(
        cls,
        leaf_to_bucket: torch.Tensor,
        bucket_sizes: Sequence[int],
        *,
        centers: torch.Tensor | None = None,
        geometry: Mapping[str, int | float] | None = None,
    ) -> MultiScaleBucketLayout:
        """Validate and invert on CPU before H2D; CUDA inputs are a convenience fallback."""
        sizes = tuple(int(n) for n in bucket_sizes)
        if not sizes or any(n < 0 for n in sizes):
            raise ValueError("Native bucket sizes must be nonnegative and include every level.")
        mapping = leaf_to_bucket.detach().cpu().numpy()
        if mapping.ndim != 2 or mapping.shape != (sum(sizes), 2):
            raise ValueError("leaf_to_bucket must have shape (sum(bucket_sizes), 2).")
        if mapping.dtype.kind not in "iu":
            raise ValueError("Native level and bucket positions must be integers.")
        if centers is not None and (
            centers.ndim != 2 or centers.shape[0] != len(mapping) or centers.shape[1] < 2
        ):
            raise ValueError("Native centers must have one coordinate row per leaf.")
        mapping = mapping.astype(np.int64, copy=False)
        levels, positions = mapping[:, 0], mapping[:, 1]
        if np.any(levels < 0) or np.any(levels >= len(sizes)):
            raise ValueError("Native mapping contains an invalid level.")
        if np.any(positions < 0) or np.any(positions >= np.asarray(sizes)[levels]):
            raise ValueError("Native mapping contains an invalid bucket position.")
        offsets = np.cumsum((0, *sizes[:-1]))
        packed_positions = offsets[levels] + positions
        if not np.all(np.bincount(packed_positions, minlength=sum(sizes)) == 1):
            raise ValueError("Native bucket positions must be unique and complete.")
        inverse = np.empty(sum(sizes), dtype=np.int64)
        inverse[packed_positions] = np.arange(sum(sizes))
        if geometry is not None:
            if centers is None or geometry["max_level_idx"] + 1 != len(sizes):
                raise ValueError("Full-field layout requires centers and matching geometry levels.")
            _validate_coverage(mapping, centers, geometry)
        packed = torch.from_numpy(inverse).to(leaf_to_bucket.device)
        return cls(
            leaf_to_bucket,
            packed,
            sizes,
            centers,
            geometry_key(geometry) if geometry is not None else None,
        )

    def validate(
        self,
        leaf_to_bucket: torch.Tensor,
        bucket_sizes: Sequence[int],
        *,
        centers: torch.Tensor | None = None,
    ) -> None:
        """Cheap ownership/shape checks; no device scalar reads in the forward path."""
        if self.leaf_to_bucket is not leaf_to_bucket or self.bucket_sizes != tuple(bucket_sizes):
            raise ValueError("Native layout belongs to a different mapping or bucket shape.")
        if centers is not None and self.centers is not centers:
            raise ValueError("Native layout belongs to different centers.")
        versions = tuple(
            _version(t) for t in (self.leaf_to_bucket, self.packed_indices, self.centers)
        )
        if versions != self._versions:
            raise ValueError("Native topology metadata was mutated; rebuild the bucket layout.")

    def to(self, device: torch.device, *, non_blocking: bool = False) -> MultiScaleBucketLayout:
        """Transfer one packed index vector; keep layout and canonical tensors paired."""
        self.validate(self.leaf_to_bucket, self.bucket_sizes)
        return MultiScaleBucketLayout(
            self.leaf_to_bucket.to(device, non_blocking=non_blocking),
            self.packed_indices.to(device, non_blocking=non_blocking),
            self.bucket_sizes,
            self.centers.to(device, non_blocking=non_blocking)
            if self.centers is not None
            else None,
            self.field_geometry,
        )


@dataclass(frozen=True)
class UniformGeometry:
    """Finest-grid geometry for ``wamrvit.loss.full_field_loss`` and native scatter.

    All fields are dataset-level constants and should be constructed once per
    training run. Use ``from_batch`` to extract them from the domain-scalar
    columns that ``Seq2SeqMapper`` already emits.
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
    ) -> UniformGeometry:
        """Read domain scalars emitted by ``Seq2SeqMapper`` as length-1 arrays."""

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


def _multi_scale_batch_tensor(arr):
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        if arr.shape != (1,):
            raise ValueError("Native batches require one sample per worker.")
        return torch.from_numpy(arr[0])
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)
    if not isinstance(arr, torch.Tensor):
        raise TypeError(f"Unsupported batch entry type {type(arr)}")
    if arr.shape[0] != 1:
        raise ValueError("Native batches require one sample per worker.")
    return arr.squeeze(0)


def unpack_multi_scale_batch(
    batch: dict[str, np.ndarray | torch.Tensor],
    num_levels: int,
    device: torch.device,
    non_blocking: bool = False,
):
    """Pull per-level input/target columns out of a Ray batch dict for rollout.

    Accepts three encodings of the leading batch=1 dimension:
      - torch.Tensor of shape (1, N_l, ...) — Ray Train's torch DataLoader.
      - np.ndarray (numeric dtype) of shape (1, N_l, ...) — fixed-shape rows.
      - np.ndarray (dtype=object) of shape (1,) wrapping the inner ndarray of
        shape (N_l, ...) — Ray Data's storage for variable-shape per-row arrays
        (the typical case for multi-scale rollout, where N_l varies
        across samples).

    Returns:
        values_by_level: {lvl: (N_l, C, T_in, H_l, W_l)}.
        targets_by_level: {lvl: (N_l, C, T_out, H_l, W_l)}.
        leaf_to_bucket:   (N, 2) long.
        centers:          (N, 3) float.
    """

    def _strip_batch(arr):
        return _multi_scale_batch_tensor(arr).to(device, non_blocking=non_blocking)

    values_by_level: dict[int, torch.Tensor] = {}
    targets_by_level: dict[int, torch.Tensor] = {}
    for lvl in range(num_levels):
        values_by_level[lvl] = _strip_batch(batch[f"input_level_{lvl}"])
        targets_by_level[lvl] = _strip_batch(batch[f"target_level_{lvl}"])

    leaf_to_bucket = _strip_batch(batch["leaf_to_bucket"]).long()
    centers = _strip_batch(batch["centers"])

    return values_by_level, targets_by_level, leaf_to_bucket, centers


def prepare_multi_scale_batch(
    batch: dict[str, np.ndarray | torch.Tensor],
    num_levels: int,
    device: torch.device,
    non_blocking: bool = False,
    *,
    geometry: UniformGeometry | None = None,
):
    """Prepare a batch-owned inverse permutation before H2D, reused for all PF calls.

    Used by training and validation. Returns input buckets, target buckets, and
    a MultiScaleBucketLayout containing the paired mapping/centers. Rollout's
    ``unpack_multi_scale_batch`` keeps its four-tensor contract. Geometry opts
    into CPU full-coverage validation.
    """
    inputs = {
        level: _multi_scale_batch_tensor(batch[f"input_level_{level}"])
        for level in range(num_levels)
    }
    targets = {
        level: _multi_scale_batch_tensor(batch[f"target_level_{level}"])
        for level in range(num_levels)
    }
    mapping = _multi_scale_batch_tensor(batch["leaf_to_bucket"])
    centers = _multi_scale_batch_tensor(batch["centers"])
    for level in range(num_levels):
        x, y = inputs[level], targets[level]
        if x.ndim != 5 or y.ndim != 5 or x.shape[:2] != y.shape[:2] or x.shape[-2:] != y.shape[-2:]:
            raise ValueError(
                "Native input/target buckets must agree in leaves, channels, and space."
            )
        if geometry is not None:
            expected = (
                geometry.base_patch_h << (num_levels - 1 - level),
                geometry.base_patch_w << (num_levels - 1 - level),
            )
            if x.shape[-2:] != expected:
                raise ValueError("Native bucket spatial shape does not match full-field geometry.")
    layout = MultiScaleBucketLayout.from_mapping(
        mapping, [inputs[level].shape[0] for level in range(num_levels)],
        centers=centers, geometry=geometry.as_kwargs() if geometry is not None else None,
    ).to(device, non_blocking=non_blocking)
    return (
        {level: x.to(device, non_blocking=non_blocking) for level, x in inputs.items()},
        {level: y.to(device, non_blocking=non_blocking) for level, y in targets.items()},
        layout,
    )


def residual_targets(
    targets_by_level: dict[int, torch.Tensor],
    inputs_by_level: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    """pred_mode='residual': return targets - last_input_frame per level."""
    return {
        lvl: tgt - inputs_by_level[lvl][:, :, -1:, :, :] for lvl, tgt in targets_by_level.items()
    }
