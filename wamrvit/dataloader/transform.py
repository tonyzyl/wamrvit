import json
import numpy as np

from wamrvit.dataloader.utils import precompute_mean_std

from typing import Any, Callable, Dict, List, Optional, Tuple

class NormalizeArray:
    def __init__(self, norm_json_path: str, field_names: List[str], channel_dim: int = -4, eps: float = 0):
        """
        Args:
            norm_json_path: Path to JSON file containing normalization parameters (mean and std) for each variable.
            field_names: List of variable names corresponding to the channels in the input array. The order should match the order of channels in the input data.
            channel_dim: The index of the channel dimension in the input array.
                         Default -4 assumes (..., C, T, H, W).
            eps: Small constant for numerical stability.
        """

        with open(norm_json_path, "r") as f:
            normalization_param_dict = json.load(f)
            
        self.mean, self.std = precompute_mean_std(
            normalization_param_dict, variable_names=field_names, return_tensor=False
        )
        self.channel_dim = channel_dim
        self.eps = eps

    def _prepare_stats(self, ndim: int):
        # Calculate how many dimensions are to the right of the channel dim
        # e.g., if ndim=5 (B, C, T, H, W) and channel_dim=-4, right_dims = 3
        pos_channel_dim = self.channel_dim if self.channel_dim >= 0 else ndim + self.channel_dim
        right_dims = ndim - pos_channel_dim - 1
        left_dims = pos_channel_dim

        # Reshape to (1, ..., 1, C, 1, ..., 1)
        new_shape = (1,) * left_dims + (-1,) + (1,) * right_dims
        return self.mean.reshape(new_shape), self.std.reshape(new_shape)

    def _check_axis(self, arr: np.ndarray) -> None:
        # Catches caller bugs at the wrong-axis site instead of letting them corrupt downstream metrics.
        pos = self.channel_dim if self.channel_dim >= 0 else arr.ndim + self.channel_dim
        if not 0 <= pos < arr.ndim:
            raise AssertionError(
                f"NormalizeArray.channel_dim={self.channel_dim} out of range for ndim={arr.ndim}"
            )
        if arr.shape[pos] != self.mean.shape[0]:
            raise AssertionError(
                f"NormalizeArray axis mismatch: arr.shape={arr.shape}, channel_dim={self.channel_dim} "
                f"resolves to axis {pos} with length {arr.shape[pos]}, but mean/std has "
                f"{self.mean.shape[0]} channels. Caller is passing the wrong layout."
            )

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        self._check_axis(arr)
        m, s = self._prepare_stats(arr.ndim)
        return (arr - m) / (s + self.eps)

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        self._check_axis(arr)
        m, s = self._prepare_stats(arr.ndim)
        return arr * (s + self.eps) + m


class PLITransform:
    def __init__(self, field_names: List[str], *args, **kwargs):
        # sum energy fields together to get the overall energy field
        self.field_names = field_names
        energy_indices = [i for i, name in enumerate(field_names) if name.startswith("energy_")]

        self.energy_field_start_idx = min(energy_indices)
        self.energy_field_end_idx = max(energy_indices)
        self.energy_slice = slice(self.energy_field_start_idx,
                                self.energy_field_end_idx + 1)

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        # field contains NaN values, need to fill them with 0 before summing
        arr = np.nan_to_num(arr, nan=0.0) # (1, N, C, T, H, W) 
        energy_sum = np.sum(arr[:, :, self.energy_slice, ...], axis=2, keepdims=True) # (1, N, 1, T, H, W)
        # strip out the original energy fields and insert the new energy sum field
        arr = np.concatenate([arr[:, :, :self.energy_field_start_idx, ...],
                            energy_sum,
                            arr[:, :, self.energy_field_end_idx + 1:, ...]], axis=2)
        return arr


class PeriodicRoll:
    """Cyclic roll along a spatial axis, for periodic-BC data augmentation.

    One shift per sample: call ``.reset()`` once at the top of each sample,
    then apply ``__call__`` to as many arrays as needed — they all share the
    same shift. A Seq2SeqMapper with a per-sample ``reset()`` invocation
    keeps input/target/target_src rolled consistently.

    Default ``axis=-2`` targets H in ``(..., C, T, H, W)`` tensors. For TRL
    (H=128 periodic, W=384 along the layer), the correct axis is -2.

    If ``.reset()`` is never called, ``__call__`` returns the array unchanged
    (identity). That's the safe default for val-side Compose chains that
    never receive a reset.
    """

    def __init__(
        self,
        axis: int = -2,
        shift_range: Optional[Tuple[int, int]] = None,
        seed: Optional[int] = None,
    ):
        self.axis = axis
        self.shift_range = shift_range
        self._rng = np.random.default_rng(seed)
        self._shift = 0       # identity until reset() is called
        self._pending = False  # True between reset() and first __call__

    def reset(self) -> None:
        """Queue a fresh random shift for the next sample.

        Shift is drawn lazily on the first ``__call__`` because the axis
        length may only be known at that point (when ``shift_range=None``).
        """
        self._pending = True

    def _draw_shift(self, axis_len: int) -> int:
        if self.shift_range is None:
            lo, hi = 0, axis_len
        else:
            lo, hi = self.shift_range
        if hi <= lo:
            return 0
        return int(self._rng.integers(lo, hi))

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        if self._pending:
            self._shift = self._draw_shift(arr.shape[self.axis])
            self._pending = False
        if self._shift == 0:
            return arr
        return np.roll(arr, self._shift, axis=self.axis)


class ComposeSampleTransform:
    """Chain of transforms applied in order, with a per-sample ``reset()``.

    Sub-transforms that define ``reset()`` (e.g. ``PeriodicRoll``) get
    resynchronized once per sample; stateless ones (e.g. ``NormalizeArray``)
    are unaffected. ``inverse_transform`` chains available inverses in
    reverse order, skipping sub-transforms that don't define one —
    augmentation steps like ``PeriodicRoll`` don't survive round-trip, so
    this is only meaningful for the normalize path.
    """

    def __init__(self, transforms: List[Dict[str, Any]]):
        from wamrvit.utils import instantiate_from_config
        self.transforms: List[Callable] = [
            instantiate_from_config(t) for t in transforms
        ]

    def reset(self) -> None:
        for t in self.transforms:
            if hasattr(t, "reset"):
                t.reset()

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            arr = t(arr)
        return arr

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        for t in reversed(self.transforms):
            if hasattr(t, "inverse_transform") and callable(t.inverse_transform):
                arr = t.inverse_transform(arr)
        return arr


def apply_transform_src(transform: Optional[Callable], arr: np.ndarray) -> np.ndarray:
    """Forward-transform a ``(..., T, C, H, W)`` full-field src tensor.

    Src tensors carry T before C (the loader emits them that way), but
    ``NormalizeArray`` expects ``(..., C, T, H, W)`` via ``channel_dim=-4``.
    Transpose T↔C around the call when the transform is ``NormalizeArray``,
    otherwise pass through (``PLITransform`` already targets axis=2 = C in
    the src layout, and pure passthrough is correct for transforms with no
    layout assumptions). Mirrored by ``inverse_transform_src``.

    Tolerates leading singleton dims (5D ``(B, T, C, H, W)`` and 6D
    ``(1, B, T, C, H, W)`` from Ray Data batching). Other shapes go through
    the transform unchanged — the assertion in ``NormalizeArray._check_axis``
    catches mistakes loudly.
    """
    if transform is None:
        return arr
    if not isinstance(transform, NormalizeArray):
        return transform(arr)
    n_leading = 0
    a = arr
    while a.ndim > 5 and a.shape[0] == 1:
        a = a.squeeze(0)
        n_leading += 1
    if a.ndim != 5:
        return transform(arr)
    a = transform(a.transpose(0, 2, 1, 3, 4)).transpose(0, 2, 1, 3, 4)
    for _ in range(n_leading):
        a = a[np.newaxis]
    return a


def inverse_transform_src(transform: Optional[Callable], arr: np.ndarray) -> np.ndarray:
    """Inverse-transform a ``(..., T, C, H, W)`` full-field src tensor.

    Mirrors :func:`apply_transform_src`. Returns ``arr`` unchanged when the
    transform has no ``inverse_transform`` (e.g. ``PLITransform``).
    """
    if transform is None:
        return arr
    if not (hasattr(transform, "inverse_transform") and callable(transform.inverse_transform)):
        return arr
    if not isinstance(transform, NormalizeArray):
        return transform.inverse_transform(arr)
    n_leading = 0
    a = arr
    while a.ndim > 5 and a.shape[0] == 1:
        a = a.squeeze(0)
        n_leading += 1
    if a.ndim != 5:
        return transform.inverse_transform(arr)
    a = transform.inverse_transform(a.transpose(0, 2, 1, 3, 4)).transpose(0, 2, 1, 3, 4)
    for _ in range(n_leading):
        a = a[np.newaxis]
    return a


class PLIRegularTransform:
    """
    PLI transform for regular-grid tensors with shape (B, C, T, H, W).

    It mirrors `PLITransform` but uses the regular channel axis (axis=1).
    """
    def __init__(self, field_names: List[str], *args, **kwargs):
        # Identify the contiguous block of PLI energy channels once at init time.
        self.field_names = field_names
        energy_indices = [i for i, name in enumerate(field_names) if name.startswith("energy_")]
        if not energy_indices:
            raise ValueError("PLIRegularTransform requires at least one channel named 'energy_*'.")

        self.energy_field_start_idx = min(energy_indices)
        self.energy_field_end_idx = max(energy_indices)
        self.energy_slice = slice(self.energy_field_start_idx, self.energy_field_end_idx + 1)

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        # Replace NaNs before any reduction so channel sums stay finite.
        arr = np.nan_to_num(arr, nan=0.0)  # (B, C, T, H, W)
        # Collapse all energy_* channels into one aggregated energy channel.
        energy_sum = np.sum(arr[:, self.energy_slice, ...], axis=1, keepdims=True)  # (B, 1, T, H, W)
        # Keep channels before/after energy block and insert the summed channel in-place.
        arr = np.concatenate(
            [
                arr[:, :self.energy_field_start_idx, ...],
                energy_sum,
                arr[:, self.energy_field_end_idx + 1:, ...],
            ],
            axis=1,
        )
        return arr
