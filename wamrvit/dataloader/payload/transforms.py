import json
import warnings
from pathlib import Path

import numpy as np
import torch

from typing import Any, Callable, Dict, List, Optional, Tuple


def precompute_mean_std(normalization_param_dict: Dict, variable_names: Optional[List[str]]=None, return_tensor:bool = True):
    """
    Precompute the mean and std tensors for the given variables. variable_names shall match the order of the input.
    Return shape: (C,)
    """
    mean_list = []
    std_list = []

    if variable_names is None:
        variable_names = list(normalization_param_dict.keys())
        warnings.warn(f"Variable names not provided. Using all variables from normalization dict: {variable_names}")

    for var_name in variable_names:
        if var_name in normalization_param_dict:
            norm_params = normalization_param_dict[var_name]

            if isinstance(norm_params["mean"], dict):
                # If the variable has level-based mean and std, create one tensor per level
                for level in norm_params["mean"].keys():
                    mean_list.append(norm_params["mean"][level])
                    std_list.append(norm_params["std"][level])
            else:
                # For regular variables, add mean and std directly
                mean_list.append(norm_params["mean"])
                std_list.append(norm_params["std"])
        else:
            # If no normalization info is found, use 0 mean and 1 std (no normalization)
            raise ValueError(
                f"No normalization parameters found for variable {var_name}."
            )
            # mean_list.append(0)
            # std_list.append(1)
            # Warning(f"No normalization parameters found for variable {var_name}. Using 0 mean and 1 std.")

    # Convert lists to PyTorch tensors
    if return_tensor:
        return torch.tensor(mean_list, dtype=torch.float32), torch.tensor(std_list, dtype=torch.float32)
    else: # np.arrays
        return np.array(mean_list, dtype=np.float32), np.array(std_list, dtype=np.float32)


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

        norm_path = Path(norm_json_path)
        if not norm_path.is_absolute() and not norm_path.is_file():
            norm_path = Path(__file__).resolve().parents[3] / norm_path
        with norm_path.open() as f:
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
        result = np.empty(arr.shape, dtype=np.result_type(arr, m, s), order="C")
        np.subtract(arr, m, out=result)
        np.divide(result, s + self.eps, out=result)
        return result

    def transform_inplace(self, arr: np.ndarray) -> np.ndarray:
        """Normalize a writable array without allocating a second full payload."""
        self._check_axis(arr)
        m, s = self._prepare_stats(arr.ndim)
        expected_dtype = np.dtype(np.result_type(arr, m, s))
        if arr.dtype != expected_dtype:
            raise ValueError(
                f"In-place normalization requires dtype {expected_dtype}; "
                f"got {arr.dtype}."
            )
        if not arr.flags.writeable:
            raise ValueError("In-place normalization requires a writable array.")
        np.subtract(arr, m, out=arr)
        np.divide(arr, s + self.eps, out=arr)
        return arr

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        self._check_axis(arr)
        m, s = self._prepare_stats(arr.ndim)
        result = np.empty(arr.shape, dtype=np.result_type(arr, m, s), order="C")
        np.multiply(arr, s + self.eps, out=result)
        np.add(result, m, out=result)
        return result


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


class PLINormalizedTransform:
    """Opt-in PLI aggregation, affine scalar scaling and asinh-energy scaling.

    Configured by: transform.target; old PLITransform configs remain unchanged.
    Calls accept (..., raw_C, T, H, W). normalize_fields/inverse_transform operate
    on the already-aggregated ten fields, including during inference regridding.
    Statistics must come from the train-only fit_pli_normalization utility.
    """

    channel_dim = -4
    regrid_in_original_space = True
    kind = "pli_asinh_standardize"
    scalar_fields = ("Uvelocity", "Wvelocity", "av_density", "energy_case")
    materials = ("case", "cushion", "maincharge", "outside_air", "striker", "throw")

    @classmethod
    def output_fields(cls, field_names):
        """Validate the raw schema and retain the historical aggregate-energy label."""
        names = list(field_names)
        energies = [f"energy_{name}" for name in cls.materials]
        expected = [*cls.scalar_fields[:3], *energies, *[f"vofm_{m}" for m in cls.materials]]
        if len(names) != len(expected) or set(names) != set(expected):
            raise ValueError("PLINormalizedTransform requires the 15 named PLI source fields.")
        first = names.index("energy_case")
        if names[first : first + len(energies)] != energies:
            raise ValueError(
                "PLI energy fields must form the canonical contiguous six-field block."
            )
        return names[:first] + ["energy_case"] + names[first + len(energies) :]

    def __init__(self, field_names: list[str], norm_json_path: str):
        self.input_fields = list(field_names)
        self.field_names = self.output_fields(field_names)
        path = Path(norm_json_path)
        if not path.is_absolute() and not path.is_file():
            path = Path(__file__).resolve().parents[3] / path
        with path.open() as stream:
            stats = json.load(stream)
        if stats.get("kind") != self.kind:
            raise ValueError("Expected a pli_asinh_standardize statistics artifact.")
        if (
            stats.get("input_fields") != self.input_fields
            or stats.get("output_fields") != self.field_names
        ):
            raise ValueError(
                "PLI normalization statistics do not match the configured field order."
            )
        self.energy_scale = float(stats["energy_scale"])
        if not np.isfinite(self.energy_scale) or self.energy_scale <= 0:
            raise ValueError("PLI energy_scale must be finite and positive.")
        if stats.get("fit", {}).get("role") != "train":
            raise ValueError(
                "PLI normalization statistics must be fitted on training sources only."
            )
        for name in self.field_names:
            mean, std = float(stats[name]["mean"]), float(stats[name]["std"])
            if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
                raise ValueError(f"Invalid PLI mean/std for {name}.")
            if name.startswith("vofm_") and (mean != 0 or std != 1):
                raise ValueError(
                    "PLI volume fractions must use identity statistics (mean=0, std=1)."
                )
        self.energy_index = self.field_names.index("energy_case")
        self.affine = NormalizeArray(str(path), self.field_names, channel_dim=-4)
        self.aggregate = PLIRegularTransform(self.input_fields)

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        """Aggregate raw fields once, then normalize; used by uniform/native mappers."""
        arr = np.asarray(arr)
        if arr.ndim < 4 or arr.shape[-4] != len(self.input_fields):
            raise ValueError("PLI preprocessing expects (..., 15, T, H, W) raw fields.")
        leading, trailing = arr.shape[:-4], arr.shape[-3:]
        folded = arr.reshape((-1, len(self.input_fields), *trailing))
        aggregated = self.aggregate(folded).reshape((*leading, len(self.field_names), *trailing))
        return self.normalize_fields(aggregated)

    def _convert(self, arr, *, inverse):
        # NumPy serves loaders/metrics; tensors keep GPU replay on-device. These
        # constants are not model parameters and consume no initialization RNG.
        import torch

        is_tensor = torch.is_tensor(arr)
        if arr.ndim < 4 or arr.shape[-4] != len(self.field_names):
            raise ValueError("PLI value conversion expects (..., 10, T, H, W) aggregated fields.")
        index = [slice(None)] * arr.ndim
        index[-4] = self.energy_index
        index = tuple(index)
        if is_tensor:
            dtype = torch.float64 if arr.dtype == torch.float64 else torch.float32
            value = arr.to(dtype=dtype).clone()
            mean, std = self.affine._prepare_stats(arr.ndim)
            mean = torch.as_tensor(mean, dtype=dtype, device=arr.device)
            std = torch.as_tensor(std, dtype=dtype, device=arr.device)
            if inverse:
                value = value * std + mean
                value[index] = self.energy_scale * torch.sinh(value[index])
            else:
                value[index] = torch.asinh(value[index] / self.energy_scale)
                value = (value - mean) / std
            finite = bool(torch.isfinite(value).all())
        else:
            value = np.array(arr, dtype=np.result_type(arr.dtype, np.float32), copy=True, order="C")
            with np.errstate(over="ignore", invalid="ignore"):
                if inverse:
                    value = self.affine.inverse_transform(value)
                    value[index] = self.energy_scale * np.sinh(value[index])
                else:
                    value[index] = np.arcsinh(value[index] / self.energy_scale)
                    value = self.affine(value)
            finite = bool(np.isfinite(value).all())
        if not finite:
            raise ValueError(
                "Nonfinite PLI normalization result; values were not clipped or zero-filled."
            )
        return value

    def normalize_fields(self, arr):
        """Normalize ten existing fields without re-aggregation; used after regridding."""
        return self._convert(arr, inverse=False)

    def inverse_transform(self, arr):
        """Restore the ten original-representation fields for metrics and regridding."""
        return self._convert(arr, inverse=True)


def transform_regrid_values(transform, values, *, inverse):
    """Convert opt-in regrid payloads, preserving all legacy transform behavior.

    Used by adaptive/combined rollout and animation around spatial regridding.
    Nonlinear normalization must not be interpolated or coarsened in model space.
    Values are (..., C, T, H, W) arrays/tensors or native per-level dictionaries.
    """
    if not getattr(transform, "regrid_in_original_space", False):
        return values
    convert = transform.inverse_transform if inverse else transform.normalize_fields
    if isinstance(values, dict):
        return {level: convert(value) for level, value in values.items()}
    return convert(values)


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
    Transpose T↔C for ``NormalizeArray`` or a transform declaring
    ``channel_dim=-4`` (e.g. ``PLINormalizedTransform``). Other transforms keep
    their historical calling convention; ``PLITransform`` targets axis=2.
    Supports bare 4D sources and arbitrary leading batch dimensions. Used by
    sequence mappers and combined rollout; mirrored by ``inverse_transform_src``.
    """
    if transform is None:
        return arr
    canonical_channels = isinstance(transform, NormalizeArray) or getattr(
        transform, "channel_dim", None
    ) == -4
    if not canonical_channels or arr.ndim < 4:
        return transform(arr)
    return np.swapaxes(transform(np.swapaxes(arr, -4, -3)), -4, -3)


def inverse_transform_src(transform: Optional[Callable], arr: np.ndarray) -> np.ndarray:
    """Inverse-transform a ``(..., T, C, H, W)`` full-field src tensor.

    Mirrors :func:`apply_transform_src`. Returns ``arr`` unchanged when the
    transform has no ``inverse_transform`` (e.g. ``PLITransform``).
    """
    if transform is None:
        return arr
    if not (hasattr(transform, "inverse_transform") and callable(transform.inverse_transform)):
        return arr
    canonical_channels = isinstance(transform, NormalizeArray) or getattr(
        transform, "channel_dim", None
    ) == -4
    if not canonical_channels or arr.ndim < 4:
        return transform.inverse_transform(arr)
    return np.swapaxes(transform.inverse_transform(np.swapaxes(arr, -4, -3)), -4, -3)


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
