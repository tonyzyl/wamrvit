"""Target-free PLI inference shared by the optional Nomad adapter and native callers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from wamrvit.checkpoint import load_model
from wamrvit.dataloader.payload.transforms import PLINormalizedTransform, PLIRegularTransform

PLI_FIELDS = (
    "Uvelocity", "Wvelocity", "av_density", "energy_case",
    "vofm_case", "vofm_cushion", "vofm_maincharge", "vofm_outside_air",
    "vofm_striker", "vofm_throw",
)
RAW_PLI_FIELDS = (
    *PLI_FIELDS[:3],
    *(f"energy_{name}" for name in PLINormalizedTransform.materials),
    *(f"vofm_{name}" for name in PLINormalizedTransform.materials),
)


def pli_source_frame_times(paths: list[str | Path]) -> list[int]:
    """Validate ordered, consecutive PLI filenames from one trajectory for the client."""
    from wamrvit.dataloader.selection.sources import group_and_sort_files, parse_frame_indices

    names = [str(path) for path in paths]
    frames = parse_frame_indices(names).tolist()
    trajectories = group_and_sort_files(names)
    if (
        len(trajectories) != 1
        or trajectories[0] != names
        or any(b - a != 1 for a, b in zip(frames, frames[1:]))
    ):
        raise ValueError("Provide ordered, consecutive PLI frames from one trajectory.")
    return frames


def load_pli_history(paths: list[str | Path], *, native: bool = False) -> torch.Tensor:
    """Read physical ``(10,T,H,W)`` fields using the published loader's precision.

    Regular loading aggregates in source dtype; native export converts samples
    to FP32 before aggregation. Preserve this distinction for FP16 NPZ inputs.
    """
    frames = []
    for path in paths:
        with np.load(path, allow_pickle=False) as snapshot:
            frame = np.stack([snapshot[name] for name in RAW_PLI_FIELDS])
            frames.append(frame.astype(np.float32) if native else frame)
    # Match Seq2SeqMapper's time-major source layout before reduction.
    raw = np.stack(frames, axis=0)[None].transpose(0, 2, 1, 3, 4)
    physical = PLIRegularTransform(list(RAW_PLI_FIELDS))(raw)[0]
    return torch.from_numpy(physical.astype(np.float32))


class PLIForecaster(nn.Module):
    """A real checkpoint plus its frozen transform; device follows its parameters.

    Input is physical/source-valued ``(C,T,H,W)`` history. Output is CPU
    ``(T_out,C,H,W)``: last observation plus the endpoint, or all future frames.
    Numerical rollout/regridding is owned by the existing public predictors.
    """

    def __init__(self, model: nn.Module, transform: PLINormalizedTransform):
        super().__init__()
        self.backbone = model.eval()
        self.transform = transform
        self.fields = tuple(transform.field_names)
        if set(self.fields) != set(PLI_FIELDS) or model.config.in_channels != len(self.fields):
            raise ValueError("Expected a normalized ten-field PLI checkpoint.")
        if model.config.adaptive and not model.config.multi_scale_patch:
            raise ValueError("PLI serving supports regular ViT and native-MS checkpoints.")

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> PLIForecaster:
        root = Path(directory)
        stats = json.loads((root / "pli_normalization.json").read_text())
        transform = PLINormalizedTransform(
            stats["input_fields"], str(root / "pli_normalization.json")
        )
        return cls(load_model(root, {}), transform)

    @property
    def history_length(self) -> int:
        return int(self.backbone.config.patch_size_t)

    def _native_batch(self, physical: np.ndarray):
        from wamrvit.quad.adapt_wavelet import adapt_on_field, expand_fine_region
        from wamrvit.quad.quad_utils import quadtree_to_tensor_native
        from wamrvit.quad.quadtree import Quadtree

        channels, frames, height, width = physical.shape
        levels = int(self.backbone.config.max_level_idx) + 1
        ph, pw = self.backbone.config.patch_size
        tile_h, tile_w = ph * 2 ** (levels - 1), pw * 2 ** (levels - 1)
        detector = [self.fields.index(name) for name in PLI_FIELDS[:3]]
        mesh = Quadtree(
            0, width, 0, height, channels=channels, max_level_idx=levels - 1,
            tile_width=tile_w, tile_height=tile_h, value_storage="native",
        )
        adapt_on_field(
            mesh, physical[:, -1], tol_frac=0.01, channel=detector, max_passes=levels - 1
        )
        mesh = expand_fine_region(mesh, 2)
        per_time = []
        for time in range(frames):
            mesh.assign_from_array(physical[:, time])
            values, mapping, meta = quadtree_to_tensor_native(
                mesh, cell_scale_mode=self.backbone.config.cell_scale_mode,
            )
            per_time.append(values)
        batch = {
            f"input_level_{level}": self.transform.normalize_fields(
                np.stack([values[level] for values in per_time], axis=2)
            )[None]
            for level in range(levels)
        }
        batch.update(
            leaf_to_bucket=mapping[None], centers=meta["centers"][None], levels=meta["levels"][None]
        )
        batch.update({key: np.array([value]) for key, value in meta["domain"].items()})
        return batch, meta, detector

    @torch.inference_mode()
    def forward(
        self, history: torch.Tensor, steps: int, *, return_trajectory: bool = False,
    ) -> torch.Tensor:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer frame count.")
        physical = history.detach().to(device="cpu", dtype=torch.float32).numpy()
        if physical.ndim != 4 or physical.shape[:2] != (len(self.fields), self.history_length):
            raise ValueError(f"Expected (10, {self.history_length}, H, W) physical PLI history.")
        if not np.isfinite(physical).all():
            raise ValueError("PLI input fields must be finite.")
        device = next(self.backbone.parameters()).device
        config = {"inference": {"predict_steps": steps, "pred_mode": "target"}}
        if self.backbone.config.multi_scale_patch:
            from wamrvit.quad.quad_utils import tensor_to_uniform_native
            from wamrvit.rollout.rollout_adaptive import AutoregressivePredictorAdaptive

            batch, meta, detector = self._native_batch(physical)
            config["file_loader"] = {"params": {
                "field_names": list(self.fields), "adapt_on_channels": detector,
                "cell_scale_mode": self.backbone.config.cell_scale_mode, "tol_frac": 0.01,
            }}
            config["inference"].update(
                regrid_interval=4, regrid_tol_frac=[0.1, 0.1, 0.3], regrid_adapt_nearby=2,
                regrid_backend="array", array_regrid_payload_backend="cpu_eager",
                array_regrid_num_threads=1,
            )
            predictor = AutoregressivePredictorAdaptive(
                config, model=self.backbone, transform=self.transform, device=device,
            )
            result = predictor(batch)
            selected = range(steps) if return_trajectory else [steps - 1]
            future = []
            for step in selected:
                values = {
                    level: value[:, :, 0]
                    for level, value in result["preds_by_level_list"][0][step].items()
                }
                step_meta = {"centers": result["centers_list"][0][step],
                             "levels": result["levels_list"][0][step], "domain": meta["domain"]}
                future.append(tensor_to_uniform_native(
                    values, result["leaf_to_bucket_list"][0][step], step_meta,
                ))
            forecast = np.stack(future, axis=1)
        else:
            from wamrvit.rollout.rollout_regular import AutoregressivePredictor

            ph, pw = self.backbone.config.patch_size
            if physical.shape[-2] % ph or physical.shape[-1] % pw:
                raise ValueError("PLI spatial dimensions must be divisible by the patch size.")
            predictor = AutoregressivePredictor(
                config, model=self.backbone, transform=self.transform, device=device,
            )
            result = predictor({"input": self.transform.normalize_fields(physical[None])})
            forecast = result["pred"][0]
            if not return_trajectory:
                forecast = forecast[:, -1:]
        if not np.isfinite(forecast).all():
            raise ValueError("Nonfinite PLI forecast; outputs were not clipped or replaced.")
        trajectory = np.concatenate([physical[:, -1:], forecast], axis=1)
        return torch.from_numpy(trajectory).permute(1, 0, 2, 3).contiguous()
