"""Optional Nomad tools for the normalized PLI checkpoints."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch
from pydantic import Field, model_validator

try:
    from nomad.fm_base_tool import TorchModuleTool
    from nomad.well_format import AutoRegressiveInput, WellFormat
except ImportError as error:
    raise ImportError(
        "Nomad integration requires wamrvit[nomad]. From the checkout, run "
        "pip install '.[nomad]' or uv sync --extra nomad."
    ) from error

from wamrvit.integrations.pli import PLI_FIELDS, PLIForecaster


class PLIForecastInput(AutoRegressiveInput):
    """Named physical PLI fields on the trained Cartesian grid."""

    duration: int = Field(
        strict=True, gt=0, le=100,
        description="Forecast horizon in source-frame steps (1–100), not seconds.",
    )
    return_trajectory: bool = Field(
        default=False,
        description=(
            "Return the last observation and all forecast frames; otherwise only "
            "the last observation and endpoint."
        ),
    )

    @model_validator(mode="after")
    def validate_pli_state(self):
        state = self.initial_state
        if state.n_spatial_dims != 2 or state.grid_type != "cartesian":
            raise ValueError("PLI requires a two-dimensional Cartesian field.")
        if set(state.t0_fields) != set(PLI_FIELDS) or state.t1_fields or state.t2_fields:
            raise ValueError(
                f"Provide exactly these ten aggregated scalar fields: {', '.join(PLI_FIELDS)}."
            )
        if state.boundary_conditions:
            raise ValueError("This checkpoint has no independent boundary-condition controls.")
        if state.dimensions.spatial_dims not in (None, ["y", "x"]):
            raise ValueError(
                "PLI scalar field axes are (history, y, x); use spatial_dims=['y', 'x']."
            )
        shapes = {tuple(value.shape) for value in state.t0_fields.values()}
        if len(shapes) != 1:
            raise ValueError("All PLI fields must have the same history and spatial shape.")
        shape = next(iter(shapes))
        if len(shape) != 3 or shape[0] < 1 or shape[1:] != (1120, 400):
            raise ValueError("PLI fields must have shape (history, 1120, 400).")
        times = state.dimensions.time
        if times:
            if len(times) != shape[0] or any(isinstance(value, list) for value in times):
                raise ValueError("Provide one source-frame time coordinate per history frame.")
            if not all(math.isfinite(value) for value in times):
                raise ValueError("Source-frame coordinates must be finite.")
            if any(abs((b - a) - 1.0) > 1e-6 for a, b in zip(times, times[1:])):
                raise ValueError(
                    "Consecutive PLI history frames must be one source-frame step apart."
                )
        return self


class PLIForecastTool(TorchModuleTool):
    """Serve regular ViT and native-MS using the same physical-field interface."""

    args_schema: type[PLIForecastInput] = PLIForecastInput
    output_schema: type[WellFormat] = WellFormat
    fm: PLIForecaster

    @classmethod
    def from_pretrained(cls, name_or_path: str):
        forecaster = PLIForecaster.from_pretrained(name_or_path)
        history = forecaster.history_length
        return cls(
            fm=forecaster, name=Path(name_or_path).name, batch_size=1,
            description=(
                f"Forecast PLI from {history} consecutive source frames on the "
                "1120x400 Cartesian grid. Provide ten named scalar fields in dataset "
                "source units, including summed energy_case. duration counts frame steps, "
                "not seconds. Returns the last observation and forecast endpoint unless "
                "return_trajectory requests all predicted frames."
            ),
        )

    def preprocess(self, inputs: Sequence[PLIForecastInput]):
        prepared = []
        for item in inputs:
            fields = item.initial_state.t0_fields
            if next(iter(fields.values())).shape[0] != self.fm.history_length:
                raise ValueError(
                    f"This checkpoint requires {self.fm.history_length} input frame(s)."
                )
            # Quadtree preparation is CPU-native; the public predictor transfers
            # normalized tensors only to the Nomad-managed model device.
            history = torch.stack([
                fields[name].detach().to(device="cpu", dtype=torch.float32)
                for name in self.fm.fields
            ])
            prepared.append((item, history))
        return prepared

    def _forward(self, model_inputs):
        # Requests may have different horizons. Process independently rather than
        # padding histories or mixing native topologies across requests.
        return [
            (item, self.fm(history, item.duration, return_trajectory=item.return_trajectory))
            for item, history in model_inputs
        ]

    def postprocess(self, model_output) -> Iterable[WellFormat]:
        for item, trajectory in model_output:
            initial = item.initial_state
            times = initial.dimensions.time
            start = float(times[-1]) if times else 0.0
            offsets = range(item.duration + 1) if item.return_trajectory else (0, item.duration)
            dimensions = initial.dimensions.model_copy(update={
                "spatial_dims": ["y", "x"], "time": [start + step for step in offsets],
            })
            yield WellFormat(
                dataset_name=initial.dataset_name, grid_type="cartesian", n_spatial_dims=2,
                dimensions=dimensions, scalars=dict(initial.scalars),
                # Nomad's torch.save codec preserves a view's whole backing storage.
                # Give each field its own compact storage, not all ten channels.
                t0_fields={
                    name: trajectory[:, channel].clone()
                    for channel, name in enumerate(self.fm.fields)
                },
            )
