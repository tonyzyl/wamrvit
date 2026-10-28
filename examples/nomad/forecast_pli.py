"""Call a PLI Nomad tool using real NPZ inputs and save a typed forecast.

Run with the optional extra installed; the server is started separately.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import torch
from fastmcp import Client
from nomad.well_format import Domain, WellFormat

from wamrvit.integrations.nomad import PLIForecastInput
from wamrvit.integrations.pli import PLI_FIELDS, load_pli_history, pli_source_frame_times


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/mcp")
    parser.add_argument("--tool", default="pli-vit-1to1-full",
                        choices=["pli-ms-2to1-1k", "pli-vit-2to1-1k", "pli-vit-1to1-full"])
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--trajectory", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, help="Optional density input/prediction/delta plot")
    args = parser.parse_args()
    times = pli_source_frame_times(args.inputs)
    history = load_pli_history(args.inputs, native=args.tool == "pli-ms-2to1-1k")
    _, _, height, width = history.shape
    state = WellFormat(
        dataset_name="PLI", grid_type="cartesian", n_spatial_dims=2,
        dimensions=Domain(
            spatial_dims=["y", "x"], time=times,
            y=list(range(height)), x=list(range(width)),
        ),
        # Compact per-field storage avoids serializing the full history per view.
        t0_fields={name: history[channel].clone() for channel, name in enumerate(PLI_FIELDS)},
    )
    request = PLIForecastInput(
        initial_state=state, duration=args.steps, return_trajectory=args.trajectory,
    )
    async with Client(args.url, timeout=600) as client:
        result = await client.call_tool(args.tool, request.model_dump(mode="json"))
    payload = result.structured_content
    if payload is None:
        raise RuntimeError("Nomad returned no structured forecast.")
    if set(payload) == {"result"}:
        payload = payload["result"]
    forecast = WellFormat.model_validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_file(args.output)
    for name, values in forecast.t0_fields.items():
        final = values[-1]
        if not bool(torch.isfinite(final).all()):
            raise ValueError(f"Nonfinite forecast field: {name}")
        print(
            f"{name}: min={final.min().item():.6g} max={final.max().item():.6g} "
            f"mean={final.mean().item():.6g}"
        )
    print("Forecast times (source-frame steps):", forecast.dimensions.time)
    print("Saved:", args.output)
    if args.plot:
        import matplotlib.pyplot as plt

        initial = forecast.t0_fields["av_density"][0].numpy()
        final = forecast.t0_fields["av_density"][-1].numpy()
        lo, hi = float(initial.min()), float(initial.max())
        delta = final - initial
        limit = max(float(abs(delta).max()), 1e-12)
        fig, axes = plt.subplots(1, 3, figsize=(10, 7), layout="constrained")
        for axis, title, image in zip(
            axes, ["Last input", "Prediction", "Delta"], [initial, final, delta]
        ):
            options = ({"vmin": lo, "vmax": hi, "cmap": "jet"} if title != "Delta" else
                       {"vmin": -limit, "vmax": limit, "cmap": "coolwarm"})
            artist = axis.imshow(image, origin="lower", **options)
            axis.set_title(title)
            fig.colorbar(artist, ax=axis, shrink=0.7)
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot)
        plt.close(fig)


if __name__ == "__main__":
    asyncio.run(main())
