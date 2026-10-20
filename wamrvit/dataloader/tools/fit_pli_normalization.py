"""Fit the opt-in PLI transform from unique training NPZ sources, without Ray.

Two bounded passes: fit the three affine scalars and estimate energy Q99 from
reproducible uniform pixel samples; then fit asinh-energy moments over every
training source-grid cell. Validation/test frames never enter either pass.
Use an owned compute allocation for production cohorts, not a login-node scan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from wamrvit.dataloader.selection.sources import prepare_source_partitions
from wamrvit.dataloader.topology_store import source_key
from wamrvit.dataloader.payload.transforms import PLINormalizedTransform


class RunningMoments:
    """Merge per-frame float64 moments; used by both fitting passes."""

    def __init__(self, channels):
        self.count = 0
        self.mean = np.zeros(channels, dtype=np.float64)
        self.m2 = np.zeros(channels, dtype=np.float64)

    def update(self, count, mean, m2):
        total = self.count + count
        delta = mean - self.mean
        self.m2 += m2 + delta * delta * (self.count * count / total)
        self.mean += delta * (count / total)
        self.count = total

    @property
    def std(self):
        return np.sqrt(self.m2 / self.count)


def _moments(values):
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=1)
    return values.shape[1], mean, np.sum((values - mean[:, None]) ** 2, axis=1)


def _field(archive, name):
    value = archive[name].astype(np.float32)
    if value.ndim != 2 or np.isinf(value).any():
        raise ValueError(f"Expected a 2D PLI field without infinities: {name}.")
    return np.nan_to_num(value, nan=0.0)


def _first_pass(path, *, fields, pixels, seed):
    with np.load(path, allow_pickle=False) as archive:
        if not set(fields) <= set(archive.files):
            raise ValueError(f"Missing PLI fields in {path}.")
        scalars = np.stack(
            [_field(archive, name) for name in PLINormalizedTransform.scalar_fields[:3]]
        )
        count = scalars.shape[1] * scalars.shape[2]
        identity = hashlib.sha256(f"{seed}\0{path}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(identity[:16], "little"))
        chosen = rng.choice(count, min(pixels, count), replace=False)
        energy = np.sum(
            np.stack(
                [
                    _field(archive, f"energy_{m}").reshape(-1)[chosen]
                    for m in PLINormalizedTransform.materials
                ]
            ),
            axis=0,
            dtype=np.float32,
        )
    if not np.isfinite(energy).all():
        raise ValueError(f"Nonfinite aggregate energy in {path}.")
    return _moments(scalars.reshape(3, -1)), np.abs(energy), count / len(chosen)


def _second_pass(path, *, scale):
    with np.load(path, allow_pickle=False) as archive:
        # Match production's FP32 six-channel sum before nonlinear processing.
        energy = np.sum(
            np.stack([_field(archive, f"energy_{m}") for m in PLINormalizedTransform.materials]),
            axis=0,
            dtype=np.float32,
        )
    if not np.isfinite(energy).all():
        raise ValueError(f"Nonfinite aggregate energy in {path}.")
    transformed = np.arcsinh(energy.astype(np.float64) / scale)
    return _moments(transformed.reshape(1, -1))


def _mapped(function, paths, workers):
    # Bound pending futures/results independently of dataset length.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(paths), 256):
            yield from pool.map(function, paths[start : start + 256])
            print(f"Processed {min(start + 256, len(paths))}/{len(paths)} frames", flush=True)


def _source_digest(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(source_key(path)[0].encode())
    return digest.hexdigest()


def fit_pli_statistics(config, *, workers=1, energy_pixels_per_frame=64, seed=42):
    """Return frozen train-only statistics/provenance; used by the CLI and tests.

    Scalar/asinh moments use all source-grid cells, not repeated windows or
    adaptive leaf weights. Energy Q99 is a deterministic weighted pixel estimate.
    """
    from wamrvit.utils import instantiate_from_config

    if workers < 1 or energy_pixels_per_frame < 1:
        raise ValueError("workers and energy_pixels_per_frame must be positive.")
    fields = list(config["file_loader"]["params"]["field_names"])
    output_fields = PLINormalizedTransform.output_fields(fields)
    parsed = instantiate_from_config(config["file_parser"])(config["data"]["glob_pattern"])
    partitions = prepare_source_partitions(config, parsed)
    paths = [path for segment in partitions["train"] for path in segment.paths]
    if not paths or any(not str(path).endswith(".npz") for path in paths):
        raise ValueError("PLI fitting requires nonempty training NPZ sources.")
    before = _source_digest(paths)
    moments = RunningMoments(3)
    samples, sample_weights = [], []
    first = partial(_first_pass, fields=fields, pixels=energy_pixels_per_frame, seed=seed)
    print(f"Pass 1: {len(paths)} unique training frames", flush=True)
    for moment, sample, weight in _mapped(first, paths, workers):
        moments.update(*moment)
        samples.append(sample)
        sample_weights.append(np.full(len(sample), weight, dtype=np.float64))
    sampled_values = np.concatenate(samples)
    scale = float(
        np.quantile(
            sampled_values, 0.99, weights=np.concatenate(sample_weights), method="inverted_cdf"
        )
    )
    sample_count = len(sampled_values)
    del samples, sample_weights, sampled_values
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training energy Q99 is not positive; cannot fit the chosen asinh policy.")
    energy = RunningMoments(1)
    print(f"Pass 2: asinh-energy moments with scale={scale:.9g}", flush=True)
    for moment in _mapped(partial(_second_pass, scale=scale), paths, workers):
        energy.update(*moment)
    if before != _source_digest(paths):
        raise RuntimeError(
            "Training sources changed during fitting; refusing to publish statistics."
        )
    if energy.count != moments.count:
        raise ValueError("PLI energy/scalar source-grid sizes differ.")
    result = {
        "kind": PLINormalizedTransform.kind,
        "input_fields": fields,
        "output_fields": output_fields,
        "energy_scale": scale,
        "fit": {
            "role": "train",
            "source_frames": len(paths),
            "source_grid_cells": moments.count,
            "source_metadata_sha256": before,
            "source_identity": "canonical path, file size and nanosecond mtime; not a content hash",
            "source_roles": {
                role: sum(len(s.paths) for s in parts) for role, parts in partitions.items()
            },
            "file_parser": config["file_parser"],
            "source_selection": config["window_generator"]["params"],
            "data": config["data"],
            "split_ratio": config["train"].get("split_ratio", 0.8),
            "weighting": "one occurrence per training source-grid cell; no window/leaf weighting",
            "energy_quantile": 0.99,
            "energy_quantile_method": "weighted inverted-CDF, uniform per-frame pixel sample",
            "energy_pixels_per_frame": energy_pixels_per_frame,
            "energy_sample_count": sample_count,
            "seed": seed,
            "energy_moments": "all train cells; FP32 aggregation, float64 asinh/moments",
        },
    }
    for name, mean, std in zip(PLINormalizedTransform.scalar_fields[:3], moments.mean, moments.std):
        result[name] = {"mean": float(mean), "std": float(std)}
    result["energy_case"] = {"mean": float(energy.mean[0]), "std": float(energy.std[0])}
    for name in output_fields:
        if name.startswith("vofm_"):
            result[name] = {"mean": 0.0, "std": 1.0}
    for name in PLINormalizedTransform.scalar_fields:
        if not np.isfinite(list(result[name].values())).all() or result[name]["std"] <= 0:
            raise ValueError(f"Degenerate/nonfinite training statistics for {name}.")
    return result


def main():
    """Fit a Hydra/standalone PLI config; publish JSON without overwriting artifacts."""
    from wamrvit.utils import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--energy-pixels-per-frame", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args, overrides = parser.parse_known_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite statistics: {output}")
    config = load_config(args, overrides)
    stats = fit_pli_statistics(
        config,
        workers=args.workers,
        energy_pixels_per_frame=args.energy_pixels_per_frame,
        seed=args.seed,
    )
    stats["fit"]["git_head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    stats["fit"]["fitter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(stats, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)  # Atomic publication; fails if destination exists.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Wrote training-only PLI statistics: {output}")


if __name__ == "__main__":
    main()
