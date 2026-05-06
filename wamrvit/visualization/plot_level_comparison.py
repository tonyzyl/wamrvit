"""Generate RMSE and VRMSE comparison line plots from a rollout_combined CSV.

CSV layout (produced by rollout_combined): one row per rollout step, with
`uniform_{RMSE,VRMSE}_{adaptive,regular}_{field}` and per-level
`level_{L}_{RMSE,VRMSE}_{adaptive,regular}_{field}` columns.

Usage:
    uv run wamrvit/visualization/plot_level_comparison.py path/to/rmse_combined_*.csv \
        [--out-dir plots] [--metrics RMSE VRMSE] [--levels 0 1 2 3]
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

DEFAULT_FIELDS = [
    "Uvelocity",
    "Wvelocity",
    "av_density",
    "energy_case",
    "vofm_case",
    "vofm_cushion",
    "vofm_maincharge",
    "vofm_outside_air",
    "vofm_striker",
    "vofm_throw",
]


def _panel_grid(n: int):
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    return fig, axes.flatten()


def _plot_scope(
    df: pd.DataFrame,
    metric: str,
    scope: str,
    fields: list[str],
    title_prefix: str,
    out_path: str,
    log_y: bool,
) -> None:
    """One figure: for each field, plot adaptive vs regular for the given scope.

    scope is either 'uniform' or 'level_{L}'.
    """
    fig, axes = _panel_grid(len(fields))
    for ax, field in zip(axes, fields):
        col_ad = f"{scope}_{metric}_adaptive_{field}"
        col_rg = f"{scope}_{metric}_regular_{field}"
        if col_ad not in df.columns or col_rg not in df.columns:
            ax.set_visible(False)
            continue
        ax.plot(df["Step"], df[col_ad], label="Adaptive", marker="o", color="tab:blue")
        ax.plot(df["Step"], df[col_rg], label="Regular", marker="x", color="tab:red")
        ax.set_title(f"{title_prefix} {metric}: {field}")
        ax.set_xlabel("Step")
        ax.set_ylabel(metric)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    for ax in axes[len(fields) :]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_comparison_plots(
    csv_path: str,
    out_dir: str,
    metrics: Iterable[str] = ("RMSE", "VRMSE"),
    levels: Iterable[int] = (0, 1, 2, 3),
    fields: Iterable[str] = DEFAULT_FIELDS,
    log_y: bool = False,
) -> list[str]:
    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(csv_path))[0]
    fields = list(fields)
    written: list[str] = []

    for metric in metrics:
        # Uniform (full-field) panel.
        out = os.path.join(out_dir, f"{stem}__uniform_{metric}.png")
        _plot_scope(df, metric, "uniform", fields, "Uniform", out, log_y)
        written.append(out)

        # Per-level panels. Skip levels not present in the CSV.
        for lvl in levels:
            if not any(f"level_{lvl}_{metric}_adaptive_{f}" in df.columns for f in fields):
                continue
            out = os.path.join(out_dir, f"{stem}__level_{lvl}_{metric}.png")
            _plot_scope(df, metric, f"level_{lvl}", fields, f"Level {lvl}", out, log_y)
            written.append(out)

    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", help="Path to rmse_combined_*.csv")
    p.add_argument("--out-dir", default="plots", help="Directory to write PNGs to")
    p.add_argument("--metrics", nargs="+", default=["RMSE", "VRMSE"])
    p.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--fields", nargs="+", default=DEFAULT_FIELDS)
    p.add_argument("--log-y", action="store_true", help="Use log scale on y-axis")
    args = p.parse_args()

    written = generate_comparison_plots(
        csv_path=args.csv,
        out_dir=args.out_dir,
        metrics=args.metrics,
        levels=args.levels,
        fields=args.fields,
        log_y=args.log_y,
    )
    print(f"Wrote {len(written)} figure(s):")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()
