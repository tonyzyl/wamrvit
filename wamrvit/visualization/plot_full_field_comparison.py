"""Compare multiple adaptive runs on the full field (uniform-space metric).

Each input CSV is a rollout_combined output with columns
`uniform_{RMSE,VRMSE}_{adaptive,regular}_{field}` + per-level columns.
This script ignores per-level columns and only plots `uniform_*` rows.

Because all runs share the same regular (full-resolution) baseline on the same
trajectory set, the `uniform_*_regular_*` columns are expected to agree across
CSVs. We draw the regular baseline once (from the first CSV) and warn if any
other CSV disagrees beyond a small tolerance.

Usage:
    uv run wamrvit/visualization/plot_full_field_comparison.py \
        run_a.csv run_b.csv run_c.csv \
        [--labels native uniform_adap something] \
        [--out-dir plots] [--metrics RMSE VRMSE] [--log-y]

If --labels is omitted, the CSV filename stem is used. Duplicate labels are
disambiguated with a numeric suffix.
"""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
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

_BASELINE_TOL = 1e-4  # relative tolerance for cross-CSV regular-column agreement


def _shorten_label(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    # Strip the common rollout_combined prefix so the label shows the run-specific part.
    stem = re.sub(r"^rmse_combined_", "", stem)
    return stem


def _dedupe_labels(labels: Sequence[str]) -> list[str]:
    seen: dict = {}
    out: list[str] = []
    for lab in labels:
        if lab not in seen:
            seen[lab] = 1
            out.append(lab)
        else:
            seen[lab] += 1
            out.append(f"{lab}#{seen[lab]}")
    return out


def _panel_grid(n: int):
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    return fig, axes.flatten()


def _check_baseline_agreement(
    dfs: Sequence[pd.DataFrame],
    labels: Sequence[str],
    metric: str,
    fields: Sequence[str],
) -> None:
    """Warn if the regular-baseline columns disagree across CSVs."""
    ref = dfs[0]
    for df, lab in zip(dfs[1:], labels[1:]):
        for field in fields:
            col = f"uniform_{metric}_regular_{field}"
            if col not in ref.columns or col not in df.columns:
                continue
            r = ref[col].to_numpy()
            o = df[col].to_numpy()
            denom = np.maximum(np.abs(r), 1e-30)
            rel = np.abs(o - r) / denom
            if rel.max() > _BASELINE_TOL:
                print(
                    f"[warn] regular baseline mismatch in '{col}' "
                    f"between '{labels[0]}' and '{lab}': max rel diff = {rel.max():.3e}. "
                    f"Using '{labels[0]}' as the reference baseline."
                )
                break


def _plot_metric(
    dfs: Sequence[pd.DataFrame],
    labels: Sequence[str],
    metric: str,
    fields: Sequence[str],
    out_path: str,
    log_y: bool,
) -> None:
    fig, axes = _panel_grid(len(fields))
    # Distinct adaptive colors; regular is always black/dashed for readability.
    cmap = plt.get_cmap("tab10")
    adaptive_colors = [cmap(i % 10) for i in range(len(dfs))]

    _check_baseline_agreement(dfs, labels, metric, fields)
    ref_df = dfs[0]

    for ax, field in zip(axes, fields):
        col_ad_any = f"uniform_{metric}_adaptive_{field}"
        col_rg = f"uniform_{metric}_regular_{field}"
        if col_ad_any not in ref_df.columns and col_rg not in ref_df.columns:
            ax.set_visible(False)
            continue

        # Adaptive lines, one per CSV.
        for df, lab, color in zip(dfs, labels, adaptive_colors):
            col = f"uniform_{metric}_adaptive_{field}"
            if col not in df.columns:
                continue
            ax.plot(
                df["Step"], df[col], label=lab, marker="o", markersize=4, color=color, linewidth=1.5
            )

        # Single regular baseline from first CSV (same across runs by construction).
        if col_rg in ref_df.columns:
            ax.plot(
                ref_df["Step"],
                ref_df[col_rg],
                label="regular (baseline)",
                marker="x",
                markersize=5,
                color="black",
                linewidth=1.5,
                linestyle="--",
            )

        ax.set_title(f"Uniform {metric}: {field}")
        ax.set_xlabel("Step")
        ax.set_ylabel(metric)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[len(fields) :]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_full_field_comparison(
    csv_paths: Sequence[str],
    out_dir: str,
    labels: Sequence[str] | None = None,
    metrics: Iterable[str] = ("RMSE", "VRMSE"),
    fields: Iterable[str] = DEFAULT_FIELDS,
    out_prefix: str = "full_field_comparison",
    log_y: bool = False,
) -> list[str]:
    if not csv_paths:
        raise ValueError("At least one CSV path is required.")

    if labels is None:
        labels = [_shorten_label(p) for p in csv_paths]
    else:
        if len(labels) != len(csv_paths):
            raise ValueError(
                f"Got {len(labels)} labels for {len(csv_paths)} CSVs; lengths must match."
            )
    labels = _dedupe_labels(labels)

    dfs = [pd.read_csv(p) for p in csv_paths]

    # Align steps — we expect them identical, but the user may have passed runs
    # with different step counts; intersect defensively.
    common_steps = set(dfs[0]["Step"].tolist())
    for df in dfs[1:]:
        common_steps &= set(df["Step"].tolist())
    common_steps = sorted(common_steps)
    if not common_steps:
        raise ValueError("No common Step values across CSVs — cannot align.")
    if any(len(df) != len(common_steps) for df in dfs):
        dfs = [
            df[df["Step"].isin(common_steps)].sort_values("Step").reset_index(drop=True)
            for df in dfs
        ]

    os.makedirs(out_dir, exist_ok=True)
    fields = list(fields)
    written: list[str] = []
    for metric in metrics:
        out = os.path.join(out_dir, f"{out_prefix}__uniform_{metric}.png")
        _plot_metric(dfs, labels, metric, fields, out, log_y)
        written.append(out)
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csvs", nargs="+", help="One or more rmse_combined_*.csv files")
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Label per CSV (must match --csvs count). Defaults to filename stem.",
    )
    p.add_argument("--out-dir", default="plots", help="Directory to write PNGs to")
    p.add_argument(
        "--out-prefix", default="full_field_comparison", help="Filename prefix for output PNGs"
    )
    p.add_argument("--metrics", nargs="+", default=["RMSE", "VRMSE"])
    p.add_argument("--fields", nargs="+", default=DEFAULT_FIELDS)
    p.add_argument("--log-y", action="store_true", help="Use log scale on y-axis")
    args = p.parse_args()

    written = generate_full_field_comparison(
        csv_paths=args.csvs,
        labels=args.labels,
        out_dir=args.out_dir,
        out_prefix=args.out_prefix,
        metrics=args.metrics,
        fields=args.fields,
        log_y=args.log_y,
    )
    print(f"Wrote {len(written)} figure(s):")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()
