"""Overlay N rollout CSVs on one set of panels (uniform + per-level).

Supports two input schemas:

1. **Combined CSV** (from ``rollout_combined.py``) — has columns
   ``uniform_{M}_{adaptive,regular}_{field}`` and per-level
   ``level_{L}_{M}_{adaptive,regular}_{field}``.
2. **Standalone adaptive CSV** (from ``rollout_adaptive.py``) — has
   ``uniform_{M}_{field}`` and per-level ``level_{L}_{M}_{field}`` with no
   ``_adaptive``/``_regular`` tag.

The uniform regular baseline is drawn *once* (black dashed). Source priority:
- If any input CSV is combined, its ``uniform_{M}_regular_*`` columns are used.
  If multiple combined CSVs disagree beyond a small tolerance we warn and use
  the first.
- Else, if ``--regular-csv`` is provided (a standalone regular CSV), that is
  used.
- Else the uniform regular line is omitted.

Per-level regular lines are drawn *per combined CSV* in the same color as its
adaptive line but dashed — because projecting the regular prediction onto each
adaptive run's quadtree topology yields different per-level values.

**Output pairs (always both):**
For every (metric, scope) combination we emit two PNGs:
- ``{prefix}__{scope}_{metric}.png`` — per-field grid (10 panels, one per field)
- ``{prefix}__{scope}_{metric}_agg.png`` — single panel, field-mean aggregate

This means every invocation produces ``len(metrics) × (1 + n_levels) × 2``
PNGs (e.g., 2 metrics × 5 panels × 2 = 20 PNGs).

Usage:
    uv run wamrvit/visualization/plot_rollout_multi.py \\
        runA.csv runB.csv [runC.csv ...] \\
        [--labels "AdaLN" "baseline" ...] \\
        [--regular-csv standalone_regular.csv] \\
        [--out-dir plots/adaln] [--out-prefix adaln_vs_baseline] \\
        [--metrics RMSE VRMSE] [--fields ...] \\
        [--no-uniform] [--log-y]
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

_BASELINE_TOL = 1e-4  # relative tol for cross-CSV regular-baseline agreement


def _shorten_label(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
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


def _detect_schema(df: pd.DataFrame) -> str:
    """Return 'combined', 'adaptive_only', 'regular_only', or 'unknown'.

    'combined' fires if adaptive + regular side-by-side columns appear in
    either uniform scope OR per-level scope (AMReX combined rollouts run
    with ``compute_uniform_metrics: False`` and only produce per-level).
    """
    cols = df.columns
    has_comb_adap_u = any(c.startswith("uniform_") and "_adaptive_" in c for c in cols)
    has_comb_reg_u = any(c.startswith("uniform_") and "_regular_" in c for c in cols)
    has_comb_adap_l = any(c.startswith("level_") and "_adaptive_" in c for c in cols)
    has_comb_reg_l = any(c.startswith("level_") and "_regular_" in c for c in cols)
    if (has_comb_adap_u and has_comb_reg_u) or (has_comb_adap_l and has_comb_reg_l):
        return "combined"
    has_lvl = any(c.startswith("level_") for c in cols)
    has_uniform = any(c.startswith("uniform_") for c in cols)
    if has_lvl and has_uniform:
        return "adaptive_only"
    if has_uniform and not has_lvl:
        return "regular_only"
    return "unknown"


def _adaptive_col(schema: str, metric: str, scope: str, field: str) -> str:
    """Return the adaptive-track column name for the given scope/metric/field.

    scope: 'uniform' or 'level_{L}'.
    """
    if schema == "combined":
        return f"{scope}_{metric}_adaptive_{field}"
    return f"{scope}_{metric}_{field}"


def _regular_col(schema: str, metric: str, scope: str, field: str) -> str | None:
    if schema == "combined":
        return f"{scope}_{metric}_regular_{field}"
    if schema == "regular_only":
        # Regular CSV only has uniform scope.
        if scope != "uniform":
            return None
        return f"{scope}_{metric}_{field}"
    return None


def _pick_uniform_regular_ref(
    dfs: Sequence[pd.DataFrame],
    labels: Sequence[str],
    schemas: Sequence[str],
    regular_csv_df: pd.DataFrame | None,
    metric: str,
    fields: Sequence[str],
) -> tuple[pd.DataFrame, str] | None:
    """Return (df, source_label) whose ``uniform_{metric}_regular_{field}`` (or
    ``uniform_{metric}_{field}`` for standalone regular) is the baseline, or
    ``None`` if no baseline is available."""
    combined_idx = [i for i, s in enumerate(schemas) if s == "combined"]
    if combined_idx:
        ref_i = combined_idx[0]
        ref_df = dfs[ref_i]
        ref_label = labels[ref_i]
        # Cross-CSV disagreement warning (same regular model → should match).
        for j in combined_idx[1:]:
            other = dfs[j]
            for field in fields:
                col = f"uniform_{metric}_regular_{field}"
                if col not in ref_df.columns or col not in other.columns:
                    continue
                r = ref_df[col].to_numpy()
                o = other[col].to_numpy()
                denom = np.maximum(np.abs(r), 1e-30)
                rel = np.abs(o - r) / denom
                if rel.max() > _BASELINE_TOL:
                    print(
                        f"[warn] uniform regular baseline mismatch in '{col}' "
                        f"between '{ref_label}' and '{labels[j]}': "
                        f"max rel diff = {rel.max():.3e}. Using '{ref_label}'."
                    )
                    break
        return ref_df, ref_label
    if regular_csv_df is not None:
        return regular_csv_df, "regular_csv"
    return None


def _plot_uniform(
    dfs: Sequence[pd.DataFrame],
    labels: Sequence[str],
    schemas: Sequence[str],
    regular_ref: tuple[pd.DataFrame, str] | None,
    metric: str,
    fields: Sequence[str],
    out_path: str,
    log_y: bool,
) -> None:
    fig, axes = _panel_grid(len(fields))
    cmap = plt.get_cmap("tab10")
    adaptive_colors = [cmap(i % 10) for i in range(len(dfs))]

    for ax, field in zip(axes, fields):
        drew = False
        for df, lab, schema, color in zip(dfs, labels, schemas, adaptive_colors):
            col = _adaptive_col(schema, metric, "uniform", field)
            if col not in df.columns:
                continue
            ax.plot(
                df["Step"], df[col], label=lab, marker="o", markersize=4, color=color, linewidth=1.5
            )
            drew = True

        if regular_ref is not None:
            ref_df, ref_label = regular_ref
            ref_schema = (
                "combined"
                if f"uniform_{metric}_regular_{field}" in ref_df.columns
                else "regular_only"
            )
            col = _regular_col(ref_schema, metric, "uniform", field)
            if col is not None and col in ref_df.columns:
                ax.plot(
                    ref_df["Step"],
                    ref_df[col],
                    label=f"regular ({ref_label})",
                    marker="x",
                    markersize=5,
                    color="black",
                    linewidth=1.5,
                    linestyle="--",
                )
                drew = True

        if not drew:
            ax.set_visible(False)
            continue
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


def _plot_uniform_aggregate(
    dfs: Sequence[pd.DataFrame],
    labels: Sequence[str],
    schemas: Sequence[str],
    regular_ref: tuple[pd.DataFrame, str] | None,
    metric: str,
    fields: Sequence[str],
    out_path: str,
    log_y: bool,
) -> None:
    """Single-panel aggregate: each CSV's adaptive is the mean across ``fields``
    of its ``uniform_{metric}_adaptive_{field}``. Regular ref averaged the same
    way if present."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    cmap = plt.get_cmap("tab10")
    adaptive_colors = [cmap(i % 10) for i in range(len(dfs))]
    drew = False

    for df, lab, schema, color in zip(dfs, labels, schemas, adaptive_colors):
        cols = [_adaptive_col(schema, metric, "uniform", f) for f in fields]
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        mean_series = df[cols].mean(axis=1)
        ax.plot(
            df["Step"], mean_series, label=lab, marker="o", markersize=4, color=color, linewidth=1.5
        )
        drew = True

    if regular_ref is not None:
        ref_df, ref_label = regular_ref
        ref_schema = (
            "combined"
            if any(f"uniform_{metric}_regular_{f}" in ref_df.columns for f in fields)
            else "regular_only"
        )
        cols = [_regular_col(ref_schema, metric, "uniform", f) for f in fields]
        cols = [c for c in cols if c is not None and c in ref_df.columns]
        if cols:
            mean_series = ref_df[cols].mean(axis=1)
            ax.plot(
                ref_df["Step"],
                mean_series,
                label=f"regular ({ref_label})",
                marker="x",
                markersize=5,
                color="black",
                linewidth=1.5,
                linestyle="--",
            )
            drew = True

    if not drew:
        plt.close(fig)
        return

    ax.set_title(f"Uniform {metric} (mean over {len(fields)} fields)")
    ax.set_xlabel("Step")
    ax.set_ylabel(f"{metric} (field-mean)")
    ax.xaxis.set_major_locator(MultipleLocator(2))
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_level_aggregate(
    dfs: Sequence[pd.DataFrame],
    labels: Sequence[str],
    schemas: Sequence[str],
    metric: str,
    level: int,
    fields: Sequence[str],
    out_path: str,
    log_y: bool,
) -> None:
    """Single-panel aggregate per level: each CSV's adaptive+reg-projection is
    the field-mean of the per-level metric."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    cmap = plt.get_cmap("tab10")
    adaptive_colors = [cmap(i % 10) for i in range(len(dfs))]
    scope = f"level_{level}"
    drew = False

    for df, lab, schema, color in zip(dfs, labels, schemas, adaptive_colors):
        ad_cols = [_adaptive_col(schema, metric, scope, f) for f in fields]
        ad_cols = [c for c in ad_cols if c in df.columns]
        if ad_cols:
            ax.plot(
                df["Step"],
                df[ad_cols].mean(axis=1),
                label=f"{lab} (adap)",
                marker="o",
                markersize=4,
                color=color,
                linewidth=1.5,
            )
            drew = True
        rg_cols = [_regular_col(schema, metric, scope, f) for f in fields]
        rg_cols = [c for c in rg_cols if c is not None and c in df.columns]
        if rg_cols:
            ax.plot(
                df["Step"],
                df[rg_cols].mean(axis=1),
                label=f"{lab} (reg→qt)",
                marker="x",
                markersize=4,
                color=color,
                linewidth=1.2,
                linestyle="--",
            )
            drew = True

    if not drew:
        plt.close(fig)
        return

    ax.set_title(f"Level {level} {metric} (mean over {len(fields)} fields)")
    ax.set_xlabel("Step")
    ax.set_ylabel(f"{metric} (field-mean)")
    ax.xaxis.set_major_locator(MultipleLocator(2))
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_level(
    dfs: Sequence[pd.DataFrame],
    labels: Sequence[str],
    schemas: Sequence[str],
    metric: str,
    level: int,
    fields: Sequence[str],
    out_path: str,
    log_y: bool,
) -> None:
    """Per-level panel: each CSV contributes adaptive (solid) + its own regular
    projection (dashed, same color) if it was a combined CSV."""
    fig, axes = _panel_grid(len(fields))
    cmap = plt.get_cmap("tab10")
    adaptive_colors = [cmap(i % 10) for i in range(len(dfs))]
    scope = f"level_{level}"

    for ax, field in zip(axes, fields):
        drew = False
        for df, lab, schema, color in zip(dfs, labels, schemas, adaptive_colors):
            ad_col = _adaptive_col(schema, metric, scope, field)
            if ad_col in df.columns:
                ax.plot(
                    df["Step"],
                    df[ad_col],
                    label=f"{lab} (adap)",
                    marker="o",
                    markersize=4,
                    color=color,
                    linewidth=1.5,
                )
                drew = True
            rg_col = _regular_col(schema, metric, scope, field)
            if rg_col is not None and rg_col in df.columns:
                ax.plot(
                    df["Step"],
                    df[rg_col],
                    label=f"{lab} (reg→qt)",
                    marker="x",
                    markersize=4,
                    color=color,
                    linewidth=1.2,
                    linestyle="--",
                )
                drew = True
        if not drew:
            ax.set_visible(False)
            continue
        ax.set_title(f"Level {level} {metric}: {field}")
        ax.set_xlabel("Step")
        ax.set_ylabel(metric)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)

    for ax in axes[len(fields) :]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_multi_overlay(
    csv_paths: Sequence[str],
    out_dir: str,
    labels: Sequence[str] | None = None,
    regular_csv: str | None = None,
    out_prefix: str = "rollout_overlay",
    metrics: Iterable[str] = ("RMSE", "VRMSE"),
    fields: Iterable[str] = DEFAULT_FIELDS,
    levels: Iterable[int] | None = None,
    plot_uniform: bool = True,
    log_y: bool = False,
) -> list[str]:
    if not csv_paths:
        raise ValueError("At least one CSV path is required.")

    if labels is None:
        labels = [_shorten_label(p) for p in csv_paths]
    elif len(labels) != len(csv_paths):
        raise ValueError(f"Got {len(labels)} labels for {len(csv_paths)} CSVs; lengths must match.")
    labels = _dedupe_labels(labels)

    dfs = [pd.read_csv(p) for p in csv_paths]
    schemas = [_detect_schema(df) for df in dfs]
    for p, s in zip(csv_paths, schemas):
        if s == "unknown":
            raise ValueError(
                f"Could not detect schema for '{p}' — "
                "expected combined or standalone adaptive/regular CSV."
            )

    # Align steps across all inputs.
    common = set(dfs[0]["Step"].tolist())
    for df in dfs[1:]:
        common &= set(df["Step"].tolist())
    common_steps = sorted(common)
    if not common_steps:
        raise ValueError("No common Step values across CSVs — cannot align.")
    dfs = [
        df[df["Step"].isin(common_steps)].sort_values("Step").reset_index(drop=True) for df in dfs
    ]

    regular_csv_df: pd.DataFrame | None = None
    if regular_csv is not None:
        regular_csv_df = pd.read_csv(regular_csv)
        regular_csv_df = (
            regular_csv_df[regular_csv_df["Step"].isin(common_steps)]
            .sort_values("Step")
            .reset_index(drop=True)
        )

    # Auto-detect present levels when --levels not given.
    if levels is None:
        lvl_set = set()
        for df in dfs:
            for c in df.columns:
                m = re.match(r"^level_(\d+)_", c)
                if m:
                    lvl_set.add(int(m.group(1)))
        levels = sorted(lvl_set)

    # Auto-disable uniform plots if no CSV exposes uniform-scope columns
    # (e.g. AMReX combined rollouts run with compute_uniform_metrics=False).
    if plot_uniform:
        any_uniform = any(any(c.startswith("uniform_") for c in df.columns) for df in dfs)
        if not any_uniform:
            plot_uniform = False

    os.makedirs(out_dir, exist_ok=True)
    fields = list(fields)
    written: list[str] = []

    for metric in metrics:
        if plot_uniform:
            ref = _pick_uniform_regular_ref(dfs, labels, schemas, regular_csv_df, metric, fields)
            out = os.path.join(out_dir, f"{out_prefix}__uniform_{metric}.png")
            _plot_uniform(dfs, labels, schemas, ref, metric, fields, out, log_y)
            written.append(out)
            out_agg = os.path.join(out_dir, f"{out_prefix}__uniform_{metric}_agg.png")
            _plot_uniform_aggregate(dfs, labels, schemas, ref, metric, fields, out_agg, log_y)
            written.append(out_agg)

        for lvl in levels:
            present = any(
                any(c.startswith(f"level_{lvl}_{metric}_") for c in df.columns) for df in dfs
            )
            if not present:
                continue
            out = os.path.join(out_dir, f"{out_prefix}__level_{lvl}_{metric}.png")
            _plot_level(dfs, labels, schemas, metric, lvl, fields, out, log_y)
            written.append(out)
            out_agg = os.path.join(out_dir, f"{out_prefix}__level_{lvl}_{metric}_agg.png")
            _plot_level_aggregate(dfs, labels, schemas, metric, lvl, fields, out_agg, log_y)
            written.append(out_agg)

    return written


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "csvs", nargs="+", help="One or more rollout CSVs (combined or standalone adaptive)."
    )
    p.add_argument(
        "--labels", nargs="+", default=None, help="One label per CSV (defaults to filename stem)."
    )
    p.add_argument(
        "--regular-csv",
        default=None,
        help="Optional standalone regular CSV for the uniform baseline "
        "(used when no combined CSV is provided).",
    )
    p.add_argument("--out-dir", default="plots", help="Directory to write PNGs to.")
    p.add_argument(
        "--out-prefix", default="rollout_overlay", help="Filename prefix for output PNGs."
    )
    p.add_argument("--metrics", nargs="+", default=["RMSE", "VRMSE"])
    p.add_argument("--fields", nargs="+", default=DEFAULT_FIELDS)
    p.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=None,
        help="Per-level panels to generate (default: all present).",
    )
    p.add_argument(
        "--no-uniform",
        action="store_true",
        help="Skip uniform (full-field) panels. Default is to plot them.",
    )
    p.add_argument("--log-y", action="store_true", help="Use log scale on y-axis.")
    args = p.parse_args()

    written = generate_multi_overlay(
        csv_paths=args.csvs,
        labels=args.labels,
        regular_csv=args.regular_csv,
        out_dir=args.out_dir,
        out_prefix=args.out_prefix,
        metrics=args.metrics,
        fields=args.fields,
        levels=args.levels,
        plot_uniform=not args.no_uniform,
        log_y=args.log_y,
    )
    print(f"Wrote {len(written)} figure(s):")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()
