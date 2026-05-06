"""Rollout overlay for N-way model comparisons with labelled dedup.

Each combined CSV (``rollout_combined.py`` schema) contributes

- one **adaptive** track (``uniform_{M}_adaptive_{field}``,
  ``level_{L}_{M}_adaptive_{field}``) — named by ``--adaptive-labels[i]``,
- one **regular** track (``uniform_{M}_regular_{field}``,
  ``level_{L}_{M}_regular_{field}``) — named by ``--regular-labels[i]``.

Labels are deduplicated by **first occurrence**: if two CSVs pass the same
adaptive or regular label, only the first CSV's track is drawn under that
label, and subsequent CSVs' same-labelled tracks are skipped. A warning is
emitted when same-labelled tracks disagree beyond ``--label-rtol``; this is
normal at per-level scopes because regular projections depend on each
adaptive run's quadtree topology.

Two emission modes:

- **aggregated** (default): field-mean, single panel per metric/scope.
- **per-field** (``--per-field``): one panel per field.

Title convention: ``uniform`` → ``Full-field``; ``level_{L}`` → ``Level {L}``,
or ``Finest level ($\\ell={L}$)`` when ``--finest-level {L}`` matches.

Usage:
    uv run wamrvit/visualization/plot_rollout_agg.py \\
        c1.csv c2.csv c3.csv \\
        --adaptive-labels "Adaptive (uniform)" "Adaptive (uniform)" "Adaptive (multi-scale)" \\
        --regular-labels  "Regular (finest)"   "Regular (mid)"     "Regular (finest)" \\
        --out-dir manuscript/figs --out-prefix pli_pf1 \\
        --finest-level 3 --levels 1 2 3     # drops level 0
"""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Iterable, Mapping, Sequence

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


def _scope_title(scope: str, finest_level: int | None) -> str:
    if scope == "uniform":
        return "Full-field"
    m = re.match(r"^level_(\d+)$", scope)
    if m:
        lvl = int(m.group(1))
        if finest_level is not None and lvl == finest_level:
            return rf"Finest level ($\ell={lvl}$)"
        if lvl == 0:
            return rf"Coarsest level ($\ell={lvl}$)"
        return f"Level {lvl}"
    return scope


def _mean_over_fields(
    df: pd.DataFrame, scope: str, metric: str, role: str, fields: Sequence[str]
) -> pd.Series | None:
    cols = [f"{scope}_{metric}_{role}_{f}" for f in fields]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return None
    return df[cols].mean(axis=1)


def _series(df: pd.DataFrame, scope: str, metric: str, role: str, field: str) -> pd.Series | None:
    col = f"{scope}_{metric}_{role}_{field}"
    return df[col] if col in df.columns else None


def _series_for(df, scope, metric, role, fields, field: str | None):
    """Dispatch: field=None → field-mean; else single field."""
    if field is None:
        return _mean_over_fields(df, scope, metric, role, fields)
    return _series(df, scope, metric, role, field)


def _check_label_agreement(
    per_label_series: dict[str, pd.Series],
    label: str,
    step_ref: pd.Series,
    candidate: pd.Series,
    scope: str,
    metric: str,
    role: str,
    source_label: str,
    rtol: float,
) -> None:
    """Warn if a new CSV's series with an already-registered label disagrees."""
    ref = per_label_series.get(label)
    if ref is None:
        return
    a = ref.to_numpy()
    b = candidate.to_numpy()
    if a.shape != b.shape:
        return
    denom = np.maximum(np.abs(a), 1e-30)
    if (np.abs(a - b) / denom).max() > rtol:
        print(
            f"[warn] {role} label '{label}' disagrees between sources at "
            f"{scope}/{metric} (using first occurrence; mismatch source: "
            f"{source_label})."
        )


def _draw_panel(
    ax,
    dfs: Sequence[pd.DataFrame],
    adaptive_labels: Sequence[str],
    regular_labels: Sequence[str],
    csv_sources: Sequence[str],
    metric: str,
    scope: str,
    fields: Sequence[str],
    field: str | None,
    rtol: float,
    log_y: bool,
    include_legend: bool,
    color_overrides: Mapping[str, object] | None = None,
    plot_regular: bool = True,
) -> bool:
    cmap = plt.get_cmap("tab10")

    # Assign a stable color per unique label in order of first occurrence,
    # honoring `color_overrides` when present (so the same variant renders in
    # the same color across PLI/AMR figures).
    color_for: dict[str, object] = {}
    legend_order: list[str] = []
    overrides = color_overrides or {}

    def _register(label: str):
        if label not in color_for:
            if label in overrides:
                color_for[label] = overrides[label]
            else:
                color_for[label] = cmap(len(color_for) % 10)
            legend_order.append(label)
        return color_for[label]

    # First pass to fix color ordering: adaptive labels first, then regular.
    for lab in adaptive_labels:
        _register(lab)
    if plot_regular:
        for lab in regular_labels:
            _register(lab)

    adap_seen: dict[str, pd.Series] = {}
    reg_seen: dict[str, pd.Series] = {}
    drew = False

    for i, df in enumerate(dfs):
        steps = df["Step"]
        alab = adaptive_labels[i]
        rlab = regular_labels[i]
        src = csv_sources[i]

        a_series = _series_for(df, scope, metric, "adaptive", fields, field)
        if a_series is not None:
            if alab not in adap_seen:
                ax.plot(
                    steps,
                    a_series,
                    label=alab,
                    color=color_for[alab],
                    marker="o",
                    markersize=4,
                    linewidth=1.8,
                )
                adap_seen[alab] = a_series
                drew = True
            else:
                _check_label_agreement(
                    adap_seen, alab, steps, a_series, scope, metric, "adaptive", src, rtol
                )

        if plot_regular:
            r_series = _series_for(df, scope, metric, "regular", fields, field)
            if r_series is not None:
                if rlab not in reg_seen:
                    ax.plot(
                        steps,
                        r_series,
                        label=rlab,
                        color=color_for[rlab],
                        marker="x",
                        markersize=5,
                        linewidth=1.5,
                        linestyle="--",
                    )
                    reg_seen[rlab] = r_series
                    drew = True
                else:
                    _check_label_agreement(
                        reg_seen, rlab, steps, r_series, scope, metric, "regular", src, rtol
                    )

    if not drew:
        return False

    ax.set_xlabel("rollout step")
    ax.set_ylabel(metric)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    if include_legend:
        # Enforce legend order: adaptive labels (deduped, first occurrence),
        # then regular labels (deduped, first occurrence).
        desired = []
        ordered_labels = list(adaptive_labels)
        if plot_regular:
            ordered_labels = ordered_labels + list(regular_labels)
        for lab in ordered_labels:
            if lab not in desired:
                desired.append(lab)
        handles, labels = ax.get_legend_handles_labels()
        label_to_handle = dict(zip(labels, handles))
        ordered = [(label_to_handle[lbl], lbl) for lbl in desired if lbl in label_to_handle]
        if ordered:
            h, lbls = zip(*ordered)
            ax.legend(h, lbls, fontsize=8)
        else:
            ax.legend(fontsize=8)
    return True


def _plot_aggregated(
    dfs,
    adaptive_labels,
    regular_labels,
    csv_sources,
    metric,
    scope,
    fields,
    finest_level,
    rtol,
    log_y,
    out,
    color_overrides=None,
    ymax=None,
    plot_regular=True,
):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ok = _draw_panel(
        ax,
        dfs,
        adaptive_labels,
        regular_labels,
        csv_sources,
        metric,
        scope,
        fields,
        None,
        rtol,
        log_y,
        True,
        color_overrides=color_overrides,
        plot_regular=plot_regular,
    )
    if not ok:
        plt.close(fig)
        return False
    ax.set_title(f"Aggregated {_scope_title(scope, finest_level).lower()} {metric}")
    if ymax is not None:
        ax.set_ylim(top=ymax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return True


def _panel_grid(n: int):
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    return fig, axes.flatten()


def _plot_per_field(
    dfs,
    adaptive_labels,
    regular_labels,
    csv_sources,
    metric,
    scope,
    fields,
    finest_level,
    rtol,
    log_y,
    out,
    color_overrides=None,
    field_groups=None,
    plot_regular=True,
):
    # ``field_groups`` overrides ``fields``: each (label, members) tuple becomes
    # one panel whose curve is the mean over `members`. When None, falls back
    # to one panel per field.
    if field_groups is None:
        groups = [(f, [f]) for f in fields]
    else:
        groups = list(field_groups)
    fig, axes = _panel_grid(len(groups))
    any_drew = False
    for ax, (label, members) in zip(axes, groups):
        ok = _draw_panel(
            ax,
            dfs,
            adaptive_labels,
            regular_labels,
            csv_sources,
            metric,
            scope,
            list(members),
            None,
            rtol,
            log_y,
            True,
            color_overrides=color_overrides,
            plot_regular=plot_regular,
        )
        if not ok:
            ax.set_visible(False)
            continue
        ax.set_title(f"{_scope_title(scope, finest_level)} {metric}: {label}")
        any_drew = True
    for ax in axes[len(groups) :]:
        ax.set_visible(False)
    if not any_drew:
        plt.close(fig)
        return False
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def generate(
    csv_paths: Sequence[str],
    adaptive_labels: Sequence[str],
    regular_labels: Sequence[str],
    out_dir: str,
    out_prefix: str,
    metrics: Iterable[str] = ("VRMSE", "RMSE"),
    fields: Iterable[str] = DEFAULT_FIELDS,
    levels: Iterable[int] | None = None,
    finest_level: int | None = None,
    plot_uniform: bool = True,
    per_field: bool = False,
    log_y: bool = False,
    label_rtol: float = 1e-2,
    color_overrides: Mapping[str, object] | None = None,
    field_groups: Sequence[tuple[str, Sequence[str]]] | None = None,
    ylim_max: Mapping[str, float] | None = None,
    plot_regular: bool = True,
) -> list[str]:
    n = len(csv_paths)
    if len(adaptive_labels) != n or len(regular_labels) != n:
        raise ValueError(
            f"Got {n} CSVs but {len(adaptive_labels)} adaptive labels and "
            f"{len(regular_labels)} regular labels; all three lengths must match."
        )

    dfs = [pd.read_csv(p) for p in csv_paths]
    common = set(dfs[0]["Step"].tolist())
    for df in dfs[1:]:
        common &= set(df["Step"].tolist())
    common = sorted(common)
    if not common:
        raise ValueError("No common Step values across CSVs.")
    dfs = [df[df["Step"].isin(common)].sort_values("Step").reset_index(drop=True) for df in dfs]

    if levels is None:
        lvl_set = set()
        for df in dfs:
            for c in df.columns:
                m = re.match(r"^level_(\d+)_", c)
                if m:
                    lvl_set.add(int(m.group(1)))
        levels = sorted(lvl_set)

    os.makedirs(out_dir, exist_ok=True)
    fields = list(fields)
    written: list[str] = []

    scopes: list[tuple[str, str]] = []
    if plot_uniform:
        scopes.append(("uniform", "uniform"))
    for lvl in levels:
        scopes.append((f"level_{lvl}", f"level_{lvl}"))

    csv_sources = [os.path.basename(p) for p in csv_paths]

    for metric in metrics:
        for scope, tag in scopes:
            out = os.path.join(out_dir, f"{out_prefix}__{tag}_{metric}.png")
            if per_field:
                ok = _plot_per_field(
                    dfs,
                    list(adaptive_labels),
                    list(regular_labels),
                    csv_sources,
                    metric,
                    scope,
                    fields,
                    finest_level,
                    label_rtol,
                    log_y,
                    out,
                    color_overrides=color_overrides,
                    field_groups=field_groups,
                    plot_regular=plot_regular,
                )
            else:
                ymax = None
                if ylim_max is not None:
                    ymax = ylim_max.get(f"{scope}_{metric}")
                ok = _plot_aggregated(
                    dfs,
                    list(adaptive_labels),
                    list(regular_labels),
                    csv_sources,
                    metric,
                    scope,
                    fields,
                    finest_level,
                    label_rtol,
                    log_y,
                    out,
                    color_overrides=color_overrides,
                    ymax=ymax,
                    plot_regular=plot_regular,
                )
            if ok:
                written.append(out)
    return written


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("csvs", nargs="+")
    p.add_argument(
        "--adaptive-labels",
        nargs="+",
        required=True,
        help="One adaptive label per CSV. Same-label CSVs dedupe on first.",
    )
    p.add_argument(
        "--regular-labels",
        nargs="+",
        required=True,
        help="One regular label per CSV. Same-label CSVs dedupe on first.",
    )
    p.add_argument(
        "--label-rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for same-label value-agreement warning.",
    )
    p.add_argument("--out-dir", default="plots")
    p.add_argument("--out-prefix", default="rollout_agg")
    p.add_argument("--metrics", nargs="+", default=["VRMSE", "RMSE"])
    p.add_argument("--fields", nargs="+", default=DEFAULT_FIELDS)
    p.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=None,
        help="Per-level panels to emit (default: all present). "
        "Pass explicit list to drop some (e.g. '--levels 1 2 3').",
    )
    p.add_argument("--finest-level", type=int, default=None)
    p.add_argument("--per-field", action="store_true")
    p.add_argument("--no-uniform", action="store_true")
    p.add_argument("--log-y", action="store_true")
    args = p.parse_args()

    written = generate(
        csv_paths=args.csvs,
        adaptive_labels=args.adaptive_labels,
        regular_labels=args.regular_labels,
        out_dir=args.out_dir,
        out_prefix=args.out_prefix,
        metrics=args.metrics,
        fields=args.fields,
        levels=args.levels,
        finest_level=args.finest_level,
        plot_uniform=not args.no_uniform,
        per_field=args.per_field,
        log_y=args.log_y,
        label_rtol=args.label_rtol,
    )
    print(f"Wrote {len(written)} figure(s):")
    for w in written:
        print(f"  {w}")


if __name__ == "__main__":
    main()
