"""Offline array-vs-object regrid comparison report. Reads two run output dirs
(produced by rollouts run with regrid_backend=object and =array) and reports regrid
speedup (+ regrid-as-%-of-rollout) and accuracy-vs-GT delta. No model / GPU."""
import argparse
import csv
import glob
import os

import pandas as pd


def _timing_files(run_dir, backend):
    return sorted(glob.glob(os.path.join(run_dir, f"regrid_timing_{backend}_*.csv")))


def _read_timing_rows(run_dir, backend):
    rows = []
    for path in _timing_files(run_dir, backend):
        with open(path) as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def _read_rollout_total(run_dir, backend):
    total = 0.0
    for path in sorted(glob.glob(os.path.join(run_dir, f"regrid_rollout_total_{backend}_*.csv"))):
        with open(path) as f:
            total += sum(float(r["total_s"]) for r in csv.DictReader(f))
    return total


def summarize_timing(object_dir, array_dir):
    obj = _read_timing_rows(object_dir, "object")
    arr = _read_timing_rows(array_dir, "array")
    obj_total = sum(float(r["seconds"]) for r in obj)
    arr_total = sum(float(r["seconds"]) for r in arr)
    n_fallback = sum(1 for r in arr if r["backend_used"] == "object")
    obj_rollout = _read_rollout_total(object_dir, "object")
    arr_rollout = _read_rollout_total(array_dir, "array")
    return {
        "object_total_s": obj_total,
        "array_total_s": arr_total,
        "speedup": (obj_total / arr_total) if arr_total > 0 else float("inf"),
        "object_n_calls": len(obj),
        "array_n_calls": len(arr),
        "array_n_fallback": n_fallback,
        "object_rollout_total_s": obj_rollout,
        "array_rollout_total_s": arr_rollout,
        # regrid as a fraction of total rollout wall-time -> contextualizes the speedup
        # (a 10x regrid speedup on 2% of runtime is ~1.8% end-to-end).
        "object_regrid_fraction": (obj_total / obj_rollout) if obj_rollout > 0 else None,
        "array_regrid_fraction": (arr_total / arr_rollout) if arr_rollout > 0 else None,
        # file lists make stale-file double-counting visible (re-runs into the same dir
        # leave extra pid-keyed files); the report prints these.
        "object_files": _timing_files(object_dir, "object"),
        "array_files": _timing_files(array_dir, "array"),
    }


def accuracy_delta(object_csv, array_csv):
    """Per-Step ``array - object`` delta for every shared numeric metric column, joined on
    the 'Step' identity column (NOT by row position). Both metric CSVs must be the same
    checkpoint/data, differing only in regrid backend. Raises if the two runs do not cover
    the same set of Steps -- a partial/misaligned run must fail loudly rather than silently
    emit spurious deltas, since this delta is the array-vs-object comparison reference."""
    obj = pd.read_csv(object_csv)
    arr = pd.read_csv(array_csv)
    if "Step" not in obj.columns or "Step" not in arr.columns:
        raise ValueError("accuracy_delta expects a 'Step' column in both metric CSVs")
    if set(obj["Step"]) != set(arr["Step"]):
        raise ValueError(
            f"Step mismatch between runs (object={sorted(obj['Step'])}, "
            f"array={sorted(arr['Step'])}); cannot compare misaligned rollouts")
    merged = obj.merge(arr, on="Step", suffixes=("_obj", "_arr")).sort_values("Step")
    out = {"Step": merged["Step"].to_numpy()}
    for c in obj.columns:
        if c == "Step" or c not in arr.columns:
            continue
        co, ca = f"{c}_obj", f"{c}_arr"
        if pd.api.types.is_numeric_dtype(merged[co]) and pd.api.types.is_numeric_dtype(merged[ca]):
            out[f"{c}_delta"] = merged[ca].to_numpy() - merged[co].to_numpy()
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser(description="Array-vs-object native regrid comparison report")
    p.add_argument("--object_dir", required=True)
    p.add_argument("--array_dir", required=True)
    p.add_argument("--object_csv", help="object-run metric CSV (accuracy)")
    p.add_argument("--array_csv", help="array-run metric CSV (accuracy)")
    args = p.parse_args()

    print("=== Regrid backend comparison ===")
    t = summarize_timing(args.object_dir, args.array_dir)
    # Visibility: list the timing files aggregated, so stale-file double-counting on a
    # re-run into the same dir is obvious (expect one file per actor PID).
    print(f"object timing files ({len(t['object_files'])}): "
          f"{[os.path.basename(f) for f in t['object_files']]}")
    print(f"array  timing files ({len(t['array_files'])}): "
          f"{[os.path.basename(f) for f in t['array_files']]}")
    print(f"object regrid: {t['object_total_s']:.4f}s over {t['object_n_calls']} calls", end="")
    if t["object_regrid_fraction"] is not None:
        print(f"  ({100 * t['object_regrid_fraction']:.2f}% of rollout)", end="")
    print()
    print(f"array  regrid: {t['array_total_s']:.4f}s over {t['array_n_calls']} calls "
          f"({t['array_n_fallback']} fell back to object)", end="")
    if t["array_regrid_fraction"] is not None:
        print(f"  ({100 * t['array_regrid_fraction']:.2f}% of rollout)", end="")
    print()
    print(f"SPEEDUP (object/array regrid): {t['speedup']:.2f}x")
    if t["object_regrid_fraction"] is None or t["array_regrid_fraction"] is None:
        print("  (rollout-total files missing -> cannot contextualize as % of rollout; "
              "regrid speedup alone can overstate end-to-end impact)")

    if args.object_csv and args.array_csv:
        print("\n--- accuracy delta (array - object) ---")
        print(accuracy_delta(args.object_csv, args.array_csv).to_string(index=False))
        print("(sanity: in combined runs the regular-side columns should be ~0 delta -- "
              "only the native regrid changed)")


if __name__ == "__main__":
    main()
