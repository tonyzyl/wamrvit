import argparse
import os

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


def _load_array(path: str, key: str) -> np.ndarray:
    """
    Load a numpy array from .npy or .npz (with a named key).
    """
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        with np.load(path) as data:
            if key not in data:
                raise KeyError(f"Key '{key}' not found in {path}. Available: {list(data.keys())}")
            return data[key]
    raise ValueError(f"Unsupported file type for {path}. Use .npy or .npz.")


def _load_fields_from_config(cfg_path: str) -> list[str]:
    """
    Read field names from a config file. Prefer data.field_names, then
    file_loader.params.field_names.
    """
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    fields = cfg.get("data", {}).get("field_names")
    if fields:
        return list(fields)
    fields = cfg.get("file_loader", {}).get("params", {}).get("field_names")
    if fields:
        return list(fields)
    raise ValueError(
        f"No field names found in {cfg_path} "
        "(expected data.field_names or file_loader.params.field_names)."
    )


def _ensure_bcthw(arr: np.ndarray) -> np.ndarray:
    """
    Normalize array to (B, C, T, H, W).
    """
    if arr.ndim == 5:
        return arr
    if arr.ndim == 4:
        # Assume (C, T, H, W) and add batch.
        return np.expand_dims(arr, 0)
    raise ValueError(f"Expected array with 4 or 5 dims (C,T,H,W) or (B,C,T,H,W), got {arr.shape}.")


def _relative_l2(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> float:
    """
    Relative L2 norm: ||gt - pred||_2 / ||gt||_2
    """
    diff = gt - pred
    num = np.linalg.norm(diff.ravel(), ord=2)
    den = np.linalg.norm(gt.ravel(), ord=2)
    return float(num / (den + eps))


def _linf(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    L-infinity norm: max |gt - pred|
    """
    return float(np.max(np.abs(gt - pred)))


def _shock_location_x(pressure: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute shock location along x for each y-row using max pressure gradient.

    pressure: (H, W) array.
    Returns: (H,) array of x-locations (float indices scaled by dx).
    """
    # dP/dx along the last axis.
    dpdx = np.gradient(pressure, dx, axis=-1)
    # Argmax over x for each y (row).
    idx = np.argmax(np.abs(dpdx), axis=-1)
    return idx.astype(np.float32) * dx


def _hrr_location_x(hrr: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute location of max heat release along x for each y-row.
    """
    idx = np.argmax(hrr, axis=-1)
    return idx.astype(np.float32) * dx


def _mass_energy(rho: np.ndarray, rhoE: np.ndarray, dx: float, dy: float) -> tuple[float, float]:
    """
    Compute total mass and total (rhoE) energy by simple Riemann sum.
    """
    cell_area = dx * dy
    mass = float(np.sum(rho) * cell_area)
    energy = float(np.sum(rhoE) * cell_area)
    return mass, energy


def compute_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    fields: list[str],
    dx: float,
    dy: float,
    pressure_field: str,
    hrr_field: str,
    rho_field: str,
    rhoE_field: str,
) -> pd.DataFrame:
    """
    Compute field errors, shock error, mass/energy conservation, and induction length error.
    Returns a DataFrame with one row per time step (averaged over batch).
    """
    gt = _ensure_bcthw(gt)
    pred = _ensure_bcthw(pred)
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt {gt.shape} vs pred {pred.shape}")

    B, C, T, H, W = gt.shape
    field_to_idx = {f: i for i, f in enumerate(fields)}

    # Validate required fields.
    for required in [pressure_field, hrr_field, rho_field, rhoE_field]:
        if required not in field_to_idx:
            raise ValueError(f"Required field '{required}' not found in fields list.")

    p_idx = field_to_idx[pressure_field]
    h_idx = field_to_idx[hrr_field]
    rho_idx = field_to_idx[rho_field]
    rhoE_idx = field_to_idx[rhoE_field]

    rows = []
    for t in range(T):
        row: dict[str, float] = {"step": t + 1}

        # Field error metrics.
        for f, c in field_to_idx.items():
            l2_vals = []
            linf_vals = []
            for b in range(B):
                gt_bt = gt[b, c, t]
                pred_bt = pred[b, c, t]
                l2_vals.append(_relative_l2(gt_bt, pred_bt))
                linf_vals.append(_linf(gt_bt, pred_bt))
            row[f"l2_{f}"] = float(np.mean(l2_vals))
            row[f"linf_{f}"] = float(np.mean(linf_vals))

        # Shock location error (RMSE across y, averaged across batch).
        shock_errs = []
        induction_errs = []
        mass_errs = []
        energy_errs = []
        for b in range(B):
            gt_p = gt[b, p_idx, t]
            pred_p = pred[b, p_idx, t]
            gt_h = gt[b, h_idx, t]
            pred_h = pred[b, h_idx, t]

            gt_shock_x = _shock_location_x(gt_p, dx)
            pred_shock_x = _shock_location_x(pred_p, dx)
            shock_err = np.sqrt(np.mean((gt_shock_x - pred_shock_x) ** 2))
            shock_errs.append(float(shock_err))

            # Induction length error: (x_shock - x_HRR) difference per y.
            gt_hrr_x = _hrr_location_x(gt_h, dx)
            pred_hrr_x = _hrr_location_x(pred_h, dx)
            gt_ind = gt_shock_x - gt_hrr_x
            pred_ind = pred_shock_x - pred_hrr_x
            ind_err = np.sqrt(np.mean((gt_ind - pred_ind) ** 2))
            induction_errs.append(float(ind_err))

            # Mass and energy conservation errors (absolute differences).
            gt_rho = gt[b, rho_idx, t]
            pred_rho = pred[b, rho_idx, t]
            gt_rhoE = gt[b, rhoE_idx, t]
            pred_rhoE = pred[b, rhoE_idx, t]
            gt_m, gt_E = _mass_energy(gt_rho, gt_rhoE, dx, dy)
            pred_m, pred_E = _mass_energy(pred_rho, pred_rhoE, dx, dy)
            mass_errs.append(abs(gt_m - pred_m))
            energy_errs.append(abs(gt_E - pred_E))

        row["shock_rmse"] = float(np.mean(shock_errs))
        row["induction_rmse"] = float(np.mean(induction_errs))
        row["mass_abs_err"] = float(np.mean(mass_errs))
        row["energy_abs_err"] = float(np.mean(energy_errs))

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute detonation metrics from rollout predictions."
    )
    parser.add_argument(
        "--pred", required=True, help="Path to prediction array (.npy or .npz with key 'pred')."
    )
    parser.add_argument(
        "--target", required=True, help="Path to target array (.npy or .npz with key 'target')."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Config file with field names (e.g., configs/amrex_ar.yaml).",
    )
    parser.add_argument("--dx", type=float, default=1.0, help="Grid spacing in x.")
    parser.add_argument("--dy", type=float, default=1.0, help="Grid spacing in y.")
    parser.add_argument("--pressure-field", default="pressure", help="Field name for pressure.")
    parser.add_argument("--hrr-field", default="HRR", help="Field name for heat release.")
    parser.add_argument("--rho-field", default="rho", help="Field name for density.")
    parser.add_argument("--rhoE-field", default="rhoE", help="Field name for total energy density.")
    parser.add_argument("--output-csv", default="detonation_metrics.csv", help="Output CSV path.")
    args = parser.parse_args()

    fields = _load_fields_from_config(args.config)
    pred = _load_array(args.pred, key="pred")
    target = _load_array(args.target, key="target")

    df = compute_metrics(
        gt=target,
        pred=pred,
        fields=fields,
        dx=args.dx,
        dy=args.dy,
        pressure_field=args.pressure_field,
        hrr_field=args.hrr_field,
        rho_field=args.rho_field,
        rhoE_field=args.rhoE_field,
    )

    out_path = os.path.abspath(args.output_csv)
    df.to_csv(out_path, index=False)
    print(f"Wrote metrics to {out_path}")


if __name__ == "__main__":
    main()
