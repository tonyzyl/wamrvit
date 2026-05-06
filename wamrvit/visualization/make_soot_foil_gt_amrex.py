import argparse
import gc
import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import yt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm

from wamrvit.quad.yt_utils import make_regular_target_from_amrex, remap_amrex_to_regular

# Hard-coded defaults based on current AMReX setup.
DEFAULT_DATA_GLOB = (
    "PATH_TO_AMREX_PLOTFILES"
)
DEFAULT_OUTPUT_DIR = "animations/soot_foil_gt"
DEFAULT_FIELD_NAME = "pressure"
DEFAULT_PATCH_SIZE = 32
DEFAULT_START_IDX = 1
DEFAULT_END_IDX = None
DEFAULT_MAX_PIXELS = 4096
DEFAULT_PRESSURE_THRESHOLD = 2e5
DEFAULT_PLOT_FRAC = (0.82, 0.92, 0.01, 0.99)  # xmin, xmax, ymin, ymax
DEFAULT_DPI = 180


def _crop_image(image, extent, x_frac=None, y_frac=None):
    def _normalize_frac(frac):
        if frac is None:
            return 0.0, 1.0
        frac0, frac1 = float(frac[0]), float(frac[1])
        if frac1 <= frac0:
            raise ValueError("plot fraction max must be greater than min.")
        return frac0, frac1

    if x_frac is not None or y_frac is not None:
        height_img, width_img = image.shape
        x0_frac, x1_frac = _normalize_frac(x_frac)
        y0_frac, y1_frac = _normalize_frac(y_frac)

        x0 = max(0, min(int(np.floor(x0_frac * width_img)), width_img - 1))
        x1 = max(x0 + 1, min(int(np.ceil(x1_frac * width_img)), width_img))
        y0 = max(0, min(int(np.floor(y0_frac * height_img)), height_img - 1))
        y1 = max(y0 + 1, min(int(np.ceil(y1_frac * height_img)), height_img))

        image = image[y0:y1, x0:x1]

        xmin, xmax, ymin, ymax = extent
        dx = xmax - xmin
        dy = ymax - ymin
        extent = (
            xmin + x0_frac * dx,
            xmin + x1_frac * dx,
            ymin + y0_frac * dy,
            ymin + y1_frac * dy,
        )

    return image, extent


def _plot_soot_foil(peak_image, extent, out_path, threshold, dpi, shock_idx=None):
    vmax = 6e6
    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)

    im = ax.imshow(
        peak_image,
        origin="lower",
        extent=extent,
        cmap="gray_r",
        vmin=threshold,
        vmax=max(vmax, threshold),
        interpolation="nearest",
    )

    # ---- ADD THIS BLOCK ----
    if shock_idx is not None:
        xmin, xmax, ymin, ymax = extent
        width = peak_image.shape[1]

        # map index -> physical x location
        x_shock = xmin + (shock_idx / width) * (xmax - xmin)

        ax.axvline(x=x_shock, color="red", linewidth=2, linestyle="--")
    # ------------------------

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Physics simulation (ground truth): peak pressure soot foil", fontsize=13, pad=10)
    fig.colorbar(im, ax=ax, pad=0.02, label="Pressure")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_surface(peak_image, extent, out_path, zmin=0.6e6, zmax=6e6, dpi=180):
    xmin, xmax, ymin, ymax = extent
    ny, nx = peak_image.shape

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    Z = peak_image

    fig = plt.figure(figsize=(8, 6), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="viridis",
        vmin=zmin,
        vmax=zmax,
        linewidth=0,
        antialiased=False,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Pressure")
    ax.set_title("3D soot foil (pressure surface)")

    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Pressure")

    ax.set_zlim(zmin, zmax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Build GT soot-foil directly from AMReX plotfiles."
    )
    parser.add_argument("--data-glob", type=str, default=DEFAULT_DATA_GLOB)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--field-name", type=str, default=DEFAULT_FIELD_NAME)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--start-idx", type=int, default=DEFAULT_START_IDX)
    parser.add_argument("--end-idx", type=int, default=DEFAULT_END_IDX)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--pressure-threshold", type=float, default=DEFAULT_PRESSURE_THRESHOLD)
    parser.add_argument("--plot-frac", type=float, nargs=4, default=DEFAULT_PLOT_FRAC)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    args = parser.parse_args()

    start_time = time.time()
    all_paths = sorted(glob.glob(args.data_glob))
    if not all_paths:
        raise ValueError(f"No plotfiles found for glob: {args.data_glob}")

    start_idx = max(0, args.start_idx)
    end_idx = len(all_paths) if args.end_idx is None else min(len(all_paths), args.end_idx)
    if end_idx <= start_idx:
        raise ValueError(f"Invalid range [{start_idx}, {end_idx}) for {len(all_paths)} files")

    selected_paths = all_paths[start_idx:end_idx]
    os.makedirs(args.output_dir, exist_ok=True)

    x_frac = (args.plot_frac[0], args.plot_frac[1])
    y_frac = (args.plot_frac[2], args.plot_frac[3])

    peak_pressure = None
    extent = None

    print(f"Found {len(all_paths)} total plotfiles")
    print(f"Processing range [{start_idx}, {end_idx}) -> {len(selected_paths)} frames")
    print(
        f"Field: {args.field_name}, max_pixels: {args.max_pixels}, "
        f"threshold: {args.pressure_threshold}"
    )

    # Build one fixed regular target geometry from the first selected snapshot.
    ds_ref = yt.load(selected_paths[0])
    tgt_geometry = make_regular_target_from_amrex(ds_ref, domain_from="domain")
    tgt_le, tgt_dx, height, width = tgt_geometry
    base_extent = (
        float(tgt_le[0, 0]),
        float(tgt_le[0, 0] + width * tgt_dx[0, 0]),
        float(tgt_le[0, 1]),
        float(tgt_le[0, 1] + height * tgt_dx[0, 1]),
    )
    del ds_ref
    gc.collect()

    for file_path in tqdm(selected_paths, desc="Accumulating peak pressure"):
        ds = yt.load(file_path)
        regular = remap_amrex_to_regular(
            ds_src=ds,
            fields=[args.field_name],
            tgt_geometry=tgt_geometry,
            return_tensor=False,
        )
        frame_img = regular[0, 0]  # (H, W)
        frame_img, extent = _crop_image(frame_img, base_extent, x_frac=x_frac, y_frac=y_frac)

        if peak_pressure is None:
            peak_pressure = np.array(frame_img, copy=True)
        else:
            peak_pressure = np.maximum(peak_pressure, frame_img)

        del regular
        del ds
        del frame_img
        gc.collect()

    out_stem = f"gt_soot_foil_{args.field_name}_idx{start_idx}_to_{end_idx - 1}"
    npy_path = os.path.join(args.output_dir, f"{out_stem}.npy")
    png_path = os.path.join(args.output_dir, f"{out_stem}.png")

    np.save(npy_path, peak_pressure)
    # _plot_soot_foil(
    #     peak_image=peak_pressure,
    #     extent=extent,
    #     out_path=png_path,
    #     threshold=args.pressure_threshold,
    #     dpi=args.dpi,
    # )
    _plot_soot_foil(
        peak_image=peak_pressure,
        extent=extent,
        out_path=png_path,
        threshold=args.pressure_threshold,
        dpi=args.dpi,
        shock_idx=316,  # 80% of 396
    )

    surface_path = os.path.join(args.output_dir, f"{out_stem}_surface.png")

    _plot_surface(
        peak_image=peak_pressure,
        extent=extent,
        out_path=surface_path,
    )

    elapsed = time.time() - start_time
    print(f"Saved array to: {npy_path}")
    print(f"Saved figure to: {png_path}")
    print(f"Done in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
