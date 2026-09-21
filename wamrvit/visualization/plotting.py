"""
Visualization functions for trajectory data
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from wamrvit.quad.quadtree import Quadtree


def compute_vrange(gt_arrays, pred_arrays, field_label, anim_cfg):
    """Shared (vmin, vmax) for GT and Pred animation colorbars.

    Outlier-robust alternative to raw min/max, which lets a single blown-up AR
    frame flatten the GT's contrast to a single color.

    Priority:
      1. Explicit `anim_vmin` / `anim_vmax` in ``anim_cfg`` — scalar applied to
         all fields, or ``{field_label: value}`` dict for per-field override.
      2. Percentile clip over concatenated GT + Pred values via
         `anim_vrange_percentile` (default ``[0.5, 99.5]``; ``None`` or
         ``[0, 100]`` disables clipping and reverts to full min/max).

    Parameters
    ----------
    gt_arrays, pred_arrays : iterable of np.ndarray
        Arbitrary-shape arrays covering all frames to be rendered. Flattened
        and concatenated internally.
    field_label : str
        Field name used to look up per-field overrides.
    anim_cfg : Mapping
        The ``animation`` config subtree.
    """

    def _override(key):
        v = anim_cfg.get(key, None)
        if v is None:
            return None
        if isinstance(v, dict):
            return v.get(field_label, None)
        return v

    pct = anim_cfg.get("anim_vrange_percentile", [0.5, 99.5])
    if pct is None:
        pct = [0, 100]
    lo_pct, hi_pct = float(pct[0]), float(pct[1])

    flat = np.concatenate([np.asarray(a).ravel() for a in list(gt_arrays) + list(pred_arrays)])
    auto_vmin = float(np.quantile(flat, lo_pct / 100.0))
    auto_vmax = float(np.quantile(flat, hi_pct / 100.0))

    vmin_o = _override("anim_vmin")
    vmax_o = _override("anim_vmax")
    vmin = auto_vmin if vmin_o is None else float(vmin_o)
    vmax = auto_vmax if vmax_o is None else float(vmax_o)
    return vmin, vmax


def visualize_channels(channels_data, channel_names=None, time_slice=None, figsize=(15, 10)):
    """
    Visualize all channels at a specific time slice

    Parameters:
    channels_data (dict): Dictionary with channel data arrays
    channel_names (list): List of channel names to visualize
    time_slice (int): Time slice to visualize (default: middle)
    figsize (tuple): Figure size
    """
    if channel_names is None:
        channel_names = list(channels_data.keys())

    # Determine time slice
    if time_slice is None:
        first_channel = list(channels_data.values())[0]
        time_slice = first_channel.shape[2] // 2

    # Create subplot layout
    n_channels = len(channel_names)
    cols = 3
    rows = (n_channels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    print(f"Visualizing time slice {time_slice}")

    for i, channel_name in enumerate(channel_names):
        if channel_name in channels_data:
            channel_data = channels_data[channel_name][:, :, time_slice]

            im = axes[i].imshow(channel_data, cmap="RdBu_r", aspect="auto", origin="lower")
            axes[i].set_title(
                f"{channel_name}\\n"
                f"(min: {np.min(channel_data):.3f}, max: {np.max(channel_data):.3f})"
            )
            axes[i].set_xlabel("Width")
            axes[i].set_ylabel("Height")
            plt.colorbar(im, ax=axes[i], shrink=0.8)
        else:
            axes[i].text(
                0.5,
                0.5,
                f"Channel {channel_name}\\nnot found",
                ha="center",
                va="center",
                transform=axes[i].transAxes,
            )
            axes[i].set_title(f"{channel_name} (missing)")

    # Remove empty subplots
    for j in range(len(channel_names), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def visualize_timestep(trajectory_data, t, channel_names=None, figsize=(15, 10)):
    """
    Visualize all channels at a specific time step

    Parameters:
    trajectory_data (numpy.ndarray): 4D trajectory array (height, width, channels, time)
    t (int): Time step to visualize
    channel_names (list): List of channel names
    figsize (tuple): Figure size
    """
    if channel_names is None:
        channel_names = ["rho", "u", "v", "p", "ogm1"]

    n_channels = min(len(channel_names), trajectory_data.shape[2])
    cols = 3
    rows = (n_channels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i in range(n_channels):
        channel_data = trajectory_data[:, :, i, t]
        channel_name = channel_names[i] if i < len(channel_names) else f"Channel_{i}"

        im = axes[i].imshow(channel_data, cmap="RdBu_r", aspect="auto", origin="lower")
        axes[i].set_title(
            f"{channel_name} at t={t}\\n"
            f"(min: {np.min(channel_data):.3f}, max: {np.max(channel_data):.3f})"
        )
        axes[i].set_xlabel("Width")
        axes[i].set_ylabel("Height")
        plt.colorbar(im, ax=axes[i], shrink=0.8)

    # Remove empty subplots
    for j in range(n_channels, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_temporal_evolution(trajectory_data, x, y, channel_names=None, figsize=(12, 8)):
    """
    Plot temporal evolution of all channels at a specific (x,y) location

    Parameters:
    trajectory_data (numpy.ndarray): 4D trajectory array or channels_data dict
    x, y (int): Spatial coordinates
    channel_names (list): List of channel names
    figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)

    if isinstance(trajectory_data, dict):
        # If it's a channels_data dictionary
        for channel_name, data in trajectory_data.items():
            temporal_data = data[y, x, :]
            plt.plot(temporal_data, label=channel_name, linewidth=2)
    else:
        # If it's a 4D trajectory array
        if channel_names is None:
            channel_names = ["rho", "u", "v", "p", "ogm1"]

        for i, channel_name in enumerate(channel_names):
            if i < trajectory_data.shape[2]:
                temporal_data = trajectory_data[y, x, i, :]
                plt.plot(temporal_data, label=channel_name, linewidth=2)

    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.gca().xaxis.set_major_locator(MultipleLocator(2))
    plt.title(f"Temporal Evolution at Location ({x}, {y})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def compare_spatial_evolution(channels_data, channel_name, points=None, figsize=(12, 8)):
    """
    Compare temporal evolution at different spatial locations for a single channel

    Parameters:
    channels_data (dict): Dictionary with channel data
    channel_name (str): Name of channel to analyze
    points (list): List of (x, y, label) tuples for comparison points
    figsize (tuple): Figure size
    """
    if channel_name not in channels_data:
        print(f"Channel '{channel_name}' not found!")
        return

    data = channels_data[channel_name]

    if points is None:
        # Default comparison points
        center_x, center_y = data.shape[1] // 2, data.shape[0] // 2
        points = [
            (center_x, center_y, "Center"),
            (data.shape[1] // 4, data.shape[0] // 4, "Bottom-left"),
            (3 * data.shape[1] // 4, 3 * data.shape[0] // 4, "Top-right"),
            (center_x, data.shape[0] // 4, "Bottom-center"),
            (center_x, 3 * data.shape[0] // 4, "Top-center"),
        ]

    plt.figure(figsize=figsize)

    for x, y, label in points:
        temporal_data = data[y, x, :]
        plt.plot(temporal_data, label=f"{label} ({x},{y})", linewidth=2)

    plt.title(f"{channel_name} - Spatial Comparison")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.gca().xaxis.set_major_locator(MultipleLocator(2))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_quadtree(
    qt: "Quadtree",
    ax=None,
    cmap_name="jet",
    title=None,
    bg=None,
    outline_color="w",
    outline_width=0.6,
    domain_color="k",
    channel: int | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = False,
    max_pixels: int = 2048,  # Target resolution bound for the FRB
    draw_outlines: bool = True,  # Added toggle to prevent Matplotlib choke
    x_frac: tuple[float, float] | None = None,
    y_frac: tuple[float, float] | None = None,
    cax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))  # slightly larger default for detail
    else:
        fig = ax.figure

    im = None

    def _normalize_frac(frac, name):
        if frac is None:
            return 0.0, 1.0
        if len(frac) != 2:
            raise ValueError(f"{name} must be a (min, max) tuple of length 2.")
        f0, f1 = float(frac[0]), float(frac[1])
        if not (0.0 <= f0 <= 1.0 and 0.0 <= f1 <= 1.0):
            raise ValueError(f"{name} values must be between 0 and 1.")
        if f1 <= f0:
            raise ValueError(f"{name} max must be greater than min.")
        return f0, f1

    x0_frac, x1_frac = _normalize_frac(x_frac, "x_frac")
    y0_frac, y1_frac = _normalize_frac(y_frac, "y_frac")
    use_crop = (x_frac is not None) or (y_frac is not None)

    # --- 1. Rasterize the Quadtree Content (Fixed Resolution Buffer) ---
    if channel is not None:
        if qt.channels <= 0:
            raise ValueError("Quadtree has no channels to plot.")

        ch = int(max(0, min(channel, qt.channels - 1)))

        L = qt.max_level_idx + 1
        Ph = getattr(qt, "patch_height", 1)
        Pw = getattr(qt, "patch_width", 1)

        # Logical grid dimensions in maximum virtual pixels
        grids_per_tile = 1 << L
        H_total_virtual = qt.ny_tiles * grids_per_tile * Ph
        W_total_virtual = qt.nx_tiles * grids_per_tile * Pw

        # Calculate scaling factor to restrict memory allocation to max_pixels
        max_virtual_dim = max(H_total_virtual, W_total_virtual)
        scale_down = min(1.0, max_pixels / max_virtual_dim)

        H_img = max(1, int(H_total_virtual * scale_down))
        W_img = max(1, int(W_total_virtual * scale_down))

        img = np.zeros((H_img, W_img), dtype=np.float32)

        # Paint each leaf using nearest-neighbor mapping
        for leaf in qt._iter_all_leaves():
            level, xi, yi = qt.cell_xy_index(leaf)
            scale = 1 << (L - level)

            # Virtual pixel coordinates
            lx0 = (leaf.tile_ix * grids_per_tile + xi * scale) * Pw
            ly0 = (leaf.tile_iy * grids_per_tile + yi * scale) * Ph
            lw = scale * Pw
            lh = scale * Ph

            # Map to target FRB image coordinates
            tx0 = int(lx0 * scale_down)
            ty0 = int(ly0 * scale_down)
            tx1 = int((lx0 + lw) * scale_down)
            ty1 = int((ly0 + lh) * scale_down)

            # Guarantee at least 1 pixel width/height to avoid dropping sub-pixel leaves
            tx0 = max(0, min(tx0, W_img - 1))
            ty0 = max(0, min(ty0, H_img - 1))
            tx1 = max(tx0 + 1, min(tx1, W_img))
            ty1 = max(ty0 + 1, min(ty1, H_img))

            th = ty1 - ty0
            tw = tx1 - tx0

            # Extract value. Native leaves store (C, Ph*s, Pw*s); uniform leaves
            # store (C, Ph, Pw). Resampling uses val_patch's real shape.
            val_patch = np.zeros((Ph, Pw), dtype=np.float32)
            if leaf.value is not None:
                v_all = np.asarray(leaf.value)
                if v_all.ndim == 3 and v_all.shape[0] > ch:
                    val_patch = v_all[ch]
                elif v_all.ndim == 2 and ch == 0:
                    val_patch = v_all

            vPh, vPw = val_patch.shape

            # Fast nearest-neighbor resampling arrays
            y_idx = np.clip((np.arange(th) * (vPh / th)).astype(int), 0, vPh - 1)
            x_idx = np.clip((np.arange(tw) * (vPw / tw)).astype(int), 0, vPw - 1)

            # Broadcast into the image buffer
            img[ty0:ty1, tx0:tx1] = val_patch[y_idx[:, None], x_idx]

        extent = (qt.xmin, qt.xmax, qt.ymin, qt.ymax)
        if use_crop:
            x0 = max(0, min(int(math.floor(x0_frac * W_img)), W_img - 1))
            x1 = max(x0 + 1, min(int(math.ceil(x1_frac * W_img)), W_img))
            y0 = max(0, min(int(math.floor(y0_frac * H_img)), H_img - 1))
            y1 = max(y0 + 1, min(int(math.ceil(y1_frac * H_img)), H_img))

            img = img[y0:y1, x0:x1]

            dx = qt.xmax - qt.xmin
            dy = qt.ymax - qt.ymin
            extent = (
                qt.xmin + x0_frac * dx,
                qt.xmin + x1_frac * dx,
                qt.ymin + y0_frac * dy,
                qt.ymin + y1_frac * dy,
            )

        im = ax.imshow(
            img,
            origin="lower",
            extent=extent,
            cmap=plt.get_cmap(cmap_name),
            interpolation="nearest",
            alpha=1.0,
            vmin=vmin,
            vmax=vmax,
        )

    # --- 2. Background Fallback ---
    elif bg is not None:
        extent = (qt.xmin, qt.xmax, qt.ymin, qt.ymax)
        if use_crop:
            dx = qt.xmax - qt.xmin
            dy = qt.ymax - qt.ymin
            extent = (
                qt.xmin + x0_frac * dx,
                qt.xmin + x1_frac * dx,
                qt.ymin + y0_frac * dy,
                qt.ymin + y1_frac * dy,
            )

        im = ax.imshow(
            bg,
            origin="lower",
            extent=extent,
            cmap=plt.get_cmap(cmap_name),
            interpolation="nearest",
            alpha=1.0,
            vmin=vmin,
            vmax=vmax,
        )

    # --- 3. Draw Outlines (QuadCells) ---
    # Warning: Adding 400k+ Rectangle patches will freeze Matplotlib.
    # Set draw_outlines=True only if you have a coarse quadtree.
    if draw_outlines:
        if use_crop:
            crop_xmin, crop_xmax = extent[0], extent[1]
            crop_ymin, crop_ymax = extent[2], extent[3]
        patches = []
        for (cx, cy, hx, hy), level, _ in qt.sample_leaves():
            x0, y0 = cx - hx, cy - hy
            x1, y1 = cx + hx, cy + hy
            if use_crop:
                if x1 <= crop_xmin or x0 >= crop_xmax or y1 <= crop_ymin or y0 >= crop_ymax:
                    continue
            patches.append(Rectangle((x0, y0), 2 * hx, 2 * hy))

        # PatchCollection is vastly faster than calling ax.add_patch in a loop
        pc = PatchCollection(
            patches, facecolor="none", edgecolor=outline_color, linewidth=outline_width
        )
        ax.add_collection(pc)

    # --- 4. Domain Boundary ---
    ax.add_patch(
        Rectangle(
            (qt.xmin, qt.ymin),
            qt.xmax - qt.xmin,
            qt.ymax - qt.ymin,
            fill=False,
            edgecolor=domain_color,
            linewidth=1.2,
        )
    )

    if colorbar and im:
        if cax is not None:
            fig.colorbar(im, cax=cax)
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.05)
            fig.colorbar(im, cax=cax)

    ax.set_aspect("equal", adjustable="box")
    if use_crop:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    else:
        ax.set_xlim(qt.xmin, qt.xmax)
        ax.set_ylim(qt.ymin, qt.ymax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)

    return fig, ax


def plot_quadtree_boundary(
    qt: "Quadtree",
    ax=None,
    cmap_name="jet",
    title=None,
    channel: int = 0,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    outline_color="w",
    outline_width=0.6,
    max_pixels: int = 2048,
    draw_outlines: bool = True,
):
    """Plot only boundary leaves; non-boundary leaves are masked to NaN."""
    raise RuntimeError(
        "plot_quadtree_boundary requires QuadCell.is_boundary to be set. "
        "mark_boundary_cells() calls were disabled because the flag is unread "
        "by the active training/inference pipeline. To use this function, "
        "call qt.mark_boundary_cells() explicitly first."
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    if qt.channels <= 0:
        raise ValueError("Quadtree has no channels to plot.")

    ch = int(max(0, min(channel, qt.channels - 1)))

    L = qt.max_level_idx + 1
    Ph = getattr(qt, "patch_height", 1)
    Pw = getattr(qt, "patch_width", 1)

    grids_per_tile = 1 << L
    H_total_virtual = qt.ny_tiles * grids_per_tile * Ph
    W_total_virtual = qt.nx_tiles * grids_per_tile * Pw

    max_virtual_dim = max(H_total_virtual, W_total_virtual)
    scale_down = min(1.0, max_pixels / max_virtual_dim)

    H_img = max(1, int(H_total_virtual * scale_down))
    W_img = max(1, int(W_total_virtual * scale_down))

    img = np.full((H_img, W_img), np.nan, dtype=np.float32)

    for leaf in qt._iter_all_leaves():
        if not leaf.is_boundary:
            continue

        level, xi, yi = qt.cell_xy_index(leaf)
        scale = 1 << (L - level)

        lx0 = (leaf.tile_ix * grids_per_tile + xi * scale) * Pw
        ly0 = (leaf.tile_iy * grids_per_tile + yi * scale) * Ph
        lw = scale * Pw
        lh = scale * Ph

        tx0 = max(0, min(int(lx0 * scale_down), W_img - 1))
        ty0 = max(0, min(int(ly0 * scale_down), H_img - 1))
        tx1 = max(tx0 + 1, min(int((lx0 + lw) * scale_down), W_img))
        ty1 = max(ty0 + 1, min(int((ly0 + lh) * scale_down), H_img))

        th = ty1 - ty0
        tw = tx1 - tx0

        val_patch = np.zeros((Ph, Pw), dtype=np.float32)
        if leaf.value is not None:
            v_all = np.asarray(leaf.value)
            if v_all.ndim == 3 and v_all.shape[0] > ch:
                val_patch = v_all[ch]
            elif v_all.ndim == 2 and ch == 0:
                val_patch = v_all

        y_idx = np.clip((np.arange(th) * (Ph / th)).astype(int), 0, Ph - 1)
        x_idx = np.clip((np.arange(tw) * (Pw / tw)).astype(int), 0, Pw - 1)
        img[ty0:ty1, tx0:tx1] = val_patch[y_idx[:, None], x_idx]

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="0.85")  # light gray for NaN (non-boundary)

    im = ax.imshow(
        img,
        origin="lower",
        extent=(qt.xmin, qt.xmax, qt.ymin, qt.ymax),
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    if draw_outlines:
        patches = []
        for leaf in qt._iter_all_leaves():
            if leaf.is_boundary:
                x0, y0 = leaf.cx - leaf.hx, leaf.cy - leaf.hy
                patches.append(Rectangle((x0, y0), 2 * leaf.hx, 2 * leaf.hy))
        if patches:
            pc = PatchCollection(
                patches,
                facecolor="none",
                edgecolor=outline_color,
                linewidth=outline_width,
            )
            ax.add_collection(pc)

    ax.add_patch(
        Rectangle(
            (qt.xmin, qt.ymin),
            qt.xmax - qt.xmin,
            qt.ymax - qt.ymin,
            fill=False,
            edgecolor="k",
            linewidth=1.2,
        )
    )

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        fig.colorbar(im, cax=cax)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(qt.xmin, qt.xmax)
    ax.set_ylim(qt.ymin, qt.ymax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)

    return fig, ax


def plot_npz_content(file_path, exclude_keys=None, cols_per_row=4, figsize_scale=3):
    """
    Robustly plots all arrays in an .npz file in a grid layout.

    Args:
        file_path (str): Path to the .npz file.
        exclude_keys (list): List of strings (keys) to ignore.
        cols_per_row (int): Number of columns in the subplot grid.
        figsize_scale (int): Multiplier for figure size to keep it readable.
    """
    if exclude_keys is None:
        exclude_keys = []

    # Check file existence
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # Load data
    with np.load(file_path) as data:
        all_keys = sorted(data.files)

        # Filter keys
        valid_keys = [k for k in all_keys if k not in exclude_keys]

        n_plots = len(valid_keys)
        if n_plots == 0:
            print("No keys left to plot after exclusion.")
            return

        # Calculate grid dimensions
        n_rows = math.ceil(n_plots / cols_per_row)

        # Setup Figure
        fig, axes = plt.subplots(
            n_rows,
            cols_per_row,
            figsize=(cols_per_row * figsize_scale, n_rows * figsize_scale),
            constrained_layout=True,
        )

        # Flatten axes array for easy iteration (handling case of single row/col)
        axes_flat = axes.flatten() if n_plots > 1 else [axes]

        print(f"Plotting {n_plots} fields from: {os.path.basename(file_path)}")

        for i, key in enumerate(valid_keys):
            ax = axes_flat[i]
            arr = data[key]

            # --- Robust Dimension Handling ---
            # If 3D (e.g., Depth, H, W), take the middle slice
            if arr.ndim == 3:
                mid_slice = arr.shape[0] // 2
                plot_data = arr[mid_slice, :, :]
                title_suffix = f" (Slice {mid_slice}/{arr.shape[0]})"
            # If 4D (e.g., Time, D, H, W), take first time, mid slice
            elif arr.ndim == 4:
                mid_slice = arr.shape[1] // 2
                plot_data = arr[0, mid_slice, :, :]
                title_suffix = f" (T=0, Z={mid_slice})"
            # If 1D, expand dims to plot as a "bar code" or just plot line
            elif arr.ndim == 1:
                # Option A: Plot as line
                ax.plot(arr)
                ax.set_title(f"{key}\nShape: {arr.shape}")
                continue
            else:
                # Standard 2D
                plot_data = arr
                title_suffix = ""

            # Handle NaNs (replace with 0 for visualization or use distinct color)
            # if np.isnan(plot_data).any():
            # plot_data = np.nan_to_num(plot_data, nan=0.0)

            # Plot Heatmap
            im = ax.imshow(plot_data, origin="lower", cmap="viridis", aspect="auto")

            # Styling
            ax.set_title(f"{key}\n{arr.shape}{title_suffix}", fontsize=9)
            ax.axis("off")  # Hide ticks for cleaner look
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Turn off empty subplots in the last row
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis("off")

        plt.suptitle(f"Contents of {os.path.basename(file_path)}", fontsize=14)
        plt.show()


## --- Usage Example ---
# file_path = '/mnt/hdd1/oceans11_cyl/cx241203_fp16_full/cx241203_id01101_pvi_idx00022.npz'
## Define keys you don't care about (e.g., coordinate grids or metadata)
# ignore_list = ["sim_time", "Rcoord", "Zcoord"]
# plot_npz_content(file_path, exclude_keys=ignore_list, cols_per_row=4)
