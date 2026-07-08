import argparse
import math
import os
import time
import warnings

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from wamrvit.dataloader.trajectory_loader import Seq2SeqMapper
from wamrvit.native_train_utils import unpack_native_batch
from wamrvit.quad.regrid_dispatch import regrid_native_dispatch, regrid_uniform_dispatch
from wamrvit.quad.quad_utils import (
    tensor_to_quadtree,
    tensor_to_quadtree_native,
    tensor_to_uniform,
    tensor_to_uniform_native,
)
from wamrvit.quadtree_transformer import QuadTreeTransformer
from wamrvit.utils import instantiate_from_config, load_config
from wamrvit.visualization import anim_style
from wamrvit.visualization.plotting import compute_vrange, plot_quadtree


def _resolve_anim_channels(anim_cfg, field_names):
    """Resolve which channels to animate as list of (label, channel_idx).

    Priority: `anim_channel_indices` > `field_names` > all channels.
    Labels fall back to `ch{i}` when an index exceeds `field_names`.
    """
    ch_indices = anim_cfg.get("anim_channel_indices", None)
    if ch_indices is not None:
        if isinstance(ch_indices, int):
            ch_indices = [ch_indices]
        pairs = []
        for c in ch_indices:
            c = int(c)
            label = field_names[c] if c < len(field_names) else f"ch{c}"
            pairs.append((label, c))
        return pairs

    anim_fields = anim_cfg.get("field_names", None)
    if anim_fields is not None:
        if isinstance(anim_fields, str):
            anim_fields = [anim_fields]
        pairs = []
        for field in anim_fields:
            if field not in field_names:
                print(f"Skipping {field}, not in data fields.")
                continue
            pairs.append((field, field_names.index(field)))
        return pairs

    return [(f, i) for i, f in enumerate(field_names)]


def _parse_save_at_indices(args, config, anim_cfg, predict_steps):
    """Parse user-requested snapshot indices (1-based) for PNG export."""
    raw = args.save_at
    if raw is None:
        raw = anim_cfg.get("save_at", None)
    if raw is None:
        raw = config.get("save_at", None)
    if raw is None:
        return []

    if isinstance(raw, str):
        cleaned = raw.strip()
        cleaned = cleaned.replace("[", "").replace("]", "")
        tokens = [tok for tok in cleaned.replace(",", " ").split() if tok]
        values = [int(tok) for tok in tokens]
    elif isinstance(raw, (int, np.integer)):
        values = [int(raw)]
    else:
        values = [int(v) for v in list(raw)]

    save_at = []
    seen = set()
    for idx in values:
        if idx in seen:
            continue
        seen.add(idx)
        if 1 <= idx <= predict_steps:
            save_at.append(idx)
        else:
            warnings.warn(
                f"Ignoring save_at={idx}: valid range is [1, {predict_steps}] for this rollout."
            )
    return save_at


def _rasterize_quadtree(
    qt,
    *,
    channel: int = 0,
    max_pixels: int = 2048,
    x_frac=None,
    y_frac=None,
):
    """Rasterize quadtree content to a fixed-resolution image buffer."""
    if qt.channels <= 0:
        raise ValueError("Quadtree has no channels to rasterize.")

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

    img = np.zeros((H_img, W_img), dtype=np.float32)

    for leaf in qt._iter_all_leaves():
        level, xi, yi = qt.cell_xy_index(leaf)
        scale = 1 << (L - level)

        lx0 = (leaf.tile_ix * grids_per_tile + xi * scale) * Pw
        ly0 = (leaf.tile_iy * grids_per_tile + yi * scale) * Ph
        lw = scale * Pw
        lh = scale * Ph

        tx0 = int(lx0 * scale_down)
        ty0 = int(ly0 * scale_down)
        tx1 = int((lx0 + lw) * scale_down)
        ty1 = int((ly0 + lh) * scale_down)

        tx0 = max(0, min(tx0, W_img - 1))
        ty0 = max(0, min(ty0, H_img - 1))
        tx1 = max(tx0 + 1, min(tx1, W_img))
        ty1 = max(ty0 + 1, min(ty1, H_img))

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

    extent = (qt.xmin, qt.xmax, qt.ymin, qt.ymax)

    def _normalize_frac(frac):
        if frac is None:
            return 0.0, 1.0
        f0, f1 = float(frac[0]), float(frac[1])
        if f1 <= f0:
            raise ValueError("plot fraction max must be greater than min.")
        return f0, f1

    if x_frac is not None or y_frac is not None:
        x0_frac, x1_frac = _normalize_frac(x_frac)
        y0_frac, y1_frac = _normalize_frac(y_frac)

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

    return img, extent


def _to_numpy(arr):
    if hasattr(arr, "detach"):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _build_quadtree_with_optional_projection(
    frame_data,
    *,
    target_centers,
    target_levels,
    domain_meta,
    cell_scale_mode,
    source_centers=None,
    source_levels=None,
    remap_mode: str = "uniform",
):
    """Build quadtree on target topology; remap from source topology if needed."""
    frame_np = np.ascontiguousarray(frame_data)
    frame_tensor = torch.from_numpy(frame_np)
    target_meta = {"centers": target_centers, "levels": target_levels, "domain": domain_meta}

    if source_centers is None or source_levels is None:
        return tensor_to_quadtree(
            data=frame_tensor, meta=target_meta, cell_scale_mode=cell_scale_mode
        )

    src_n = int(frame_np.shape[0])
    tgt_n = int(_to_numpy(target_levels).shape[0])

    # Fast path when topology size matches.
    if src_n == tgt_n:
        return tensor_to_quadtree(
            data=frame_tensor, meta=target_meta, cell_scale_mode=cell_scale_mode
        )

    source_meta = {"centers": source_centers, "levels": source_levels, "domain": domain_meta}
    zeros = np.zeros(
        (tgt_n, frame_np.shape[1], frame_np.shape[2], frame_np.shape[3]), dtype=np.float32
    )
    dst_qt = tensor_to_quadtree(data=zeros, meta=target_meta, cell_scale_mode=cell_scale_mode)

    if remap_mode == "uniform":
        # Conservative remap: source topology -> dense uniform field -> target topology.
        src_uniform = tensor_to_uniform(
            data=frame_np,
            meta=source_meta,
            cell_scale_mode=cell_scale_mode,
            return_tensor=False,
        )
        dst_qt.assign_from_array(
            src_uniform,
            extent=(
                domain_meta["xmin"],
                domain_meta["xmax"],
                domain_meta["ymin"],
                domain_meta["ymax"],
            ),
        )
    elif remap_mode == "center_copy":
        # Legacy behavior: center-point copy from source leaf to target leaf.
        src_qt = tensor_to_quadtree(
            data=frame_tensor, meta=source_meta, cell_scale_mode=cell_scale_mode
        )
        dst_qt.project_values_from(src_qt)
    else:
        raise ValueError(f"Unsupported remap_mode={remap_mode!r}; use 'uniform' or 'center_copy'.")

    return dst_qt


def save_single_gif(
    seq,
    centers_list,
    levels_list,
    domain_meta,
    T,
    title_prefix,
    filename,
    cell_scale_mode,
    dpi=150,
    fps=4,
    vmin=None,
    vmax=None,
    *,
    outline_width: float = 0.3,
    draw_outlines: bool = True,
    x_frac=None,
    y_frac=None,
    max_pixels: int | None = None,
    source_centers=None,
    source_levels=None,
    remap_mode: str = "uniform",
    show_axes: bool = True,
    aspect: float = 1.0,
):
    """Renders the sequence to a GIF, handling dynamically changing grids."""
    if vmin is None:
        vmin = float(np.min(seq))
    if vmax is None:
        vmax = float(np.max(seq))

    _figsize, _ax_rect, _cax_rect = anim_style.compute_layout(aspect, show_axes=show_axes)
    fig = plt.figure(figsize=_figsize, dpi=dpi)

    def update(frame):
        fig.clf()
        ax, cax = anim_style.make_aligned_axes(fig, _ax_rect, _cax_rect)
        ax.grid(False)

        # Extract frame data and corresponding dynamic grid metadata
        frame_data = np.ascontiguousarray(seq[:, :, frame, :, :])
        centers_frame = centers_list[frame] if isinstance(centers_list, list) else centers_list
        levels_frame = levels_list[frame] if isinstance(levels_list, list) else levels_list
        source_centers_frame = (
            source_centers[frame] if isinstance(source_centers, list) else source_centers
        )
        source_levels_frame = (
            source_levels[frame] if isinstance(source_levels, list) else source_levels
        )

        qt = _build_quadtree_with_optional_projection(
            frame_data,
            target_centers=centers_frame,
            target_levels=levels_frame,
            domain_meta=domain_meta,
            cell_scale_mode=cell_scale_mode,
            source_centers=source_centers_frame,
            source_levels=source_levels_frame,
            remap_mode=remap_mode,
        )

        plot_quadtree(
            qt,
            ax=ax,
            channel=0,
            outline_width=outline_width,
            colorbar=True,
            cax=cax,
            vmin=vmin,
            vmax=vmax,
            draw_outlines=draw_outlines,
            x_frac=x_frac,
            y_frac=y_frac,
            max_pixels=max_pixels if max_pixels is not None else 2048,
        )

        anim_style.finalize_axes(
            ax, title=f"{title_prefix} (Frame {frame})", show_axes=show_axes
        )
        return []

    anim = animation.FuncAnimation(fig, update, frames=T, blit=False)
    anim.save(filename, writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)


def _animate_native(
    *,
    model,
    batch,
    config,
    device,
    transform,
    predict_steps,
    num_levels,
    cell_scale_mode,
    adapt_on_channels,
    regrid_interval,
    regrid_adapt_nearby,
    regrid_tol_frac,
    pred_mode,
    anim_cfg,
    anim_dir,
    anim_dpi,
    anim_fps,
    model_name,
    data_config,
    outline_width,
    gt_grid_mode: str = "adaptive",
    draw_outlines,
    x_frac,
    y_frac,
    max_pixels,
    pred_max_pixels,
    args,
    save_at_indices,
):
    """Native-mode animation: uses tensor_to_uniform_native + imshow."""

    show_axes = anim_cfg.get("show_axes", True)

    def _unpack(arr):
        return arr[0] if arr.dtype == object else arr

    inputs_by_level, targets_by_level, leaf_to_bucket, centers = unpack_native_batch(
        batch,
        num_levels,
        device,
        non_blocking=False,
    )

    domain = {
        k: _unpack(batch[k]).item()
        for k in ["xmin", "xmax", "ymin", "ymax", "max_level_idx", "tile_width", "tile_height"]
    }
    ref_centers_np = _unpack(batch["centers"])
    ref_levels_np = _unpack(batch["levels"])
    if ref_centers_np.ndim == 3:
        ref_centers_np = ref_centers_np.squeeze(0)
    if ref_levels_np.ndim == 2:
        ref_levels_np = ref_levels_np.squeeze(0)
    ref_meta = {"centers": ref_centers_np, "levels": ref_levels_np, "domain": domain}
    meta = ref_meta

    native_aspect = anim_style.resolve_aspect(
        domain["ymax"] - domain["ymin"], domain["xmax"] - domain["xmin"],
        x_frac=x_frac, y_frac=y_frac, override=anim_cfg.get("data_aspect"),
    )
    native_figsize, native_ax_rect, native_cax_rect = anim_style.compute_layout(
        native_aspect, show_axes=show_axes
    )
    print(f"[layout] native: data aspect H/W = {native_aspect:.4f}")

    R = model.config.return_seq_len
    T_in = next(v.shape[2] for v in inputs_by_level.values() if v.shape[0] > 0)
    num_forward_calls = math.ceil(predict_steps / R)

    curr_by_level = {lvl: v.clone() for lvl, v in inputs_by_level.items()}

    all_preds_by_level = []
    all_leaf_to_bucket = []
    all_centers = []
    all_levels = []

    allow_coarsening = config["inference"].get("allow_coarsening", True)
    regrid_backend = config["inference"].get("regrid_backend", "object")

    print(
        f"\nStarting native inference (Steps: {predict_steps}, "
        f"Forward calls: {num_forward_calls}, R: {R})..."
    )
    inf_start_time = time.time()

    with torch.no_grad():
        for call_idx in range(num_forward_calls):
            timestep_idx = call_idx * R
            steps_this_call = min(R, predict_steps - timestep_idx)

            if (
                regrid_interval is not None
                and timestep_idx > 0
                and timestep_idx % regrid_interval == 0
            ):
                C_orig = T_orig = None
                for arr in curr_by_level.values():
                    if arr.shape[0] > 0:
                        C_orig, T_orig = arr.shape[1], arr.shape[2]
                        break
                n_leaves_before = int(leaf_to_bucket.shape[0])
                np_by_level = {lvl: arr.cpu().numpy() for lvl, arr in curr_by_level.items()}
                new_np, new_l2b, meta = regrid_native_dispatch(
                    np_by_level,
                    leaf_to_bucket.cpu().numpy(),
                    meta,
                    backend=regrid_backend,
                    C=C_orig,
                    T=T_orig,
                    tol_frac=regrid_tol_frac,
                    cell_scale_mode=cell_scale_mode,
                    adapt_on_channels=adapt_on_channels,
                    adapt_nearby=regrid_adapt_nearby,
                    allow_coarsening=allow_coarsening,
                )
                curr_by_level = {
                    lvl: torch.from_numpy(arr).to(device, dtype=torch.float32)
                    for lvl, arr in new_np.items()
                }
                leaf_to_bucket = torch.from_numpy(new_l2b).to(device).long()
                centers = torch.from_numpy(meta["centers"]).to(device, dtype=torch.float32)
                n_leaves_after = int(new_l2b.shape[0])
                print(
                    f"Timestep {timestep_idx}: Regrid triggered - "
                    f"N_leaves changed from {n_leaves_before} to {n_leaves_after}"
                )

            pred_by_level = model.forward_multi_scale(curr_by_level, leaf_to_bucket, centers)

            if pred_mode == "residual":
                for lvl in pred_by_level:
                    pred_by_level[lvl] = pred_by_level[lvl] + curr_by_level[lvl][:, :, -1:]

            for t in range(steps_this_call):
                step_pred = {}
                for lvl, arr in pred_by_level.items():
                    slice_np = arr[:, :, t : t + 1].cpu().numpy()
                    if hasattr(transform, "inverse_transform") and callable(
                        transform.inverse_transform
                    ):
                        slice_np = transform.inverse_transform(slice_np)
                    step_pred[lvl] = slice_np
                all_preds_by_level.append(step_pred)
                all_leaf_to_bucket.append(leaf_to_bucket.cpu().numpy())
                all_centers.append(meta["centers"])
                all_levels.append(meta["levels"])

            if call_idx < num_forward_calls - 1:
                num_from_input = max(T_in - R, 0)
                for lvl in curr_by_level:
                    if num_from_input > 0:
                        curr_by_level[lvl] = torch.cat(
                            (curr_by_level[lvl][:, :, -num_from_input:], pred_by_level[lvl]),
                            dim=2,
                        )
                    else:
                        curr_by_level[lvl] = pred_by_level[lvl][:, :, -T_in:]

    inf_end_time = time.time()
    print(f"Inference completed in {inf_end_time - inf_start_time:.2f}s.")

    # --- GT ---
    gt_by_level = {}
    for lvl, arr in targets_by_level.items():
        gt_np = arr.cpu().numpy()
        if hasattr(transform, "inverse_transform") and callable(transform.inverse_transform):
            gt_np = transform.inverse_transform(gt_np)
        gt_by_level[lvl] = gt_np

    # Reference leaf_to_bucket for GT (topology unchanged).
    ref_l2b = leaf_to_bucket.cpu().numpy() if not all_leaf_to_bucket else all_leaf_to_bucket[0]

    # --- RENDERING ---
    print("\nStarting native animation rendering...")
    anim_start_time = time.time()

    anim_pairs = _resolve_anim_channels(anim_cfg, data_config["field_names"])

    for field, c_idx in anim_pairs:
        print(f"Rendering {field}...")

        base_filename = os.path.join(
            anim_dir, f"{model_name}_{field}_traj{args.traj_idx}_frame{args.frame_idx}"
        )
        gt_base_filename = os.path.join(
            anim_dir, f"GT_{field}_traj{args.traj_idx}_frame{args.frame_idx}"
        )
        # Title prefixes: omit the "name: " segment when the display name is
        # empty, matching animate_regular.py so an empty name yields just the
        # field (no leading ": ").
        gt_gif_prefix = f"{args.gt_display_name}: {field}" if args.gt_display_name else field
        pred_gif_prefix = f"{args.display_name}: {field}" if args.display_name else field

        # Compute global value range via flat uniform reconstruction (cheap).
        gt_vals = []
        pred_vals = []
        T_gt = next(v.shape[2] for v in gt_by_level.values() if v.shape[0] > 0)
        for t in range(min(predict_steps, T_gt)):
            gt_frame = {lvl: arr[:, c_idx : c_idx + 1, t] for lvl, arr in gt_by_level.items()}
            gt_uniform = tensor_to_uniform_native(gt_frame, ref_l2b, ref_meta)
            gt_vals.append(gt_uniform[0])
        for t in range(predict_steps):
            p = all_preds_by_level[t]
            pred_frame = {lvl: arr[:, c_idx : c_idx + 1, 0] for lvl, arr in p.items()}
            pred_uniform = tensor_to_uniform_native(
                pred_frame,
                all_leaf_to_bucket[t],
                {"centers": all_centers[t], "levels": all_levels[t], "domain": domain},
            )
            pred_vals.append(pred_uniform[0])

        vmin_global, vmax_global = compute_vrange(
            gt_vals,
            pred_vals,
            field,
            anim_cfg,
        )

        def _build_gt_qt_native(gt_frame_per_level, frame, c_idx):
            """Build GT quadtree; if gt_grid_mode==adaptive, snap to the
            prediction's per-step topology by routing through the dense
            uniform field and re-binning via assign_from_array."""
            if gt_grid_mode == "adaptive":
                gt_uniform = tensor_to_uniform_native(
                    gt_frame_per_level, ref_l2b, ref_meta
                )
                seed_frame = {
                    lvl: arr[:, c_idx : c_idx + 1, 0]
                    for lvl, arr in all_preds_by_level[frame].items()
                }
                step_meta = {
                    "centers": all_centers[frame],
                    "levels": all_levels[frame],
                    "domain": domain,
                }
                qt = tensor_to_quadtree_native(
                    seed_frame,
                    all_leaf_to_bucket[frame],
                    step_meta,
                    cell_scale_mode=cell_scale_mode,
                )
                qt.assign_from_array(
                    gt_uniform,
                    extent=(
                        domain["xmin"], domain["xmax"],
                        domain["ymin"], domain["ymax"],
                    ),
                )
                return qt
            return tensor_to_quadtree_native(
                gt_frame_per_level, ref_l2b, ref_meta, cell_scale_mode=cell_scale_mode
            )

        if not args.png_only:
            # --- GT GIF (quadtree overlay via plot_quadtree) ---
            fig = plt.figure(figsize=native_figsize, dpi=anim_dpi)

            def _update_gt(frame, fig=fig, c_idx=c_idx):
                fig.clf()
                ax, cax = anim_style.make_aligned_axes(fig, native_ax_rect, native_cax_rect)
                ax.grid(False)
                gt_frame = {lvl: arr[:, c_idx : c_idx + 1, frame] for lvl, arr in gt_by_level.items()}
                if gt_grid_mode == "overlay":
                    pred_frame_per_level = {
                        lvl: arr[:, c_idx : c_idx + 1, 0]
                        for lvl, arr in all_preds_by_level[frame].items()
                    }
                    step_meta = {
                        "centers": all_centers[frame],
                        "levels": all_levels[frame],
                        "domain": domain,
                    }
                    pred_qt = tensor_to_quadtree_native(
                        pred_frame_per_level,
                        all_leaf_to_bucket[frame],
                        step_meta,
                        cell_scale_mode=cell_scale_mode,
                    )
                    gt_uniform = tensor_to_uniform_native(gt_frame, ref_l2b, ref_meta)
                    plot_quadtree(
                        pred_qt,
                        ax=ax,
                        channel=None,
                        bg=gt_uniform[0],
                        outline_width=outline_width,
                        colorbar=True,
                        cax=cax,
                        vmin=vmin_global,
                        vmax=vmax_global,
                        draw_outlines=draw_outlines,
                        x_frac=x_frac,
                        y_frac=y_frac,
                    )
                else:
                    qt = _build_gt_qt_native(gt_frame, frame, c_idx)
                    plot_quadtree(
                        qt,
                        ax=ax,
                        channel=0,
                        outline_width=outline_width,
                        colorbar=True,
                        cax=cax,
                        vmin=vmin_global,
                        vmax=vmax_global,
                        draw_outlines=draw_outlines,
                        max_pixels=max_pixels if max_pixels is not None else 2048,
                        x_frac=x_frac,
                        y_frac=y_frac,
                    )
                anim_style.finalize_axes(
                    ax,
                    title=f"{gt_gif_prefix} (Frame {frame})",
                    show_axes=show_axes,
                )
                return []

            anim_gt = animation.FuncAnimation(
                fig, _update_gt, frames=min(predict_steps, T_gt), blit=False
            )
            anim_gt.save(f"{gt_base_filename}.gif", writer="pillow", fps=anim_fps, dpi=anim_dpi)
            plt.close(fig)

            # --- Prediction GIF (quadtree overlay via plot_quadtree) ---
            fig = plt.figure(figsize=native_figsize, dpi=anim_dpi)

            def _update_pred(frame, fig=fig, c_idx=c_idx):
                fig.clf()
                ax, cax = anim_style.make_aligned_axes(fig, native_ax_rect, native_cax_rect)
                ax.grid(False)
                p = all_preds_by_level[frame]
                pred_frame = {lvl: arr[:, c_idx : c_idx + 1, 0] for lvl, arr in p.items()}
                step_meta = {
                    "centers": all_centers[frame],
                    "levels": all_levels[frame],
                    "domain": domain,
                }
                qt = tensor_to_quadtree_native(
                    pred_frame, all_leaf_to_bucket[frame], step_meta, cell_scale_mode=cell_scale_mode
                )
                plot_quadtree(
                    qt,
                    ax=ax,
                    channel=0,
                    outline_width=outline_width,
                    colorbar=True,
                    cax=cax,
                    vmin=vmin_global,
                    vmax=vmax_global,
                    draw_outlines=draw_outlines,
                    max_pixels=pred_max_pixels,
                    x_frac=x_frac,
                    y_frac=y_frac,
                )
                anim_style.finalize_axes(
                    ax,
                    title=f"{pred_gif_prefix} (Frame {frame})",
                    show_axes=show_axes,
                )
                return []

            anim_pred = animation.FuncAnimation(fig, _update_pred, frames=predict_steps, blit=False)
            anim_pred.save(f"{base_filename}_Pred.gif", writer="pillow", fps=anim_fps, dpi=anim_dpi)
            plt.close(fig)

        # --- Optional snapshot PNGs ---
        for idx1 in save_at_indices:
            frame = idx1 - 1
            if frame < min(predict_steps, T_gt):
                fig, ax, cax = anim_style.make_aligned_figure(native_aspect, anim_dpi, show_axes=show_axes)
                ax.grid(False)
                gt_frame = {
                    lvl: arr[:, c_idx : c_idx + 1, frame] for lvl, arr in gt_by_level.items()
                }
                if gt_grid_mode == "overlay":
                    pred_frame_per_level = {
                        lvl: arr[:, c_idx : c_idx + 1, 0]
                        for lvl, arr in all_preds_by_level[frame].items()
                    }
                    step_meta = {
                        "centers": all_centers[frame],
                        "levels": all_levels[frame],
                        "domain": domain,
                    }
                    pred_qt = tensor_to_quadtree_native(
                        pred_frame_per_level,
                        all_leaf_to_bucket[frame],
                        step_meta,
                        cell_scale_mode=cell_scale_mode,
                    )
                    gt_uniform = tensor_to_uniform_native(gt_frame, ref_l2b, ref_meta)
                    plot_quadtree(
                        pred_qt,
                        ax=ax,
                        channel=None,
                        bg=gt_uniform[0],
                        outline_width=outline_width,
                        colorbar=True,
                        cax=cax,
                        vmin=vmin_global,
                        vmax=vmax_global,
                        draw_outlines=draw_outlines,
                        x_frac=x_frac,
                        y_frac=y_frac,
                    )
                else:
                    qt = _build_gt_qt_native(gt_frame, frame, c_idx)
                    plot_quadtree(
                        qt,
                        ax=ax,
                        channel=0,
                        outline_width=outline_width,
                        colorbar=True,
                        cax=cax,
                        vmin=vmin_global,
                        vmax=vmax_global,
                        draw_outlines=draw_outlines,
                        max_pixels=max_pixels if max_pixels is not None else 2048,
                        x_frac=x_frac,
                        y_frac=y_frac,
                    )
                anim_style.finalize_axes(ax, title=args.gt_display_name, show_axes=show_axes)
                fig.savefig(f"{gt_base_filename}_saveat{idx1}.png", dpi=anim_dpi)
                plt.close(fig)

            if frame < predict_steps:
                fig, ax, cax = anim_style.make_aligned_figure(native_aspect, anim_dpi, show_axes=show_axes)
                ax.grid(False)
                p = all_preds_by_level[frame]
                pred_frame = {lvl: arr[:, c_idx : c_idx + 1, 0] for lvl, arr in p.items()}
                step_meta = {
                    "centers": all_centers[frame],
                    "levels": all_levels[frame],
                    "domain": domain,
                }
                qt = tensor_to_quadtree_native(
                    pred_frame,
                    all_leaf_to_bucket[frame],
                    step_meta,
                    cell_scale_mode=cell_scale_mode,
                )
                plot_quadtree(
                    qt,
                    ax=ax,
                    channel=0,
                    outline_width=outline_width,
                    colorbar=True,
                    cax=cax,
                    vmin=vmin_global,
                    vmax=vmax_global,
                    draw_outlines=draw_outlines,
                    max_pixels=pred_max_pixels,
                    x_frac=x_frac,
                    y_frac=y_frac,
                )
                anim_style.finalize_axes(ax, title=args.display_name, show_axes=show_axes)
                fig.savefig(f"{base_filename}_Pred_saveat{idx1}.png", dpi=anim_dpi)
                plt.close(fig)

    anim_end_time = time.time()
    print(f"Animation rendering completed in {anim_end_time - anim_start_time:.2f}s.")
    print(f"Total time: {anim_end_time - inf_start_time:.2f}s.")


def main():
    parser = argparse.ArgumentParser(description="Standalone Adaptive Animation Generator")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to standalone yaml file (legacy). Omit to use Hydra config groups.",
    )
    parser.add_argument("--traj_idx", type=int, required=True, help="Trajectory index to animate.")
    parser.add_argument("--frame_idx", type=int, required=True, help="Frame index to animate.")
    parser.add_argument(
        "--aux_plot", action="store_true", help="Generate auxiliary plots (shock pressure history)."
    )
    parser.add_argument(
        "--save_at",
        type=str,
        default=None,
        help="1-based snapshot indices to save as PNG (e.g., '25' or '10,25,40').",
    )
    parser.add_argument(
        "--display_name",
        type=str,
        default="AR model",
        help="Title prefix for the prediction frames (e.g., 'WAMRViT', 'ViT-finest').",
    )
    parser.add_argument(
        "--gt_display_name",
        type=str,
        default="GT",
        help="Title prefix for the ground-truth frames.",
    )
    parser.add_argument(
        "--png_only",
        action="store_true",
        help="Skip GIF rendering; only emit the --save_at snapshot PNGs (and aux soot foil if --aux_plot).",
    )
    args, unknown = parser.parse_known_args()

    config = load_config(args, unknown)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. SETUP OUTPUT DIRECTORY ---
    anim_cfg = config.get("animation", {})
    base_dir = os.getcwd()
    anim_dir = anim_cfg.get("output_dir", "animations")
    if not os.path.isabs(anim_dir):
        anim_dir = os.path.abspath(os.path.join(base_dir, anim_dir))
    os.makedirs(anim_dir, exist_ok=True)
    anim_dpi = anim_cfg.get("dpi", 150)
    anim_fps = anim_cfg.get("fps", 4)
    show_axes = anim_cfg.get("show_axes", True)

    model_name = config["inference"]["checkpoint_path"].split("/")[-2]

    # --- 2. FIND THE REQUESTED WINDOW ---
    print(f"Searching for Trajectory {args.traj_idx}, Frame {args.frame_idx}...")
    data_config = config["data"]
    file_parser = instantiate_from_config(config["file_parser"])
    all_paths = file_parser(data_config["glob_pattern"])

    window_config = dict(config["window_generator"])
    predict_steps = config["inference"].get("predict_steps", 1)
    window_config["params"]["return_seq_len"] = predict_steps
    window_config["params"]["file_path_list"] = all_paths

    windows = instantiate_from_config(window_config)

    target_window = None
    for w in windows:
        if (
            isinstance(w, dict)
            and w.get("traj_idx") == args.traj_idx
            and w.get("frame_idx") == args.frame_idx
        ):
            target_window = w
            break

    if target_window is None:
        raise ValueError(
            f"Could not find window with traj_idx={args.traj_idx} and frame_idx={args.frame_idx}."
        )
    print("Window found. Loading data...")

    # --- 3. LOAD DATA (Single Batch) ---
    # Disable resample augmentation during animation — it's training-only
    loader_cfg = config["file_loader"]
    loader_params = loader_cfg.get("params", {})
    for key in ("resample_coarsen_ratio", "resample_refine_ratio"):
        if loader_params.get(key, 0.0) > 0.0:
            warnings.warn(
                f"Animate: overriding {key}={loader_params[key]} → 0.0 "
                "(augmentation disabled during inference)"
            )
            loader_params[key] = 0.0

    file_loader = instantiate_from_config(loader_cfg)
    transform = instantiate_from_config(config["transform"])
    mapper = Seq2SeqMapper(loader=file_loader, transform=transform)

    # Simulate a Ray batch of size 1 so the mapper parses it correctly
    ray_mock_batch = {k: [v] for k, v in target_window.items()}
    batch = mapper(ray_mock_batch)

    # Extract configs (Check inference block first for easy param tuning, then fallback)
    cell_scale_mode = file_loader.cell_scale_mode
    adapt_on_channels = file_loader.adapt_on_channels
    tol_frac = config["inference"].get("tol_frac", file_loader.tol_frac)
    regrid_interval = config["inference"].get("regrid_interval", None)
    regrid_adapt_nearby = config["inference"].get("regrid_adapt_nearby", 0)
    regrid_tol_frac = config["inference"].get("regrid_tol_frac", tol_frac)
    regrid_backend = config["inference"].get("regrid_backend", "object")
    pred_mode = config["inference"].get("pred_mode", "target")
    outline_width = float(anim_cfg.get("outline_width", 0.3))
    draw_outlines = bool(anim_cfg.get("draw_outlines", True))
    max_pixels = anim_cfg.get("max_pixels", None)
    pred_max_pixels = anim_cfg.get(
        "pred_max_pixels", max_pixels if max_pixels is not None else 8192
    )
    aux_max_pixels = anim_cfg.get("aux_max_pixels", max_pixels if max_pixels is not None else 4096)
    aux_pressure_threshold = float(anim_cfg.get("aux_pressure_threshold", 2.0e5))
    save_at_indices = _parse_save_at_indices(args, config, anim_cfg, predict_steps)
    if save_at_indices:
        print(f"Additional snapshots requested at (1-based): {save_at_indices}")

    def _parse_plot_frac(cfg):
        plot_frac = cfg.get("plot_frac", None)
        if plot_frac is not None:
            vals = list(plot_frac)
            if len(vals) != 4:
                raise ValueError(
                    "animation.plot_frac must be [xmin_frac, xmax_frac, ymin_frac, ymax_frac]."
                )
            return (vals[0], vals[1]), (vals[2], vals[3])
        x_frac = cfg.get("x_frac", None)
        y_frac = cfg.get("y_frac", None)
        if x_frac is not None:
            x_frac = tuple(x_frac)
        if y_frac is not None:
            y_frac = tuple(y_frac)
        return x_frac, y_frac

    x_frac, y_frac = _parse_plot_frac(anim_cfg)
    gt_grid_mode = str(anim_cfg.get("gt_grid_mode", "adaptive")).strip().lower()
    if gt_grid_mode not in {"adaptive", "static", "overlay"}:
        warnings.warn(
            f"animation.gt_grid_mode={gt_grid_mode!r} is invalid; falling back to 'adaptive'."
        )
        gt_grid_mode = "adaptive"
    gt_adaptive_remap = str(anim_cfg.get("gt_adaptive_remap", "uniform")).strip().lower()
    if gt_adaptive_remap not in {"uniform", "center_copy"}:
        warnings.warn(
            f"animation.gt_adaptive_remap={gt_adaptive_remap!r} is invalid; "
            "falling back to 'uniform'."
        )
        gt_adaptive_remap = "uniform"

    # --- 4. LOAD MODEL ---
    inf_cfg = config["inference"]
    if inf_cfg.get("is_diffusers", False):
        model = QuadTreeTransformer.from_pretrained(inf_cfg["checkpoint_path"])
    else:
        model = QuadTreeTransformer(**config["model"])
        state_dict = torch.load(inf_cfg["checkpoint_path"], map_location="cpu")
        if "module." in list(state_dict.keys())[0]:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    multi_scale = getattr(model.config, "multi_scale_patch", False)
    num_levels = int(model.config.max_level_idx) + 1 if multi_scale else 0

    model.to(device)
    model.eval()

    # --- 5. PREPARE TENSORS ---
    def _unpack(arr):
        return arr[0] if arr.dtype == object else arr

    # Detect native-mode batch (per-level columns).
    is_native_batch = "input_level_0" in batch

    if is_native_batch and not multi_scale:
        raise ValueError("Batch has native per-level columns but model.multi_scale_patch is False.")

    if is_native_batch:
        # Delegate to native rendering path.
        _animate_native(
            model=model,
            batch=batch,
            config=config,
            device=device,
            transform=transform,
            predict_steps=predict_steps,
            num_levels=num_levels,
            cell_scale_mode=cell_scale_mode,
            adapt_on_channels=adapt_on_channels,
            regrid_interval=regrid_interval,
            regrid_adapt_nearby=regrid_adapt_nearby,
            regrid_tol_frac=regrid_tol_frac,
            pred_mode=pred_mode,
            anim_cfg=anim_cfg,
            anim_dir=anim_dir,
            anim_dpi=anim_dpi,
            anim_fps=anim_fps,
            model_name=model_name,
            data_config=data_config,
            outline_width=outline_width,
            gt_grid_mode=gt_grid_mode,
            draw_outlines=draw_outlines,
            x_frac=x_frac,
            y_frac=y_frac,
            max_pixels=max_pixels,
            pred_max_pixels=pred_max_pixels,
            args=args,
            save_at_indices=save_at_indices,
        )
        return

    inputs = torch.from_numpy(_unpack(batch["input"])).to(device, dtype=torch.float32)
    targets = torch.from_numpy(_unpack(batch["target"])).to(device, dtype=torch.float32)

    if inputs.ndim == 6:
        inputs = inputs.squeeze(0)
        targets = targets.squeeze(0)

    ref_centers_np = _unpack(batch["centers"])
    if ref_centers_np.ndim == 3:
        ref_centers_np = ref_centers_np.squeeze(0)
    ref_levels_np = _unpack(batch["levels"])
    if ref_levels_np.ndim == 2:
        ref_levels_np = ref_levels_np.squeeze(0)

    centers = torch.from_numpy(ref_centers_np).to(device, dtype=torch.float32)

    domain = {
        "xmin": _unpack(batch["xmin"]).item(),
        "xmax": _unpack(batch["xmax"]).item(),
        "ymin": _unpack(batch["ymin"]).item(),
        "ymax": _unpack(batch["ymax"]).item(),
        "max_level_idx": _unpack(batch["max_level_idx"]).item(),
        "tile_width": _unpack(batch["tile_width"]).item(),
        "tile_height": _unpack(batch["tile_height"]).item(),
    }
    meta = {"centers": ref_centers_np, "levels": ref_levels_np, "domain": domain}

    uniform_aspect = anim_style.resolve_aspect(
        domain["ymax"] - domain["ymin"], domain["xmax"] - domain["xmin"],
        x_frac=x_frac, y_frac=y_frac, override=anim_cfg.get("data_aspect"),
    )
    uniform_figsize, uniform_ax_rect, uniform_cax_rect = anim_style.compute_layout(
        uniform_aspect, show_axes=show_axes
    )
    print(f"[layout] uniform: data aspect H/W = {uniform_aspect:.4f}")

    N_grids, C, T_in, H, W = inputs.shape
    R = model.config.return_seq_len
    num_forward_calls = math.ceil(predict_steps / R)
    curr_input_seq = inputs.clone()

    if regrid_interval is not None and regrid_interval % R != 0:
        warnings.warn(
            f"regrid_interval ({regrid_interval}) is not divisible by model return_seq_len "
            f"({R}). Regridding will only occur at forward-call boundaries (every {R} timesteps)."
        )

    all_preds = []
    all_centers = []
    all_levels = []

    # --- 6. AUTOREGRESSIVE INFERENCE ---
    print(
        f"\nStarting inference loop (Steps: {predict_steps}, "
        f"Forward calls: {num_forward_calls}, R: {R}, "
        f"Regrid Interval: {regrid_interval}, Tol: {tol_frac})..."
    )
    inf_start_time = time.time()

    with torch.no_grad():
        for call_idx in range(num_forward_calls):
            timestep_idx = call_idx * R
            steps_this_call = min(R, predict_steps - timestep_idx)

            if (
                regrid_interval is not None
                and timestep_idx > 0
                and timestep_idx % regrid_interval == 0
            ):
                N, c_in, t_in, h, w = curr_input_seq.shape
                flat_input = (
                    curr_input_seq.transpose(1, 2)
                    .contiguous()
                    .view(N, t_in * c_in, h, w)
                    .cpu()
                    .numpy()
                )
                ch_offset = (t_in - 1) * c_in
                use_channels = (
                    [ch + ch_offset for ch in adapt_on_channels]
                    if adapt_on_channels is not None
                    else list(range(ch_offset, t_in * c_in))
                )

                new_input_np, meta = regrid_uniform_dispatch(
                    flat_input,
                    meta,
                    backend=regrid_backend,
                    max_passes=10,
                    cell_scale_mode=cell_scale_mode,
                    tol_frac=regrid_tol_frac,
                    channel=use_channels,
                    adapt_nearby=regrid_adapt_nearby,
                    allow_coarsening=True,
                )
                centers = torch.from_numpy(meta["centers"]).to(device, dtype=torch.float32)

                new_N = new_input_np.shape[0]
                print(
                    f"Timestep {timestep_idx}: Regrid triggered - "
                    f"N_grids changed from {curr_input_seq.shape[0]} to {new_N}"
                )
                curr_input_seq = torch.from_numpy(
                    np.ascontiguousarray(new_input_np)
                    .reshape(new_N, t_in, c_in, h, w)
                    .transpose(0, 2, 1, 3, 4)
                ).to(device, dtype=torch.float32)

            pred_full = model(curr_input_seq, centers)
            if pred_mode == "residual":
                pred_full = pred_full + curr_input_seq[:, :, -1].unsqueeze(2)

            pred_to_store = pred_full[:, :, :steps_this_call]
            pred_np = pred_to_store.cpu().numpy()
            if hasattr(transform, "inverse_transform") and callable(transform.inverse_transform):
                pred_phys = transform.inverse_transform(pred_np)
            else:
                pred_phys = pred_np

            # Store per-timestep
            for t in range(steps_this_call):
                all_preds.append(pred_phys[:, :, t : t + 1])
                all_centers.append(meta["centers"])
                all_levels.append(meta["levels"])

            if call_idx < num_forward_calls - 1:
                num_from_input = max(T_in - R, 0)
                if num_from_input > 0:
                    curr_input_seq = torch.cat(
                        (curr_input_seq[:, :, -num_from_input:], pred_full), dim=2
                    )
                else:
                    curr_input_seq = pred_full[:, :, -T_in:]

    inf_end_time = time.time()
    print(f"Inference Loop Completed in {inf_end_time - inf_start_time:.2f} seconds.")

    # --- 7. ANIMATION RENDERING ---
    print("\nStarting animation rendering...")
    anim_start_time = time.time()

    gt_np = targets.cpu().numpy()
    if hasattr(transform, "inverse_transform") and callable(transform.inverse_transform):
        gt_phys = transform.inverse_transform(gt_np)
    else:
        gt_phys = gt_np

    anim_pairs = _resolve_anim_channels(anim_cfg, data_config["field_names"])

    for field, c_idx in anim_pairs:
        print(f"Rendering {field}...")

        # Extract sequences as lists of arrays
        pred_field_seqs = [p[:, c_idx : c_idx + 1, :, :, :] for p in all_preds]
        gt_field_seq = gt_phys[:, c_idx : c_idx + 1, :, :, :]

        base_filename = os.path.join(
            anim_dir, f"{model_name}_{field}_traj{args.traj_idx}_frame{args.frame_idx}"
        )
        gt_base_filename = os.path.join(
            anim_dir, f"GT_{field}_traj{args.traj_idx}_frame{args.frame_idx}"
        )
        # Title prefixes: omit the "name: " segment when the display name is
        # empty, matching animate_regular.py so an empty name yields just the
        # field (no leading ": ").
        gt_gif_prefix = f"{args.gt_display_name}: {field}" if args.gt_display_name else field
        pred_gif_prefix = f"{args.display_name}: {field}" if args.display_name else field

        vmin_global, vmax_global = compute_vrange(
            [gt_field_seq],
            pred_field_seqs,
            field,
            anim_cfg,
        )

        if gt_grid_mode == "adaptive":
            gt_centers_seq = all_centers
            gt_levels_seq = all_levels
            gt_source_centers = ref_centers_np
            gt_source_levels = ref_levels_np
        else:
            gt_centers_seq = ref_centers_np
            gt_levels_seq = ref_levels_np
            gt_source_centers = None
            gt_source_levels = None

        if not args.png_only:
            if gt_grid_mode == "overlay":
                # Dense GT imshow + per-step prediction outlines overlaid.
                fig_gt = plt.figure(figsize=uniform_figsize, dpi=anim_dpi)

                def _update_gt_overlay(frame, fig=fig_gt):
                    fig.clf()
                    ax, cax = anim_style.make_aligned_axes(fig, uniform_ax_rect, uniform_cax_rect)
                    ax.grid(False)
                    gt_frame_data = np.ascontiguousarray(
                        gt_field_seq[:, :, frame, :, :]
                    )
                    pred_frame_data = np.ascontiguousarray(
                        pred_field_seqs[frame][:, :, 0, :, :]
                    )
                    pred_qt = tensor_to_quadtree(
                        data=torch.from_numpy(pred_frame_data),
                        meta={
                            "centers": all_centers[frame],
                            "levels": all_levels[frame],
                            "domain": domain,
                        },
                        cell_scale_mode=cell_scale_mode,
                    )
                    plot_quadtree(
                        pred_qt,
                        ax=ax,
                        channel=None,
                        bg=gt_frame_data[0, 0],
                        outline_width=outline_width,
                        colorbar=True,
                        cax=cax,
                        vmin=vmin_global,
                        vmax=vmax_global,
                        draw_outlines=draw_outlines,
                        x_frac=x_frac,
                        y_frac=y_frac,
                    )
                    anim_style.finalize_axes(
                        ax,
                        title=f"{gt_gif_prefix} (Frame {frame})",
                        show_axes=show_axes,
                    )
                    return []

                anim_gt = animation.FuncAnimation(
                    fig_gt, _update_gt_overlay, frames=predict_steps, blit=False
                )
                anim_gt.save(
                    f"{gt_base_filename}.gif",
                    writer="pillow",
                    fps=anim_fps,
                    dpi=anim_dpi,
                )
                plt.close(fig_gt)
            else:
                # Save Ground Truth (configurable: static or adaptive grid)
                save_single_gif(
                    gt_field_seq,
                    gt_centers_seq,
                    gt_levels_seq,
                    domain,
                    predict_steps,
                    gt_gif_prefix,
                    f"{gt_base_filename}.gif",
                    cell_scale_mode,
                    dpi=anim_dpi,
                    fps=anim_fps,
                    vmin=vmin_global,
                    vmax=vmax_global,
                    outline_width=outline_width,
                    draw_outlines=draw_outlines,
                    x_frac=x_frac,
                    y_frac=y_frac,
                    max_pixels=max_pixels,
                    source_centers=gt_source_centers,
                    source_levels=gt_source_levels,
                    remap_mode=gt_adaptive_remap,
                    show_axes=show_axes,
                    aspect=uniform_aspect,
                )

            # Render Prediction step-by-step (per-timestep storage: each entry has T=1)
            def update_dynamic(frame, ax, cax, vmax, vmin):
                frame_data = np.ascontiguousarray(pred_field_seqs[frame][:, :, 0, :, :])
                frame_tensor = torch.from_numpy(frame_data)

                step_meta = {
                    "centers": all_centers[frame],
                    "levels": all_levels[frame],
                    "domain": domain,
                }
                qt = tensor_to_quadtree(
                    data=frame_tensor, meta=step_meta, cell_scale_mode=cell_scale_mode
                )
                ax.grid(False)
                plot_quadtree(
                    qt,
                    ax=ax,
                    channel=0,
                    outline_width=outline_width,
                    colorbar=True,
                    cax=cax,
                    vmin=vmin,
                    vmax=vmax,
                    draw_outlines=draw_outlines,
                    max_pixels=pred_max_pixels,
                    x_frac=x_frac,
                    y_frac=y_frac,
                )
                anim_style.finalize_axes(
                    ax,
                    title=f"{pred_gif_prefix} (Frame {frame})",
                    show_axes=show_axes,
                )
                return []

            fig = plt.figure(figsize=uniform_figsize, dpi=anim_dpi)

            def wrapper_update(frame):
                fig.clf()
                ax, cax = anim_style.make_aligned_axes(fig, uniform_ax_rect, uniform_cax_rect)
                result = update_dynamic(frame, ax, cax, vmax_global, vmin_global)
                return result

            anim = animation.FuncAnimation(fig, wrapper_update, frames=predict_steps, blit=False)
            anim.save(f"{base_filename}_Pred.gif", writer="pillow", fps=anim_fps, dpi=anim_dpi)
            plt.close(fig)

        # Optional snapshot PNGs (1-based indices mapped to 0-based frames).
        for idx1 in save_at_indices:
            frame = idx1 - 1
            if frame < predict_steps:
                gt_frame_data = np.ascontiguousarray(gt_field_seq[:, :, frame, :, :])
                fig, ax, cax = anim_style.make_aligned_figure(uniform_aspect, anim_dpi, show_axes=show_axes)
                ax.grid(False)
                if gt_grid_mode == "overlay":
                    pred_frame_data = np.ascontiguousarray(
                        pred_field_seqs[frame][:, :, 0, :, :]
                    )
                    pred_qt = tensor_to_quadtree(
                        data=torch.from_numpy(pred_frame_data),
                        meta={
                            "centers": all_centers[frame],
                            "levels": all_levels[frame],
                            "domain": domain,
                        },
                        cell_scale_mode=cell_scale_mode,
                    )
                    plot_quadtree(
                        pred_qt,
                        ax=ax,
                        channel=None,
                        bg=gt_frame_data[0, 0],
                        outline_width=outline_width,
                        colorbar=True,
                        cax=cax,
                        vmin=vmin_global,
                        vmax=vmax_global,
                        draw_outlines=draw_outlines,
                        x_frac=x_frac,
                        y_frac=y_frac,
                    )
                else:
                    if gt_grid_mode == "adaptive":
                        gt_qt = _build_quadtree_with_optional_projection(
                            gt_frame_data,
                            target_centers=all_centers[frame],
                            target_levels=all_levels[frame],
                            domain_meta=domain,
                            cell_scale_mode=cell_scale_mode,
                            source_centers=ref_centers_np,
                            source_levels=ref_levels_np,
                            remap_mode=gt_adaptive_remap,
                        )
                    else:
                        gt_qt = tensor_to_quadtree(
                            data=torch.from_numpy(gt_frame_data),
                            meta={"centers": ref_centers_np, "levels": ref_levels_np, "domain": domain},
                            cell_scale_mode=cell_scale_mode,
                        )
                    plot_quadtree(
                        gt_qt,
                        ax=ax,
                        channel=0,
                        outline_width=outline_width,
                        colorbar=True,
                        cax=cax,
                        vmin=vmin_global,
                        vmax=vmax_global,
                        draw_outlines=draw_outlines,
                        max_pixels=max_pixels if max_pixels is not None else 2048,
                        x_frac=x_frac,
                        y_frac=y_frac,
                    )
                anim_style.finalize_axes(ax, title=args.gt_display_name, show_axes=show_axes)
                fig.savefig(f"{gt_base_filename}_saveat{idx1}.png", dpi=anim_dpi)
                plt.close(fig)

                pred_frame_data = np.ascontiguousarray(pred_field_seqs[frame][:, :, 0, :, :])
                pred_qt = tensor_to_quadtree(
                    data=torch.from_numpy(pred_frame_data),
                    meta={
                        "centers": all_centers[frame],
                        "levels": all_levels[frame],
                        "domain": domain,
                    },
                    cell_scale_mode=cell_scale_mode,
                )
                fig, ax, cax = anim_style.make_aligned_figure(uniform_aspect, anim_dpi, show_axes=show_axes)
                ax.grid(False)
                plot_quadtree(
                    pred_qt,
                    ax=ax,
                    channel=0,
                    outline_width=outline_width,
                    colorbar=True,
                    cax=cax,
                    vmin=vmin_global,
                    vmax=vmax_global,
                    draw_outlines=draw_outlines,
                    max_pixels=pred_max_pixels,
                    x_frac=x_frac,
                    y_frac=y_frac,
                )
                anim_style.finalize_axes(ax, title=args.display_name, show_axes=show_axes)
                fig.savefig(f"{base_filename}_Pred_saveat{idx1}.png", dpi=anim_dpi)
                plt.close(fig)

    # --- 8. AUXILIARY PLOTS ---
    if args.aux_plot:
        if "pressure" not in data_config["field_names"]:
            print("Aux plot requested but 'pressure' field not found in data fields.")
        else:
            print("\nGenerating numerical soot-foil plots...")
            p_idx = data_config["field_names"].index("pressure")

            # Peak-hold (max-pressure) soot foil in physical space.
            gt_press_seq = gt_phys[:, p_idx : p_idx + 1, :, :, :]
            gt_pmax = None
            gt_extent = None

            for t in range(predict_steps):
                frame_data = np.ascontiguousarray(gt_press_seq[:, :, t, :, :])
                frame_tensor = torch.from_numpy(frame_data)
                step_meta = {"centers": ref_centers_np, "levels": ref_levels_np, "domain": domain}
                qt = tensor_to_quadtree(
                    data=frame_tensor, meta=step_meta, cell_scale_mode=cell_scale_mode
                )
                img, extent = _rasterize_quadtree(
                    qt, channel=0, max_pixels=aux_max_pixels, x_frac=x_frac, y_frac=y_frac
                )
                if gt_pmax is None:
                    gt_pmax = np.array(img, copy=True)
                else:
                    gt_pmax = np.maximum(gt_pmax, img)
                gt_extent = extent

            pred_pmax = None
            pred_extent = None
            pred_press_seqs = [p[:, p_idx : p_idx + 1, :, :, :] for p in all_preds]

            for t in range(predict_steps):
                frame_data = np.ascontiguousarray(pred_press_seqs[t][:, :, 0, :, :])
                frame_tensor = torch.from_numpy(frame_data)
                step_meta = {"centers": all_centers[t], "levels": all_levels[t], "domain": domain}
                qt = tensor_to_quadtree(
                    data=frame_tensor, meta=step_meta, cell_scale_mode=cell_scale_mode
                )
                img, extent = _rasterize_quadtree(
                    qt, channel=0, max_pixels=aux_max_pixels, x_frac=x_frac, y_frac=y_frac
                )
                if pred_pmax is None:
                    pred_pmax = np.array(img, copy=True)
                else:
                    pred_pmax = np.maximum(pred_pmax, img)
                pred_extent = extent

            # Shared color scale (pressure)
            vmin_soot = aux_pressure_threshold
            vmax_soot = float(np.max([gt_pmax.max(), pred_pmax.max()]))

            def _plot_soot(pmax_img, extent, title, out_path):
                fig, ax = plt.subplots(figsize=(7, 5), dpi=anim_dpi)
                im = ax.imshow(
                    pmax_img,
                    origin="lower",
                    extent=extent,
                    cmap="gray_r",
                    vmin=vmin_soot,
                    vmax=vmax_soot,
                    interpolation="nearest",
                )
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_title(title, fontsize=14, pad=10)
                fig.colorbar(im, ax=ax, pad=0.02, label="Pressure")
                fig.tight_layout()
                fig.savefig(out_path, dpi=anim_dpi, bbox_inches="tight")
                plt.close(fig)

            base_filename = os.path.join(
                anim_dir, f"{model_name}_pressure_traj{args.traj_idx}_frame{args.frame_idx}"
            )
            gt_base_filename = os.path.join(
                anim_dir, f"GT_pressure_traj{args.traj_idx}_frame{args.frame_idx}"
            )

            _plot_soot(
                gt_pmax,
                gt_extent,
                args.gt_display_name,
                f"{gt_base_filename}_soot_foil.png",
            )
            _plot_soot(
                pred_pmax,
                pred_extent,
                args.display_name,
                f"{base_filename}_Pred_soot_foil.png",
            )

    anim_end_time = time.time()
    print(f"Animation Rendering Completed in {anim_end_time - anim_start_time:.2f} seconds.")
    print(f"Total time: {anim_end_time - inf_start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
