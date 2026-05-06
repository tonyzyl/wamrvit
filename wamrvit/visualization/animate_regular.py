import argparse
import math
import os
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from wamrvit.dataloader.trajectory_loader import Seq2SeqMapper
from wamrvit.quad.yt_utils import make_regular_centers
from wamrvit.quadtree_transformer import QuadTreeTransformer
from wamrvit.utils import instantiate_from_config, load_config
from wamrvit.visualization.plotting import compute_vrange


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


def save_single_gif(seq, T, title_prefix, vmin, vmax, filename, dpi=150, fps=4):
    """Renders a regular-grid sequence to a GIF using imshow."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)

    im = ax.imshow(seq[0], origin="lower", cmap="jet", aspect="equal", vmin=vmin, vmax=vmax)
    ax.axis("off")

    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)

    def update(frame):
        ax.set_title(f"{title_prefix} (Frame {frame})", fontsize=16, pad=15)
        im.set_data(seq[frame])
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=T, blit=False)
    fig.tight_layout()
    anim.save(filename, writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Standalone Regular-Grid Animation Generator")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to standalone yaml file (legacy). Omit to use Hydra config groups.",
    )
    parser.add_argument("--traj_idx", type=int, required=True, help="Trajectory index to animate.")
    parser.add_argument("--frame_idx", type=int, required=True, help="Frame index to animate.")
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
    if "file_loader" in config:
        file_loader = instantiate_from_config(config["file_loader"])
    else:
        from wamrvit.dataloader.loader import YTAmReXRegularLoader

        fallback_fields = data_config.get("field_names", data_config.get("fields"))
        file_loader = YTAmReXRegularLoader(
            field_names=fallback_fields,
            domain_from=data_config.get("domain_from", "domain"),
        )
    transform = instantiate_from_config(config["transform"])
    mapper = Seq2SeqMapper(loader=file_loader, transform=transform)

    # Simulate a Ray batch of size 1 so the mapper parses it correctly
    ray_mock_batch = {k: [v] for k, v in target_window.items()}
    batch = mapper(ray_mock_batch)

    pred_mode = config["inference"].get("pred_mode", "target")

    # --- 4. LOAD MODEL ---
    inf_cfg = config["inference"]

    # When loading via diffusers, read _class_name from the checkpoint's
    # config.json so the YAML model block doesn't need to stay in sync.
    model_class_name = config["model"].get("_class_name", "QuadTreeTransformer")
    if inf_cfg.get("is_diffusers", False):
        import json

        ckpt_config_path = os.path.join(inf_cfg["checkpoint_path"], "config.json")
        if os.path.exists(ckpt_config_path):
            with open(ckpt_config_path) as f:
                model_class_name = json.load(f).get("_class_name", model_class_name)

    if model_class_name == "SwinV2Transformer":
        from wamrvit.swin_transformer import SwinV2Transformer

        model_cls = SwinV2Transformer
    else:
        model_cls = QuadTreeTransformer

    if inf_cfg.get("is_diffusers", False):
        model = model_cls.from_pretrained(inf_cfg["checkpoint_path"])
    else:
        model = model_cls(**config["model"])
        state_dict = torch.load(inf_cfg["checkpoint_path"], map_location="cpu")
        if "module." in list(state_dict.keys())[0]:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # --- 5. PREPARE TENSORS ---
    inputs = torch.from_numpy(batch["input"]).to(device, dtype=torch.float32)
    targets = torch.from_numpy(batch["target"]).to(device, dtype=torch.float32)

    B, C, T_in, H, W = inputs.shape
    R = model.config.return_seq_len
    num_forward_calls = math.ceil(predict_steps / R)
    centers = make_regular_centers(H, W, p=model.config.patch_size, device=device)

    curr_input_seq = inputs.clone()

    all_preds = []
    all_gts = []

    # --- 6. AUTOREGRESSIVE INFERENCE ---
    print(
        f"\nStarting inference loop (Steps: {predict_steps}, "
        f"Forward calls: {num_forward_calls}, R: {R})..."
    )
    inf_start_time = time.time()

    with torch.no_grad():
        for call_idx in range(num_forward_calls):
            timestep_idx = call_idx * R
            steps_this_call = min(R, predict_steps - timestep_idx)

            pred_full = model(curr_input_seq, centers)  # (B, C, R, H, W)

            if pred_mode == "residual":
                pred_full = pred_full + curr_input_seq[:, :, -1].unsqueeze(2)

            pred_to_store = pred_full[:, :, :steps_this_call]
            gt = targets[:, :, timestep_idx : timestep_idx + steps_this_call]
            pred_np = pred_to_store.cpu().numpy()
            gt_np = gt.cpu().numpy()

            if hasattr(transform, "inverse_transform") and callable(transform.inverse_transform):
                pred_phys = transform.inverse_transform(pred_np)
                gt_phys = transform.inverse_transform(gt_np)
            else:
                pred_phys = pred_np
                gt_phys = gt_np

            all_preds.append(pred_phys)
            all_gts.append(gt_phys)

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

    # Concatenate all steps: (B, C, T_total, H, W)
    pred_full = np.concatenate(all_preds, axis=2)
    gt_full = np.concatenate(all_gts, axis=2)
    T_total = pred_full.shape[2]

    # --- 7. ANIMATION RENDERING ---
    print("\nStarting animation rendering...")
    anim_start_time = time.time()

    anim_pairs = _resolve_anim_channels(anim_cfg, data_config["field_names"])

    # Use first batch element (b=0)
    for field, c_idx in anim_pairs:
        print(f"Rendering {field}...")

        # Extract (T, H, W) sequences for the chosen field and batch element 0
        pred_seq = pred_full[0, c_idx]  # (T_total, H, W)
        gt_seq = gt_full[0, c_idx]  # (T_total, H, W)

        # Shared colorscale across GT and Pred
        vmin_global, vmax_global = compute_vrange(
            [gt_seq],
            [pred_seq],
            field,
            anim_cfg,
        )

        gt_filename = os.path.join(
            anim_dir, f"GT_{field}_traj{args.traj_idx}_frame{args.frame_idx}.gif"
        )
        pred_filename = os.path.join(
            anim_dir, f"{model_name}_{field}_traj{args.traj_idx}_frame{args.frame_idx}_Pred.gif"
        )

        save_single_gif(
            gt_seq,
            T_total,
            f"GT: {field}",
            vmin_global,
            vmax_global,
            gt_filename,
            dpi=anim_dpi,
            fps=anim_fps,
        )
        save_single_gif(
            pred_seq,
            T_total,
            f"Pred: {field}",
            vmin_global,
            vmax_global,
            pred_filename,
            dpi=anim_dpi,
            fps=anim_fps,
        )

    anim_end_time = time.time()
    print(f"Animation Rendering Completed in {anim_end_time - anim_start_time:.2f} seconds.")
    print(f"Total time: {anim_end_time - inf_start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
