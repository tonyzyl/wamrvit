"""Standalone script to build and cache quadtree topology for a given config.

Usage:
    # Hydra-style
    uv run wamrvit/dataloader/build_topology_cache.py dataset=amrex_adaptive mode=train

    # Legacy single-file
    uv run wamrvit/dataloader/build_topology_cache.py --config configs/amrex_ar.yaml

    # Override cache directory
    uv run wamrvit/dataloader/build_topology_cache.py --config configs/amrex_ar.yaml \
        --topology_cache_dir path_to_topo_cache
"""

import argparse
import os
import sys

import ray

# Ensure project root is importable when invoked directly
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from wamrvit.utils import (
    load_config,
    instantiate_from_config,
    filter_valid_kwargs,
    get_topology_cache_key,
    _write_topology_cache_meta,
)
from wamrvit.dataloader.loader import TopologyBuilder


def main():
    parser = argparse.ArgumentParser(description="Build and cache quadtree topology without launching training.")
    parser.add_argument("--config", type=str, default=None, help="Path to standalone yaml (legacy). Omit for Hydra.")
    parser.add_argument("--topology_cache_dir", type=str, default=None, help="Override topology cache directory.")
    parser.add_argument("--debug", action="store_true", help="Limit to 10 snapshots for quick testing.")
    parser.add_argument("--actor_pool_size", type=int, default=None, help="Override Ray actor pool size.")
    parser.add_argument("--map_batch_size", type=int, default=None, help="Override map batch size.")
    args, unknown = parser.parse_known_args()

    config = load_config(args, unknown)

    # ---- Ray init ----
    if not ray.is_initialized():
        ray.init(
            object_store_memory=config["general"].get("ray_object_store_memory", 8) * 1024 * 1024 * 1024,
            object_spilling_directory=config["general"].get("ray_spill_dir", None),
        )

    # ---- Data discovery (mirrors train_ray_AR_amrex.py) ----
    data_config = config["data"]
    file_parser = instantiate_from_config(config["file_parser"])
    all_paths = file_parser(data_config["glob_pattern"])
    if args.debug:
        all_paths = all_paths[:10]

    val_paths = None
    if "val_glob_pattern" in data_config:
        val_paths = file_parser(data_config["val_glob_pattern"])
        if args.debug:
            val_paths = val_paths[:10]
        print(f"Found {len(val_paths)} validation snapshots.")

    print(f"Found {len(all_paths)} snapshots.")

    # ---- Window generation ----
    window_config = dict(config["window_generator"])
    window_config["params"]["return_seq_len"] *= config["train"].get("num_push_forward_steps", 1)
    params = dict(window_config.get("params", {}))
    params["file_path_list"] = all_paths
    window_config["params"] = params
    windows = instantiate_from_config(window_config)

    if val_paths is not None:
        val_window_config = window_config.copy()
        val_window_config["params"] = {**params, "file_path_list": val_paths}
        val_windows = instantiate_from_config(val_window_config)
        train_windows = windows
    else:
        split_ratio = config["train"].get("split_ratio", 0.8)
        split_idx = int(len(windows) * split_ratio)
        train_windows = windows[:split_idx]
        val_windows = windows[split_idx:]

    print(f"Train windows: {len(train_windows)} | Val windows: {len(val_windows)}")

    # ---- Topology builder params ----
    fl_params = dict(config["file_loader"].get("params", {}))
    topo_builder_params = filter_valid_kwargs(TopologyBuilder, fl_params)

    # ---- Cache directory ----
    default_ckpt_dir = os.path.join(
        config["general"]["save_dir"],
        config["general"].get("run_name", "amrex_run"),
    )
    topology_cache_dir_base = (
        args.topology_cache_dir
        or config["file_loader"].get("topology_cache_dir")
        or os.path.join(default_ckpt_dir, "topology_cache")
    )

    map_batch_size = args.map_batch_size or config["train"].get("map_batch_size", 1)
    actor_pool_size = args.actor_pool_size or config["train"].get("actor_pool_size", None)
    mem_per_task = 10 * 1024 * 1024 * 1024

    def build_role_cache(windows, role):
        if not windows:
            return
        cache_key = get_topology_cache_key(windows, topo_builder_params, role)
        cache_dir = os.path.join(topology_cache_dir_base, cache_key)

        if os.path.exists(cache_dir) and any(
            f.endswith(".parquet") for f in os.listdir(cache_dir)
        ):
            print(f"{role.capitalize()} cache already exists at {cache_dir}")
            meta_path = os.path.join(cache_dir, "meta.json")
            if not os.path.exists(meta_path):
                _write_topology_cache_meta(
                    cache_dir, cache_key, topo_builder_params,
                    config, windows, role,
                )
            else:
                print("meta.json already present.")
            return

        topo_builder = TopologyBuilder(**topo_builder_params)
        ds = ray.data.from_items(windows)
        if role == "train":
            ds = ds.random_shuffle(seed=config["train"].get("seed", 42))

        print(f"Building {role} quadtree topologies (Stage 1)...")
        ds = ds.map_batches(
            topo_builder, batch_size=map_batch_size, batch_format="numpy",
            compute=ray.data.TaskPoolStrategy(size=actor_pool_size),
            memory=mem_per_task,
        )
        ds = ds.materialize()

        print(f"Writing {role} topology cache to {cache_dir}...")
        os.makedirs(cache_dir, exist_ok=True)
        ds.write_parquet(cache_dir)

        _write_topology_cache_meta(
            cache_dir, cache_key, topo_builder_params,
            config, windows, role,
        )
        print(f"  {role} cache_key: {cache_key}")
        print(f"  {cache_dir}")

    build_role_cache(train_windows, "train")
    build_role_cache(val_windows, "val")

    ray.shutdown()


if __name__ == "__main__":
    main()
