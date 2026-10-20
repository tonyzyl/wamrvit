"""Build missing reference topologies without launching training.

Use the shared environment from AGENTS.md, for example:
    uv run --no-sync -m wamrvit.dataloader.tools.build_topology_cache \
        +experiment=pli_adaptive_train --topology_cache_dir /path/to/cache

Legacy window-Parquet caches are read only by the explicit topology_store
migration command, never by this builder or the training runtime.
"""

import argparse
import os

import ray

def main():
    """CLI: prepare source/window descriptors and build the shared reference store."""
    from wamrvit.dataloader.payload.loaders import TopologyBuilder
    from wamrvit.dataloader.selection.windows import prepare_window_splits
    from wamrvit.dataloader.selection.sources import prepare_source_partitions
    from wamrvit.dataloader.pipeline import prepare_reference_store
    from wamrvit.dataloader.topology_store import validate_topology_store_options
    from wamrvit.utils import filter_valid_kwargs, instantiate_from_config, load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Standalone YAML; otherwise use Hydra.")
    parser.add_argument("--topology_cache_dir", default=None)
    parser.add_argument(
        "--debug", action="store_true", help="Limit parsed training input to ten entries."
    )
    parser.add_argument("--actor_pool_size", type=int, default=None)
    parser.add_argument("--map_batch_size", type=int, default=None)
    parser.add_argument("--role", choices=("all", "train", "val", "test"), default="all")
    args, unknown = parser.parse_known_args()
    config = load_config(args, unknown)
    validate_topology_store_options(config.get("topology_store", {}))
    batch_size = args.map_batch_size or config["train"].get("map_batch_size", 1)
    if batch_size != 1:
        raise ValueError("Reference topology building requires map_batch_size=1.")
    file_parser = instantiate_from_config(config["file_parser"])
    paths = file_parser(config["data"]["glob_pattern"])
    if args.debug:
        paths = paths[:10]
    partitions = prepare_source_partitions(config, paths)
    _, windows_by_role = prepare_window_splits(config, partitions)
    selected = [
        window
        for role, rows in windows_by_role.items()
        if args.role in ("all", role)
        for window in rows
    ]
    if not selected:
        raise ValueError(f"No windows selected for --role={args.role}.")
    params = filter_valid_kwargs(TopologyBuilder, config["file_loader"].get("params", {}))
    cache_base = (
        args.topology_cache_dir
        or config["file_loader"].get("topology_cache_dir")
        or os.path.join(
            config["general"]["save_dir"],
            config["general"].get("run_name", "train_run"),
            "topology_cache",
        )
    )
    try:
        if not ray.is_initialized():
            ray.init(
                object_store_memory=int(
                    config["general"].get("ray_object_store_memory", 8) * 1024**3
                ),
                object_spilling_directory=config["general"].get("ray_spill_dir"),
            )
        _, store_path = prepare_reference_store(
            selected,
            params,
            config,
            cache_base,
            actor_pool_size=args.actor_pool_size or config["train"].get("actor_pool_size"),
            memory=10 * 1024**3,
        )
        print(f"Shared reference topology store: {store_path}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
