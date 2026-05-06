import hashlib
import importlib
import inspect
import json
import math
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR


def _to_tensor(
    arr: np.ndarray | torch.Tensor, device: torch.device, non_blocking: bool = False
) -> torch.Tensor:
    """Convert numpy or torch input to a torch tensor on *device*."""
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr).to(device, non_blocking=non_blocking)
    return arr.to(device, non_blocking=non_blocking)


def _register_resolvers():
    """Register custom OmegaConf resolvers for derived config values.

    Resolvers are idempotent — safe to call multiple times.
    """
    if OmegaConf.has_resolver("sub"):
        return  # already registered

    # Arithmetic helpers
    OmegaConf.register_new_resolver("sub", lambda a, b: int(a) - int(b))
    OmegaConf.register_new_resolver("idiv", lambda a, b: int(a) // int(b))
    OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b))

    # rope_axes_dim for adaptive grids (3D: [xy, xy, cell_scale])
    # Rule: last_dim = 4*(num_levels+1), first two = (head_dim - last_dim) // 2
    def _rope_axes_dim_adaptive(head_dim, num_levels):
        head_dim, num_levels = int(head_dim), int(num_levels)
        last = 4 * (num_levels + 1)
        first = (head_dim - last) // 2
        return [first, first, last]

    # rope_axes_dim for regular grids (2D: [x, y])
    def _rope_axes_dim_regular(head_dim):
        half = int(head_dim) // 2
        return [half, half]

    OmegaConf.register_new_resolver("rope_axes_dim_adaptive", _rope_axes_dim_adaptive)
    OmegaConf.register_new_resolver("rope_axes_dim_regular", _rope_axes_dim_regular)


# Register resolvers at import time so they're available when configs are loaded
_register_resolvers()


def is_main_process():
    """Check if current worker is rank 0."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def load_hydra_config(config_dir=None, config_name="config", overrides=None):
    """Load a composed Hydra config from config groups.

    Args:
        config_dir: Absolute path to the configs directory. Defaults to
            ``<repo_root>/configs``.
        config_name: Name of the root config file (without ``.yaml``).
        overrides: List of Hydra override strings, e.g.
            ``["dataset=pli_adaptive", "env=hpc", "train.num_push_forward_steps=4"]``.

    Returns:
        A plain Python dict (resolved OmegaConf container).
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if config_dir is None:
        config_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
    config_dir = os.path.abspath(config_dir)

    # Clear any previous Hydra state (allows repeated calls in the same process)
    GlobalHydra.instance().clear()

    # Convert value overrides (key=val) to force-add (++key=val) so they work
    # regardless of struct mode.  Group overrides (no dots, e.g. dataset=X) and
    # overrides already prefixed with + or ~ are left untouched.
    resolved = []
    for o in overrides or []:
        if o.startswith("+") or o.startswith("~"):
            resolved.append(o)
        elif "=" in o:
            key = o.split("=", 1)[0]
            if "." in key:
                # dotted key  →  force-add so struct mode doesn't block it
                resolved.append("++" + o)
            else:
                resolved.append(o)
        else:
            resolved.append(o)

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=resolved)

    OmegaConf.set_struct(cfg, False)

    return OmegaConf.to_container(cfg, resolve=True)


def load_config(args, unknown_args):
    """Unified config loader with backward compatibility.

    If ``args.config`` is provided, loads a standalone YAML file directly
    (legacy mode). Otherwise, uses Hydra to compose config from groups
    using ``unknown_args`` as overrides.

    Args:
        args: Parsed argparse namespace (may have ``config`` attribute).
        unknown_args: Remaining CLI args forwarded to Hydra as overrides.

    Returns:
        A plain Python dict.
    """
    config_path = getattr(args, "config", None)
    if config_path:
        # Legacy mode: load a standalone YAML file directly
        config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    else:
        # Hydra compose mode
        config = load_hydra_config(overrides=unknown_args)

    # Store CLI flags in config for downstream use
    config["args"] = {k: v for k, v in vars(args).items() if k != "config"}
    return config


def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        raise Exception("target not in config! ", config)
    target = get_obj_from_str(config["target"])
    params = dict(config.get("params", {}))
    # Filter kwargs against the target's signature so Hydra dict-merges
    # (e.g. a base `loss_fn.params: {d, p, ...}` in mode/train.yaml merged
    # with an experiment's `loss_fn.params: {reduction: mean}`) don't
    # crash on unexpected kwargs. Invalid keys are logged by
    # filter_valid_kwargs so real typos stay visible.
    params = filter_valid_kwargs(target, params, verbose=True)
    return target(**params)


def _introspect_defaults(target_str: str) -> dict[str, Any]:
    """Return the default kwargs of ``target_str``'s ``__init__`` signature.

    Any parameter whose default is ``inspect.Parameter.empty`` (i.e. required)
    or which cannot be JSON-serialised is skipped so the result is safe to log
    to W&B. Silently returns an empty dict if the target cannot be imported.
    """
    try:
        obj = get_obj_from_str(target_str)
    except Exception:
        return {}
    try:
        sig = inspect.signature(obj.__init__ if inspect.isclass(obj) else obj)
    except (TypeError, ValueError):
        return {}
    defaults: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name in ("self", "cls", "args", "kwargs"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is inspect.Parameter.empty:
            continue
        try:
            json.dumps(param.default)
        except (TypeError, ValueError):
            # Non-serialisable default (e.g. a sentinel object) — record its
            # repr so W&B still sees something informative.
            defaults[name] = repr(param.default)
            continue
        defaults[name] = param.default
    return defaults


def resolve_effective_config(config: Any) -> Any:
    """Walk a config tree and materialise effective kwargs for any ``{target, params}``
    entry by merging ``inspect.signature`` defaults under the user-provided params.

    This makes the config logged to W&B reflect the actual instantiation kwargs,
    not just what appeared in YAML. If a class default changes between runs, the
    diff shows up in W&B instead of being silently absorbed.

    Keys outside of ``{target, params}`` blocks are recursed into but otherwise
    left untouched. The input is not mutated.
    """
    if isinstance(config, dict):
        if "target" in config and isinstance(config.get("params", {}), dict):
            defaults = _introspect_defaults(config["target"])
            user_params = dict(config.get("params", {}) or {})
            merged = {**defaults, **user_params}  # user overrides defaults
            out = {k: v for k, v in config.items() if k not in ("params",)}
            out["params"] = {k: resolve_effective_config(v) for k, v in merged.items()}
            out["_params_source"] = {
                k: ("user" if k in user_params else "default") for k in merged.keys()
            }
            return out
        return {k: resolve_effective_config(v) for k, v in config.items()}
    if isinstance(config, list):
        return [resolve_effective_config(v) for v in config]
    return config


def get_git_info(repo_dir: str | None = None) -> dict[str, Any]:
    """Return a small dict of git state for W&B logging.

    Fields: ``commit``, ``commit_short``, ``branch``, ``dirty`` (bool),
    ``dirty_files`` (list of changed paths, truncated to 20).
    Silently returns ``{"available": False}`` outside a git checkout or if
    ``git`` is unavailable. Never raises.
    """
    import subprocess

    def _run(args):
        return subprocess.run(
            args,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()

    try:
        commit = _run(["git", "rev-parse", "HEAD"])
        commit_short = _run(["git", "rev-parse", "--short", "HEAD"])
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        status = _run(["git", "status", "--porcelain"])
    except Exception:
        return {"available": False}

    # Porcelain v1 lines look like "XY path" (XY = 2-char status + space).
    # Use split(maxsplit=1) so we are robust to rename entries ("R  old -> new")
    # and to any stray whitespace in the prefix.
    dirty_files = []
    for line in status.splitlines():
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            dirty_files.append(parts[1])
    dirty_files = dirty_files[:20]
    return {
        "available": True,
        "commit": commit,
        "commit_short": commit_short,
        "branch": branch,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files,
    }


def get_scheduler_with_min_lr(
    name: str,
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    min_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    power: float = 1.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Wrapper around diffusers get_scheduler that enforces a minimum learning rate.

    Args:
        base_lr: The starting learning rate (must match optimizer's initial lr).
        min_lr: The target minimum learning rate.
    """
    min_lr_ratio = min_lr / base_lr
    if name == "polynomial":
        from diffusers.optimization import get_polynomial_decay_schedule_with_warmup

        return get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            lr_end=min_lr,  # Native support
            power=power,
            last_epoch=last_epoch,
        )

    if name == "cosine":

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            if current_step > num_training_steps:
                return min_lr_ratio

            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))  # Standard 1->0

            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    if name == "cosine_with_restarts":

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            if current_step > num_training_steps:
                return min_lr_ratio

            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )

            if progress >= 1.0:
                return min_lr_ratio

            return min_lr_ratio + (1.0 - min_lr_ratio) * (
                0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0)))
            )

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    # 4. Fallback for others (Constant, etc. don't need min_lr usually)
    return get_scheduler(
        name,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        power=power,
    )


class UniformLossMaskScheduler:
    def __init__(
        self,
        min_keep_ratio: float,
        max_step_ratio: float,
        schedule_type: str = "cosine",
        min_keep_patch: int = 100,
    ):
        """
        Generates a uniform random mask for loss calculation, gradually increasing
        the percentage of patches kept over time.

        Args:
            min_keep_ratio: Starting probability of keeping a patch (e.g., 0.15).
            max_step_ratio: The training progress ratio (0 to 1) at which the mask reaches 1.0.
            schedule_type: "linear" or "cosine"
            min_keep_patch: Absolute minimum number of patches to keep in any forward pass.
        """
        self.min_keep_ratio = min_keep_ratio
        self.max_step_ratio = max_step_ratio
        self.schedule_type = schedule_type
        self.min_keep_patch = min_keep_patch

        assert 0.0 <= min_keep_ratio <= 1.0, (
            f"min_keep_ratio should be in [0, 1], got {min_keep_ratio}"
        )
        assert 0.0 < max_step_ratio <= 1.0, (
            f"max_step_ratio should be in (0, 1], got {max_step_ratio}"
        )
        assert schedule_type in ["linear", "cosine"], "schedule_type must be 'linear' or 'cosine'"
        assert min_keep_patch >= 1, f"min_keep_patch must be at least 1, got {min_keep_patch}"

    def get_ratio(self, current_step: int, total_steps: int) -> float:
        # 1. Calculate how far along we are in the training process
        current_step_ratio = current_step / total_steps

        # 2. Calculate progress specifically relative to our masking schedule, clamping at 1.0
        progress = min(1.0, current_step_ratio / self.max_step_ratio)

        if progress == 1.0:
            return 1.0

        if self.schedule_type == "linear":
            return self.min_keep_ratio + (1.0 - self.min_keep_ratio) * progress
        elif self.schedule_type == "cosine":
            # Cosine annealing: starts slow, accelerates in the middle, slows down near 1.0
            return self.min_keep_ratio + (1.0 - self.min_keep_ratio) * 0.5 * (
                1.0 - math.cos(math.pi * progress)
            )

    def get_mask(
        self, num_patches: int, current_step: int, total_steps: int, device: torch.device
    ) -> torch.Tensor:
        """
        Generates the 1D boolean/float mask tensor for the current step.
        """
        current_ratio = self.get_ratio(current_step, total_steps)

        # Fast path for full grid
        if current_ratio >= 1.0:
            return torch.ones(num_patches, device=device)

        # Create binary mask (1 = keep, 0 = drop)
        mask = (torch.rand(num_patches, device=device) < current_ratio).float()

        # Safety fallback: Ensure at least `min_keep_patch` patches are selected
        current_keep_count = mask.sum().item()

        if current_keep_count < self.min_keep_patch:
            # Prevent requesting more patches than actually exist
            target_count = min(self.min_keep_patch, num_patches)
            shortfall = int(target_count - current_keep_count)

            if shortfall > 0:
                # Find indices that are currently 0 (dropped)
                zero_indices = (mask == 0).nonzero(as_tuple=True)[0]
                # Randomly select exactly `shortfall` number of these indices and flip them to 1
                chosen_zeros = zero_indices[
                    torch.randperm(len(zero_indices), device=device)[:shortfall]
                ]
                mask[chosen_zeros] = 1.0

        return mask


def filter_valid_kwargs(
    target: Callable | type, params: dict[str, Any], verbose: bool = True
) -> dict[str, Any]:
    """
    Filters a dictionary to only include keys that are valid arguments
    for the provided function or class initialization.

    Args:
        target: A class (inspects __init__) or a callable function.
        params: The dictionary of keyword arguments to filter.
        verbose: If True, prints diagnostic info about dropped keys.

    Returns:
        A new dictionary containing only the valid keyword arguments.
    """
    # Helper to safely get a printable name for the target
    target_name = getattr(target, "__name__", str(target))

    if inspect.isclass(target):
        target_func = getattr(target, "__init__", target)
    else:
        target_func = target

    try:
        sig = inspect.signature(target_func)
    except ValueError:
        if verbose:
            print(
                f"[Warning] Could not inspect signature for {target_name}. Returning params as-is."
            )
        return params

    # If the target explicitly accepts **kwargs, it accepts everything.
    has_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )
    if has_kwargs:
        if verbose:
            print(f"[Info] '{target_name}' accepts **kwargs. Keeping all keys.")
        return params

    # Filter the dictionary
    valid_keys = set(sig.parameters.keys())
    filtered_params = {k: v for k, v in params.items() if k in valid_keys}

    # Handle verbose output for dropped keys
    if verbose:
        dropped_keys = set(params.keys()) - valid_keys
        if dropped_keys:
            print(f"[Info] Dropped invalid keys for '{target_name}': {dropped_keys}")
        else:
            print(f"[Info] No keys needed to be dropped for '{target_name}'.")

    return filtered_params


def get_topology_cache_key(windows, params, role):
    """Generate a reproducible MD5 hash for one role's topology cache.

    Hashes the windows, TopologyBuilder parameters, and the role tag. String
    values (recursively, inside dicts and lists) are reduced to their
    ``os.path.basename`` so that the same data files mounted under different
    parent directories produce the same cache key. This assumes filenames are
    unique within each dataset (true for PLI ``id*idx*.npz`` and TRL
    ``*.hdf5``; potentially collision-prone for AMReX ``plt*`` across runs).

    Args:
        windows (list): Window descriptors / file pointers for this role.
        params (dict): Configuration passed to the TopologyBuilder map step.
        role (str): "train" or "val".

    Returns:
        str: An MD5 generated hex string encoding the per-role topology dependencies.
    """

    def normalize(obj):
        if isinstance(obj, dict):
            return {k: normalize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [normalize(v) for v in obj]
        if isinstance(obj, str):
            return os.path.basename(obj)
        return obj

    norm_windows = [normalize(w) for w in (windows or [])]
    norm_params = normalize(params) if params else {}

    try:
        cache_str = json.dumps(
            {
                "windows": norm_windows,
                "params": norm_params,
                "role": role,
            },
            sort_keys=True,
        )
    except Exception:
        # Fallback if params contains non-serializable objects
        cache_str = str(
            {
                "windows": norm_windows,
                "params": norm_params,
                "role": role,
            }
        )
    return hashlib.md5(cache_str.encode("utf-8")).hexdigest()


def _write_topology_cache_meta(
    topology_cache_dir,
    cache_key,
    topo_builder_params,
    config,
    windows,
    role,
):
    """Write a human-readable meta.json into the per-role topology cache directory."""
    from datetime import datetime

    meta = {
        "cache_key": cache_key,
        "role": role,
        "created_at": datetime.now().isoformat(),
        "dataset": config.get("dataset", {}).get("name", None)
        or config.get("general", {}).get("run_name", "unknown"),
        "topo_builder_params": {
            k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
            for k, v in topo_builder_params.items()
        },
        "file_loader": {
            k: str(v) if not isinstance(v, (int, float, bool, str, list, type(None))) else v
            for k, v in config.get("file_loader", {}).get("params", {}).items()
        },
        "num_windows": len(windows) if windows else 0,
        "git": get_git_info(),
    }
    meta_path = os.path.join(topology_cache_dir, "meta.json")
    try:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Topology cache metadata written to {meta_path}")
    except Exception as e:
        print(f"Warning: failed to write topology cache metadata: {e}")


def apply_two_stage_topology_pipeline(
    ds,
    windows,
    role,
    config,
    args,
    map_batch_size,
    actor_pool_size,
    mem_per_task,
    transform,
):
    """
    Apply the two-stage quadtree pipeline to a single Ray dataset for one role
    (train or val). Call once per role.

    - **Stage 1 (Topology Building)**: deterministic; materializes quadtree
      structures and persists them as Parquet under
      ``<topology_cache_dir>/<cache_key>/`` where ``cache_key`` is derived from
      this role's windows, the TopologyBuilder params, and the role tag. Used as
      an ultra-fast loading proxy on future runs with identical inputs.
    - **Stage 2 (Augmentation and Exporting)**: lazily iterates the cached
      topology each epoch. ``augment`` is enabled for ``role="train"`` and
      disabled for ``role="val"`` so validation runs on the canonical
      (deterministic) topology.

    Args:
        ds: Ray Dataset for this role, constructed natively from file shards.
        windows: Pre-split window descriptors / file pointers for this role.
        role: "train" or "val".
        config: Current run execution parameters defined through Hydra.
        args: Argparse configuration pointing parameters.
        map_batch_size: Memory constraints setting chunk quantities passed
            through the Actor node.
        actor_pool_size: TaskPool strategy parallelization limits.
        mem_per_task: Expected upper threshold object store allocations.
        transform: External module mapping data into PyTorch objects per batch.

    Returns:
        Ray Dataset: The mapped and initialized dataset ready for TorchTrainer.
    """
    import ray

    from wamrvit.dataloader.loader import FillAndExportMapper, TopologyBuilder
    from wamrvit.dataloader.trajectory_loader import CachedSeq2SeqMapper
    from wamrvit.utils import filter_valid_kwargs

    fl_params = dict(config["file_loader"].get("params", {}))

    # Stage 1: Build topology (deterministic, materializable)
    topo_builder_params = filter_valid_kwargs(TopologyBuilder, fl_params)

    default_ckpt_dir = os.path.join(
        config["general"]["save_dir"], config["general"].get("run_name", "amrex_run")
    )
    topology_cache_dir_base = (
        getattr(args, "topology_cache_dir", None)
        or config["file_loader"].get("topology_cache_dir")
        or os.path.join(default_ckpt_dir, "topology_cache")
    )

    loaded_from_cache = False
    cache_key = None
    cache_dir = None

    if topology_cache_dir_base:
        cache_key = get_topology_cache_key(windows, topo_builder_params, role)
        cache_dir = os.path.join(topology_cache_dir_base, cache_key)

        if os.path.exists(cache_dir) and any(
            f.endswith(".parquet") for f in os.listdir(cache_dir)
        ):
            print(f"Loading cached {role} quadtree topologies from {cache_dir}...")
            try:
                ds = ray.data.read_parquet(cache_dir)
                ds = ds.materialize()
                loaded_from_cache = True
                print(f"{role.capitalize()} topology cache successfully loaded.")
                # Backfill meta.json for caches created before this feature
                meta_path = os.path.join(cache_dir, "meta.json")
                if not os.path.exists(meta_path):
                    _write_topology_cache_meta(
                        cache_dir,
                        cache_key,
                        topo_builder_params,
                        config,
                        windows,
                        role,
                    )
            except Exception as e:
                print(f"Failed to load cached {role} topology ({e}). Rebuilding from scratch...")

    if not loaded_from_cache:
        topo_builder = TopologyBuilder(**topo_builder_params)

        print(f"Building and materializing {role} quadtree topologies (Stage 1)...")
        ds = ds.map_batches(
            topo_builder,
            batch_size=map_batch_size,
            batch_format="numpy",
            compute=ray.data.TaskPoolStrategy(size=actor_pool_size),
            memory=mem_per_task,
        )
        ds = ds.materialize()

        if topology_cache_dir_base:
            print(f"Writing {role} topologies to cache at {cache_dir}...")
            os.makedirs(cache_dir, exist_ok=True)
            ds.write_parquet(cache_dir)
            _write_topology_cache_meta(
                cache_dir,
                cache_key,
                topo_builder_params,
                config,
                windows,
                role,
            )

        print(f"{role.capitalize()} topology materialization complete.")

    # Stage 2: Fill + augment (lazy, re-runs each epoch)
    fill_common_params = filter_valid_kwargs(FillAndExportMapper, fl_params)
    augment = (role == "train")
    fill_params = dict(fill_common_params, augment=augment)
    fill_mapper = FillAndExportMapper(**fill_params)

    cache_pipeline = config.setdefault("__cache_topology_pipeline__", {})
    cache_pipeline["topo_builder_params"] = topo_builder_params
    cache_pipeline[f"fill_{role}_params"] = fill_params

    mapper = CachedSeq2SeqMapper(fill_mapper=fill_mapper, transform=transform)

    ds = ds.map_batches(
        mapper,
        batch_size=map_batch_size,
        batch_format="numpy",
        compute=ray.data.TaskPoolStrategy(size=actor_pool_size),
        memory=mem_per_task,
    )

    if getattr(args, "load_in_ram", False):
        print(f"Materializing final {role} tensors into RAM...")
        ds = ds.materialize()
        print(f"{role.capitalize()} materialization complete. Data is now pinned in memory.")

    return ds
