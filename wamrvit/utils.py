import importlib
import inspect
import json
import os
from collections.abc import Callable
from typing import Any

import torch.distributed as dist
from omegaconf import OmegaConf


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
