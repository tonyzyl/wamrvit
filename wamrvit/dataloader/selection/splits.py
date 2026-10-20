"""Config-reproducible evaluation splits and optional source bookkeeping.

Source regions are authoritative, not saved JSON. Internal evaluation shares the
trainer's partition builder, including fixed-window training boundaries.
Explicit external test sources use prepartitioned evaluation. No checkpoint metadata is required.
This module has only standard-library import-time dependencies.
"""

import copy
import hashlib
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path

TRAINING_MANIFEST = "training_split.json"

_WINDOW_GENERATOR_TARGET_RENAMES = {
    "wamrvit.dataloader.trajectory_loader.generate_ar_windows": (
        "wamrvit.dataloader.selection.windows.generate_ar_windows"
    ),
    "wamrvit.dataloader.trajectory_loader.generate_multi_trajectory_windows": (
        "wamrvit.dataloader.selection.windows.generate_multi_trajectory_windows"
    ),
}


def _manifest_for_membership_comparison(manifest):
    """Canonicalize only known built-in target moves for record comparison."""
    result = copy.deepcopy(manifest)
    window_generator = result.get("window_generator")
    if isinstance(window_generator, dict):
        target = window_generator.get("target")
        if target in _WINDOW_GENERATOR_TARGET_RENAMES:
            window_generator["target"] = _WINDOW_GENERATOR_TARGET_RENAMES[target]
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _inventory(all_paths):
    """Preserve parser order, normalizing only the shared physical mount prefix.

    Relative subdirectories and HDF5 virtual trajectory/frame suffixes are kept;
    basenames alone are not sufficient to identify frames in multiple simulations.
    """
    trajectories = [all_paths] if all_paths and isinstance(all_paths[0], str) else all_paths
    if not trajectories or any(not trajectory for trajectory in trajectories):
        raise ValueError("A split inventory requires nonempty trajectories")
    physical = [os.path.abspath(p.split(":::", 1)[0]) for t in trajectories for p in t]
    root = os.path.commonpath([os.path.dirname(p) for p in physical])
    inventory, positions = [], {}
    for traj_idx, trajectory in enumerate(trajectories):
        frames = []
        for frame_idx, path in enumerate(trajectory):
            if path in positions:
                raise ValueError(f"Duplicate frame in split inventory: {path}")
            physical_path, separator, virtual = path.partition(":::")
            frames.append(
                os.path.relpath(os.path.abspath(physical_path), root) + separator + virtual
            )
            positions[path] = (traj_idx, frame_idx)
        inventory.append(frames)
    return inventory, positions


def _window_positions(window, positions):
    inputs = [positions[p] for p in window["input_paths"]]
    targets = [positions[p] for p in window["target_paths"]]
    if not inputs or not targets:
        raise ValueError("Windows must have both input and target frames")
    trajectory = inputs[0][0]
    if any(t != trajectory for t, _ in inputs + targets):
        raise ValueError("A window crosses trajectory boundaries")
    if window.get("traj_idx", trajectory) != trajectory:
        raise ValueError("Window trajectory ID disagrees with the source inventory")
    frame_indices = [i for _, i in inputs + targets]
    if any(a >= b for a, b in zip(frame_indices, frame_indices[1:])):
        raise ValueError("Window inputs and targets must be strictly chronological")
    return trajectory, [i for _, i in inputs], [i for _, i in targets]


def _window_digest(windows, positions):
    digest = hashlib.sha256()
    for window in windows:
        trajectory, inputs, targets = _window_positions(window, positions)
        digest.update(json.dumps([trajectory, inputs, targets]).encode() + b"\n")
    return digest.hexdigest()


def make_training_manifest(all_paths, train_windows, *, run_names, window_config, split_ratio):
    """Record the selected training windows without changing their membership.

    The frontier includes *all* training inputs and targets. Unused trajectories
    have frontier -1. The window hash authenticates reconstruction of legacy runs;
    it is not a hash of file contents or of the checkpoint weights.
    """
    inventory, positions = _inventory(all_paths)
    frontiers = [-1] * len(inventory)
    for window in train_windows:
        trajectory, inputs, targets = _window_positions(window, positions)
        frontiers[trajectory] = max(frontiers[trajectory], *inputs, *targets)
    if not train_windows:
        raise ValueError("Cannot establish a training boundary from empty training windows")
    return {
        "schema": "training_split_v1",
        "checkpoint_run_names": list(run_names),
        "inventory": inventory,
        "inventory_sha256": _digest(inventory),
        "last_training_frame": frontiers,
        "training_window_count": len(train_windows),
        "training_windows_sha256": _window_digest(train_windows, positions),
        "window_generator": {
            "target": window_config["target"],
            "params": {
                k: v for k, v in window_config.get("params", {}).items() if k != "file_path_list"
            },
        },
        "training_split_ratio": split_ratio,
    }


def _source_regions(manifest, inventory):
    """Validate recorded role regions and their conservative training frontier."""
    partitions = manifest.get("source_partitions")
    if not isinstance(partitions, dict) or set(partitions) != {"train", "val", "test"}:
        raise ValueError("Source membership must record train/val/test partitions")
    seen, frontiers = set(), [-1] * len(inventory)
    regions = {role: defaultdict(list) for role in partitions}
    for role, segments in partitions.items():
        for segment in segments:
            trajectory, positions = segment["trajectory"], segment["positions"]
            if type(trajectory) is not int or not 0 <= trajectory < len(inventory):
                raise ValueError("Invalid source-region trajectory")
            if (
                not positions
                or any(
                    type(p) is not int or not 0 <= p < len(inventory[trajectory]) for p in positions
                )
                or any(a >= b for a, b in zip(positions, positions[1:]))
            ):
                raise ValueError("Source-region positions must be nonempty, ordered and in range")
            identities = {(trajectory, p) for p in positions}
            if seen.intersection(identities):
                raise ValueError("Recorded source regions overlap across segments or roles")
            seen.update(identities)
            regions[role][trajectory].append(frozenset(positions))
            if role == "train":
                frontiers[trajectory] = max(frontiers[trajectory], positions[-1])
    if max(frontiers, default=-1) < 0:
        raise ValueError("The recorded training source partition is empty")
    if frontiers != manifest["last_training_frame"]:
        raise ValueError("Training frontier disagrees with permitted training source regions")
    return regions


def make_source_partition_manifest(all_paths, partitions, config, *, run_name):
    """Freeze permitted source regions; used by both training entrypoints.

    ``partitions`` contains the already-resolved SourceSegment objects. No source
    selection is repeated. Raw discovery retains unsampled frames and reserved
    tails; explicit external role segments are appended as separate trajectories.
    Internal evaluation must supply that same complete ordered inventory.
    """
    trajectories = [all_paths] if all_paths and isinstance(all_paths[0], str) else all_paths
    trajectories = [list(map(str, paths)) for paths in trajectories]
    positions = {p: (t, i) for t, paths in enumerate(trajectories) for i, p in enumerate(paths)}
    if set(partitions) - {"train", "val", "test"}:
        raise ValueError("Unknown source partition role")
    regions = {role: [] for role in ("train", "val", "test")}
    for role in regions:
        for segment in partitions.get(role, []):
            paths = list(map(str, segment.paths))
            if not paths:
                continue
            known = [p in positions for p in paths]
            if not any(known):
                trajectory = len(trajectories)
                trajectories.append(paths)
                positions.update({p: (trajectory, i) for i, p in enumerate(paths)})
            elif not all(known):
                raise ValueError("A source segment mixes discovered and external frames")
            trajectory = positions[paths[0]][0]
            if any(positions[p][0] != trajectory for p in paths):
                raise ValueError("A source segment crosses discovered trajectories")
            regions[role].append(
                {
                    "trajectory": trajectory,
                    "positions": [positions[p][1] for p in paths],
                    "source_indices": list(map(int, segment.indices)),
                    "trajectory_id": segment.trajectory_id,
                }
            )
    inventory, _ = _inventory(trajectories)
    frontiers = [-1] * len(inventory)
    for segment in regions["train"]:
        t = segment["trajectory"]
        frontiers[t] = max(frontiers[t], max(segment["positions"]))
    record = {
        # Additive extension: legacy consumers still see a safe, conservative
        # frontier. Updated consumers also support exact declared role regions.
        "schema": "training_split_v1",
        "membership_basis": "source_partitions",
        "checkpoint_run_names": [run_name],
        "inventory": inventory,
        "inventory_sha256": _digest(inventory),
        "last_training_frame": frontiers,
        "source_partitions": regions,
        "window_generator": {
            "target": config["window_generator"]["target"],
            "params": {
                k: v
                for k, v in config["window_generator"]["params"].items()
                if k != "file_path_list"
            },
        },
        "training_split_ratio": config["train"].get("split_ratio", 0.8),
        "num_push_forward_steps": config["train"].get("num_push_forward_steps", 1),
        "data_time": config.get("data", {}).get("time", {"source": "index", "step": 1.0}),
    }
    _source_regions(record, inventory)
    return record


def _write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def record_source_partitions(all_paths, partitions, config, *, default_run_name="train_run"):
    """Persist full permitted source regions before windowing in new training runs."""
    general = config["general"]
    run_name = general.get("run_name", default_run_name)
    manifest = make_source_partition_manifest(all_paths, partitions, config, run_name=run_name)
    _record_manifest(general["save_dir"], run_name, manifest)


def _record_manifest(save_dir, run_name, manifest):
    path = Path(save_dir) / run_name / TRAINING_MANIFEST
    if path.exists():
        saved = json.loads(path.read_text())
        if _manifest_for_membership_comparison(saved) != _manifest_for_membership_comparison(
            manifest
        ):
            raise ValueError(f"Training membership changed; refusing to overwrite {path}")
    else:
        # Missing bookkeeping is reconstructible from the supplied training config.
        _write_new_json(path, manifest)
    print(f"Training split manifest: {path}")


def _configured_training_manifest(all_paths, config):
    """Reconstruct membership without loading checkpoints or bookkeeping files."""
    from wamrvit.dataloader.selection.sources import prepare_source_partitions

    recipe = copy.deepcopy(config)
    inference = config["inference"]
    train = recipe.setdefault("train", {})
    # Accept the familiar rollout override, otherwise use the training key/default.
    ratio = inference.get("split_ratio", train.get("split_ratio", 0.8))
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 < ratio <= 1:
        raise ValueError("Training split_ratio must be in (0, 1]")
    train["split_ratio"] = ratio
    recipe.setdefault("data", {})
    partitions = prepare_source_partitions(recipe, all_paths)
    return make_source_partition_manifest(all_paths, partitions, recipe, run_name="configured")


def _check_saved_membership(all_paths, selected, inference):
    """Advisory checks only: missing/stale metadata never changes selected windows."""
    checks = []
    for key in ("checkpoint_path", "checkpoint_path_adaptive", "checkpoint_path_regular"):
        if not inference.get(key):
            continue
        run_dir = Path(inference[key]).parent
        path = run_dir / TRAINING_MANIFEST
        if not path.exists():
            checks.append({"checkpoint": key, "status": "unavailable"})
            continue
        try:
            records = json.loads(path.read_text())
            records = records if isinstance(records, list) else [records]
            if not records or any(not isinstance(record, dict) for record in records):
                raise ValueError("saved split must contain a record or list of records")
            matches = [r for r in records if run_dir.name in r.get("checkpoint_run_names", [])]
            if len(matches) != 1:
                raise ValueError("record does not identify this checkpoint unambiguously")
            eligible, _ = chronological_windows(
                all_paths,
                selected,
                matches,
                stride=1,
                eval_full=True,
                require_unseen_inputs=inference.get("require_unseen_inputs", True),
                evaluation_role=inference.get("evaluation_role"),
                require_unseen_trajectories=inference.get("require_unseen_trajectories", False),
            )
            if len(eligible) != len(selected):
                raise ValueError("configured evaluation overlaps recorded training membership")
        except (OSError, ValueError, KeyError, TypeError) as error:
            warnings.warn(
                f"Saved split cross-check for {inference[key]} failed: {error}. "
                "Using the configured split; held-out status for this checkpoint is not verified.",
                stacklevel=2,
            )
            checks.append({"checkpoint": key, "status": "unverified", "reason": str(error)})
        else:
            checks.append({"checkpoint": key, "status": "verified"})
    return checks


def chronological_windows(
    all_paths,
    windows,
    manifests,
    *,
    stride,
    include_tail=True,
    require_unseen_inputs=True,
    eval_full=False,
    evaluation_role=None,
    require_unseen_trajectories=False,
):
    """Filter by recorded training frontiers, then sample independently per trajectory.

    An explicit evaluation_role (val/test) instead requires every input and target
    inside one recorded region of that role, for each compared checkpoint. This
    supports held-out interior segments without inventing a chronological split.

    ``stride`` is in positions in the original parser-ordered source inventory,
    not compressed valid-window indices. Missing-frame rejection stays with the
    window generator. Tail inclusion can overlap other *evaluation* targets.
    """
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("Evaluation start stride must be a positive integer")
    inventory, positions = _inventory(all_paths)
    if not manifests:
        raise ValueError("Chronological evaluation requires training membership")
    if evaluation_role not in (None, "val", "test"):
        raise ValueError("evaluation_role must be val or test")
    if evaluation_role is not None:
        require_unseen_inputs = True
    frontiers = [-1] * len(inventory)
    role_regions = []
    bases = []
    for manifest in manifests:
        if manifest.get("schema") != "training_split_v1":
            raise ValueError("Unsupported training split manifest schema")
        basis = manifest.get("membership_basis", "fixed_windows")
        if basis not in {"fixed_windows", "source_partitions"}:
            raise ValueError("Unsupported training membership basis")
        bases.append(basis)
        if manifest.get("inventory_sha256") != _digest(inventory) or (
            "inventory" in manifest and manifest["inventory"] != inventory
        ):
            raise ValueError("Source inventory differs from the frozen training inventory")
        if basis == "source_partitions":
            regions = _source_regions(manifest, inventory)
            if evaluation_role is not None:
                role_regions.append(regions[evaluation_role])
        elif evaluation_role is not None:
            raise ValueError(
                "Explicit evaluation_role requires source-partition records for every checkpoint"
            )
        recorded = manifest["last_training_frame"]
        if len(recorded) != len(inventory):
            raise ValueError("Invalid training frontier count")
        for t, end in enumerate(recorded):
            if type(end) is not int or not -1 <= end < len(inventory[t]):
                raise ValueError("Invalid training frontier position")
            frontiers[t] = max(frontiers[t], end)

    eligible = defaultdict(list)
    for window in windows:
        trajectory, inputs, targets = _window_positions(window, positions)
        checked = inputs + targets if require_unseen_inputs else targets
        allowed = (
            all(
                any(set(checked) <= region for region in regions[trajectory])
                for regions in role_regions
            )
            if evaluation_role is not None
            else min(checked) > frontiers[trajectory]
        )
        if allowed and (not require_unseen_trajectories or frontiers[trajectory] < 0):
            eligible[trajectory].append((inputs[0], window))
    selected = []
    for trajectory in sorted(eligible):
        candidates = sorted(eligible[trajectory], key=lambda item: item[0])
        if eval_full:
            selected.extend(w for _, w in candidates)
            continue
        last_start = None
        for start, window in candidates:
            if last_start is None or start - last_start >= stride:
                selected.append(window)
                last_start = start
        if include_tail and last_start != candidates[-1][0]:
            selected.append(candidates[-1][1])
    if not selected:
        raise ValueError("No complete evaluation windows remain after the training boundary")
    return selected, {
        "inventory_sha256": _digest(inventory),
        "last_training_frame": frontiers,
        "last_training_paths": [
            frames[end] if end >= 0 else None for frames, end in zip(inventory, frontiers)
        ],
        "eligible_window_count": sum(len(group) for group in eligible.values()),
        "start_stride": stride,
        "include_tail": include_tail,
        "require_unseen_inputs": require_unseen_inputs,
        "evaluation_role": evaluation_role,
        "require_unseen_trajectories": require_unseen_trajectories,
        "membership_bases": bases,
    }


def _window_cohort(all_paths, windows, inference, manifests, *, eval_full):
    """Preserve PLI's global suffix/stride convention, with an independent safety filter.

    The suffix is a cohort-selection convention, never the training boundary.
    On the established PLI inventory it reproduces all 761 historical windows.
    Other inventories are still filtered against reconstructed training membership.
    """
    if inference.get("eval_start_stride") is not None or inference.get("include_eval_tail", False):
        raise ValueError(
            "Window-cohort sampling uses its global horizon stride, without tail insertion"
        )
    ratio = manifests[0]["training_split_ratio"]
    candidates = windows[int(len(windows) * ratio) :]
    eligible, metadata = chronological_windows(
        all_paths,
        candidates,
        manifests,
        stride=1,
        include_tail=False,
        eval_full=True,
        require_unseen_inputs=inference.get("require_unseen_inputs", True),
        evaluation_role=inference.get("evaluation_role"),
        require_unseen_trajectories=inference.get("require_unseen_trajectories", False),
    )
    if len(eligible) != len(candidates):
        warnings.warn(
            f"Removed {len(candidates) - len(eligible)} cohort windows that overlap "
            "the configured training region.",
            stacklevel=2,
        )
    stride = inference.get("predict_steps", 1)
    metadata.update(
        sampling="window_cohort", stride_windows=stride, candidate_window_count=len(candidates)
    )
    return eligible if eval_full else eligible[::stride], metadata


def evaluation_window_config(config, all_paths):
    """Build evaluation candidates without mutating/reusing training's temporal slice."""
    window_config = copy.deepcopy(config["window_generator"])
    params = window_config.setdefault("params", {})
    params["file_path_list"] = all_paths
    params["return_seq_len"] = config["inference"].get("predict_steps", 1)
    # These fractions specify the training source region, not an evaluation crop.
    params.pop("trajectory_start_fraction", None)
    params.pop("trajectory_stop_fraction", None)
    for name in ("sampling_interval", "interval_between_pred"):
        if name in config["inference"]:
            params[name] = config["inference"][name]
    return window_config


def prepare_evaluation_windows(all_paths, config, *, eval_full=False, debug=False):
    """Prepare the shared fixed-window rollout cohort, then attach global IDs.

    Used by adaptive, combined and regular rollout entrypoints. Selection writes
    its provenance before IDs are attached, as in the original entrypoints.
    Source clocks and experimental temporal conditioning are not attached.
    """
    from wamrvit.utils import instantiate_from_config

    windows = instantiate_from_config(evaluation_window_config(config, all_paths))
    windows = select_evaluation_windows(
        all_paths, windows, config, eval_full=eval_full, debug=debug,
    )
    trajectories = set()
    for index, window in enumerate(windows):
        if isinstance(window, dict):
            window["idx"] = index
            trajectories.add(window.get("traj_idx", 0))
    print(
        f"Total samples in validation split: {len(windows)}, "
        f"Trajectories: {len(trajectories)}\n"
    )
    return windows


def select_evaluation_windows(all_paths, windows, config, *, eval_full=False, debug=False):
    """Reconstruct the configured split, select windows, and record output provenance.

    internal (default) shares training's fixed-window partition builder.
    prepartitioned uses an explicitly supplied external test dataset. Saved metadata
    only cross-checks the selected cohort and is never an execution prerequisite.
    """
    inference = config["inference"]
    steps = inference.get("predict_steps", 1)
    policy = inference.get("split_policy", "internal")
    if policy not in {"internal", "prepartitioned"}:
        raise ValueError(f"Unsupported split_policy {policy!r}; use internal or prepartitioned")
    if type(steps) is not int or steps < 1:
        raise ValueError("predict_steps must be a positive integer")
    if any(len(w["target_paths"]) != steps for w in windows):
        raise ValueError("Evaluation windows do not match the requested forecast horizon")
    training_recipe = None
    if policy != "prepartitioned":
        manifests = [_configured_training_manifest(all_paths, config)]
        training_recipe = {
            key: manifests[0][key]
            for key in (
                "window_generator",
                "training_split_ratio",
                "num_push_forward_steps",
            )
        }
        if inference.get("preserve_window_cohort", False):
            selected, metadata = _window_cohort(
                all_paths,
                windows,
                inference,
                manifests,
                eval_full=eval_full,
            )
        else:
            params = evaluation_window_config(config, all_paths)["params"]
            stride = inference.get("eval_start_stride")
            if stride is None:
                stride = (
                    steps
                    * params.get("sampling_interval", 1)
                    * params.get("interval_between_pred", 1)
                )
            selected, metadata = chronological_windows(
                all_paths,
                windows,
                manifests,
                stride=stride,
                include_tail=inference.get("include_eval_tail", True),
                require_unseen_inputs=inference.get("require_unseen_inputs", True),
                eval_full=eval_full,
                evaluation_role=inference.get("evaluation_role"),
                require_unseen_trajectories=inference.get("require_unseen_trajectories", False),
            )
            metadata["sampling"] = "source_position_stride"
    else:
        if inference.get("split_ratio", 0) != 0:
            raise ValueError("prepartitioned evaluation cannot apply a nonzero split_ratio")
        if (
            inference.get("preserve_window_cohort", False)
            or inference.get("eval_start_stride") is not None
            or inference.get("include_eval_tail", False)
            or inference.get("evaluation_role") is not None
        ):
            raise ValueError(
                "prepartitioned evaluation uses all test windows and its horizon stride"
            )
        inventory, positions = _inventory(all_paths)
        for window in windows:
            _window_positions(window, positions)
        selected = windows if eval_full else windows[::steps]
        metadata = {
            "inventory_sha256": _digest(inventory),
            "separation_basis": "externally_provided_partition",
            "source_glob": config.get("data", {}).get("glob_pattern"),
            "eligible_window_count": len(windows),
            "sampling": "window_stride",
            "stride_windows": steps,
            "include_tail": False,
        }
    if not selected:
        raise ValueError("The evaluation partition has no complete windows")
    if debug:
        selected = selected[:100]
    csv_path = Path(inference["output_csv"])
    if csv_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {csv_path}")
    manifest_path = str(csv_path) + ".windows.json"
    _write_new_json(
        manifest_path,
        {
            "schema": "evaluation_windows_v1",
            "split_policy": policy,
            "predict_steps": steps,
            "eval_full": eval_full,
            "debug": debug,
            **metadata,
            "training_recipe": training_recipe,
            "saved_membership_checks": (
                _check_saved_membership(all_paths, selected, inference)
                if policy != "prepartitioned"
                else []
            ),
            "checkpoints": {k: v for k, v in inference.items() if k.startswith("checkpoint_path")},
            "selected_window_count": len(selected),
            "windows": selected,
        },
    )
    print(f"{policy} evaluation: {len(selected)} windows; manifest: {manifest_path}")
    if not eval_full:
        print(f"Sparse evaluation. Total evaluation samples: {len(selected)}")
    return selected
