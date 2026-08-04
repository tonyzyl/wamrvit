"""Backend-neutral detector selection and payload-lineage representation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch


SOURCE_OP = np.uint8(0)
REFINE_OP = np.uint8(1)
COARSEN_OP = np.uint8(2)


@dataclass(frozen=True)
class PayloadOp:
    kind: Literal["source", "refine", "coarsen"]
    source_slot: int | None = None
    parent_op: int | None = None
    child_ops: tuple[int, int, int, int] | None = None
    quadrant: int | None = None
    phase: str = "source"


@dataclass(frozen=True)
class DetectorSpec:
    physical_channels: tuple[int, ...]
    field_names: tuple[str, ...]

    def __post_init__(self) -> None:
        channels = tuple(int(channel) for channel in self.physical_channels)
        names = tuple(str(name) for name in self.field_names)
        if not channels:
            raise ValueError("DetectorSpec requires at least one physical channel.")
        if len(set(channels)) != len(channels):
            raise ValueError(f"Detector channels must be unique, got {channels}.")
        if any(channel < 0 for channel in channels):
            raise ValueError(f"Detector channels must be non-negative, got {channels}.")
        if len(names) != len(channels):
            raise ValueError(
                "Detector field_names must have one entry per physical channel, "
                f"got {len(names)} names for {len(channels)} channels."
            )
        object.__setattr__(self, "physical_channels", channels)
        object.__setattr__(self, "field_names", names)

    def validate_channel_count(self, channels: int) -> None:
        invalid = [channel for channel in self.physical_channels if channel >= int(channels)]
        if invalid:
            raise ValueError(
                f"Detector channels {invalid} are outside payload channel count {channels}."
            )

    def flattened_latest(self, channels: int, timesteps: int) -> tuple[int, ...]:
        self.validate_channel_count(channels)
        if int(timesteps) <= 0:
            raise ValueError(f"timesteps must be positive, got {timesteps}.")
        offset = (int(timesteps) - 1) * int(channels)
        return tuple(offset + channel for channel in self.physical_channels)


@dataclass(frozen=True)
class LineagePlan:
    kind: np.ndarray
    parent: np.ndarray
    children: np.ndarray
    quadrant: np.ndarray
    final_slots: np.ndarray
    initial_slots: int
    patch_shape: tuple[int, int]

    def __post_init__(self) -> None:
        validate_lineage_plan(self)


def extract_latest_detectors(
    sequence: np.ndarray | torch.Tensor,
    spec: DetectorSpec,
) -> np.ndarray | torch.Tensor:
    """Select ``(N,K,H,W)`` latest-frame detector planes from ``(N,C,T,H,W)``."""
    if sequence.ndim != 5:
        raise ValueError(f"Expected sequence shape (N,C,T,H,W), got {tuple(sequence.shape)}.")
    spec.validate_channel_count(int(sequence.shape[1]))
    if int(sequence.shape[2]) <= 0:
        raise ValueError("Sequence must contain at least one timestep.")
    if isinstance(sequence, torch.Tensor):
        indices = torch.tensor(
            spec.physical_channels, dtype=torch.long, device=sequence.device
        )
        return torch.index_select(sequence[:, :, -1], 1, indices)
    return np.take(np.asarray(sequence)[:, :, -1], spec.physical_channels, axis=1)


def lineage_plan_from_payload_ops(
    payload_ops: Sequence[PayloadOp | None],
    *,
    next_slot: int,
    initial_slots: int,
    final_slots: np.ndarray,
    patch_shape: tuple[int, int],
) -> LineagePlan:
    """Convert source-array shadow records into compact array lineage."""
    next_slot = int(next_slot)
    initial_slots = int(initial_slots)
    if next_slot < initial_slots or len(payload_ops) < next_slot:
        raise ValueError(
            f"Invalid lineage bounds: initial={initial_slots}, next={next_slot}, "
            f"records={len(payload_ops)}."
        )
    kind = np.full(next_slot, 255, dtype=np.uint8)
    parent = np.full(next_slot, -1, dtype=np.int32)
    children = np.full((next_slot, 4), -1, dtype=np.int32)
    quadrant = np.full(next_slot, -1, dtype=np.int8)
    op_codes = {"source": SOURCE_OP, "refine": REFINE_OP, "coarsen": COARSEN_OP}

    for slot in range(next_slot):
        op = payload_ops[slot]
        if op is None:
            raise ValueError(f"Missing payload lineage operation for slot {slot}.")
        if op.kind not in op_codes:
            raise ValueError(f"Unknown payload lineage kind {op.kind!r} at slot {slot}.")
        kind[slot] = op_codes[op.kind]
        if op.kind == "source":
            if int(op.source_slot) != slot:
                raise ValueError(
                    f"Source lineage slot mismatch at {slot}: {op.source_slot}."
                )
        elif op.kind == "refine":
            if op.parent_op is None or op.quadrant is None:
                raise ValueError(f"Incomplete refine lineage at slot {slot}.")
            parent[slot] = int(op.parent_op)
            quadrant[slot] = int(op.quadrant)
        else:
            if op.child_ops is None or len(op.child_ops) != 4:
                raise ValueError(f"Incomplete coarsen lineage at slot {slot}.")
            children[slot] = np.asarray(op.child_ops, dtype=np.int32)

    return LineagePlan(
        kind=kind,
        parent=parent,
        children=children,
        quadrant=quadrant,
        final_slots=np.asarray(final_slots, dtype=np.int32),
        initial_slots=initial_slots,
        patch_shape=(int(patch_shape[0]), int(patch_shape[1])),
    )


def validate_lineage_plan(plan: LineagePlan) -> None:
    count = int(plan.kind.shape[0])
    if plan.kind.dtype != np.uint8 or plan.kind.ndim != 1:
        raise ValueError("Lineage kind must be a one-dimensional uint8 array.")
    if plan.parent.shape != (count,) or plan.parent.dtype != np.int32:
        raise ValueError("Lineage parent must be int32 with shape (N,).")
    if plan.children.shape != (count, 4) or plan.children.dtype != np.int32:
        raise ValueError("Lineage children must be int32 with shape (N,4).")
    if plan.quadrant.shape != (count,) or plan.quadrant.dtype != np.int8:
        raise ValueError("Lineage quadrant must be int8 with shape (N,).")
    if plan.final_slots.ndim != 1 or plan.final_slots.dtype != np.int32:
        raise ValueError("Lineage final_slots must be a one-dimensional int32 array.")
    if not 0 <= int(plan.initial_slots) <= count:
        raise ValueError("Lineage initial_slots is outside operation bounds.")
    if len(plan.patch_shape) != 2 or min(plan.patch_shape) <= 0:
        raise ValueError(f"Invalid lineage patch shape {plan.patch_shape}.")
    if np.any(plan.final_slots < 0) or np.any(plan.final_slots >= count):
        raise ValueError("Lineage final_slots contains an out-of-range slot.")

    for slot in range(count):
        op = plan.kind[slot]
        if slot < plan.initial_slots:
            if op != SOURCE_OP:
                raise ValueError(f"Initial slot {slot} is not a source operation.")
        elif op == REFINE_OP:
            dependency = int(plan.parent[slot])
            if dependency < 0 or dependency >= slot:
                raise ValueError(
                    f"Refine dependency {dependency} must precede slot {slot}."
                )
            if int(plan.quadrant[slot]) not in (0, 1, 2, 3):
                raise ValueError(f"Invalid refine quadrant at slot {slot}.")
        elif op == COARSEN_OP:
            dependencies = plan.children[slot]
            if np.any(dependencies < 0) or np.any(dependencies >= slot):
                raise ValueError(
                    f"Coarsen dependencies must precede slot {slot}: {dependencies.tolist()}."
                )
        else:
            raise ValueError(f"Invalid lineage operation code {int(op)} at slot {slot}.")
