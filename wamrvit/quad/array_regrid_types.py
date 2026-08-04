from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np

from wamrvit.quad.regrid_lineage import PayloadOp

Tol = Union[float, Sequence[float], np.ndarray]
Channel = Optional[Union[int, Sequence[int]]]
FineBounds = tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass
class FlatTopology:
    values: np.ndarray
    tile_ix: np.ndarray
    tile_iy: np.ndarray
    level_idx: np.ndarray
    x_idx: np.ndarray
    y_idx: np.ndarray
    domain: dict[str, Any]
    cell_scale_mode: Optional[str]


@dataclass
class ActiveTopology:
    values: np.ndarray
    tile_ix: np.ndarray
    tile_iy: np.ndarray
    level_idx: np.ndarray
    x_idx: np.ndarray
    y_idx: np.ndarray
    active_slots: np.ndarray
    active_count: int
    next_slot: int
    domain: dict[str, Any]
    cell_scale_mode: Optional[str]


@dataclass
class SequenceSourceActiveTopology(ActiveTopology):
    source_sequence: np.ndarray
    initial_slots: int
    sequence_channels: int
    sequence_timesteps: int
    source_sequence_copied: bool
    payload_ops: list[PayloadOp | None] | None
    lineage_phase: str


class ArrayFallback(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason
