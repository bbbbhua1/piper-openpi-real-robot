#!/usr/bin/env python3
"""Pure-Python measured Piper state-history helpers for the ROS client."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
import math
from typing import Any, Deque, List


PHYSICAL_STATE_DIM = 14
HISTORY_LENGTH = 16


def _finite_vector(value: Any, length: int, field_name: str) -> List[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a {length}-value sequence")
    if len(value) != length:
        raise ValueError(
            f"{field_name} must contain exactly {length} values, got {len(value)}"
        )
    row: List[float] = []
    for index, item in enumerate(value):
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}[{index}] is not numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"{field_name}[{index}] must be finite")
        row.append(number)
    return row


def state_to_model_row(state: Mapping[str, Any]) -> List[float]:
    """Map Piper [left7, right7] state to model physical14 order."""
    if not isinstance(state, Mapping):
        raise ValueError("state must be an object with left_arm and right_arm")
    left = _finite_vector(state.get("left_arm"), 7, "state.left_arm")
    right = _finite_vector(state.get("right_arm"), 7, "state.right_arm")
    return [*left[:6], *right[:6], left[6], right[6]]


class MeasuredStateHistory:
    """Bounded chronological buffer of real robot state rows."""

    def __init__(self, max_length: int = HISTORY_LENGTH) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self._rows: Deque[List[float]] = deque(maxlen=int(max_length))

    def append(self, state: Mapping[str, Any]) -> None:
        self._rows.append(state_to_model_row(state))

    def rows(self) -> List[List[float]]:
        return [list(row) for row in self._rows]

    def __len__(self) -> int:
        return len(self._rows)
