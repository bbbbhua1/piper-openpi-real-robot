#!/usr/bin/env python3
"""Robot-local HDF5 recording for replayable Piper policy trajectories."""

from __future__ import annotations

import math
from pathlib import Path
import re
import time
from typing import Any, List, Mapping, Optional, Union


PIPER_WIRE_DIM = 14
PIPER_WIRE_LAYOUT = "[L1..L6,Lg,R1..R6,Rg]"


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return (safe or "recording")[:120]


def _finite_vector(value: Any, field_name: str, expected_len: int) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) != expected_len:
        raise ValueError(
            "{} must contain exactly {} numeric values".format(
                field_name, expected_len
            )
        )
    result = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError("{} contains a boolean".format(field_name))
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("{} contains a non-numeric value".format(field_name)) from exc
        if not math.isfinite(number):
            raise ValueError("{} contains a non-finite value".format(field_name))
        result.append(number)
    return result


def piper_state_to_wire_row(state: Mapping[str, Any]) -> List[float]:
    left = _finite_vector(state.get("left_arm"), "left_arm", 7)
    right = _finite_vector(state.get("right_arm"), "right_arm", 7)
    return left + right


class TrajectoryHDF5Recorder:
    """Append actual absolute Piper commands and measured states to HDF5."""

    def __init__(
        self,
        root_dir: Union[str, Path],
        *,
        session_name: Optional[str] = None,
        session_dir: Optional[Union[str, Path]] = None,
        instruction: str = "",
    ) -> None:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "--record-trajectory requires h5py; install h5py in the robot Python environment"
            ) from exc

        if session_dir is None:
            root = Path(root_dir).expanduser()
            root.mkdir(parents=True, exist_ok=True)
            chosen_name = _safe_component(
                session_name or time.strftime("piper_%Y%m%d_%H%M%S")
            )
            self.session_dir = root / chosen_name
            self.session_dir.mkdir(parents=False, exist_ok=False)
        else:
            self.session_dir = Path(session_dir).expanduser()
            self.session_dir.mkdir(parents=True, exist_ok=True)
            chosen_name = _safe_component(session_name or self.session_dir.name)

        self.path = self.session_dir / "trajectory.hdf5"
        if self.path.exists():
            raise FileExistsError("trajectory file already exists: {}".format(self.path))

        self._h5 = h5py.File(str(self.path), "w")
        self._closed = False
        self._start_monotonic = time.monotonic()
        self._last_timestamp: Optional[float] = None
        self._rows = 0

        float_chunks = (256, PIPER_WIRE_DIM)
        scalar_chunks = (256,)
        self._h5.create_dataset(
            "action",
            shape=(0, PIPER_WIRE_DIM),
            maxshape=(None, PIPER_WIRE_DIM),
            dtype="f8",
            chunks=float_chunks,
        )
        self._h5.create_dataset(
            "state",
            shape=(0, PIPER_WIRE_DIM),
            maxshape=(None, PIPER_WIRE_DIM),
            dtype="f8",
            chunks=float_chunks,
        )
        for name in ("timestamp", "dt"):
            self._h5.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                dtype="f8",
                chunks=scalar_chunks,
            )
        for name in ("request_index", "waypoint_index"):
            self._h5.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                dtype="i8",
                chunks=scalar_chunks,
            )
        self._h5.create_dataset(
            "phase",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            chunks=scalar_chunks,
        )
        self._h5.create_dataset(
            "initial_state",
            shape=(PIPER_WIRE_DIM,),
            dtype="f8",
        )

        self._h5.attrs["format_version"] = 1
        self._h5.attrs["action_type"] = "absolute_joint_position"
        self._h5.attrs["units"] = "radians"
        self._h5.attrs["layout"] = PIPER_WIRE_LAYOUT
        self._h5.attrs["state_layout"] = PIPER_WIRE_LAYOUT
        self._h5.attrs["scope"] = "policy_execution"
        self._h5.attrs["instruction"] = str(instruction)
        self._h5.attrs["session_name"] = chosen_name
        self._h5.attrs["created_at_unix"] = float(time.time())
        self._h5.attrs["initial_state_recorded"] = False
        self._h5.attrs["num_rows"] = 0

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("trajectory recorder is already closed")

    def set_initial_state(self, state: Mapping[str, Any]) -> None:
        self._ensure_open()
        self._h5["initial_state"][...] = piper_state_to_wire_row(state)
        self._h5.attrs["initial_state_recorded"] = True
        self._h5.flush()

    def append(
        self,
        action: Mapping[str, Any],
        state: Mapping[str, Any],
        *,
        request_index: int,
        waypoint_index: int,
        phase: str = "policy",
    ) -> None:
        self._ensure_open()
        action_row = piper_state_to_wire_row(action)
        state_row = piper_state_to_wire_row(state)
        timestamp = time.monotonic() - self._start_monotonic
        dt = 0.0 if self._last_timestamp is None else timestamp - self._last_timestamp
        self._last_timestamp = timestamp

        row = self._rows
        for name in ("action", "state"):
            dataset = self._h5[name]
            dataset.resize((row + 1, PIPER_WIRE_DIM))
        for name in ("timestamp", "dt", "request_index", "waypoint_index", "phase"):
            dataset = self._h5[name]
            dataset.resize((row + 1,))

        self._h5["action"][row, :] = action_row
        self._h5["state"][row, :] = state_row
        self._h5["timestamp"][row] = timestamp
        self._h5["dt"][row] = dt
        self._h5["request_index"][row] = int(request_index)
        self._h5["waypoint_index"][row] = int(waypoint_index)
        self._h5["phase"][row] = str(phase)
        self._rows += 1
        self._h5.attrs["num_rows"] = self._rows
        self._h5.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._h5.attrs["closed_at_unix"] = float(time.time())
        self._h5.flush()
        self._h5.close()
        self._closed = True

    def __enter__(self) -> "TrajectoryHDF5Recorder":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
