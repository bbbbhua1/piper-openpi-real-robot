#!/usr/bin/env python3
"""HDF5 recorder for the raw model actions returned by the Piper server.

The authoritative ``action`` dataset uses the model's physical 14-D layout:
``[left_arm6, right_arm6, left_gripper, right_gripper]``.  A second dataset
contains the same raw values mapped to Piper's legacy 7+7 wire layout so a
future replay tool does not have to duplicate the layout conversion.
"""

from __future__ import annotations

import math
from pathlib import Path
from collections.abc import Sequence
import time
from typing import Any, Mapping


PHYSICAL_ACTION_DIM = 14


def _finite_vector(value: Sequence[Any], field_name: str) -> list[float]:
    if isinstance(value, (str, bytes, bytearray)) or not hasattr(value, "__len__"):
        raise ValueError(f"{field_name} must be a 14-value sequence")
    if len(value) != PHYSICAL_ACTION_DIM:
        raise ValueError(
            f"{field_name} must contain exactly {PHYSICAL_ACTION_DIM} values, "
            f"got {len(value)}"
        )
    result: list[float] = []
    for index, item in enumerate(value):
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{field_name}[{index}] must be finite")
        result.append(number)
    return result


def model_row_to_piper_wire(row: Sequence[Any]) -> list[float]:
    """Map model physical14 ``arms_then_grippers`` to Piper 7+7 order."""
    values = _finite_vector(row, "model action")
    return [*values[:6], values[12], *values[6:12], values[13]]


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"could not find an unused HDF5 path beside {path}")


class RawModelActionHDF5Recorder:
    """Append the raw model-action prefix from each successful server response."""

    def __init__(self, path: str | Path, *, max_steps_per_request: int = 8) -> None:
        if max_steps_per_request <= 0:
            raise ValueError("max_steps_per_request must be positive")

        try:
            import h5py  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised at deployment
            raise RuntimeError(
                "server-side raw-action HDF5 recording requires h5py in the "
                "uva-dit-piper environment"
            ) from exc

        requested_path = Path(path).expanduser()
        requested_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = _next_available_path(requested_path)
        self.max_steps_per_request = int(max_steps_per_request)
        self._request_index = 0
        self._closed = False
        self._h5 = h5py.File(self.path, "w")
        string_dtype = h5py.string_dtype(encoding="utf-8")

        self._h5.attrs.update(
            {
                "format": "piper_uva_dit_raw_model_actions_v1",
                "source": "server_model_output_before_robot_safety_postprocess",
                "raw_model_output": True,
                "action_type": "absolute_joint_position",
                "units": "radians",
                "model_action_dim": PHYSICAL_ACTION_DIM,
                "model_action_layout": (
                    "left_arm6,right_arm6,left_gripper,right_gripper"
                ),
                "piper_wire_layout": "left_arm6,left_gripper,right_arm6,right_gripper",
                "saved_prefix_steps": self.max_steps_per_request,
            }
        )
        self._action = self._h5.create_dataset(
            "action",
            shape=(0, PHYSICAL_ACTION_DIM),
            maxshape=(None, PHYSICAL_ACTION_DIM),
            dtype="f8",
            chunks=True,
        )
        self._action_piper_wire = self._h5.create_dataset(
            "action_piper_wire",
            shape=(0, PHYSICAL_ACTION_DIM),
            maxshape=(None, PHYSICAL_ACTION_DIM),
            dtype="f8",
            chunks=True,
        )
        self._current_state = self._h5.create_dataset(
            "current_state",
            shape=(0, PHYSICAL_ACTION_DIM),
            maxshape=(None, PHYSICAL_ACTION_DIM),
            dtype="f8",
            chunks=True,
        )
        self._timestamp = self._h5.create_dataset(
            "timestamp", shape=(0,), maxshape=(None,), dtype="f8", chunks=True
        )
        self._dt = self._h5.create_dataset(
            "dt", shape=(0,), maxshape=(None,), dtype="f8", chunks=True
        )
        self._request_index_ds = self._h5.create_dataset(
            "request_index", shape=(0,), maxshape=(None,), dtype="i8", chunks=True
        )
        self._waypoint_index = self._h5.create_dataset(
            "waypoint_index", shape=(0,), maxshape=(None,), dtype="i8", chunks=True
        )
        self._step = self._h5.create_dataset(
            "step", shape=(0,), maxshape=(None,), dtype="i8", chunks=True
        )
        self._request_id = self._h5.create_dataset(
            "request_id", shape=(0,), maxshape=(None,), dtype=string_dtype, chunks=True
        )
        self._instruction = self._h5.create_dataset(
            "instruction", shape=(0,), maxshape=(None,), dtype=string_dtype, chunks=True
        )
        self._initial_state = self._h5.create_dataset(
            "initial_state", shape=(PHYSICAL_ACTION_DIM,), dtype="f8"
        )
        self._initial_state[:] = float("nan")
        self._h5.attrs["initial_state_valid"] = False
        self._h5.attrs["num_requests"] = 0

    def record(
        self,
        *,
        request_id: str,
        step: int,
        instruction: str,
        raw_actions: Sequence[Sequence[Any]],
        current_state: Sequence[Any],
        action_dt: float,
        timing: Mapping[str, Any] | None = None,
    ) -> int:
        """Append at most the configured prefix and return the rows written."""
        if self._closed:
            raise RuntimeError("raw-action HDF5 recorder is closed")
        if not math.isfinite(float(action_dt)) or float(action_dt) <= 0:
            raise ValueError("action_dt must be finite and > 0")
        state = _finite_vector(current_state, "current_state")
        rows = [
            _finite_vector(row, f"raw_actions[{index}]")
            for index, row in enumerate(raw_actions[: self.max_steps_per_request])
        ]
        if not rows:
            return 0

        if not bool(self._h5.attrs["initial_state_valid"]):
            self._initial_state[:] = state
            self._h5.attrs["initial_state_valid"] = True

        start = self._action.shape[0]
        end = start + len(rows)
        for dataset in (
            self._action,
            self._action_piper_wire,
            self._current_state,
            self._timestamp,
            self._dt,
            self._request_index_ds,
            self._waypoint_index,
            self._step,
            self._request_id,
            self._instruction,
        ):
            dataset.resize((end, *dataset.shape[1:]))

        recorded_at = time.time()
        self._action[start:end] = rows
        self._action_piper_wire[start:end] = [model_row_to_piper_wire(row) for row in rows]
        self._current_state[start:end] = [state] * len(rows)
        self._timestamp[start:end] = [
            recorded_at + float(action_dt) * index for index in range(len(rows))
        ]
        self._dt[start:end] = [float(action_dt)] * len(rows)
        self._request_index_ds[start:end] = [self._request_index] * len(rows)
        self._waypoint_index[start:end] = list(range(len(rows)))
        self._step[start:end] = [int(step)] * len(rows)
        self._request_id[start:end] = [str(request_id)] * len(rows)
        self._instruction[start:end] = [str(instruction)] * len(rows)

        self._request_index += 1
        self._h5.attrs["num_requests"] = self._request_index
        if timing:
            mode = timing.get("mode")
            if mode is not None:
                self._h5.attrs["last_inference_mode"] = str(mode)
        self._h5.flush()
        return len(rows)

    def close(self) -> None:
        if self._closed:
            return
        self._h5.flush()
        self._h5.close()
        self._closed = True

    def __enter__(self) -> "RawModelActionHDF5Recorder":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
