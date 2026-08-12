#!/usr/bin/env python3
"""Replay only the right-arm portion of a Piper UVA-DiT raw-action HDF5 file.

The server-side raw-action file stores model physical14 rows as:
``[left_arm6, right_arm6, left_gripper, right_gripper]``.
This script publishes only ``[right_arm6, right_gripper]`` to the Piper right
action topic.  It never creates or publishes a left-arm ROS topic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Union


MODEL_ACTION_DIM = 14
RIGHT_ACTION_DIM = 7
DEFAULT_RIGHT_ACTION_TOPIC = "/master/joint_right"
DEFAULT_ENABLE_TOPIC = "/enable_flag"


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _finite_row(row: Sequence[Any], field_name: str) -> List[float]:
    if len(row) != MODEL_ACTION_DIM:
        raise ValueError(
            "{} must contain exactly {} values, got {}".format(
                field_name, MODEL_ACTION_DIM, len(row)
            )
        )
    result: List[float] = []
    for index, value in enumerate(row):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("{}[{}] is not numeric".format(field_name, index)) from exc
        if not math.isfinite(number):
            raise ValueError("{}[{}] is not finite".format(field_name, index))
        result.append(number)
    return result


def model_action_to_right_wire(row: Sequence[Any]) -> List[float]:
    """Extract raw model physical14 into Piper right-arm 7D wire order."""
    values = _finite_row(row, "model action")
    return [*values[6:12], values[13]]


@dataclass(frozen=True)
class ReplayData:
    path: Path
    right_actions: List[List[float]]
    request_indices: List[int]
    metadata: Dict[str, str]


def load_replay_data(
    path: Union[str, Path],
    *,
    start_index: int = 0,
    max_steps: Optional[int] = None,
) -> ReplayData:
    """Load and validate the raw ``/action`` dataset and extract right rows."""
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be > 0 when provided")

    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "replay requires h5py; install h5py in the robot Python environment"
        ) from exc

    hdf5_path = Path(path).expanduser().resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError("HDF5 file not found: {}".format(hdf5_path))

    with h5py.File(str(hdf5_path), "r") as h5_file:
        if "action" not in h5_file:
            raise ValueError("HDF5 file does not contain /action")
        action_dataset = h5_file["action"]
        if len(action_dataset.shape) != 2 or action_dataset.shape[1] != MODEL_ACTION_DIM:
            raise ValueError(
                "/action must have shape [N, 14], got {}".format(action_dataset.shape)
            )

        source = _as_text(h5_file.attrs.get("source", ""))
        layout = _as_text(h5_file.attrs.get("model_action_layout", ""))
        raw_output = h5_file.attrs.get("raw_model_output", True)
        if isinstance(raw_output, bool) and not raw_output:
            raise ValueError("/action is marked as non-raw output; refusing replay")
        if layout and "right_arm6" not in layout:
            raise ValueError(
                "unexpected model action layout {}; expected right_arm6".format(layout)
            )

        total_steps = int(action_dataset.shape[0])
        if start_index >= total_steps:
            raise ValueError(
                "start_index {} is outside /action with {} rows".format(
                    start_index, total_steps
                )
            )
        end_index = total_steps
        if max_steps is not None:
            end_index = min(end_index, start_index + max_steps)

        raw_rows = action_dataset[start_index:end_index]
        right_actions = [
            model_action_to_right_wire(row)
            for row in raw_rows
        ]
        if "request_index" in h5_file:
            request_indices = [
                int(value) for value in h5_file["request_index"][start_index:end_index]
            ]
        else:
            request_indices = [-1] * len(right_actions)

        metadata = {
            "source": source,
            "model_action_layout": layout,
            "action_type": _as_text(h5_file.attrs.get("action_type", "")),
            "units": _as_text(h5_file.attrs.get("units", "")),
            "total_rows": str(total_steps),
            "selected_start": str(start_index),
            "selected_rows": str(len(right_actions)),
        }

    if not right_actions:
        raise ValueError("selected replay range is empty")
    return ReplayData(
        path=hdf5_path,
        right_actions=right_actions,
        request_indices=request_indices,
        metadata=metadata,
    )


class RightArmRosPublisher:
    """ROS publisher that owns only the right-arm action topic."""

    def __init__(
        self,
        *,
        action_topic: str,
        enable_topic: str,
        enable_on_start: bool,
        node_name: str,
    ) -> None:
        try:
            import rospy  # type: ignore
            from sensor_msgs.msg import JointState  # type: ignore
            from std_msgs.msg import Bool  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "ROS replay requires rospy, sensor_msgs, and std_msgs on the robot"
            ) from exc

        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True, disable_signals=True)
        self.rospy = rospy
        self.JointState = JointState
        self.Bool = Bool
        self.action_pub = rospy.Publisher(
            action_topic, JointState, queue_size=1, tcp_nodelay=True
        )
        self.enable_pub = rospy.Publisher(
            enable_topic, Bool, queue_size=1, latch=True
        )
        self.last_action: Optional[List[float]] = None
        if enable_on_start:
            time.sleep(0.5)
            self.set_enable(True, "raw-action-replay-start")

    def set_enable(self, enabled: bool, reason: str = "") -> None:
        suffix = " ({})".format(reason) if reason else ""
        self.enable_pub.publish(self.Bool(data=bool(enabled)))
        print("[enable] published enable={}{}".format(enabled, suffix), flush=True)

    def _raise_if_shutdown(self) -> None:
        if self.rospy.is_shutdown():
            raise KeyboardInterrupt("ROS shutdown requested")

    def publish_right(self, action: Sequence[float]) -> None:
        values = [float(value) for value in action]
        if len(values) != RIGHT_ACTION_DIM:
            raise ValueError("right action must contain exactly 7 values")
        message = self.JointState()
        message.header.stamp = self.rospy.Time.now()
        message.name = ["joint{}".format(index) for index in range(RIGHT_ACTION_DIM)]
        message.position = values
        message.velocity = [0.0] * RIGHT_ACTION_DIM
        message.effort = [0.0] * RIGHT_ACTION_DIM
        self.action_pub.publish(message)
        self.last_action = values

    def disable(self) -> None:
        self.set_enable(False, "raw-action-replay-exit")


def _sleep_until(publisher: RightArmRosPublisher, deadline: float) -> None:
    while True:
        publisher._raise_if_shutdown()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.05))


def replay_actions(
    publisher: RightArmRosPublisher,
    data: ReplayData,
    *,
    control_hz: float,
    hold_on_exit: bool,
    disable_on_exit: bool,
) -> None:
    if control_hz <= 0 or not math.isfinite(control_hz):
        raise ValueError("control_hz must be finite and > 0")
    period = 1.0 / control_hz
    start_time = time.monotonic()
    total = len(data.right_actions)
    print(
        "[replay] publishing {} right-arm waypoints at {:.3f} Hz".format(
            total, control_hz
        ),
        flush=True,
    )
    try:
        for index, action in enumerate(data.right_actions):
            _sleep_until(publisher, start_time + index * period)
            publisher.publish_right(action)
            print(
                "[replay] waypoint {}/{} request={} right={}".format(
                    index + 1,
                    total,
                    data.request_indices[index],
                    [round(value, 6) for value in action],
                ),
                flush=True,
            )

        if hold_on_exit:
            if publisher.last_action is None:
                return
            print("[replay] finished; holding right arm, press Ctrl-C to stop", flush=True)
            while True:
                publisher._raise_if_shutdown()
                publisher.publish_right(publisher.last_action)
                time.sleep(period)
    except KeyboardInterrupt:
        print("[replay] interrupted", flush=True)
    finally:
        if disable_on_exit:
            publisher.disable()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True, help="server raw-action HDF5 file")
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="publish to ROS; without this flag only validate and preview the data",
    )
    parser.add_argument(
        "--right-action-topic", default=DEFAULT_RIGHT_ACTION_TOPIC
    )
    parser.add_argument("--enable-topic", default=DEFAULT_ENABLE_TOPIC)
    parser.add_argument("--node-name", default="piper_raw_action_replay")
    parser.add_argument(
        "--enable-on-start",
        action="store_true",
        help="publish enable=True before replay; disabled by default",
    )
    parser.add_argument(
        "--hold-on-exit",
        action="store_true",
        help="keep publishing the final right waypoint after replay",
    )
    parser.add_argument(
        "--disable-on-exit",
        action="store_true",
        help="publish enable=False after replay or interruption",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.control_hz <= 0 or not math.isfinite(args.control_hz):
        raise ValueError("--control-hz must be finite and > 0")
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    data = load_replay_data(
        args.hdf5,
        start_index=args.start_index,
        max_steps=args.max_steps,
    )
    print("[replay] file={}".format(data.path), flush=True)
    print("[replay] metadata={}".format(data.metadata), flush=True)
    print(
        "[replay] first_right_action={}".format(
            [round(value, 6) for value in data.right_actions[0]]
        ),
        flush=True,
    )

    if not args.execute:
        preview_count = min(3, len(data.right_actions))
        print(
            "[dry-run] validated {} rows; previewing first {} right-arm rows".format(
                len(data.right_actions), preview_count
            ),
            flush=True,
        )
        for index in range(preview_count):
            print(
                "[dry-run] {} {}".format(
                    index, [round(value, 6) for value in data.right_actions[index]]
                ),
                flush=True,
            )
        return

    publisher = RightArmRosPublisher(
        action_topic=args.right_action_topic,
        enable_topic=args.enable_topic,
        enable_on_start=args.enable_on_start,
        node_name=args.node_name,
    )
    replay_actions(
        publisher,
        data,
        control_hz=args.control_hz,
        hold_on_exit=args.hold_on_exit,
        disable_on_exit=args.disable_on_exit,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nraw action replay stopped", flush=True)
