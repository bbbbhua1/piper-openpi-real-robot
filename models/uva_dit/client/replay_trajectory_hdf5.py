#!/usr/bin/env python3
"""Replay a robot-executed Piper trajectory HDF5 file.

The client trajectory format stores absolute Piper wire-order rows:
``[left_arm7, right_arm7]``.  Dry-run is the default.  Physical publishing
requires the explicit ``--execute`` flag.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import threading
import time
from typing import Any, List, Optional, Sequence, Tuple, Union


WIRE_DIM = 14
ARM_DIM = 7
DEFAULT_LEFT_ACTION_TOPIC = "/master/joint_left"
DEFAULT_RIGHT_ACTION_TOPIC = "/master/joint_right"
DEFAULT_ENABLE_TOPIC = "/enable_flag"


def _finite_vector(value: Sequence[Any], name: str, length: int) -> List[float]:
    if isinstance(value, (str, bytes, bytearray)) or len(value) != length:
        raise ValueError("{} must contain exactly {} values".format(name, length))
    result = []
    for index, item in enumerate(value):
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("{}[{}] is not numeric".format(name, index)) from exc
        if not math.isfinite(number):
            raise ValueError("{}[{}] is not finite".format(name, index))
        result.append(number)
    return result


def _as_text(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def load_trajectory(path: Union[str, Path], *, start_index: int = 0, max_steps: Optional[int] = None):
    """Read and validate the client-recorded absolute action trajectory."""
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be > 0")
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("trajectory replay requires h5py") from exc

    hdf5_path = Path(path).expanduser().resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError(str(hdf5_path))

    with h5py.File(str(hdf5_path), "r") as h5_file:
        required = ("action", "initial_state", "dt")
        missing = [name for name in required if name not in h5_file]
        if missing:
            raise ValueError("trajectory HDF5 missing datasets: {}".format(", ".join(missing)))
        action = h5_file["action"]
        initial = _finite_vector(h5_file["initial_state"][:], "/initial_state", WIRE_DIM)
        if len(action.shape) != 2 or action.shape[1] != WIRE_DIM:
            raise ValueError("/action must have shape [N, 14], got {}".format(action.shape))
        total = int(action.shape[0])
        if total == 0:
            raise ValueError("/action is empty")
        if start_index >= total:
            raise ValueError("start_index {} is outside {} rows".format(start_index, total))
        end = total if max_steps is None else min(total, start_index + max_steps)
        rows = [_finite_vector(row, "/action[{}]".format(i), WIRE_DIM)
                for i, row in enumerate(action[start_index:end], start=start_index)]
        dts = []
        for i, value in enumerate(h5_file["dt"][start_index:end], start=start_index):
            dt = float(value)
            if not math.isfinite(dt) or dt < 0:
                raise ValueError("/dt[{}] must be finite and >= 0".format(i))
            dts.append(dt)
        layout = _as_text(h5_file.attrs.get("layout", ""))
        if layout and layout != "[L1..L6,Lg,R1..R6,Rg]":
            raise ValueError("unexpected trajectory layout: {}".format(layout))
        if "phase" in h5_file:
            phases = [_as_text(value) for value in h5_file["phase"][start_index:end]]
            if any(phase != "policy" for phase in phases):
                raise ValueError("selected trajectory contains non-policy phases")

        metadata = {
            "session_name": _as_text(h5_file.attrs.get("session_name", "")),
            "instruction": _as_text(h5_file.attrs.get("instruction", "")),
            "layout": layout,
            "units": _as_text(h5_file.attrs.get("units", "")),
            "total_rows": str(total),
            "selected_start": str(start_index),
            "selected_rows": str(len(rows)),
        }
    return hdf5_path, initial, rows, dts, metadata


def smoothstep(alpha: float) -> float:
    return alpha * alpha * (3.0 - 2.0 * alpha)


def preview(path: Union[str, Path], *, start_index: int = 0, max_steps: Optional[int] = None) -> None:
    hdf5_path, initial, rows, dts, metadata = load_trajectory(
        path, start_index=start_index, max_steps=max_steps
    )
    print("[dry-run] file={}".format(hdf5_path))
    print("[dry-run] metadata={}".format(metadata))
    print("[dry-run] initial_state={}".format([round(x, 6) for x in initial]))
    print("[dry-run] rows={} recorded_duration={:.3f}s".format(len(rows), sum(dts)))
    print("[dry-run] first_action={}".format([round(x, 6) for x in rows[0]]))
    print("[dry-run] last_action={}".format([round(x, 6) for x in rows[-1]]))
    print("[dry-run] no ROS action was published")


class DualArmRosPublisher:
    def __init__(
        self,
        left_topic: str,
        right_topic: str,
        enable_topic: str,
        node_name: str,
        left_state_topic: str,
        right_state_topic: str,
    ):
        try:
            import rospy  # type: ignore
            from sensor_msgs.msg import JointState  # type: ignore
            from std_msgs.msg import Bool  # type: ignore
        except ImportError as exc:
            raise RuntimeError("ROS replay requires rospy, sensor_msgs, and std_msgs") from exc
        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True, disable_signals=True)
        self.rospy = rospy
        self.JointState = JointState
        self.Bool = Bool
        self.left_pub = rospy.Publisher(left_topic, JointState, queue_size=1, tcp_nodelay=True)
        self.right_pub = rospy.Publisher(right_topic, JointState, queue_size=1, tcp_nodelay=True)
        self.enable_pub = rospy.Publisher(enable_topic, Bool, queue_size=1, latch=True)
        self._state_lock = threading.Lock()
        self._current_left: Optional[List[float]] = None
        self._current_right: Optional[List[float]] = None
        rospy.Subscriber(left_state_topic, JointState, self._left_state_callback, queue_size=1)
        rospy.Subscriber(right_state_topic, JointState, self._right_state_callback, queue_size=1)
        self.last_pair: Optional[Tuple[List[float], List[float]]] = None

    def _state_callback(self, msg: Any, side: str) -> None:
        values = _finite_vector(msg.position, "{} state".format(side), ARM_DIM)
        with self._state_lock:
            if side == "left":
                self._current_left = values
            else:
                self._current_right = values

    def _left_state_callback(self, msg: Any) -> None:
        self._state_callback(msg, "left")

    def _right_state_callback(self, msg: Any) -> None:
        self._state_callback(msg, "right")

    def current_state(self, timeout: float) -> List[float]:
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("state timeout must be finite and > 0")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.raise_if_shutdown()
            with self._state_lock:
                if self._current_left is not None and self._current_right is not None:
                    return list(self._current_left) + list(self._current_right)
            time.sleep(0.05)
        raise RuntimeError(
            "did not receive both measured states within {:.1f}s".format(timeout)
        )

    def _message(self, values: Sequence[float]) -> Any:
        msg = self.JointState()
        msg.header.stamp = self.rospy.Time.now()
        msg.name = ["joint{}".format(i) for i in range(ARM_DIM)]
        msg.position = list(values)
        msg.velocity = [0.0] * ARM_DIM
        msg.effort = [0.0] * ARM_DIM
        return msg

    def set_enable(self, enabled: bool, reason: str) -> None:
        self.enable_pub.publish(self.Bool(data=bool(enabled)))
        print("[enable] published enable={} ({})".format(enabled, reason), flush=True)

    def publish_pair(self, row: Sequence[float]) -> None:
        values = _finite_vector(row, "action", WIRE_DIM)
        left, right = values[:ARM_DIM], values[ARM_DIM:]
        self.left_pub.publish(self._message(left))
        self.right_pub.publish(self._message(right))
        self.last_pair = (left, right)

    def raise_if_shutdown(self) -> None:
        if self.rospy.is_shutdown():
            raise KeyboardInterrupt("ROS shutdown requested")


def move_to_initial(publisher: DualArmRosPublisher, current: Sequence[float], target: Sequence[float], *, hz: float, min_duration: float, max_duration: float, joint_speed: float) -> None:
    start = _finite_vector(current, "current_state", WIRE_DIM)
    goal = _finite_vector(target, "initial_state", WIRE_DIM)
    max_delta = max(abs(a - b) for a, b in zip(start, goal))
    if joint_speed <= 0 or hz <= 0:
        raise ValueError("alignment hz and joint speed must be > 0")
    duration = max(min_duration, max_delta / joint_speed)
    if duration > max_duration:
        raise ValueError("initial alignment requires {:.2f}s, exceeds {:.2f}s".format(duration, max_duration))
    steps = max(2, int(duration * hz))
    print("[align] moving both arms to initial_state; max_delta={:.4f}, duration={:.2f}s, steps={}".format(max_delta, duration, steps), flush=True)
    publisher.set_enable(True, "trajectory-align")
    for index in range(steps + 1):
        publisher.raise_if_shutdown()
        alpha = smoothstep(float(index) / float(steps))
        row = [(1.0 - alpha) * a + alpha * b for a, b in zip(start, goal)]
        publisher.publish_pair(row)
        time.sleep(1.0 / hz)
    print("[align] initial_state command published", flush=True)


def replay(publisher: DualArmRosPublisher, rows: List[List[float]], dts: List[float], *, hold_on_exit: bool, disable_on_exit: bool) -> None:
    publisher.set_enable(True, "trajectory-replay")
    print("[replay] publishing {} dual-arm waypoints".format(len(rows)), flush=True)
    start = time.monotonic()
    elapsed = 0.0
    try:
        for index, (row, dt) in enumerate(zip(rows, dts)):
            elapsed += dt
            deadline = start + elapsed
            while time.monotonic() < deadline:
                publisher.raise_if_shutdown()
                time.sleep(min(0.01, deadline - time.monotonic()))
            publisher.publish_pair(row)
            if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(rows):
                print("[replay] waypoint {}/{} t={:.3f}s".format(index + 1, len(rows), elapsed), flush=True)
        if hold_on_exit and publisher.last_pair is not None:
            print("[replay] finished; holding final dual-arm waypoint, press Ctrl-C to stop", flush=True)
            while True:
                publisher.raise_if_shutdown()
                publisher.publish_pair(publisher.last_pair[0] + publisher.last_pair[1])
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("[replay] interrupted", flush=True)
    finally:
        if disable_on_exit:
            publisher.set_enable(False, "trajectory-replay-exit")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--execute", action="store_true", help="allow physical ROS publishing")
    parser.add_argument("--left-action-topic", default=DEFAULT_LEFT_ACTION_TOPIC)
    parser.add_argument("--right-action-topic", default=DEFAULT_RIGHT_ACTION_TOPIC)
    parser.add_argument("--left-state-topic", default="/puppet/joint_left")
    parser.add_argument("--right-state-topic", default="/puppet/joint_right")
    parser.add_argument("--enable-topic", default=DEFAULT_ENABLE_TOPIC)
    parser.add_argument("--node-name", default="piper_trajectory_hdf5_replay")
    parser.add_argument("--align-hz", type=float, default=20.0)
    parser.add_argument("--align-min-duration", type=float, default=2.0)
    parser.add_argument("--align-max-duration", type=float, default=60.0)
    parser.add_argument("--align-joint-speed", type=float, default=0.12)
    parser.add_argument("--state-timeout", type=float, default=10.0)
    parser.add_argument("--hold-on-exit", action="store_true")
    parser.add_argument("--disable-on-exit", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    hdf5_path, initial, rows, dts, metadata = load_trajectory(
        args.hdf5, start_index=args.start_index, max_steps=args.max_steps
    )
    if not args.execute:
        preview(hdf5_path, start_index=args.start_index, max_steps=args.max_steps)
        return

    import rospy  # type: ignore
    publisher = DualArmRosPublisher(
        args.left_action_topic,
        args.right_action_topic,
        args.enable_topic,
        args.node_name,
        args.left_state_topic,
        args.right_state_topic,
    )
    rospy.sleep(0.5)
    current = publisher.current_state(args.state_timeout)
    print("[align] current_state={}".format([round(x, 6) for x in current]), flush=True)
    move_to_initial(
        publisher, current, initial,
        hz=args.align_hz,
        min_duration=args.align_min_duration,
        max_duration=args.align_max_duration,
        joint_speed=args.align_joint_speed,
    )
    replay(publisher, rows, dts, hold_on_exit=args.hold_on_exit, disable_on_exit=args.disable_on_exit)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ntrajectory replay stopped", flush=True)
