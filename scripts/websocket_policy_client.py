#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Host A websocket policy client for Piper ROS.

The client reads local observations, sends one request to the policy server,
receives one chunked action, executes it locally, and repeats.

Use `--source mock --executor mock` to test the websocket loop without ROS or
robot hardware.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import websockets

from ws_policy_protocol import CAMERA_NAMES, build_policy_request


def _normalize_numeric_vector(values: Any, field_name: str, expected_len: int) -> List[float]:
    if not isinstance(values, (list, tuple)):
        raise ValueError("{} must be a list with {} numbers".format(field_name, expected_len))

    vector = []
    for value in values:
        try:
            vector.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError("{} contains a non-numeric value: {!r}".format(field_name, value)) from exc

    if len(vector) != expected_len:
        raise ValueError(
            "{} must contain exactly {} values, got {}".format(
                field_name,
                expected_len,
                len(vector),
            )
        )
    return vector


def _normalize_vector_chunk(chunk: Any, field_name: str, vector_len: int) -> List[List[float]]:
    if not isinstance(chunk, (list, tuple)) or len(chunk) == 0:
        raise ValueError("{} must be a non-empty list of waypoint vectors".format(field_name))

    return [
        _normalize_numeric_vector(waypoint, "{}[{}]".format(field_name, index), vector_len)
        for index, waypoint in enumerate(chunk)
    ]


def _normalize_scalar_chunk(chunk: Any, field_name: str) -> List[List[float]]:
    if not isinstance(chunk, (list, tuple)) or len(chunk) == 0:
        raise ValueError("{} must be a non-empty list of scalar waypoints".format(field_name))

    normalized = []
    for index, waypoint in enumerate(chunk):
        if isinstance(waypoint, (list, tuple)):
            normalized.append(
                _normalize_numeric_vector(waypoint, "{}[{}]".format(field_name, index), 1)
            )
            continue

        try:
            normalized.append([float(waypoint)])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "{}[{}] must be a scalar or single-value list, got {!r}".format(
                    field_name,
                    index,
                    waypoint,
                )
            ) from exc
    return normalized


def _build_time_list(action: Dict[str, Any], num_waypoints: int, default_dt: float) -> List[float]:
    raw_time_list = action.get("time_list")
    if raw_time_list is not None:
        time_list = _normalize_numeric_vector(raw_time_list, "time_list", num_waypoints)
    else:
        raw_dt = action.get("dt", default_dt)
        try:
            dt = float(raw_dt)
        except (TypeError, ValueError) as exc:
            raise ValueError("dt must be numeric, got {!r}".format(raw_dt)) from exc
        if dt <= 0:
            raise ValueError("dt must be > 0, got {}".format(dt))
        time_list = [dt * (index + 1) for index in range(num_waypoints)]

    previous_time = 0.0
    for index, arrival_time in enumerate(time_list):
        if arrival_time <= previous_time:
            raise ValueError(
                "time_list must be strictly increasing and > 0, got {} at index {}".format(
                    arrival_time,
                    index,
                )
            )
        previous_time = arrival_time
    return time_list


def _smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


class SafetyTrip(RuntimeError):
    pass


def parse_chunked_action(
    action: Dict[str, Any],
    default_dt: float,
    left_arm_dim: int = 7,
    right_arm_dim: int = 7,
) -> Dict[str, Any]:
    if not isinstance(action, dict):
        raise ValueError("chunked action must be a dict, got {}".format(type(action).__name__))

    left_arm_chunk = _normalize_vector_chunk(action.get("left_arm"), "left_arm", left_arm_dim)
    right_arm_chunk = _normalize_vector_chunk(action.get("right_arm"), "right_arm", right_arm_dim)

    num_waypoints = len(left_arm_chunk)
    if len(right_arm_chunk) != num_waypoints:
        raise ValueError(
            "left_arm and right_arm must have the same number of waypoints: {} vs {}".format(
                num_waypoints,
                len(right_arm_chunk),
            )
        )

    left_gripper_chunk = None
    if "left_gripper" in action and action["left_gripper"] is not None:
        left_gripper_chunk = _normalize_scalar_chunk(action["left_gripper"], "left_gripper")
        if len(left_gripper_chunk) != num_waypoints:
            raise ValueError("left_gripper must have the same number of waypoints as left_arm")

    right_gripper_chunk = None
    if "right_gripper" in action and action["right_gripper"] is not None:
        right_gripper_chunk = _normalize_scalar_chunk(action["right_gripper"], "right_gripper")
        if len(right_gripper_chunk) != num_waypoints:
            raise ValueError("right_gripper must have the same number of waypoints as right_arm")

    return {
        "left_arm": left_arm_chunk,
        "right_arm": right_arm_chunk,
        "left_gripper": left_gripper_chunk,
        "right_gripper": right_gripper_chunk,
        "time_list": _build_time_list(action, num_waypoints, default_dt),
        "num_waypoints": num_waypoints,
    }


class MockObservationSource:
    def __init__(self) -> None:
        self.step = 0

    def wait_until_ready(self, timeout_s: float = 5.0) -> None:
        return None

    def get_observation(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        import numpy as np

        self.step += 1
        images = {}
        for index, name in enumerate(CAMERA_NAMES):
            image = np.zeros((240, 320, 3), dtype=np.uint8)
            image[:, :, index % 3] = min(255, 20 * self.step)
            images[name] = image

        phase = 0.05 * self.step
        state = {
            "left_arm": [float(np.sin(phase + index * 0.1)) for index in range(7)],
            "right_arm": [float(np.cos(phase + index * 0.1)) for index in range(7)],
        }
        return images, state


class MockExecutor:
    def __init__(self, chunk_dt: float, left_arm_dim: int, right_arm_dim: int) -> None:
        self.chunk_dt = chunk_dt
        self.left_arm_dim = left_arm_dim
        self.right_arm_dim = right_arm_dim

    def execute_action(self, action: Dict[str, Any]) -> float:
        chunk = parse_chunked_action(
            action,
            default_dt=self.chunk_dt,
            left_arm_dim=self.left_arm_dim,
            right_arm_dim=self.right_arm_dim,
        )
        print(
            "[mock executor] {}".format(
                json.dumps(
                    {
                        "num_waypoints": chunk["num_waypoints"],
                        "time_list": chunk["time_list"],
                        "left_first": chunk["left_arm"][0],
                        "right_first": chunk["right_arm"][0],
                    },
                    ensure_ascii=False,
                )
            )
        )
        return chunk["time_list"][-1]


class RosObservationSource:
    ARM_STATUS_TOPICS = (
        "/puppet/arm_status",
        "/puppet/arm_status_left",
        "/puppet/arm_status_right",
    )
    ARM_STATUS_FAULT_FIELDS = (
        "joint_1_angle_limit",
        "joint_2_angle_limit",
        "joint_3_angle_limit",
        "joint_4_angle_limit",
        "joint_5_angle_limit",
        "joint_6_angle_limit",
        "communication_status_joint_1",
        "communication_status_joint_2",
        "communication_status_joint_3",
        "communication_status_joint_4",
        "communication_status_joint_5",
        "communication_status_joint_6",
    )

    def __init__(
        self,
        left_joint_topic: str,
        right_joint_topic: str,
        camera_topics: Dict[str, str],
        compressed_images: bool,
        require_images: bool,
        enable_topic: str,
    ) -> None:
        import rospy
        from piper_msgs.msg import PiperStatusMsg
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool

        self.rospy = rospy
        self.Bool = Bool
        self.require_images = require_images
        self.camera_topics = camera_topics
        self.enable_topic = enable_topic
        self.lock = threading.Lock()
        self.left_joint = None
        self.right_joint = None
        self.images = {}
        self.bridge = None
        self.safety_tripped = False
        self.safety_reason = ""

        if not rospy.core.is_initialized():
            rospy.init_node("websocket_policy_client", anonymous=True, disable_signals=True)
        self.enable_pub = rospy.Publisher(enable_topic, Bool, queue_size=1, latch=True)
        for topic in self.ARM_STATUS_TOPICS:
            rospy.Subscriber(topic, PiperStatusMsg, self._arm_status_callback, callback_args=topic, queue_size=1)
        rospy.Subscriber(left_joint_topic, JointState, self._left_joint_callback, queue_size=1)
        rospy.Subscriber(right_joint_topic, JointState, self._right_joint_callback, queue_size=1)

        if camera_topics:
            if compressed_images:
                from sensor_msgs.msg import CompressedImage

                for camera_name, topic in camera_topics.items():
                    rospy.Subscriber(
                        topic,
                        CompressedImage,
                        self._compressed_image_callback,
                        callback_args=camera_name,
                        queue_size=1,
                    )
            else:
                from cv_bridge import CvBridge
                from sensor_msgs.msg import Image

                self.bridge = CvBridge()
                for camera_name, topic in camera_topics.items():
                    rospy.Subscriber(
                        topic,
                        Image,
                        self._raw_image_callback,
                        callback_args=camera_name,
                        queue_size=1,
                    )

    def _arm_status_fault_reasons(self, msg: Any) -> List[str]:
        reasons = []
        err_code = int(getattr(msg, "err_code", 0))
        if err_code != 0:
            reasons.append("err_code={}".format(err_code))
        for field_name in self.ARM_STATUS_FAULT_FIELDS:
            if bool(getattr(msg, field_name, False)):
                reasons.append("{}=True".format(field_name))
        return reasons

    def _arm_status_callback(self, msg: Any, topic: str) -> None:
        reasons = self._arm_status_fault_reasons(msg)
        if reasons:
            self.trip_safety("{} fault: {}".format(topic, ", ".join(reasons)))

    def trip_safety(self, reason: str) -> None:
        with self.lock:
            if self.safety_tripped:
                return
            self.safety_tripped = True
            self.safety_reason = reason

        print("[safety] {}; publishing {}=False".format(reason, self.enable_topic))
        try:
            self.enable_pub.publish(self.Bool(data=False))
        except Exception as exc:
            print("[safety] failed to publish {}=False: {}".format(self.enable_topic, exc))

    def is_safety_tripped(self) -> bool:
        with self.lock:
            return self.safety_tripped

    def raise_if_safety_tripped(self) -> None:
        with self.lock:
            tripped = self.safety_tripped
            reason = self.safety_reason
        if tripped:
            raise SafetyTrip(reason)

    def _left_joint_callback(self, msg: Any) -> None:
        with self.lock:
            self.left_joint = msg

    def _right_joint_callback(self, msg: Any) -> None:
        with self.lock:
            self.right_joint = msg

    def _compressed_image_callback(self, msg: Any, camera_name: str) -> None:
        with self.lock:
            self.images[camera_name] = {
                "encoding": "jpeg",
                "data": bytes(msg.data),
            }

    def _raw_image_callback(self, msg: Any, camera_name: str) -> None:
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self.lock:
            self.images[camera_name] = image.copy()

    def wait_until_ready(self, timeout_s: float = 5.0) -> None:
        start_time = time.time()
        while time.time() - start_time < timeout_s:
            self.raise_if_safety_tripped()
            with self.lock:
                joints_ready = self.left_joint is not None and self.right_joint is not None
                images_ready = (not self.require_images) or all(
                    name in self.images for name in self.camera_topics
                )
            if joints_ready and images_ready:
                return
            time.sleep(0.05)
        raise TimeoutError("ROS observations are not ready within {} seconds".format(timeout_s))

    def _joint_to_state(self, msg: Any) -> List[float]:
        return [float(value) for value in msg.position]

    def get_state(self) -> Dict[str, Any]:
        self.raise_if_safety_tripped()
        with self.lock:
            if self.left_joint is None or self.right_joint is None:
                raise RuntimeError("missing joint observations")
            return {
                "left_arm": self._joint_to_state(self.left_joint),
                "right_arm": self._joint_to_state(self.right_joint),
            }

    def get_observation(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.raise_if_safety_tripped()
        with self.lock:
            if self.left_joint is None or self.right_joint is None:
                raise RuntimeError("missing joint observations")
            images = dict(self.images)
            state = {
                "left_arm": self._joint_to_state(self.left_joint),
                "right_arm": self._joint_to_state(self.right_joint),
            }
        return images, state


class RosJointExecutor:
    def __init__(
        self,
        left_action_topic: str,
        right_action_topic: str,
        enable_topic: str,
        chunk_dt: float,
        left_arm_dim: int,
        right_arm_dim: int,
        enable_on_start: bool,
        waypoint_sleep: float,
    ) -> None:
        import rospy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool

        if not rospy.core.is_initialized():
            rospy.init_node("websocket_policy_client", anonymous=True, disable_signals=True)

        self.rospy = rospy
        self.JointState = JointState
        self.chunk_dt = chunk_dt
        self.left_arm_dim = left_arm_dim
        self.right_arm_dim = right_arm_dim
        self.waypoint_sleep = waypoint_sleep
        self.Bool = Bool
        self.left_pub = rospy.Publisher(left_action_topic, JointState, queue_size=1, tcp_nodelay=True)
        self.right_pub = rospy.Publisher(right_action_topic, JointState, queue_size=1, tcp_nodelay=True)
        self.enable_pub = rospy.Publisher(enable_topic, Bool, queue_size=1, latch=True)

        if enable_on_start:
            time.sleep(0.5)
            self.set_enable(True, "start")

    def set_enable(self, enabled: bool, reason: str = "") -> None:
        try:
            self.enable_pub.publish(self.Bool(data=enabled))
            suffix = " ({})".format(reason) if reason else ""
            print("[enable] published enable={}{}".format(enabled, suffix))
        except Exception as exc:
            print("[enable] failed to publish enable={}: {}".format(enabled, exc))

    def disable_robot(self) -> None:
        self.set_enable(False, "explicit disable-on-exit")

    def _raise_if_shutdown(self) -> None:
        if self.rospy.is_shutdown():
            raise KeyboardInterrupt("ROS shutdown requested")

    def _interruptible_sleep(self, duration_s: float) -> None:
        deadline = time.time() + max(0.0, duration_s)
        while time.time() < deadline:
            self._raise_if_shutdown()
            time.sleep(min(0.05, deadline - time.time()))
        self._raise_if_shutdown()

    def _build_joint_msg(self, positions: List[float]) -> Any:
        msg = self.JointState()
        msg.header.stamp = self.rospy.Time.now()
        msg.name = ["joint{}".format(index) for index in range(len(positions))]
        msg.position = positions
        msg.velocity = [0.0] * len(positions)
        msg.effort = [0.0] * len(positions)
        return msg

    def _state_pair(self, state: Dict[str, Any]) -> Tuple[List[float], List[float]]:
        left = _normalize_numeric_vector(state.get("left_arm"), "state.left_arm", self.left_arm_dim)
        right = _normalize_numeric_vector(state.get("right_arm"), "state.right_arm", self.right_arm_dim)
        return left, right

    def _publish_pair(self, left: List[float], right: List[float]) -> None:
        self.left_pub.publish(self._build_joint_msg(left))
        self.right_pub.publish(self._build_joint_msg(right))

    def move_home(
        self,
        state: Dict[str, Any],
        home_hz: float,
        min_duration: float,
        max_duration: float,
        joint_speed: float,
        tolerance: float,
    ) -> float:
        if home_hz <= 0:
            raise ValueError("--home-hz must be > 0")
        if min_duration < 0 or max_duration < 0:
            raise ValueError("home durations must be >= 0")
        if max_duration < min_duration:
            raise ValueError("--home-max-duration must be >= --home-min-duration")
        if joint_speed <= 0:
            raise ValueError("--home-joint-speed must be > 0")

        start_left, start_right = self._state_pair(state)
        home_left = [0.0] * self.left_arm_dim
        home_right = [0.0] * self.right_arm_dim
        max_delta = max([abs(value) for value in start_left + start_right] or [0.0])

        self.set_enable(True, "home")
        period_s = 1.0 / home_hz
        if max_delta <= tolerance:
            print("[home] already near home; max_delta={:.6f}".format(max_delta))
            for _ in range(max(3, int(0.5 * home_hz))):
                self._raise_if_shutdown()
                self._publish_pair(home_left, home_right)
                time.sleep(period_s)
            return 0.0

        duration_s = max(min_duration, min(max_duration, max_delta / joint_speed))
        num_steps = max(2, int(duration_s * home_hz))
        print(
            "[home] moving to zero joints; max_delta={:.4f}, duration={:.2f}s, steps={}".format(
                max_delta,
                duration_s,
                num_steps,
            )
        )

        for step in range(num_steps + 1):
            self._raise_if_shutdown()
            alpha = _smoothstep(float(step) / float(num_steps))
            left = [(1.0 - alpha) * value for value in start_left]
            right = [(1.0 - alpha) * value for value in start_right]
            self._publish_pair(left, right)
            time.sleep(period_s)

        for _ in range(max(3, int(0.5 * home_hz))):
            self._raise_if_shutdown()
            self._publish_pair(home_left, home_right)
            time.sleep(period_s)

        print("[home] reached zero joints")
        return duration_s

    def hold_position(self, state: Dict[str, Any], hold_hz: float) -> None:
        if hold_hz <= 0:
            raise ValueError("--hold-hz must be > 0")
        left, right = self._state_pair(state)
        period_s = 1.0 / hold_hz
        self.set_enable(True, "hold")
        print("[hold] holding current joint position; press Ctrl-C again to leave hold loop")
        try:
            while True:
                self._publish_pair(left, right)
                time.sleep(period_s)
        except KeyboardInterrupt:
            print("[hold] hold loop stopped; robot remains enabled")

    def _merge_gripper(
        self,
        arm: List[float],
        gripper_chunk: Optional[List[List[float]]],
        index: int,
    ) -> List[float]:
        if len(arm) == 7:
            return arm
        if len(arm) == 6 and gripper_chunk is not None:
            return arm + [float(gripper_chunk[index][0])]
        return arm

    def execute_action(self, action: Dict[str, Any]) -> float:
        chunk = parse_chunked_action(
            action,
            default_dt=self.chunk_dt,
            left_arm_dim=self.left_arm_dim,
            right_arm_dim=self.right_arm_dim,
        )

        previous_time = 0.0
        start_time = time.time()
        for index, arrival_time in enumerate(chunk["time_list"]):
            self._raise_if_shutdown()
            sleep_s = max(0.0, start_time + arrival_time - time.time())
            if sleep_s > 0:
                self._interruptible_sleep(sleep_s)

            self._raise_if_shutdown()
            left = self._merge_gripper(chunk["left_arm"][index], chunk["left_gripper"], index)
            right = self._merge_gripper(chunk["right_arm"][index], chunk["right_gripper"], index)
            self.left_pub.publish(self._build_joint_msg(left))
            self.right_pub.publish(self._build_joint_msg(right))
            print(
                "[execute] waypoint {}/{} published; sleep={:.3f}s".format(
                    index + 1,
                    chunk["num_waypoints"],
                    self.waypoint_sleep,
                )
            )
            if self.waypoint_sleep > 0:
                self._interruptible_sleep(self.waypoint_sleep)
            previous_time = arrival_time

        return max(previous_time, time.time() - start_time)


def parse_camera_topics(items: List[str]) -> Dict[str, str]:
    topics = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--camera-topic must look like name=/topic")
        name, topic = item.split("=", 1)
        topics[name] = topic
    return topics


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "on")


def create_source(args: argparse.Namespace) -> Any:
    if args.source == "mock":
        return MockObservationSource()
    return RosObservationSource(
        left_joint_topic=args.left_observation_topic,
        right_joint_topic=args.right_observation_topic,
        camera_topics=parse_camera_topics(args.camera_topic),
        compressed_images=args.compressed_images,
        require_images=args.require_images,
        enable_topic=args.enable_topic,
    )


def create_executor(args: argparse.Namespace) -> Any:
    chunk_dt = 1.0 / args.control_hz
    if args.executor == "mock":
        return MockExecutor(chunk_dt, args.left_arm_dim, args.right_arm_dim)
    return RosJointExecutor(
        left_action_topic=args.left_action_topic,
        right_action_topic=args.right_action_topic,
        enable_topic=args.enable_topic,
        chunk_dt=chunk_dt,
        left_arm_dim=args.left_arm_dim,
        right_arm_dim=args.right_arm_dim,
        enable_on_start=args.enable_on_start,
        waypoint_sleep=args.waypoint_sleep,
    )


def get_current_state(source: Any) -> Dict[str, Any]:
    get_state = getattr(source, "get_state", None)
    if get_state is not None:
        return get_state()
    _, state = source.get_observation()
    return state


def raise_if_safety_tripped(source: Any) -> None:
    checker = getattr(source, "raise_if_safety_tripped", None)
    if checker is not None:
        checker()


def is_safety_tripped(source: Any) -> bool:
    checker = getattr(source, "is_safety_tripped", None)
    return bool(checker()) if checker is not None else False


async def run_loop(args: argparse.Namespace) -> None:
    if args.control_hz <= 0:
        raise ValueError("--control-hz must be > 0")

    source = create_source(args)
    executor = create_executor(args)

    try:
        print("waiting for local observation source...")
        source.wait_until_ready(timeout_s=args.ready_timeout)
        print("observation source is ready")

        if args.execute_actions and args.home_on_start:
            move_home = getattr(executor, "move_home", None)
            if move_home is not None:
                move_home(
                    get_current_state(source),
                    home_hz=args.home_hz,
                    min_duration=args.home_min_duration,
                    max_duration=args.home_max_duration,
                    joint_speed=args.home_joint_speed,
                    tolerance=args.home_tolerance,
                )

        async with websockets.connect(
            args.uri,
            max_size=args.max_size_mb * 1024 * 1024,
            ping_interval=20,
            ping_timeout=60,
        ) as websocket:
            print("connected to policy server: {}".format(args.uri))
            step = 0
            while args.max_steps is None or step < args.max_steps:
                loop_start = time.time()
                images, state = source.get_observation()
                request = build_policy_request(
                    step=step,
                    instruction=args.instruction,
                    state=state,
                    images=images,
                    jpeg_quality=args.jpeg_quality,
                    resize_scale=args.resize_scale,
                )
                print("state", state)

                await websocket.send(json.dumps(request, ensure_ascii=False))
                response = json.loads(await websocket.recv())
                if not response.get("ok", False):
                    raise RuntimeError("policy server error: {}".format(response.get("error")))
                print("action", json.dumps(response["action"],ensure_ascii=False,indent=2))
                raise_if_safety_tripped(source)
                if args.execute_actions:
                    chunk_duration_s = executor.execute_action(response["action"])
                else:
                    print("[dry-run] action execution disabled; pass --execute-actions to publish commands")
                    chunk_duration_s = 0
                elapsed = time.time() - loop_start
                print(
                    "[step {}] round_trip={:.3f}s request_id={}".format(
                        step,
                        elapsed,
                        request["request_id"],
                    )
                )

                step += 1
                min_cycle_s = max(1.0 / args.control_hz, chunk_duration_s)
                sleep_s = max(0.0, min_cycle_s - elapsed)
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
    except SafetyTrip as exc:
        print("[safety] client stopped: {}".format(exc))
    finally:
        if args.execute_actions and is_safety_tripped(source):
            print("[safety] skipping hold/disable-on-exit cleanup after arm_status fault")
        elif args.execute_actions:
            if args.hold_on_exit:
                hold_position = getattr(executor, "hold_position", None)
                if hold_position is not None:
                    try:
                        hold_position(get_current_state(source), hold_hz=args.hold_hz)
                    except Exception as exc:
                        print("[hold] skipped because current state was unavailable: {}".format(exc))
            if args.disable_on_exit:
                disable_robot = getattr(executor, "disable_robot", None)
                if disable_robot is not None:
                    disable_robot()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Piper websocket policy client")
    parser.add_argument("--uri", required=True, help="e.g. ws://192.168.1.20:7081")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--source", choices=["ros", "mock"], default="ros")
    parser.add_argument("--executor", choices=["ros", "mock"], default="ros")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--resize-scale", type=float, default=1.0)
    parser.add_argument("--ready-timeout", type=float, default=10.0)
    parser.add_argument("--max-size-mb", type=int, default=32)
    parser.add_argument("--left-arm-dim", type=int, default=7)
    parser.add_argument("--right-arm-dim", type=int, default=7)
    parser.add_argument("--left-observation-topic", default="/puppet/joint_left")
    parser.add_argument("--right-observation-topic", default="/puppet/joint_right")
    parser.add_argument("--left-action-topic", default="/master/joint_left")
    parser.add_argument("--right-action-topic", default="/master/joint_right")
    parser.add_argument("--enable-topic", default="/enable_flag")
    parser.add_argument("--enable-on-start", nargs="?", const=True, default=False, type=parse_bool)
    parser.set_defaults(home_on_start=True, hold_on_exit=True)
    parser.add_argument(
        "--no-home-on-start",
        action="store_false",
        dest="home_on_start",
        help="skip the default smooth move to zero joints before executing policy actions",
    )
    parser.add_argument("--home-hz", type=float, default=20.0)
    parser.add_argument("--home-min-duration", type=float, default=2.0)
    parser.add_argument("--home-max-duration", type=float, default=20.0)
    parser.add_argument("--home-joint-speed", type=float, default=0.25)
    parser.add_argument("--home-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--no-hold-on-exit",
        action="store_false",
        dest="hold_on_exit",
        help="skip the default hold-position loop on exit",
    )
    parser.add_argument("--hold-hz", type=float, default=20.0)
    parser.add_argument(
        "--disable-on-exit",
        action="store_true",
        help="publish enable=False after exit handling; dangerous if the arm needs torque to hold position",
    )
    parser.add_argument(
        "--execute-actions",
        action="store_true",
        help="publish returned action chunks to ROS action topics; default is dry-run",
    )
    parser.add_argument(
        "--waypoint-sleep",
        type=float,
        default=0.0,
        help="extra seconds to sleep after publishing each waypoint when --execute-actions is set",
    )
    parser.add_argument(
        "--camera-topic",
        action="append",
        default=[],
        help="repeatable name=/topic, e.g. head=/camera/color/image_raw/compressed",
    )
    parser.add_argument("--compressed-images", action="store_true")
    parser.add_argument("--require-images", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(run_loop(args))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\npolicy client stopped")
