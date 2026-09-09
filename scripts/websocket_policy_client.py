#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Host A websocket policy client for Piper ROS.

The client reads local observations, sends one request to the policy server,
receives one chunked action, executes it locally, and repeats synchronously.

Use `--source mock --executor mock` to test the websocket loop without ROS or
robot hardware.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import websockets
except ImportError:  # pragma: no cover - allows protocol/CLI inspection without extras
    websockets = None  # type: ignore[assignment]

from camera_video_recorder import (
    CameraVideoRecorder,
    DEFAULT_RECORD_DIR,
    DEFAULT_UPLOAD_KEY,
    DEFAULT_UPLOAD_DIR,
    DEFAULT_UPLOAD_HOST,
    DEFAULT_UPLOAD_PORT,
)
from piper_start_pose import (
    StartPoseConfig,
    load_start_pose_config,
    pose_errors,
    pose_is_reached,
    validate_state_within_bounds,
)
from ws_policy_protocol import (
    CAMERA_NAMES,
    FASTWAM_TRANSPORT_IMAGE_SIZES,
    build_policy_request,
)


class SubtaskOverlay:
    """Thread-safe Q1/Q2 state used to annotate the head-camera recording."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}

    def update(self, response: Dict[str, Any]) -> None:
        progress = response.get("subtask_progress")
        if isinstance(progress, dict):
            with self._lock:
                self._data = dict(progress)

    def transform(self, payload: bytes) -> bytes:
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return payload
        with self._lock:
            data = dict(self._data)
        goals = data.get("subtasks") or []
        current = int(data.get("current_index", 0) or 0)
        completed = int(data.get("completed_count", 0) or 0)
        total = max(len(goals), completed, 1)
        current_goal = str(data.get("current_goal") or "No active subtask")
        q2 = data.get("q2_status") or {}
        q2_text = "Q2: done={} confidence={}".format(q2.get("done", False), q2.get("confidence", "-"))
        height, width = image.shape[:2]
        bar_h = max(42, height // 14)
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, bar_h + 62), (18, 24, 30), -1)
        image = cv2.addWeighted(overlay, 0.82, image, 0.18, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "Subtask {}/{}: {}".format(min(current + 1, total), total, current_goal[:110]), (12, 25), font, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, q2_text[:110], (12, 52), font, 0.5, (205, 220, 230), 1, cv2.LINE_AA)
        x0, y0, x1, y1 = 12, bar_h + 20, width - 12, bar_h + 36
        cv2.rectangle(image, (x0, y0), (x1, y1), (75, 80, 85), -1)
        cv2.rectangle(image, (x0, y0), (x0 + int((x1 - x0) * min(1.0, completed / total)), y1), (45, 190, 95), -1)
        cv2.putText(image, "Q1/Q2 progress: {}/{} completed".format(completed, total), (12, y1 + 23), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        return encoded.tobytes() if ok else payload


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


def _lerp(start: Sequence[float], end: Sequence[float], alpha: float) -> List[float]:
    return [(1.0 - alpha) * a + alpha * b for a, b in zip(start, end)]


def _embedded_gripper_index(values: Sequence[float]) -> Optional[int]:
    if len(values) >= 7:
        return len(values) - 1
    return None


def _lerp_joints(
    start: Sequence[float],
    end: Sequence[float],
    alpha: float,
    *,
    smooth_gripper: bool,
    snap_gripper: bool,
    stepwise_all: bool = False,
) -> List[float]:
    if smooth_gripper:
        return _lerp(start, end, alpha)
    if stepwise_all:
        return list(end) if snap_gripper else list(start)
    gripper_index = _embedded_gripper_index(start)
    if gripper_index is None or gripper_index >= len(end):
        return _lerp(start, end, alpha)
    joints = _lerp(start[:gripper_index], end[:gripper_index], alpha)
    gripper = end[gripper_index] if snap_gripper else start[gripper_index]
    return joints + [float(gripper)] + list(start[gripper_index + 1 :])


def _interpolate_waypoints(
    waypoints: List[List[float]],
    time_list: List[float],
    factor: int,
    *,
    smooth_gripper: bool = True,
    stepwise_all: bool = False,
) -> Tuple[List[List[float]], List[float]]:
    if factor == 1 or len(waypoints) < 2:
        return waypoints, list(time_list)

    interpolated = [list(waypoints[0])]
    interpolated_times = [float(time_list[0])]
    for index in range(len(waypoints) - 1):
        start = waypoints[index]
        end = waypoints[index + 1]
        start_time = time_list[index]
        end_time = time_list[index + 1]
        for step in range(1, factor + 1):
            alpha = float(step) / float(factor)
            interpolated.append(
                _lerp_joints(
                    start,
                    end,
                    alpha,
                    smooth_gripper=smooth_gripper,
                    snap_gripper=step == factor,
                    stepwise_all=stepwise_all,
                )
            )
            interpolated_times.append(start_time + alpha * (end_time - start_time))
    return interpolated, interpolated_times


def _validate_interp_factor(interp_factor: int) -> None:
    if (
        isinstance(interp_factor, bool)
        or not isinstance(interp_factor, int)
        or interp_factor <= 0
    ):
        raise ValueError("interp_factor must be a positive integer")


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
    max_waypoints: Optional[int] = None,
    interp_factor: int = 1,
    smooth_gripper: bool = False,
) -> Dict[str, Any]:
    _validate_interp_factor(interp_factor)
    if max_waypoints is not None:
        if (
            isinstance(max_waypoints, bool)
            or not isinstance(max_waypoints, int)
            or max_waypoints <= 0
        ):
            raise ValueError("max_waypoints must be a positive integer")

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

    time_list = _build_time_list(action, num_waypoints, default_dt)
    source_num_waypoints = num_waypoints
    if max_waypoints is not None:
        num_waypoints = min(num_waypoints, max_waypoints)
        left_arm_chunk = left_arm_chunk[:num_waypoints]
        right_arm_chunk = right_arm_chunk[:num_waypoints]
        time_list = time_list[:num_waypoints]
        if left_gripper_chunk is not None:
            left_gripper_chunk = left_gripper_chunk[:num_waypoints]
        if right_gripper_chunk is not None:
            right_gripper_chunk = right_gripper_chunk[:num_waypoints]

    executed_waypoints = num_waypoints
    if interp_factor > 1 and num_waypoints > 1:
        left_arm_chunk, interpolated_times = _interpolate_waypoints(
            left_arm_chunk,
            time_list,
            interp_factor,
            smooth_gripper=smooth_gripper,
        )
        right_arm_chunk, _ = _interpolate_waypoints(
            right_arm_chunk,
            time_list,
            interp_factor,
            smooth_gripper=smooth_gripper,
        )
        if left_gripper_chunk is not None:
            left_gripper_chunk, _ = _interpolate_waypoints(
                left_gripper_chunk,
                time_list,
                interp_factor,
                smooth_gripper=smooth_gripper,
                stepwise_all=not smooth_gripper,
            )
        if right_gripper_chunk is not None:
            right_gripper_chunk, _ = _interpolate_waypoints(
                right_gripper_chunk,
                time_list,
                interp_factor,
                smooth_gripper=smooth_gripper,
                stepwise_all=not smooth_gripper,
            )
        time_list = interpolated_times
        num_waypoints = len(left_arm_chunk)

    return {
        "left_arm": left_arm_chunk,
        "right_arm": right_arm_chunk,
        "left_gripper": left_gripper_chunk,
        "right_gripper": right_gripper_chunk,
        "time_list": time_list,
        "num_waypoints": num_waypoints,
        "source_num_waypoints": source_num_waypoints,
        "executed_waypoints": executed_waypoints,
        "interp_factor": interp_factor,
        "blend_steps": 0,
        "smooth_gripper": smooth_gripper,
    }


def _validate_blend_steps(blend_steps: int) -> None:
    if (
        isinstance(blend_steps, bool)
        or not isinstance(blend_steps, int)
        or blend_steps < 0
    ):
        raise ValueError("blend_steps must be a non-negative integer")


def _merged_chunk_waypoint(
    chunk: Dict[str, Any],
    index: int,
    left_arm_dim: int,
    right_arm_dim: int,
) -> Tuple[List[float], List[float]]:
    left = list(chunk["left_arm"][index])
    right = list(chunk["right_arm"][index])
    if len(left) < left_arm_dim and chunk["left_gripper"] is not None:
        left = left + [float(chunk["left_gripper"][index][0])]
    if len(right) < right_arm_dim and chunk["right_gripper"] is not None:
        right = right + [float(chunk["right_gripper"][index][0])]
    return left, right


def _chunk_step_dt(chunk: Dict[str, Any], default_dt: float) -> float:
    time_list = chunk["time_list"]
    if len(time_list) >= 2:
        step_dt = time_list[1] - time_list[0]
    elif time_list:
        step_dt = time_list[0]
    else:
        step_dt = default_dt
    if step_dt <= 0:
        raise ValueError("blend step dt must be > 0, got {}".format(step_dt))
    return step_dt


def blend_action_chunks(
    chunk: Dict[str, Any],
    previous_left: Sequence[float],
    previous_right: Sequence[float],
    blend_steps: int,
    default_dt: float,
    left_arm_dim: int,
    right_arm_dim: int,
    smooth_gripper: bool = False,
) -> Dict[str, Any]:
    _validate_blend_steps(blend_steps)
    blended = dict(chunk)
    blended["blend_steps"] = blend_steps
    if blend_steps == 0:
        return blended

    first_left, first_right = _merged_chunk_waypoint(
        chunk, 0, left_arm_dim, right_arm_dim
    )
    previous_left = [float(value) for value in previous_left]
    previous_right = [float(value) for value in previous_right]
    if len(previous_left) != len(first_left) or len(previous_right) != len(first_right):
        raise ValueError(
            "previous chunk pose dims must match the next chunk: "
            "left {} vs {}, right {} vs {}".format(
                len(previous_left),
                len(first_left),
                len(previous_right),
                len(first_right),
            )
        )

    step_dt = _chunk_step_dt(chunk, default_dt)
    blend_left = []
    blend_right = []
    blend_times = []
    for step in range(1, blend_steps + 1):
        alpha = float(step) / float(blend_steps)
        snap_gripper = step == blend_steps
        blend_left.append(
            _lerp_joints(
                previous_left,
                first_left,
                alpha,
                smooth_gripper=smooth_gripper,
                snap_gripper=snap_gripper,
            )
        )
        blend_right.append(
            _lerp_joints(
                previous_right,
                first_right,
                alpha,
                smooth_gripper=smooth_gripper,
                snap_gripper=snap_gripper,
            )
        )
        blend_times.append(step_dt * step)

    rest_offset = blend_times[-1] - chunk["time_list"][0]
    rest_left = []
    rest_right = []
    rest_times = []
    for index in range(1, chunk["num_waypoints"]):
        left, right = _merged_chunk_waypoint(
            chunk, index, left_arm_dim, right_arm_dim
        )
        rest_left.append(left)
        rest_right.append(right)
        rest_times.append(chunk["time_list"][index] + rest_offset)

    blended["left_arm"] = blend_left + rest_left
    blended["right_arm"] = blend_right + rest_right
    blended["left_gripper"] = None
    blended["right_gripper"] = None
    blended["time_list"] = blend_times + rest_times
    blended["num_waypoints"] = len(blended["left_arm"])
    blended["smooth_gripper"] = smooth_gripper
    return blended


def prepare_executed_chunk(
    action: Dict[str, Any],
    default_dt: float,
    left_arm_dim: int,
    right_arm_dim: int,
    max_waypoints: Optional[int] = None,
    interp_factor: int = 1,
    blend_steps: int = 0,
    smooth_gripper: bool = False,
    previous_left: Optional[Sequence[float]] = None,
    previous_right: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    _validate_blend_steps(blend_steps)
    chunk = parse_chunked_action(
        action,
        default_dt=default_dt,
        left_arm_dim=left_arm_dim,
        right_arm_dim=right_arm_dim,
        max_waypoints=max_waypoints,
        interp_factor=interp_factor,
        smooth_gripper=smooth_gripper,
    )
    if (
        blend_steps > 0
        and previous_left is not None
        and previous_right is not None
    ):
        chunk = blend_action_chunks(
            chunk,
            previous_left,
            previous_right,
            blend_steps,
            default_dt,
            left_arm_dim,
            right_arm_dim,
            smooth_gripper=smooth_gripper,
        )
    return chunk


def log_executed_chunk(chunk: Dict[str, Any]) -> None:
    if chunk["executed_waypoints"] < chunk["source_num_waypoints"]:
        print(
            "[execute] limiting action chunk to {}/{} waypoints".format(
                chunk["executed_waypoints"], chunk["source_num_waypoints"]
            )
        )
    if chunk["interp_factor"] > 1:
        print(
            "[execute] interpolated action chunk x{}".format(chunk["interp_factor"])
        )
    if chunk["blend_steps"] > 0:
        print(
            "[execute] blended from previous chunk with {} steps to {} waypoints".format(
                chunk["blend_steps"],
                chunk["num_waypoints"],
            )
        )
    if (chunk["interp_factor"] > 1 or chunk["blend_steps"] > 0) and not chunk.get(
        "smooth_gripper", False
    ):
        print("[execute] gripper held stepwise; only joints are smoothed")


def slice_action_chunk(action: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    """Return a relative-time action slice for intermediate observation cadence."""
    left = action.get("left_arm", [])[start:end]
    right = action.get("right_arm", [])[start:end]
    if not left or not right or len(left) != len(right):
        raise ValueError("action chunk does not contain matching arm waypoints")
    raw_times = action.get("time_list") or []
    if len(raw_times) >= end:
        base = float(raw_times[start - 1]) if start else 0.0
        times = [float(raw_times[index]) - base for index in range(start, end)]
    else:
        dt = float(action.get("dt", 1.0 / 30.0))
        times = [dt * (index + 1) for index in range(end - start)]
    sliced = {"left_arm": left, "right_arm": right, "time_list": times}
    for key in ("left_gripper", "right_gripper"):
        if key in action:
            sliced[key] = action[key][start:end]
    sliced["dt"] = float(action.get("dt", times[0] if times else 1.0 / 30.0))
    return sliced


async def _read_policy_responses(
    websocket: Any,
    response_queues: Dict[str, "asyncio.Queue[Dict[str, Any]]"],
    observation_queue: "asyncio.Queue[Dict[str, Any]]",
) -> None:
    """Route concurrent server replies without stealing action responses."""
    try:
        async for raw in websocket:
            response = json.loads(raw)
            request_id = str(response.get("request_id", ""))
            if response.get("observation_only", False):
                await observation_queue.put(response)
                continue
            queue = response_queues.get(request_id)
            if queue is not None:
                await queue.put(response)
    except asyncio.CancelledError:
        raise


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
    def __init__(
        self,
        chunk_dt: float,
        left_arm_dim: int,
        right_arm_dim: int,
        max_waypoints: Optional[int] = None,
        interp_factor: int = 1,
        blend_steps: int = 0,
        smooth_gripper: bool = False,
    ) -> None:
        self.chunk_dt = chunk_dt
        self.left_arm_dim = left_arm_dim
        self.right_arm_dim = right_arm_dim
        self.max_waypoints = max_waypoints
        self.interp_factor = interp_factor
        self.blend_steps = blend_steps
        self.smooth_gripper = smooth_gripper
        self._last_published_target: Optional[Tuple[List[float], List[float]]] = None

    def execute_action(self, action: Dict[str, Any]) -> float:
        previous = self._last_published_target
        chunk = prepare_executed_chunk(
            action,
            default_dt=self.chunk_dt,
            left_arm_dim=self.left_arm_dim,
            right_arm_dim=self.right_arm_dim,
            max_waypoints=self.max_waypoints,
            interp_factor=self.interp_factor,
            blend_steps=self.blend_steps,
            smooth_gripper=self.smooth_gripper,
            previous_left=None if previous is None else previous[0],
            previous_right=None if previous is None else previous[1],
        )
        log_executed_chunk(chunk)
        last_left, last_right = _merged_chunk_waypoint(
            chunk,
            chunk["num_waypoints"] - 1,
            self.left_arm_dim,
            self.right_arm_dim,
        )
        self._last_published_target = (last_left, last_right)
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
        video_recorder: Optional[CameraVideoRecorder] = None,
        overlay_recorder: Optional[CameraVideoRecorder] = None,
        subtask_overlay: Optional[SubtaskOverlay] = None,
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
        self.video_recorder = video_recorder
        self.overlay_recorder = overlay_recorder
        self.subtask_overlay = subtask_overlay
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
        payload = bytes(msg.data)
        if self.video_recorder is not None:
            self.video_recorder.submit(camera_name, payload)
        overlay_recorder = getattr(self, "overlay_recorder", None)
        if camera_name == "head" and overlay_recorder is not None:
            overlay_recorder.submit(camera_name, payload)
        with self.lock:
            self.images[camera_name] = {
                "encoding": "jpeg",
                "data": payload,
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

    def finalize_recording(self) -> None:
        recorder = self.video_recorder
        if recorder is None:
            return
        self.video_recorder = None
        overlay_recorder = getattr(self, "overlay_recorder", None)
        self.overlay_recorder = None

        print("[record] finalizing videos in {}".format(recorder.session_dir))
        recorder.close()
        if overlay_recorder is not None:
            overlay_recorder.close()
        manifest_path = recorder.write_manifest()
        print("[record] manifest written: {}".format(manifest_path))
        try:
            remote_path = recorder.upload()
            print("[record] uploaded recording: {}".format(remote_path))
        except Exception as exc:
            print("[record] upload failed: {}".format(exc))
            print("[record] local recording preserved: {}".format(recorder.session_dir))

    def update_recording_metadata(self, response: Dict[str, Any]) -> None:
        subtask_overlay = getattr(self, "subtask_overlay", None)
        if subtask_overlay is not None:
            subtask_overlay.update(response)


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
        max_waypoints: Optional[int] = None,
        interp_factor: int = 1,
        blend_steps: int = 0,
        smooth_gripper: bool = False,
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
        self.max_waypoints = max_waypoints
        self.interp_factor = interp_factor
        self.blend_steps = blend_steps
        self.smooth_gripper = smooth_gripper
        self.Bool = Bool
        self.left_pub = rospy.Publisher(left_action_topic, JointState, queue_size=1, tcp_nodelay=True)
        self.right_pub = rospy.Publisher(right_action_topic, JointState, queue_size=1, tcp_nodelay=True)
        self.enable_pub = rospy.Publisher(enable_topic, Bool, queue_size=1, latch=True)
        # Keep the last command separate from ROS observations. After a short
        # waypoint, the feedback topic can still contain the pre-command pose;
        # holding that stale observation would command the arm back immediately.
        self._last_published_target: Optional[Tuple[List[float], List[float]]] = None

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
        left_target = [float(value) for value in left]
        right_target = [float(value) for value in right]
        self.left_pub.publish(self._build_joint_msg(left_target))
        self.right_pub.publish(self._build_joint_msg(right_target))
        self._last_published_target = (left_target, right_target)

    def move_home(
        self,
        state: Dict[str, Any],
        home_hz: float,
        min_duration: float,
        max_duration: float,
        joint_speed: float,
        tolerance: float,
    ) -> float:
        home_left = [0.0] * self.left_arm_dim
        home_right = [0.0] * self.right_arm_dim
        return self.move_to_pose(
            state,
            home_left,
            home_right,
            move_hz=home_hz,
            min_duration=min_duration,
            max_duration=max_duration,
            joint_speed=joint_speed,
            safety_check=lambda: None,
            label="home",
            already_at_tolerance=tolerance,
        )

    def move_to_pose(
        self,
        state: Dict[str, Any],
        target_left: Sequence[float],
        target_right: Sequence[float],
        *,
        move_hz: float,
        min_duration: float,
        max_duration: float,
        joint_speed: float,
        safety_check: Callable[[], None],
        label: str,
        already_at_tolerance: float = 0.0,
    ) -> float:
        if move_hz <= 0:
            raise ValueError("move_hz must be > 0")
        if min_duration < 0 or max_duration < 0:
            raise ValueError("move durations must be >= 0")
        if max_duration < min_duration:
            raise ValueError("max_duration must be >= min_duration")
        if joint_speed <= 0:
            raise ValueError("joint_speed must be > 0")
        if already_at_tolerance < 0:
            raise ValueError("already_at_tolerance must be >= 0")

        start_left, start_right = self._state_pair(state)
        left_target = _normalize_numeric_vector(
            target_left, "target.left_arm", self.left_arm_dim
        )
        right_target = _normalize_numeric_vector(
            target_right, "target.right_arm", self.right_arm_dim
        )
        max_delta = max(
            abs(start - target)
            for start, target in zip(
                start_left + start_right, left_target + right_target
            )
        )

        required_duration_s = max_delta / joint_speed
        if required_duration_s > max_duration:
            raise ValueError(
                "{} requires {:.2f}s at configured speed, exceeding max_duration {:.2f}s".format(
                    label,
                    required_duration_s,
                    max_duration,
                )
            )

        self.set_enable(True, label)
        period_s = 1.0 / move_hz
        if max_delta <= already_at_tolerance:
            print("[{}] already at target; max_delta={:.6f}".format(label, max_delta))
            for _ in range(max(3, int(0.5 * move_hz))):
                self._raise_if_shutdown()
                safety_check()
                self._publish_pair(left_target, right_target)
                time.sleep(period_s)
            return 0.0

        duration_s = max(min_duration, required_duration_s)
        num_steps = max(2, int(duration_s * move_hz))
        print(
            "[{}] moving to target; max_delta={:.4f}, duration={:.2f}s, steps={}".format(
                label,
                max_delta,
                duration_s,
                num_steps,
            )
        )

        for step in range(num_steps + 1):
            self._raise_if_shutdown()
            safety_check()
            alpha = _smoothstep(float(step) / float(num_steps))
            left = [
                (1.0 - alpha) * start + alpha * target
                for start, target in zip(start_left, left_target)
            ]
            right = [
                (1.0 - alpha) * start + alpha * target
                for start, target in zip(start_right, right_target)
            ]
            self._publish_pair(left, right)
            time.sleep(period_s)

        for _ in range(max(3, int(0.5 * move_hz))):
            self._raise_if_shutdown()
            safety_check()
            self._publish_pair(left_target, right_target)
            time.sleep(period_s)

        print("[{}] target command published".format(label))
        return duration_s

    def hold_position(self, state: Dict[str, Any], hold_hz: float) -> None:
        if hold_hz <= 0:
            raise ValueError("--hold-hz must be > 0")
        if self._last_published_target is None:
            left, right = self._state_pair(state)
            target_source = "latest observed state"
        else:
            left = list(self._last_published_target[0])
            right = list(self._last_published_target[1])
            target_source = "last published waypoint"
        period_s = 1.0 / hold_hz
        self.set_enable(True, "hold")
        print(
            "[hold] holding {}; press Ctrl-C again to leave hold loop".format(
                target_source
            )
        )
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
        previous = self._last_published_target
        chunk = prepare_executed_chunk(
            action,
            default_dt=self.chunk_dt,
            left_arm_dim=self.left_arm_dim,
            right_arm_dim=self.right_arm_dim,
            max_waypoints=getattr(self, "max_waypoints", None),
            interp_factor=getattr(self, "interp_factor", 1),
            blend_steps=getattr(self, "blend_steps", 0),
            smooth_gripper=getattr(self, "smooth_gripper", False),
            previous_left=None if previous is None else previous[0],
            previous_right=None if previous is None else previous[1],
        )
        log_executed_chunk(chunk)
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
            self._last_published_target = (list(left), list(right))
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
    camera_topics = parse_camera_topics(args.camera_topic)
    video_recorder = None
    overlay_recorder = None
    subtask_overlay = None
    if args.record_videos:
        if not args.compressed_images:
            raise ValueError("--record-videos requires --compressed-images")
        if not camera_topics:
            raise ValueError("--record-videos requires at least one --camera-topic")
        video_recorder = CameraVideoRecorder(
            camera_names=camera_topics.keys(),
            root_dir=args.record_dir,
            fps=args.record_fps,
            queue_size=args.record_queue_size,
            session_name=args.record_session_name,
            upload_host=args.record_upload_host,
            upload_port=args.record_upload_port,
            upload_key=args.record_upload_key,
            upload_dir=args.record_upload_dir,
        )
        if "head" in camera_topics:
            subtask_overlay = SubtaskOverlay()
            overlay_recorder = CameraVideoRecorder(
                camera_names=("head",),
                root_dir=args.record_dir,
                session_name=args.record_session_name,
                session_dir=video_recorder.session_dir,
                output_filename="head_subtask_overlay.mp4",
                frame_transform=subtask_overlay.transform,
                fps=args.record_fps,
                queue_size=args.record_queue_size,
                upload_host=args.record_upload_host,
                upload_port=args.record_upload_port,
                upload_key=args.record_upload_key,
                upload_dir=args.record_upload_dir,
            )
        print("[record] recording session: {}".format(video_recorder.session_dir))

    try:
        return RosObservationSource(
            left_joint_topic=args.left_observation_topic,
            right_joint_topic=args.right_observation_topic,
            camera_topics=camera_topics,
            compressed_images=args.compressed_images,
            require_images=args.require_images,
            enable_topic=args.enable_topic,
            video_recorder=video_recorder,
            overlay_recorder=overlay_recorder,
            subtask_overlay=subtask_overlay,
        )
    except Exception:
        if video_recorder is not None:
            video_recorder.close()
            if overlay_recorder is not None:
                overlay_recorder.close()
            video_recorder.write_manifest()
        raise


def create_executor(args: argparse.Namespace) -> Any:
    chunk_dt = 1.0 / args.control_hz
    if args.executor == "mock":
        return MockExecutor(
            chunk_dt,
            args.left_arm_dim,
            args.right_arm_dim,
            max_waypoints=args.max_waypoints,
            interp_factor=args.interp_factor,
            blend_steps=args.blend_steps,
            smooth_gripper=args.smooth_gripper,
        )
    return RosJointExecutor(
        left_action_topic=args.left_action_topic,
        right_action_topic=args.right_action_topic,
        enable_topic=args.enable_topic,
        chunk_dt=chunk_dt,
        left_arm_dim=args.left_arm_dim,
        right_arm_dim=args.right_arm_dim,
        enable_on_start=args.enable_on_start,
        waypoint_sleep=args.waypoint_sleep,
        max_waypoints=args.max_waypoints,
        interp_factor=args.interp_factor,
        blend_steps=args.blend_steps,
        smooth_gripper=args.smooth_gripper,
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


def validate_initialization_args(args: argparse.Namespace) -> None:
    if args.start_pose_config and not args.home_on_start:
        raise ValueError(
            "--start-pose-config cannot be combined with --no-home-on-start"
        )


def wait_until_start_pose(
    source: Any,
    config: StartPoseConfig,
    *,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> Dict[str, Any]:
    deadline = monotonic() + config.feedback_timeout_s
    last_arm_error = float("inf")
    last_gripper_error = float("inf")
    while monotonic() < deadline:
        raise_if_safety_tripped(source)
        state = get_current_state(source)
        last_arm_error, last_gripper_error = pose_errors(state, config)
        if pose_is_reached(state, config):
            print(
                "[start-pose] reached source_episode={} "
                "arm_error={:.6f} gripper_error={:.6f}".format(
                    config.source_episode,
                    last_arm_error,
                    last_gripper_error,
                )
            )
            return state
        sleep(0.05)
    raise RuntimeError(
        "start-pose feedback did not reach configured tolerances within {:.2f}s; "
        "last_arm_error={:.6f} last_gripper_error={:.6f}".format(
            config.feedback_timeout_s,
            last_arm_error,
            last_gripper_error,
        )
    )


async def run_loop(args: argparse.Namespace) -> None:
    if websockets is None:
        raise RuntimeError(
            "websockets is required for the Piper client; install the robot client dependencies"
        )
    if args.control_hz <= 0:
        raise ValueError("--control-hz must be > 0")
    validate_initialization_args(args)
    start_pose_config = (
        load_start_pose_config(args.start_pose_config)
        if args.start_pose_config
        else None
    )

    source = create_source(args)
    executor = create_executor(args)
    transport_image_sizes = (
        FASTWAM_TRANSPORT_IMAGE_SIZES
        if args.transport_image_profile == "fastwam"
        else None
    )

    try:
        print("waiting for local observation source...")
        source.wait_until_ready(timeout_s=args.ready_timeout)
        print("observation source is ready")
        if transport_image_sizes is not None:
            print(
                "[transport] FastWAM JPEG resize before WebSocket: "
                "head=320x256, left_wrist=160x128, right_wrist=160x128"
            )

        if args.execute_actions and start_pose_config is not None:
            move_to_pose = getattr(executor, "move_to_pose", None)
            if move_to_pose is None:
                raise RuntimeError(
                    "selected executor cannot move to a configured start pose"
            )
            current_state = get_current_state(source)
            current_state = validate_state_within_bounds(
                current_state, start_pose_config
            )
            move_to_pose(
                current_state,
                start_pose_config.left_arm,
                start_pose_config.right_arm,
                move_hz=start_pose_config.move_hz,
                min_duration=start_pose_config.min_duration_s,
                max_duration=start_pose_config.max_duration_s,
                joint_speed=start_pose_config.max_joint_speed,
                safety_check=lambda: raise_if_safety_tripped(source),
                label=start_pose_config.name,
            )
            wait_until_start_pose(source, start_pose_config)
        elif args.execute_actions and args.home_on_start:
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
            response_queues: Dict[str, asyncio.Queue] = {}
            observation_queue: asyncio.Queue = asyncio.Queue()
            response_reader = asyncio.create_task(
                _read_policy_responses(websocket, response_queues, observation_queue)
            )

            async def request_response(request_payload: Dict[str, Any]) -> Dict[str, Any]:
                request_id = str(request_payload["request_id"])
                queue: asyncio.Queue = asyncio.Queue(maxsize=1)
                response_queues[request_id] = queue
                try:
                    await websocket.send(json.dumps(request_payload, ensure_ascii=False))
                    return await queue.get()
                finally:
                    response_queues.pop(request_id, None)

            if args.reset_before_start:
                reset_request = {
                    "request_id": "client-reset-{}".format(int(time.time() * 1000)),
                    "type": "reset",
                    "step": 0,
                }
                reset_response = await request_response(reset_request)
                if not reset_response.get("ok", False) or not reset_response.get("reset", False):
                    raise RuntimeError(
                        "policy server reset failed: {}".format(reset_response.get("error"))
                    )
                print("policy server memory reset")
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
                    transport_image_sizes=transport_image_sizes,
                )
                # Tell adaptive servers how many returned waypoints will be
                # physically executed before this client's next observation.
                execute_horizon = int(args.max_waypoints or 32)
                if execute_horizon <= 0:
                    raise ValueError("client execution horizon must be positive")
                request["client_execution_horizon"] = execute_horizon
                request["motion_mode"] = args.motion_mode
                print("state", state)

                response = await request_response(request)
                if not response.get("ok", False):
                    raise RuntimeError("policy server error: {}".format(response.get("error")))
                update_recording_metadata = getattr(source, "update_recording_metadata", None)
                if update_recording_metadata is not None:
                    update_recording_metadata(response)
                if response.get("final_task_done", False):
                    print("[task] server Q2 marked the final subtask complete; stopping execution")
                    break
                print("action", json.dumps(response["action"],ensure_ascii=False,indent=2))
                raise_if_safety_tripped(source)
                if args.execute_actions:
                    action = response["action"]
                    available = len(action.get("left_arm", []))
                    if available < execute_horizon:
                        raise RuntimeError(
                            "policy returned {} waypoints, cannot execute requested {}".format(
                                available, execute_horizon
                            )
                        )
                    # Execute the returned chunk as one unit, matching the
                    # Pi05 client. Inter-chunk blending must happen once at
                    # the boundary, not once for every four-waypoint slice.
                    chunk_duration_s = executor.execute_action(action)
                    step += execute_horizon
                    images, state = source.get_observation()
                    update = build_policy_request(
                        step=step,
                        instruction=args.instruction,
                        state=state,
                        images=images,
                        jpeg_quality=args.jpeg_quality,
                        resize_scale=args.resize_scale,
                        transport_image_sizes=transport_image_sizes,
                    )
                    update["observation_only"] = True
                    update["client_execution_horizon"] = execute_horizon
                    update["motion_mode"] = args.motion_mode
                    await websocket.send(json.dumps(update, ensure_ascii=False))
                    await asyncio.sleep(0)
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

                if not args.execute_actions:
                    step += execute_horizon
                min_cycle_s = max(1.0 / args.control_hz, chunk_duration_s)
                sleep_s = max(0.0, min_cycle_s - elapsed)
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
    except SafetyTrip as exc:
        print("[safety] client stopped: {}".format(exc))
    finally:
        if "response_reader" in locals():
            response_reader.cancel()
            try:
                await response_reader
            except asyncio.CancelledError:
                pass
        try:
            if args.execute_actions and is_safety_tripped(source):
                print("[safety] skipping hold/disable-on-exit cleanup after arm_status fault")
            elif args.execute_actions:
                if args.home_on_exit:
                    move_home = getattr(executor, "move_home", None)
                    if move_home is not None:
                        try:
                            print("[exit] returning to home pose")
                            move_home(
                                get_current_state(source),
                                home_hz=args.home_hz,
                                min_duration=args.home_min_duration,
                                max_duration=args.home_max_duration,
                                joint_speed=args.home_joint_speed,
                                tolerance=args.home_tolerance,
                            )
                        except KeyboardInterrupt:
                            print("[exit] home interrupted")
                        except Exception as exc:
                            print("[exit] home failed: {}".format(exc))
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
        finally:
            finalize_recording = getattr(source, "finalize_recording", None)
            if finalize_recording is not None:
                finalize_recording()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Piper websocket policy client")
    parser.add_argument("--uri", required=True, help="e.g. ws://192.168.1.20:7081")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--source", choices=["ros", "mock"], default="ros")
    parser.add_argument("--executor", choices=["ros", "mock"], default="ros")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--reset-before-start",
        action="store_true",
        help="clear policy/Wan/VLM episode state after connecting, without stopping the server",
    )
    parser.add_argument(
        "--execution-horizon",
        "--max-waypoints",
        dest="max_waypoints",
        type=int,
        default=None,
        metavar="N",
        help=(
            "execute only the first N waypoints from each returned action chunk; "
            "--max-waypoints is a deprecated alias; default executes the complete chunk"
        ),
    )
    parser.add_argument(
        "--action-interp-factor",
        dest="interp_factor",
        type=int,
        default=1,
        metavar="N",
        help=(
            "linearly upsample each executed action chunk by N; "
            "N=1 sends the original waypoints, N=5 inserts 4 intermediate "
            "points between each pair and keeps the original chunk duration"
        ),
    )
    parser.add_argument(
        "--action-chunk-blend-steps",
        dest="blend_steps",
        type=int,
        default=0,
        metavar="N",
        help=(
            "linearly blend from the last published waypoint of the previous "
            "chunk to the first waypoint of the next chunk using N steps; "
            "N=0 disables inter-chunk blending"
        ),
    )
    parser.add_argument("--motion-mode", choices=["smooth", "interp"], default="smooth")
    parser.set_defaults(smooth_gripper=False)
    parser.add_argument(
        "--smooth-gripper",
        action="store_true",
        dest="smooth_gripper",
        help="also interpolate/blend gripper values; default keeps gripper stepwise",
    )
    parser.add_argument(
        "--no-smooth-gripper",
        action="store_false",
        dest="smooth_gripper",
        help="keep gripper commands stepwise and only smooth arm joints",
    )
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--resize-scale", type=float, default=1.0)
    parser.add_argument(
        "--transport-image-profile",
        choices=("none", "fastwam"),
        default="none",
        help=(
            "resize JPEGs before WebSocket transport; 'fastwam' sends the exact "
            "320x256/160x128 tiles required by FastWAM without changing OpenPI defaults"
        ),
    )
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
    parser.add_argument(
        "--start-pose-config",
        default=None,
        help="move to and verify a task-specific pose before physical policy execution",
    )
    parser.add_argument("--home-hz", type=float, default=20.0)
    parser.add_argument("--home-min-duration", type=float, default=2.0)
    parser.add_argument("--home-max-duration", type=float, default=20.0)
    parser.add_argument("--home-joint-speed", type=float, default=0.25)
    parser.add_argument("--home-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--home-on-exit",
        action="store_true",
        help="smoothly return to zero joints after Ctrl-C or loop exit, before hold",
    )
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
    parser.add_argument(
        "--record-videos",
        action="store_true",
        help="record every compressed camera callback to one MP4 per camera",
    )
    parser.add_argument("--record-dir", default=DEFAULT_RECORD_DIR)
    parser.add_argument("--record-fps", type=float, default=30.0)
    parser.add_argument("--record-queue-size", type=int, default=90)
    parser.add_argument("--record-upload-host", default=DEFAULT_UPLOAD_HOST)
    parser.add_argument("--record-upload-port", type=int, default=DEFAULT_UPLOAD_PORT)
    parser.add_argument(
        "--record-upload-key",
        default=DEFAULT_UPLOAD_KEY,
        help="SSH private key used for recording upload",
    )
    parser.add_argument("--record-upload-dir", default=DEFAULT_UPLOAD_DIR)
    parser.add_argument("--record-session-name", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(run_loop(args))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\npolicy client stopped")
