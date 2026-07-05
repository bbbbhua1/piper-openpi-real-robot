#!/usr/bin/env python3
"""
Host A client:
1. read head + left wrist + right wrist images
2. read current arm joints + gripper state as state
3. send one policy request to host B
4. receive one chunked action from host B
5. execute the action chunk locally via `move_joints_waypoints`
6. repeat

Default source/executor use Astribot directly on host A.
Use `--source mock --executor mock` to test the network loop without robot hardware.

Expected response schema:
{
    "left_arm": [[...], [...], ...],
    "right_arm": [[...], [...], ...],
    "left_gripper": [0.0, 0.0, ...],        # optional
    "right_gripper": [0.0, 0.0, ...],       # optional
    "time_list": [0.1, 0.2, 0.3, ...],      # optional
    "dt": 0.1,                              # optional fallback when time_list is absent
}

When both `time_list` and `dt` are absent, the client derives waypoint timing from
`1 / --control-hz`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from typing import Any

import numpy as np
import websockets

from ws_policy_protocol import CAMERA_NAMES, build_policy_request


def _normalize_numeric_vector(values: Any, field_name: str, expected_len: int) -> list[float]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be a list with {expected_len} numbers")

    vector = []
    for value in values:
        try:
            vector.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} contains a non-numeric value: {value!r}") from exc

    if len(vector) != expected_len:
        raise ValueError(
            f"{field_name} must contain exactly {expected_len} values, got {len(vector)}"
        )
    return vector


def _normalize_vector_chunk(chunk: Any, field_name: str, vector_len: int) -> list[list[float]]:
    if not isinstance(chunk, (list, tuple)) or len(chunk) == 0:
        raise ValueError(f"{field_name} must be a non-empty list of waypoint vectors")

    return [
        _normalize_numeric_vector(waypoint, f"{field_name}[{index}]", vector_len)
        for index, waypoint in enumerate(chunk)
    ]


def _normalize_scalar_chunk(chunk: Any, field_name: str) -> list[list[float]]:
    if not isinstance(chunk, (list, tuple)) or len(chunk) == 0:
        raise ValueError(f"{field_name} must be a non-empty list of scalar waypoints")

    normalized = []
    for index, waypoint in enumerate(chunk):
        if isinstance(waypoint, (list, tuple)):
            normalized.append(
                _normalize_numeric_vector(waypoint, f"{field_name}[{index}]", expected_len=1)
            )
            continue

        try:
            normalized.append([float(waypoint)])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name}[{index}] must be a scalar or a single-value list, got {waypoint!r}"
            ) from exc
    return normalized


def _build_time_list(action: dict[str, Any], num_waypoints: int, default_dt: float) -> list[float]:
    raw_time_list = action.get("time_list")
    if raw_time_list is not None:
        time_list = _normalize_numeric_vector(raw_time_list, "time_list", num_waypoints)
    else:
        raw_dt = action.get("dt", default_dt)
        try:
            dt = float(raw_dt)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"dt must be numeric, got {raw_dt!r}") from exc
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        time_list = [dt * (index + 1) for index in range(num_waypoints)]

    previous_time = 0.0
    for index, arrival_time in enumerate(time_list):
        if arrival_time <= previous_time:
            raise ValueError(
                f"time_list must be strictly increasing and > 0, got {arrival_time} at index {index}"
            )
        previous_time = arrival_time
    return time_list


def _parse_chunked_action(
    action: dict[str, Any],
    default_dt: float,
    left_arm_dim: int = 7,
    right_arm_dim: int = 7,
) -> dict[str, Any]:
    if not isinstance(action, dict):
        raise ValueError(f"chunked action must be a dict, got {type(action).__name__}")

    left_arm_chunk = _normalize_vector_chunk(action.get("left_arm"), "left_arm", vector_len=left_arm_dim)
    right_arm_chunk = _normalize_vector_chunk(action.get("right_arm"), "right_arm", vector_len=right_arm_dim)

    num_waypoints = len(left_arm_chunk)
    if len(right_arm_chunk) != num_waypoints:
        raise ValueError(
            "left_arm and right_arm must have the same number of waypoints: "
            f"{num_waypoints} vs {len(right_arm_chunk)}"
        )

    left_gripper_chunk = None
    if "left_gripper" in action and action["left_gripper"] is not None:
        left_gripper_chunk = _normalize_scalar_chunk(action["left_gripper"], "left_gripper")
        if len(left_gripper_chunk) != num_waypoints:
            raise ValueError(
                "left_gripper must have the same number of waypoints as left_arm: "
                f"{len(left_gripper_chunk)} vs {num_waypoints}"
            )

    right_gripper_chunk = None
    if "right_gripper" in action and action["right_gripper"] is not None:
        right_gripper_chunk = _normalize_scalar_chunk(action["right_gripper"], "right_gripper")
        if len(right_gripper_chunk) != num_waypoints:
            raise ValueError(
                "right_gripper must have the same number of waypoints as right_arm: "
                f"{len(right_gripper_chunk)} vs {num_waypoints}"
            )

    return {
        "left_arm": left_arm_chunk,
        "right_arm": right_arm_chunk,
        "left_gripper": left_gripper_chunk,
        "right_gripper": right_gripper_chunk,
        "time_list": _build_time_list(action, num_waypoints=num_waypoints, default_dt=default_dt),
        "num_waypoints": num_waypoints,
    }


class MockObservationSource:
    """Local smoke-test source without robot dependencies."""

    def __init__(self) -> None:
        self.step = 0

    def wait_until_ready(self, timeout_s: float = 5.0) -> None:
        return None

    def get_observation(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.step += 1
        images = {}
        for idx, name in enumerate(CAMERA_NAMES):
            image = np.zeros((240, 320, 3), dtype=np.uint8)
            image[:, :, idx % 3] = min(255, 40 * self.step)
            images[name] = image

        phase = 0.05 * self.step
        state = {
            "left_arm": [float(np.sin(phase + i * 0.1)) for i in range(7)],
            "right_arm": [float(np.cos(phase + i * 0.1)) for i in range(7)],
            "left_gripper": float(min(100.0, self.step)),
            "right_gripper": float(max(0.0, 100.0 - self.step)),
        }
        return images, state


class MockExecutor:
    """Local smoke-test executor for chunked actions."""

    def __init__(self, chunk_dt: float, left_arm_dim: int = 7, right_arm_dim: int = 7):
        self.chunk_dt = chunk_dt
        self.left_arm_dim = left_arm_dim
        self.right_arm_dim = right_arm_dim

    def execute_action(self, action: dict[str, Any]) -> float:
        chunk = _parse_chunked_action(
            action,
            default_dt=self.chunk_dt,
            left_arm_dim=self.left_arm_dim,
            right_arm_dim=self.right_arm_dim,
        )
        summary = {
            "num_waypoints": chunk["num_waypoints"],
            "time_list": chunk["time_list"],
            "left_arm_first": chunk["left_arm"][0],
            "right_arm_first": chunk["right_arm"][0],
        }
        if chunk["left_gripper"] is not None:
            summary["left_gripper_first"] = chunk["left_gripper"][0][0]
        if chunk["right_gripper"] is not None:
            summary["right_gripper_first"] = chunk["right_gripper"][0][0]
        print("[mock executor] chunk =", json.dumps(summary, ensure_ascii=False))
        return chunk["time_list"][-1]


class AstribotObservationSource:
    """Read three camera views, arm joints, and gripper state on host A."""

    def __init__(self, wait_timeout: float = 10.0):
        import rospy
        from core.astribot_api.astribot_client import Astribot

        self.rospy = rospy
        self.astribot = Astribot()
        self.wait_timeout = wait_timeout
        self.image_lock = threading.Lock()
        self.images: dict[str, np.ndarray] = {}
        self.subscribers = []

        self._activate_and_subscribe()

    def _activate_and_subscribe(self) -> None:
        self.astribot.activate_camera()
        start_time = time.time()
        while time.time() - start_time < self.wait_timeout:
            cameras_info = self.astribot.get_cameras_info()
            ready = all(
                cameras_info.get(name, {}).get("activate", False)
                for name in CAMERA_NAMES
            )
            if ready:
                break
            time.sleep(0.5)

        cameras_info = self.astribot.get_cameras_info()
        for camera_name in CAMERA_NAMES:
            if not cameras_info.get(camera_name, {}).get("activate", False):
                raise RuntimeError(f"camera {camera_name} is not activated")

            sub = self.astribot.register_image_callback(
                camera_name,
                "color",
                self._image_callback,
                True,
            )
            if sub is not None:
                self.subscribers.append(sub)

    def _image_callback(self, topic_name, msg, width, height, array: np.ndarray):
        if msg.format.lower() != "jpeg":
            return

        camera_name = self.astribot.get_camera_name_from_topic_name(topic_name)
        if camera_name not in CAMERA_NAMES:
            return

        with self.image_lock:
            self.images[camera_name] = array.copy()

    def wait_until_ready(self, timeout_s: float = 5.0) -> None:
        start_time = time.time()
        while time.time() - start_time < timeout_s:
            with self.image_lock:
                ready = all(name in self.images for name in CAMERA_NAMES)
            if ready:
                return
            time.sleep(0.1)
        raise TimeoutError(f"did not receive all required images within {timeout_s} seconds")

    def get_observation(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        with self.image_lock:
            missing = [name for name in CAMERA_NAMES if name not in self.images]
            if missing:
                raise RuntimeError(f"missing live images: {missing}")
            images = {name: self.images[name].copy() for name in CAMERA_NAMES}

        names = [
            self.astribot.arm_left_name,
            self.astribot.arm_right_name,
            self.astribot.effector_left_name,
            self.astribot.effector_right_name,
        ]
        joint_positions = self.astribot.get_current_joints_position(names=names)
        state = {
            "left_arm": [float(v) for v in joint_positions[0]],
            "right_arm": [float(v) for v in joint_positions[1]],
            # Effector range is 0-100. 100 means fully closed, 0 means fully open.
            "left_gripper": float(joint_positions[2][0]),
            "right_gripper": float(joint_positions[3][0]),
        }
        return images, state


class AstribotExecutor:
    """Execute chunked arm and optional gripper actions on host A."""

    def __init__(
        self,
        chunk_dt: float,
        use_wbc: bool = True,
        left_arm_dim: int = 7,
        right_arm_dim: int = 7,
    ):
        from core.astribot_api.astribot_client import Astribot

        self.astribot = Astribot()
        self.chunk_dt = chunk_dt
        self.use_wbc = use_wbc
        self.left_arm_dim = left_arm_dim
        self.right_arm_dim = right_arm_dim

    def execute_action(self, action: dict[str, Any]) -> float:
        chunk = _parse_chunked_action(
            action,
            default_dt=self.chunk_dt,
            left_arm_dim=self.left_arm_dim,
            right_arm_dim=self.right_arm_dim,
        )

        names = [self.astribot.arm_left_name, self.astribot.arm_right_name]
        include_left_gripper = chunk["left_gripper"] is not None
        include_right_gripper = chunk["right_gripper"] is not None
        if include_left_gripper:
            names.append(self.astribot.effector_left_name)
        if include_right_gripper:
            names.append(self.astribot.effector_right_name)

        waypoints = []
        for index in range(chunk["num_waypoints"]):
            waypoint = [chunk["left_arm"][index], chunk["right_arm"][index]]
            if include_left_gripper:
                waypoint.append(chunk["left_gripper"][index])
            if include_right_gripper:
                waypoint.append(chunk["right_gripper"][index])
            waypoints.append(waypoint)

        self.astribot.move_joints_waypoints(
            names,
            waypoints,
            chunk["time_list"],
            use_wbc=self.use_wbc,
        )
        return chunk["time_list"][-1]


def create_source(source_name: str, wait_timeout: float):
    if source_name == "mock":
        return MockObservationSource()
    if source_name == "astribot":
        return AstribotObservationSource(wait_timeout=wait_timeout)
    raise ValueError(f"unsupported source: {source_name}")


def create_executor(
    executor_name: str,
    chunk_dt: float,
    use_wbc: bool,
    left_arm_dim: int = 7,
    right_arm_dim: int = 7,
):
    if executor_name == "mock":
        return MockExecutor(chunk_dt=chunk_dt, left_arm_dim=left_arm_dim, right_arm_dim=right_arm_dim)
    if executor_name == "astribot":
        return AstribotExecutor(
            chunk_dt=chunk_dt,
            use_wbc=use_wbc,
            left_arm_dim=left_arm_dim,
            right_arm_dim=right_arm_dim,
        )
    raise ValueError(f"unsupported executor: {executor_name}")


async def run_loop(args) -> None:
    if args.control_hz <= 0:
        raise ValueError(f"--control-hz must be > 0, got {args.control_hz}")

    source = create_source(args.source, wait_timeout=args.ready_timeout)
    executor = create_executor(
        args.executor,
        chunk_dt=1.0 / args.control_hz,
        use_wbc=args.use_wbc,
        left_arm_dim=args.left_arm_dim,
        right_arm_dim=args.right_arm_dim,
    )

    print("waiting for local observation source...")
    source.wait_until_ready(timeout_s=args.ready_timeout)
    print("observation source is ready")

    async with websockets.connect(
        args.uri,
        max_size=args.max_size_mb * 1024 * 1024,
        ping_interval=20,
        ping_timeout=60,
    ) as websocket:
        print(f"connected to policy server: {args.uri}")
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

            await websocket.send(json.dumps(request, ensure_ascii=False))
            raw_response = await websocket.recv()
            response = json.loads(raw_response)

            if not response.get("ok", False):
                raise RuntimeError(
                    f"policy server returned error at step {step}: {response.get('error')}"
                )

            action_chunk = response["action"]
            chunk_duration_s = executor.execute_action(action_chunk)

            elapsed = time.time() - loop_start
            print(
                f"[step {step}] round_trip={elapsed:.3f}s "
                f"request_id={request['request_id']}"
            )
            step += 1

            min_cycle_s = max((1.0 / args.control_hz), chunk_duration_s)
            sleep_s = max(0.0, min_cycle_s - elapsed)
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Host A policy client")
    parser.add_argument("--uri", required=True, help="e.g. ws://192.168.1.20:7133")
    parser.add_argument("--instruction", required=True, help="task instruction sent to host B")
    parser.add_argument("--source", choices=["astribot", "mock"], default="astribot")
    parser.add_argument("--executor", choices=["astribot", "mock"], default="astribot")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--control-hz",
        type=float,
        default=10.0,
        help="default per-waypoint frequency used when the policy omits time_list or dt",
    )
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--resize-scale", type=float, default=1.0)
    parser.add_argument("--ready-timeout", type=float, default=10.0)
    parser.add_argument(
        "--use-wbc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to use WBC when executor=astribot",
    )
    parser.add_argument("--max-size-mb", type=int, default=32)
    parser.add_argument(
        "--left-arm-dim",
        type=int,
        default=6,
        help="左臂关节数（不含 gripper），须与推理服务器 --left_arm_dim 一致",
    )
    parser.add_argument(
        "--right-arm-dim",
        type=int,
        default=6,
        help="右臂关节数（不含 gripper），须与推理服务器 --right_arm_dim 一致",
    )
    args = parser.parse_args()

    await run_loop(args)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\npolicy client stopped")
