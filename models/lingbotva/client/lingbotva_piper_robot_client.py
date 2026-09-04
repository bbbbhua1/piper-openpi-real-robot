#!/usr/bin/env python3
"""Piper ROS client for the LingBot-VA JSON WebSocket server."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
from typing import Any

import websockets


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import websocket_policy_client as common  # noqa: E402
from ws_policy_protocol import build_policy_request  # noqa: E402


DEFAULT_INSTRUCTION = "Complete the jigsaw puzzle on the table."
LINGBOTVA_TRANSPORT_IMAGE_SIZES = {
    "head": (320, 256),
    "left_wrist": (320, 256),
    "right_wrist": (320, 256),
}


class LingBotVARosExecutor(common.RosJointExecutor):
    """Publish waypoints while sampling key frames for the next VA cache."""

    def __init__(self, *args: Any, observation_source: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.observation_source = observation_source

    def execute_action_with_history(
        self, action: dict[str, Any], *, sample_every: int = 16
    ) -> tuple[float, list[dict[str, Any]]]:
        chunk = common.parse_chunked_action(
            action,
            default_dt=self.chunk_dt,
            left_arm_dim=self.left_arm_dim,
            right_arm_dim=self.right_arm_dim,
        )
        start_time = time.time()
        history: list[dict[str, Any]] = []
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
            if (index + 1) % sample_every == 0 or index == chunk["num_waypoints"] - 1:
                images, _ = self.observation_source.get_observation()
                history.append(images)
            if self.waypoint_sleep > 0:
                self._interruptible_sleep(self.waypoint_sleep)
        return max(chunk["time_list"][-1], time.time() - start_time), history


def truncate_action(
    action: dict[str, Any], execution_horizon: int, *, skip_initial_frame: bool = False
) -> dict[str, Any]:
    """Return the first N synchronized waypoints from a Piper action response."""

    if not isinstance(action, dict):
        raise ValueError("server action must be a JSON object")
    if execution_horizon <= 0 or execution_horizon > 32:
        raise ValueError("execution_horizon must be between 1 and 32")
    left = action.get("left_arm")
    right = action.get("right_arm")
    if not isinstance(left, list) or not isinstance(right, list) or not left or not right:
        raise ValueError("server action must contain non-empty left_arm and right_arm chunks")
    if len(left) != len(right):
        raise ValueError("left_arm and right_arm chunks must have equal length")
    start = 16 if skip_initial_frame else 0
    if start >= len(left):
        raise ValueError("server action does not contain a post-conditioning frame")
    horizon = min(execution_horizon, len(left) - start)
    result = dict(action)
    result["left_arm"] = left[start : start + horizon]
    result["right_arm"] = right[start : start + horizon]
    if action.get("left_gripper") is not None:
        result["left_gripper"] = action["left_gripper"][start : start + horizon]
    if action.get("right_gripper") is not None:
        result["right_gripper"] = action["right_gripper"][start : start + horizon]
    if action.get("time_list") is not None:
        time_list = action["time_list"]
        if not isinstance(time_list, list) or len(time_list) != len(left):
            raise ValueError("time_list must match the returned action chunk")
        if skip_initial_frame:
            dt = float(action.get("dt", 1 / 30))
            if not dt > 0:
                raise ValueError("action dt must be positive")
            # Rebuild the rebased schedule instead of subtracting large
            # floating-point timestamps, which can make equal-period points
            # appear out of order at the client validator.
            result["time_list"] = [dt * (index + 1) for index in range(horizon)]
        else:
            result["time_list"] = time_list[start : start + horizon]
    return result


async def run_loop(args: argparse.Namespace) -> None:
    if args.control_hz <= 0:
        raise ValueError("--control-hz must be > 0")
    if args.execution_horizon <= 0 or args.execution_horizon > 32:
        raise ValueError("--execution-horizon must be between 1 and 32")
    common.validate_initialization_args(args)
    start_pose_config = (
        common.load_start_pose_config(args.start_pose_config)
        if args.start_pose_config
        else None
    )
    source = common.create_source(args)
    if args.executor == "ros":
        executor = LingBotVARosExecutor(
            left_action_topic=args.left_action_topic,
            right_action_topic=args.right_action_topic,
            enable_topic=args.enable_topic,
            chunk_dt=1.0 / args.control_hz,
            left_arm_dim=args.left_arm_dim,
            right_arm_dim=args.right_arm_dim,
            enable_on_start=args.enable_on_start,
            waypoint_sleep=args.waypoint_sleep,
            observation_source=source,
        )
    else:
        executor = common.create_executor(args)
    transport_sizes = (
        LINGBOTVA_TRANSPORT_IMAGE_SIZES
        if args.transport_image_profile == "lingbotva"
        else None
    )
    pending_history_images: list[dict[str, Any]] = []

    try:
        print("waiting for local observation source...")
        source.wait_until_ready(timeout_s=args.ready_timeout)
        print("observation source is ready")
        if transport_sizes is not None:
            print("[transport] LingBot-VA JPEG resize: all cameras=320x256")

        if args.execute_actions and start_pose_config is not None:
            move_to_pose = getattr(executor, "move_to_pose", None)
            if move_to_pose is None:
                raise RuntimeError("selected executor cannot move to a configured start pose")
            current_state = common.validate_state_within_bounds(
                common.get_current_state(source), start_pose_config
            )
            move_to_pose(
                current_state,
                start_pose_config.left_arm,
                start_pose_config.right_arm,
                move_hz=start_pose_config.move_hz,
                min_duration=start_pose_config.min_duration_s,
                max_duration=start_pose_config.max_duration_s,
                joint_speed=start_pose_config.max_joint_speed,
                safety_check=lambda: common.raise_if_safety_tripped(source),
                label=start_pose_config.name,
            )
            common.wait_until_start_pose(source, start_pose_config)
        elif args.execute_actions and args.home_on_start:
            move_home = getattr(executor, "move_home", None)
            if move_home is not None:
                move_home(
                    common.get_current_state(source),
                    home_hz=args.home_hz,
                    min_duration=args.home_min_duration,
                    max_duration=args.home_max_duration,
                    joint_speed=args.home_joint_speed,
                    tolerance=args.home_tolerance,
                )

        async with websockets.connect(
            args.uri,
            max_size=args.max_size_mb * 1024 * 1024,
            ping_interval=None,
            ping_timeout=None,
        ) as websocket:
            print("connected to LingBot-VA server: {}".format(args.uri))
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
                    transport_image_sizes=transport_sizes,
                )
                request["reset"] = step == 0
                if step > 0:
                    request["history_images"] = pending_history_images
                await websocket.send(json.dumps(request, ensure_ascii=False))
                response = json.loads(await websocket.recv())
                if not response.get("ok", False):
                    raise RuntimeError(
                        "LingBot-VA server error: {}".format(response.get("error"))
                    )
                action = truncate_action(
                    response["action"],
                    args.execution_horizon,
                    skip_initial_frame=step == 0
                    and len(response["action"].get("left_arm", [])) >= 32,
                )
                print("action", json.dumps(action, ensure_ascii=False, indent=2))
                common.raise_if_safety_tripped(source)
                if args.execute_actions:
                    execute_with_history = getattr(executor, "execute_action_with_history", None)
                    if execute_with_history is not None:
                        chunk_duration_s, history = execute_with_history(action)
                    else:
                        chunk_duration_s = executor.execute_action(action)
                        history = []
                else:
                    print("[dry-run] action execution disabled; pass --execute-actions")
                    chunk_duration_s = 0.0
                    history = []
                if not history:
                    history_count = max(1, min(2, (len(action["left_arm"]) + 15) // 16))
                    latest_images, _ = source.get_observation()
                    history = [latest_images for _ in range(history_count)]
                request_history = []
                for history_images in history:
                    history_request = build_policy_request(
                        step=step,
                        instruction=args.instruction,
                        state=state,
                        images=history_images,
                        jpeg_quality=args.jpeg_quality,
                        resize_scale=args.resize_scale,
                        transport_image_sizes=transport_sizes,
                    )
                    request_history.append(history_request["images"])
                # The history belongs to the action just executed and is sent
                # with the next observation request.
                pending_history_images = request_history
                elapsed = time.time() - loop_start
                print(
                    "[step {}] round_trip={:.3f}s request_id={}".format(
                        step, elapsed, request["request_id"]
                    )
                )
                step += 1
                sleep_s = max(0.0, max(1.0 / args.control_hz, chunk_duration_s) - elapsed)
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
    except common.SafetyTrip as exc:
        print("[safety] client stopped: {}".format(exc))
    finally:
        try:
            if args.execute_actions and common.is_safety_tripped(source):
                print("[safety] skipping exit motion after arm_status fault")
            elif args.execute_actions:
                if args.hold_on_exit:
                    hold_position = getattr(executor, "hold_position", None)
                    if hold_position is not None:
                        try:
                            hold_position(
                                common.get_current_state(source), hold_hz=args.hold_hz
                            )
                        except Exception as exc:
                            print("[hold] skipped: {}".format(exc))
                if args.disable_on_exit:
                    disable_robot = getattr(executor, "disable_robot", None)
                    if disable_robot is not None:
                        disable_robot()
        finally:
            finalize_recording = getattr(source, "finalize_recording", None)
            if finalize_recording is not None:
                finalize_recording()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = common.build_arg_parser()
    parser.description = "Piper ROS client for LingBot-VA"
    for action in parser._actions:
        if action.dest == "instruction":
            action.required = False
            action.default = DEFAULT_INSTRUCTION
        elif action.dest == "control_hz":
            action.default = 30.0
        elif action.dest == "max_size_mb":
            action.default = 64
        elif action.dest == "transport_image_profile":
            action.choices = ("none", "lingbotva")
            action.default = "lingbotva"
            action.help = "resize all three cameras to 320x256 for LingBot-VA"
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=32,
        help="execute at most this many waypoints from each 32-point model chunk",
    )
    return parser


def main() -> None:
    asyncio.run(run_loop(build_arg_parser().parse_args()))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nLingBot-VA policy client stopped")
