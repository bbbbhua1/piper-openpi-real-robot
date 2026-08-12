#!/usr/bin/env python3
"""Strict, ROS-free validation for task-specific Piper start poses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


PIPER_WIRE_ORDER = "[L1,L2,L3,L4,L5,L6,Lg,R1,R2,R3,R4,R5,R6,Rg]"
PIPER_VECTOR_SIZE = 14
ARM_VECTOR_SIZE = 7
VERIFIED_MAX_JOINT_SPEED = 0.3


def _finite_vector(value: Any, name: str, size: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must contain exactly {size} finite numeric values")
    result = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"{name} must contain exactly {size} finite numeric values")
        try:
            number = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{name} must contain exactly {size} finite numeric values"
            ) from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain exactly {size} finite numeric values")
        result.append(number)
    return tuple(result)


def _positive_number(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a finite positive number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{key} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{key} must be a finite positive number")
    return number


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


@dataclass(frozen=True)
class StartPoseConfig:
    name: str
    source_dataset: str
    source_episode: int
    piper_wire_order: str
    left_arm: tuple[float, ...]
    right_arm: tuple[float, ...]
    joint_lower: tuple[float, ...]
    joint_upper: tuple[float, ...]
    move_hz: float
    max_joint_speed: float
    min_duration_s: float
    max_duration_s: float
    feedback_timeout_s: float
    arm_tolerance: float
    gripper_tolerance: float


def load_start_pose_config(path: str | Path) -> StartPoseConfig:
    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load start-pose config {config_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("start-pose config must contain a JSON object")

    name = _required_string(payload, "name")
    source_dataset = _required_string(payload, "source_dataset")
    source_episode_raw = payload.get("source_episode")
    if isinstance(source_episode_raw, bool) or not isinstance(source_episode_raw, int):
        raise ValueError("source_episode must be a non-negative integer")
    if source_episode_raw < 0:
        raise ValueError("source_episode must be a non-negative integer")

    piper_wire_order = _required_string(payload, "piper_wire_order")
    if piper_wire_order != PIPER_WIRE_ORDER:
        raise ValueError(f"piper_wire_order must be exactly {PIPER_WIRE_ORDER}")

    left_arm = _finite_vector(payload.get("left_arm"), "left_arm", ARM_VECTOR_SIZE)
    right_arm = _finite_vector(payload.get("right_arm"), "right_arm", ARM_VECTOR_SIZE)
    lower = _finite_vector(payload.get("joint_lower"), "joint_lower", PIPER_VECTOR_SIZE)
    upper = _finite_vector(payload.get("joint_upper"), "joint_upper", PIPER_VECTOR_SIZE)
    if any(low >= high for low, high in zip(lower, upper)):
        raise ValueError("joint_lower must be strictly less than joint_upper")

    target = left_arm + right_arm
    for index, (value, low, high) in enumerate(zip(target, lower, upper)):
        if value < low or value > high:
            raise ValueError(
                "start-pose target is outside verified bounds at "
                f"piper_index={index}: value={value} not in [{low}, {high}]"
            )

    move_hz = _positive_number(payload, "move_hz")
    max_joint_speed = _positive_number(payload, "max_joint_speed")
    if max_joint_speed >= VERIFIED_MAX_JOINT_SPEED:
        raise ValueError(
            "max_joint_speed must stay below the verified 0.3 rad/s hardware speed limit"
        )
    min_duration_s = _positive_number(payload, "min_duration_s")
    max_duration_s = _positive_number(payload, "max_duration_s")
    if max_duration_s < min_duration_s:
        raise ValueError("max_duration_s must be greater than or equal to min_duration_s")

    return StartPoseConfig(
        name=name,
        source_dataset=source_dataset,
        source_episode=source_episode_raw,
        piper_wire_order=piper_wire_order,
        left_arm=left_arm,
        right_arm=right_arm,
        joint_lower=lower,
        joint_upper=upper,
        move_hz=move_hz,
        max_joint_speed=max_joint_speed,
        min_duration_s=min_duration_s,
        max_duration_s=max_duration_s,
        feedback_timeout_s=_positive_number(payload, "feedback_timeout_s"),
        arm_tolerance=_positive_number(payload, "arm_tolerance"),
        gripper_tolerance=_positive_number(payload, "gripper_tolerance"),
    )


def pose_errors(
    state: Mapping[str, Any], config: StartPoseConfig
) -> tuple[float, float]:
    left, right = _state_vectors(state)
    arm_errors = [abs(left[index] - config.left_arm[index]) for index in range(6)]
    arm_errors.extend(abs(right[index] - config.right_arm[index]) for index in range(6))
    gripper_errors = (
        abs(left[6] - config.left_arm[6]),
        abs(right[6] - config.right_arm[6]),
    )
    return max(arm_errors), max(gripper_errors)


def pose_is_reached(state: Mapping[str, Any], config: StartPoseConfig) -> bool:
    arm_error, gripper_error = pose_errors(state, config)
    return (
        arm_error <= config.arm_tolerance
        and gripper_error <= config.gripper_tolerance
    )


def _state_vectors(state: Mapping[str, Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not isinstance(state, Mapping):
        raise ValueError("state must contain left_arm and right_arm")
    left = _finite_vector(state.get("left_arm"), "state.left_arm", ARM_VECTOR_SIZE)
    right = _finite_vector(state.get("right_arm"), "state.right_arm", ARM_VECTOR_SIZE)
    return left, right


def validate_state_within_bounds(
    state: Mapping[str, Any], config: StartPoseConfig
) -> dict[str, list[float]]:
    """Validate a measured start state and clamp only tiny gripper zero drift.

    Arm readings remain strictly checked against their verified physical range.
    Piper gripper encoders can report a small signed value at their mechanical
    zero, so an out-of-range gripper value no larger than the configured
    ``gripper_tolerance`` is accepted and clamped to the verified endpoint
    before trajectory interpolation begins.  No command is widened beyond the
    verified physical ranges.
    """
    left, right = _state_vectors(state)
    sanitized = list(left + right)
    for index, (value, low, high) in enumerate(
        zip(left + right, config.joint_lower, config.joint_upper)
    ):
        is_gripper = index in (6, 13)
        boundary_tolerance = config.gripper_tolerance if is_gripper else 0.0
        if value < low - boundary_tolerance or value > high + boundary_tolerance:
            raise ValueError(
                "current state is outside verified bounds at "
                f"piper_index={index}: value={value} not in [{low}, {high}]"
            )
        sanitized[index] = min(max(value, low), high)
    return {
        "left_arm": sanitized[:ARM_VECTOR_SIZE],
        "right_arm": sanitized[ARM_VECTOR_SIZE:],
    }
