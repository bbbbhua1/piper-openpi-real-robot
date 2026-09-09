#!/usr/bin/env python3
"""CogWAM policy server for the Piper JSON WebSocket client.

This adapter is intentionally separate from the OpenPI/Pi05 and FastWAM
servers.  It loads CogWAM's RobotWin WAM policy with the real-robot 14-D
joint configuration, keeps Wan memory on the four-action grid, and optionally
uses VLM Q1/Q2 subtask planning and completion checks.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)
ACTION_HORIZON = 32
MEMORY_STRIDE = 4
WAN_MEMORY_FRAME_COUNT = 3
VLM_MEMORY_FRAMES = 5
VLM_MEMORY_FRAME_STRIDE = 4
VLM_COMPLETED_MEMORY_STRIDE = 16
VLM_CURRENT_MEMORY_STRIDE = 4
PIPER_LOWER = np.asarray(
    [-2.617993878, 0.0, -2.967059728, -1.745329252, -1.221730476, -3.141592654, 0.0,
     -2.617993878, 0.0, -2.967059728, -1.745329252, -1.221730476, -3.141592654, 0.0],
    dtype=np.float32,
)
PIPER_UPPER = np.asarray(
    [2.617993878, 3.141592654, 0.0, 1.745329252, 1.221730476, 3.141592654, 0.08,
     2.617993878, 3.141592654, 0.0, 1.745329252, 1.221730476, 3.141592654, 0.08],
    dtype=np.float32,
)
PIPER_MAX_DELTA = np.asarray(
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.005,
     0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.005],
    dtype=np.float32,
)


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for CogWAM deployment") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"CogWAM config must be a YAML object: {path}")
    return dict(value)


def build_policy_response(request_id: str, step: int, action: dict[str, Any], error: str | None = None) -> dict[str, Any]:
    response = {"ok": error is None, "request_id": request_id, "step": int(step), "action": action}
    if error is not None:
        response["error"] = str(error)
    return response


def _decode_image(payload: Any, input_color: str) -> np.ndarray:
    if isinstance(payload, Mapping):
        if "image" in payload and "data" not in payload:
            return _decode_image(payload["image"], input_color)
        if "array" in payload:
            image = np.asarray(payload["array"])
        elif "data" in payload:
            raw = payload["data"]
            if isinstance(raw, str):
                raw = base64.b64decode(raw.split(",", 1)[-1])
            elif isinstance(raw, list):
                raw = bytes(raw)
            from PIL import Image
            image = np.asarray(Image.open(io.BytesIO(bytes(raw))).convert("RGB"))
            input_color = "rgb"
        else:
            raise ValueError("image object requires data or array")
    elif isinstance(payload, str):
        from PIL import Image
        image = np.asarray(Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB"))
        input_color = "rgb"
    else:
        image = np.asarray(payload)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"expected HWC image, got {image.shape}")
    image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if input_color == "bgr":
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image)


def _piper_state(raw: Mapping[str, Any]) -> np.ndarray:
    left = raw.get("left_arm")
    right = raw.get("right_arm")
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        raise ValueError("state.left_arm and state.right_arm are required")
    left = [float(x) for x in left]
    right = [float(x) for x in right]
    if len(left) != 7 or len(right) != 7:
        raise ValueError("CogWAM requires 7 values per Piper arm")
    return np.asarray(left + right, dtype=np.float32)


def request_to_robotwin(request: Mapping[str, Any], *, input_color: str) -> tuple[dict[str, Any], np.ndarray]:
    images = request.get("images")
    if not isinstance(images, Mapping):
        raise ValueError("request.images must be an object")
    aliases = {
        "head": ("head", "image", "cam_high", "base"),
        "left_wrist": ("left_wrist", "wrist_image_left", "cam_left_wrist"),
        "right_wrist": ("right_wrist", "wrist_image_right", "cam_right_wrist"),
    }
    decoded: dict[str, np.ndarray] = {}
    for target, names in aliases.items():
        payload = next((images[name] for name in names if name in images), None)
        if payload is None:
            raise ValueError(f"missing required image for {target}")
        decoded[target] = _decode_image(payload, input_color)
    state = _piper_state(request.get("state", {}))
    observation = {
        "observation": {
            "head_camera": {"rgb": decoded["head"]},
            "left_camera": {"rgb": decoded["left_wrist"]},
            "right_camera": {"rgb": decoded["right_wrist"]},
        },
        "joint_action": {"vector": state.copy()},
    }
    return observation, state


def _to_client_action(actions: np.ndarray, *, horizon: int, dt: float) -> dict[str, Any]:
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 14:
        raise ValueError(f"CogWAM must return [T,14] joint actions, got {values.shape}")
    values = values[:horizon]
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("CogWAM returned an empty or non-finite action chunk")
    return {
        "time_list": [float(dt * (i + 1)) for i in range(len(values))],
        "dt": float(dt),
        "left_arm": values[:, :7].tolist(),
        "right_arm": values[:, 7:].tolist(),
    }


def _validated_safety_limits(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the CogWAM-local Piper boundary and fail closed on bad vectors."""
    vectors = []
    for key in ("joint_lower", "joint_upper", "max_delta"):
        value = config.get(key)
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{key} is required for CogWAM physical inference")
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (14,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{key} must contain exactly 14 finite values")
        vectors.append(vector)
    lower, upper, max_delta = vectors
    if not np.all(lower < upper):
        raise ValueError("joint_lower must be strictly less than joint_upper")
    if not np.all(max_delta > 0):
        raise ValueError("max_delta must be strictly positive")
    return lower, upper, max_delta


def project_cogwam_actions(actions: np.ndarray, current_state: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    """Clamp absolute 14-D predictions and rate-limit each published waypoint."""
    lower, upper, max_delta = _validated_safety_limits(config)
    values = np.asarray(actions, dtype=np.float32)
    current = np.asarray(current_state, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 14 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError(f"CogWAM actions must be finite [T,14], got {values.shape}")
    if current.shape != (14,) or not np.all(np.isfinite(current)):
        raise ValueError("CogWAM current state must be finite [14]")
    bounded = np.clip(values, lower, upper)
    result = np.empty_like(bounded)
    previous = np.clip(current, lower, upper)
    for index, prediction in enumerate(bounded):
        result[index] = np.clip(prediction, previous - max_delta, previous + max_delta)
        previous = result[index]
    return result


class CogWAMPiperRuntime:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        if str(self.config.get("wan_memory_attention_mode")) != "static_kv":
            raise ValueError("CogWAM deployment requires wan_memory_attention_mode=static_kv")
        if not bool(self.config.get("wan_memory_enabled", True)):
            raise ValueError("CogWAM deployment requires Wan memory to be enabled")
        if str(self.config.get("wan_memory_strategy")) != "short_term_only":
            raise ValueError("CogWAM deployment requires wan_memory_strategy=short_term_only")
        if int(self.config.get("wan_memory_short_term_frame_count", WAN_MEMORY_FRAME_COUNT)) != WAN_MEMORY_FRAME_COUNT:
            raise ValueError("CogWAM deployment requires wan_memory_short_term_frame_count=3")
        expected_vlm_memory = {
            "vlm_memory_strategy": "subtask_cls_recent_images",
            "vlm_memory_frames": VLM_MEMORY_FRAMES,
            "vlm_memory_frame_stride": VLM_MEMORY_FRAME_STRIDE,
            "vlm_completed_subtask_memory_stride": VLM_COMPLETED_MEMORY_STRIDE,
            "vlm_current_subtask_memory_stride": VLM_CURRENT_MEMORY_STRIDE,
        }
        for key, expected in expected_vlm_memory.items():
            if self.config.get(key, expected) != expected:
                raise ValueError(f"CogWAM requires {key}={expected!r} to match training memory")
        # Keep the CogWAM/Wan memory timeline on its fixed four-action grid.
        # This is independent of the number of waypoints returned to Piper and
        # is not a Q1 planning trigger.
        self.memory_stride = MEMORY_STRIDE
        self.execution_horizon = int(self.config.get("execution_horizon", ACTION_HORIZON))
        if self.execution_horizon <= 0 or self.execution_horizon % MEMORY_STRIDE:
            raise ValueError("execution_horizon must be a positive multiple of 4")
        self.step_count = 0
        self.task_goal = str(self.config.get("task_label", ""))
        self.subtasks: list[dict[str, Any]] = []
        self.completed_subtasks: list[dict[str, Any]] = []
        self.current_subtask_pos = 0
        self.vlm_q2_check_interval = int(self.config.get("vlm_q2_check_interval", 4))
        if self.vlm_q2_check_interval <= 0:
            raise ValueError("vlm_q2_check_interval must be positive")
        self._vlm_history: list[tuple[int, dict[str, Any]]] = []
        self._vlm_last_q2_step = 0
        self._vlm_done_streak = 0
        self.last_q2_status: dict[str, Any] | None = None
        self.final_task_done = False
        self.stop_current_chunk = False
        self.vlm = None
        self._smoother = None
        self._speed_adapter = None
        self._load_motion_postprocessors()
        self.policy = self._load_policy()
        if bool(self.config.get("vlm_enabled", False)):
            self.vlm = self._load_q1_planner()

    def _load_motion_postprocessors(self) -> None:
        overlay = Path(__file__).resolve().parents[1] / "vendor" / "cogwam_overlay" / "flexpi_smoothing"
        if bool(self.config.get("temporal_smoother_enabled", False)) or bool(self.config.get("speed_adapter_enabled", False)):
            if not overlay.is_dir():
                raise FileNotFoundError(f"FlexPi smoothing overlay not found: {overlay}")
            if bool(self.config.get("speed_adapter_enabled", False)):
                from importlib.util import module_from_spec, spec_from_file_location
                spec = spec_from_file_location("cogwam_speed_adapter", overlay / "speed_adapter.py")
                module = module_from_spec(spec); assert spec.loader is not None; sys.modules[spec.name] = module; spec.loader.exec_module(module)
                self._speed_adapter = module.SpeedAdapter(module.SpeedAdapterConfig(
                    mode=str(self.config.get("speed_adapter_mode", "heuristic")),
                    factor_lo=float(self.config.get("speed_factor_lo", 0.75)),
                    factor_hi=float(self.config.get("speed_factor_hi", 1.25)),
                    heuristic_alpha=float(self.config.get("speed_heuristic_alpha", 1.0)),
                    heuristic_v_ref=float(self.config.get("speed_heuristic_v_ref", 0.05)),
                    smooth_window=int(self.config.get("speed_smooth_window", 5)),
                ))
            if bool(self.config.get("temporal_smoother_enabled", False)):
                from importlib.util import module_from_spec, spec_from_file_location
                spec = spec_from_file_location("cogwam_temporal_smoother", overlay / "temporal_smoother.py")
                module = module_from_spec(spec); assert spec.loader is not None; sys.modules[spec.name] = module; spec.loader.exec_module(module)
                self._smoother = module.TemporalSmoother(module.SmootherConfig(
                    dt_ref=float(self.config.get("temporal_smoother_dt_ref", 1 / 30)),
                    dt_min=float(self.config.get("temporal_smoother_dt_min", 1 / 60)),
                    dt_max=float(self.config.get("temporal_smoother_dt_max", 1 / 15)),
                    lambda_acc=float(self.config.get("temporal_smoother_lambda_acc", 10.0)),
                    lambda_time=float(self.config.get("temporal_smoother_lambda_time", 1.0)),
                    horizon=int(self.config.get("temporal_smoother_horizon", 32)),
                    stride=int(self.config.get("temporal_smoother_stride", 16)),
                    optim_dims=(0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12),
                ))

    def postprocess_actions(self, actions: np.ndarray, mode: str = "smooth") -> np.ndarray:
        values = np.asarray(actions, dtype=np.float32)
        if mode == "interp":
            return values
        if mode != "smooth":
            raise ValueError(f"unsupported motion_mode: {mode}")
        if self._speed_adapter is not None:
            factors = self._speed_adapter.factors(values)
        else:
            factors = None
        if self._smoother is not None:
            values = self._smoother.smooth(values, factors)
        return values

    def _load_policy(self) -> Any:
        root = Path(str(self.config["cogwam_root"])).expanduser().resolve()
        if not (root / "experiments/robotwin/fastwam_policy/deploy_policy.py").is_file():
            raise FileNotFoundError(f"CogWAM policy source not found under {root}")
        overlay = Path(__file__).resolve().parents[1] / "vendor" / "cogwam_overlay"
        if overlay.is_dir():
            sys.path.insert(0, str(overlay))
            qwen_overlay = overlay / "qwen-vl-progress-estimation" / "qwen-vl-finetune"
            if qwen_overlay.is_dir():
                sys.path.insert(1, str(qwen_overlay))
        sys.path.insert(1 if overlay.is_dir() else 0, str(root))
        from omegaconf import OmegaConf
        from experiments.robotwin.fastwam_policy.deploy_policy import WorldActionRobotWinPolicy

        cfg_path = Path(str(self.config.get("wam_config", root / "real_robot/configs/wam_joint.yaml"))).expanduser()
        checkpoint = Path(str(self.config["checkpoint_path"])).expanduser().resolve()
        stats = Path(str(self.config["dataset_stats_path"])).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"CogWAM WAM checkpoint not found: {checkpoint}")
        if not stats.is_file():
            raise FileNotFoundError(f"CogWAM dataset stats not found: {stats}")
        _validated_safety_limits(self.config)
        # The WAM YAML is shared with training and contains mandatory W&B
        # interpolations. Inference never initializes W&B, so provide inert
        # values solely to let OmegaConf resolve the shared model config.
        os.environ.setdefault("WANDB_ENTITY", "inference")
        os.environ.setdefault("WANDB_PROJECT", "cogwam-inference")
        os.environ.setdefault("WANDB_MODE", "disabled")
        cfg = OmegaConf.load(cfg_path)
        OmegaConf.resolve(cfg)
        model_dtype = self._torch_dtype(str(self.config.get("mixed_precision", "bf16")))
        return WorldActionRobotWinPolicy(
            model_cfg=cfg.model,
            processor_cfg=cfg.data.train.processor,
            checkpoint_path=str(checkpoint),
            dataset_stats_path=stats,
            device=str(self.config.get("device", "cuda:0")),
            model_dtype=model_dtype,
            action_horizon=int(self.config.get("model_action_horizon", ACTION_HORIZON)),
            # The upstream constructor uses this legacy name for its action
            # queue/memory cadence. Q1 planning is controlled by subtask
            # transitions below, not by this legacy constructor argument.
            replan_steps=self.memory_stride,
            num_inference_steps=int(self.config.get("num_inference_steps", 10)),
            sigma_shift=None,
            seed=int(self.config.get("seed", 42)),
            text_cfg_scale=float(self.config.get("text_cfg_scale", 1.0)),
            negative_prompt="",
            rand_device=str(self.config.get("rand_device", "cpu")),
            tiled=False,
            timing_enabled=True,
            num_video_frames=int(self.config.get("num_video_frames", 9)),
            robotwin_action_type="qpos",
            # CogWAM's qpos policy is a single merged 14-D vector.  Passing
            # arm dimensions here would make the RobotWin helper expect a
            # 16-D EE-style vector.
            left_arm_dim=None,
            right_arm_dim=None,
            wan_memory_enabled=True,
            wan_memory_attention_mode=str(self.config.get("wan_memory_attention_mode", "static_kv")),
            wan_memory_short_term_offset=int(self.config.get("wan_memory_short_term_offset", 4)),
            vlm_monitor=None,
        )

    @staticmethod
    def _torch_dtype(name: str) -> Any:
        import torch

        return {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                "fp16": torch.float16, "float16": torch.float16,
                "no": torch.float32, "fp32": torch.float32}[name.lower()]

    def _load_q1_planner(self) -> Any:
        monitor_path = (
            Path(__file__).resolve().parents[1]
            / "vendor"
            / "cogwam_overlay"
            / "experiments"
            / "robotwin"
            / "fastwam_policy"
            / "vlm_monitor.py"
        )
        if not monitor_path.is_file():
            raise FileNotFoundError(f"CogWAM VLM monitor not found: {monitor_path}")
        monitor_spec = importlib.util.spec_from_file_location(
            "cogwam_piper_vlm_monitor", monitor_path
        )
        if monitor_spec is None or monitor_spec.loader is None:
            raise ImportError(f"Could not load CogWAM VLM monitor: {monitor_path}")
        monitor_module = importlib.util.module_from_spec(monitor_spec)
        sys.modules[monitor_spec.name] = monitor_module
        monitor_spec.loader.exec_module(monitor_module)
        RobotWinVLMSubtaskMonitor = monitor_module.RobotWinVLMSubtaskMonitor

        for key in ("vlm_repo_path", "vlm_base_model", "vlm_checkpoint", "vlm_q1_adapter", "vlm_q2_adapter"):
            path = Path(str(self.config[key])).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"CogWAM VLM {key} not found: {path}")
        q1 = Path(str(self.config["vlm_q1_adapter"])).expanduser().resolve()
        if not (q1 / "adapter_config.json").is_file():
            raise FileNotFoundError(f"CogWAM Q1 adapter is incomplete: {q1}")
        q2 = Path(str(self.config["vlm_q2_adapter"])).expanduser().resolve()
        if not (q2 / "adapter_config.json").is_file():
            raise FileNotFoundError(f"CogWAM Q2 adapter is incomplete: {q2}")

        return RobotWinVLMSubtaskMonitor(
            repo_path=str(self.config["vlm_repo_path"]),
            base_model=str(self.config["vlm_base_model"]),
            checkpoint=str(self.config["vlm_checkpoint"]),
            q1_adapter_path=str(self.config["vlm_q1_adapter"]),
            load_q2_adapter=True,
            device=str(self.config.get("vlm_device", self.config.get("device", "cuda:0"))),
            require_dual_lora=True,
            q2_state_enabled=bool(self.config.get("vlm_q2_state_enabled", True)),
            failure_query_enabled=bool(self.config.get("vlm_failure_query_enabled", True)),
            done_threshold=float(self.config.get("vlm_done_threshold", 0.5)),
            expected_memory_frames=VLM_MEMORY_FRAMES,
            memory_strategy="subtask_cls_recent_images",
            memory_frame_stride=VLM_MEMORY_FRAME_STRIDE,
            completed_subtask_memory_stride=VLM_COMPLETED_MEMORY_STRIDE,
            current_subtask_memory_stride=VLM_CURRENT_MEMORY_STRIDE,
            plan_enabled=True,
        )

    def infer(
        self,
        observation: dict[str, Any],
        request_step: int,
    ) -> np.ndarray:
        policy_step = int(request_step)
        if policy_step < 0 or policy_step % MEMORY_STRIDE:
            raise ValueError("request step must be a non-negative multiple of 4")
        self.policy.step_count = policy_step
        self._record_vlm_observation(policy_step, observation)
        instruction = self.task_goal
        if self.subtasks:
            current = self.subtasks[min(self.current_subtask_pos, len(self.subtasks) - 1)]
            instruction = f"Overall task: {self.task_goal}\nCurrent subtask: {current['subtask_goal']}"
        return self.policy._infer_action_chunk(observation, instruction)

    def observe(self, observation: dict[str, Any], policy_step: int) -> None:
        """Record an intermediate four-action-grid observation without inference."""
        policy_step = int(policy_step)
        if policy_step < 0 or policy_step % MEMORY_STRIDE:
            raise ValueError("observation step must be a non-negative multiple of 4")
        self.policy.step_count = policy_step
        self._record_vlm_observation(policy_step, observation)
        if getattr(self.policy, "wan_memory_enabled", False):
            self.policy._record_wan_memory_observation(observation)

    def check_q2(self, policy_step: int, observation_batch: Any) -> None:
        """Run Q2 and any Q1 transition from a server background task."""
        self._check_q2(int(policy_step), observation_batch)

    def consume_stop_current_chunk(self) -> bool:
        value = self.stop_current_chunk
        self.stop_current_chunk = False
        return value

    def _check_q2(self, policy_step: int, observation_batch: Any) -> None:
        if (
            self.vlm is None
            or not self.subtasks
            or policy_step <= 0
            or policy_step - self._vlm_last_q2_step < self.vlm_q2_check_interval
            or self.current_subtask_pos >= len(self.subtasks)
        ):
            return
        q2_started_at = time.monotonic()
        status = self.vlm.check_current_done(
            self.task_goal,
            self.completed_subtasks,
            self.subtasks[self.current_subtask_pos],
            observation_batch,
        )
        q2_duration_s = time.monotonic() - q2_started_at
        self.last_q2_status = status
        self._vlm_last_q2_step = policy_step
        LOG.info(
            "CogWAM Q2 check step=%d subtask=%d/%d done=%s done_prob=%s "
            "threshold=%s memory_cached_step=%s observation_steps=%s duration=%.3fs",
            policy_step,
            self.current_subtask_pos + 1,
            len(self.subtasks),
            bool(status.get("done", False)),
            status.get("done_prob", "-"),
            status.get("threshold", self.config.get("vlm_done_threshold", "-")),
            status.get("memory_cached_step", "-"),
            [step for step, _ in observation_batch.prompt],
            q2_duration_s,
        )
        if bool(status.get("done", False)):
            self._vlm_done_streak += 1
        else:
            self._vlm_done_streak = 0
        if self._vlm_done_streak < int(self.config.get("vlm_done_consecutive_count", 3)):
            return
        completed = dict(self.subtasks[self.current_subtask_pos])
        self.completed_subtasks.append(completed)
        self.current_subtask_pos += 1
        self._vlm_done_streak = 0
        self.stop_current_chunk = True
        if self.current_subtask_pos >= len(self.subtasks):
            self.final_task_done = True
            LOG.info("CogWAM Q2 marked final subtask done at step %d: %s", policy_step, completed["subtask_goal"])
            return
        if getattr(self.policy, "wan_memory_enabled", False):
            self.policy._record_wan_memory_subtask_start()
        future = self.vlm.plan_subtasks(
            self.task_goal,
            observation_batch,
            completed_subtasks=self.completed_subtasks,
            strict=True,
            allow_empty=self.current_subtask_pos >= len(self.subtasks) - 1,
        )
        if future:
            self.subtasks = [
                *self.completed_subtasks,
                *[
                    {**item, "subtask_index": len(self.completed_subtasks) + index}
                    for index, item in enumerate(future)
                ],
            ]
            self.current_subtask_pos = len(self.completed_subtasks)
        LOG.info("CogWAM Q2 marked subtask done at step %d: %s", policy_step, completed["subtask_goal"])

    def reset(self) -> None:
        """Start a fresh task while keeping the loaded model and server alive."""
        reset_policy = getattr(self.policy, "reset", None)
        if callable(reset_policy):
            reset_policy()
        self.step_count = 0
        self.task_goal = str(self.config.get("task_label", ""))
        self.subtasks = []
        self.completed_subtasks = []
        self.current_subtask_pos = 0
        self._vlm_history.clear()
        self._vlm_last_q2_step = 0
        self._vlm_done_streak = 0
        self.last_q2_status = None
        self.final_task_done = False
        self.stop_current_chunk = False

    def _record_vlm_observation(self, step: int, observation: dict[str, Any]) -> None:
        if self._vlm_history and self._vlm_history[-1][0] == step:
            self._vlm_history[-1] = (step, observation)
        else:
            self._vlm_history.append((step, observation))

    def _vlm_observation_batch(self):
        from experiments.robotwin.fastwam_policy.vlm_monitor import VLMObservationBatch

        return VLMObservationBatch(
            prompt=self._vlm_history[-(VLM_MEMORY_FRAMES + 1):],
            cache=list(self._vlm_history),
        )

    def q2_observation_batch(self):
        """Freeze the Q2 input at the four-action observation that scheduled it."""
        return self._vlm_observation_batch()

    def plan_q1_once(self, observation: dict[str, Any]) -> None:
        if self.vlm is None or self.subtasks:
            return
        from experiments.robotwin.fastwam_policy.vlm_monitor import VLMObservationBatch

        self._record_vlm_observation(0, observation)
        batch = VLMObservationBatch(prompt=[(0, observation)], cache=[(0, observation)], reset_memory=True)
        self.subtasks = self.vlm.plan_subtasks(self.task_goal, batch, completed_subtasks=(), strict=True)
        self.completed_subtasks = []
        self.current_subtask_pos = 0
        self._vlm_last_q2_step = 0
        self._vlm_done_streak = 0
        LOG.info("CogWAM Q1 plan (Q2 enabled): %s", [x["subtask_goal"] for x in self.subtasks])


class CogWAMServer:
    def __init__(self, runtime: CogWAMPiperRuntime, config: Mapping[str, Any]) -> None:
        self.runtime = runtime
        self.config = dict(config)
        self.final_done = False
        self._q2_task: asyncio.Task[Any] | None = None
        self._q2_task_step: int | None = None
        self._q2_skipped_observations = 0
        self._memory_task: asyncio.Task[Any] | None = None

    def _schedule_q2(self, step: int, websocket: Any = None, send_lock: asyncio.Lock | None = None, request_id: str = "") -> None:
        """Start Q2 in a worker; observation acknowledgements stay low latency."""
        if self._q2_task is not None and not self._q2_task.done():
            self._q2_skipped_observations += 1
            return

        if self._q2_skipped_observations:
            LOG.info(
                "CogWAM Q2 scheduling resumes at step %d after %d four-step observations "
                "were skipped while Q2 for step %s was running",
                step,
                self._q2_skipped_observations,
                self._q2_task_step,
            )
            self._q2_skipped_observations = 0
        self._q2_task_step = int(step)
        observation_batch = self.runtime.q2_observation_batch()
        LOG.info("CogWAM Q2 scheduled at step %d", step)

        async def run() -> None:
            await asyncio.to_thread(self.runtime.check_q2, int(step), observation_batch)
            if websocket is not None and (bool(getattr(self.runtime, "stop_current_chunk", False)) or bool(
                getattr(self.runtime, "final_task_done", False)
            )):
                response = build_policy_response(request_id, int(step), {})
                response.update({
                    "observation_only": True,
                    "stop_current_chunk": True,
                    "final_task_done": bool(getattr(self.runtime, "final_task_done", False)),
                    "q2_enabled": getattr(self.runtime, "vlm", None) is not None,
                    "q2_status": getattr(self.runtime, "last_q2_status", None),
                    "subtask_progress": self._subtask_progress(),
                    "wan_memory_attention_mode": self.runtime.config.get("wan_memory_attention_mode", "static_kv"),
                })
                if send_lock is None:
                    await websocket.send(json.dumps(response, ensure_ascii=False))
                else:
                    async with send_lock:
                        await websocket.send(json.dumps(response, ensure_ascii=False))

        self._q2_task = asyncio.create_task(run())

    def _schedule_memory_encode(self) -> None:
        if self._memory_task is not None and not self._memory_task.done():
            return
        policy = getattr(self.runtime, "policy", None)
        if not getattr(policy, "wan_memory_enabled", False):
            return

        async def encode() -> None:
            while getattr(policy, "_wan_memory_pending_observations", {}):
                await asyncio.to_thread(
                    policy._encode_pending_wan_memory_observations
                )

        self._memory_task = asyncio.create_task(encode())

    async def _wait_memory(self) -> None:
        task = self._memory_task
        if task is not None:
            await task
            self._memory_task = None

    async def _finish_q2_if_ready(self) -> None:
        task = self._q2_task
        if task is not None and task.done():
            await task
            self._q2_task = None

    def _subtask_progress(self) -> dict[str, Any]:
        subtasks = list(getattr(self.runtime, "subtasks", []) or [])
        completed = list(getattr(self.runtime, "completed_subtasks", []) or [])
        current_index = int(getattr(self.runtime, "current_subtask_pos", 0) or 0)
        current_goal = ""
        if subtasks and current_index < len(subtasks):
            current_goal = str(subtasks[current_index].get("subtask_goal", ""))
        return {
            "subtasks": [str(item.get("subtask_goal", "")) for item in subtasks],
            "completed_count": len(completed),
            "current_index": current_index,
            "current_goal": current_goal,
            "q2_status": getattr(self.runtime, "last_q2_status", None),
        }

    async def handle(
        self,
        message: str,
        websocket: Any = None,
        send_lock: asyncio.Lock | None = None,
    ) -> dict[str, Any]:
        request_id, step = "unknown", -1
        try:
            request = json.loads(message)
            if not isinstance(request, Mapping):
                raise ValueError("request must be a JSON object")
            request_id, step = str(request.get("request_id", request_id)), int(request.get("step", 0))
            if bool(request.get("reset", False)) or request.get("type") == "reset":
                self.runtime.reset()
                self.final_done = False
                response = build_policy_response(request_id, step, {})
                response.update({
                    "reset": True,
                    "final_task_done": False,
                    "q2_enabled": getattr(self.runtime, "vlm", None) is not None,
                    "q2_status": None,
                    "subtask_progress": self._subtask_progress(),
                    "wan_memory_attention_mode": self.runtime.config.get("wan_memory_attention_mode", "static_kv"),
                })
                return response
            observation, current_state = request_to_robotwin(
                request, input_color=str(self.config.get("input_color", "bgr"))
            )
            if bool(request.get("observation_only", False)):
                await asyncio.to_thread(self.runtime.observe, observation, step)
                self._schedule_memory_encode()
                self._schedule_q2(step, websocket, send_lock, request_id)
                runtime_done = bool(getattr(self.runtime, "final_task_done", False))
                progress = self._subtask_progress()
                q2_status = getattr(self.runtime, "last_q2_status", None)
                stop_current_chunk = bool(
                    getattr(self.runtime, "consume_stop_current_chunk", lambda: False)()
                )
                if runtime_done:
                    self.final_done = True
                    self.runtime.reset()
                response = build_policy_response(request_id, step, {})
                response.update({
                    "observation_only": True,
                    "stop_current_chunk": stop_current_chunk or runtime_done,
                    "final_task_done": self.final_done or runtime_done,
                    "q2_enabled": getattr(self.runtime, "vlm", None) is not None,
                    "q2_status": q2_status,
                    "subtask_progress": progress,
                    "wan_memory_attention_mode": self.runtime.config.get("wan_memory_attention_mode", "static_kv"),
                })
                return response
            # Q1 is only needed for the first request or after an already
            # detected subtask transition.  It is allowed to finish before
            # WAM inference, while Q2 itself remains asynchronous.
            # A prefetched inference may arrive before the matching
            # observation_only message has finished encoding. Register the
            # inference observation here as well so memory is self-contained.
            if getattr(self.runtime, "policy", None) is not None:
                await asyncio.to_thread(self.runtime.observe, observation, step)
                self._schedule_memory_encode()
            await self._finish_q2_if_ready()
            await self._wait_memory()
            if getattr(self.runtime, "stop_current_chunk", False):
                await asyncio.to_thread(self.runtime.plan_q1_once, observation)
            else:
                await asyncio.to_thread(self.runtime.plan_q1_once, observation)
            done_file = self.config.get("final_done_file")
            if bool(request.get("final_task_done", False)) or (
                done_file and Path(str(done_file)).expanduser().exists()
            ):
                self.final_done = True
            actions = await asyncio.to_thread(self.runtime.infer, observation, step)
            actions = self.runtime.postprocess_actions(actions, str(request.get("motion_mode", "smooth")))
            runtime_done = bool(getattr(self.runtime, "final_task_done", False))
            progress = self._subtask_progress()
            q2_status = getattr(self.runtime, "last_q2_status", None)
            stop_current_chunk = bool(
                getattr(self.runtime, "consume_stop_current_chunk", lambda: False)()
            )
            if runtime_done:
                self.final_done = True
                self.runtime.reset()
            actions = project_cogwam_actions(actions, current_state, self.config)
            action = _to_client_action(actions, horizon=int(self.config.get("execution_horizon", ACTION_HORIZON)), dt=float(self.config.get("action_dt", 1 / 30)))
            response = build_policy_response(request_id, step, action)
            response["final_task_done"] = self.final_done or runtime_done
            response["q2_enabled"] = getattr(self.runtime, "vlm", None) is not None
            response["q2_status"] = q2_status
            response["subtask_progress"] = progress
            response["stop_current_chunk"] = stop_current_chunk or runtime_done
            response["wan_memory_attention_mode"] = self.runtime.config.get("wan_memory_attention_mode", "static_kv")
            return response
        except Exception as exc:
            LOG.exception("CogWAM request failed")
            return build_policy_response(request_id, step, {}, error=str(exc))

    async def serve(self) -> None:
        host, port = str(self.config.get("host", "127.0.0.1")), int(self.config.get("port", 7085))
        max_size = int(self.config.get("max_message_size_mb", 64)) * 1024 * 1024
        try:
            import websockets
        except ImportError:
            from aiohttp import web

            async def handler(request: Any) -> Any:
                websocket = web.WebSocketResponse(max_msg_size=max_size)
                await websocket.prepare(request)
                async for message in websocket:
                    if message.type == web.WSMsgType.TEXT:
                        await websocket.send_str(json.dumps(await self.handle(message.data), ensure_ascii=False))
                return websocket

            app = web.Application()
            app.router.add_get("/", handler)
            runner = web.AppRunner(app)
            await runner.setup()
            await web.TCPSite(runner, host, port).start()
            LOG.info("CogWAM Piper policy server listening on ws://%s:%s (aiohttp)", host, port)
            await asyncio.Future()
        else:
            async def handler(websocket: Any, *_unused: Any) -> None:
                send_lock = asyncio.Lock()

                async def process(message: str) -> None:
                    response = await self.handle(message, websocket, send_lock)
                    async with send_lock:
                        await websocket.send(json.dumps(response, ensure_ascii=False))

                tasks: set[asyncio.Task[Any]] = set()
                observation_tasks: set[asyncio.Task[Any]] = set()
                async for message in websocket:
                    is_observation = False
                    try:
                        is_observation = bool(json.loads(message).get("observation_only", False))
                    except (TypeError, ValueError):
                        pass

                    async def ordered_process(payload: str, wait_for_observations: bool) -> None:
                        if wait_for_observations and observation_tasks:
                            await asyncio.gather(*tuple(observation_tasks), return_exceptions=False)
                        await process(payload)

                    task = asyncio.create_task(ordered_process(message, not is_observation))
                    tasks.add(task)
                    if is_observation:
                        observation_tasks.add(task)
                        task.add_done_callback(observation_tasks.discard)
                    task.add_done_callback(tasks.discard)
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            async with websockets.serve(handler, host, port, max_size=max_size):
                LOG.info("CogWAM Piper policy server listening on ws://%s:%s", host, port)
                await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(description="CogWAM Piper policy server (Q1/Q2)")
    parser.add_argument("--config", default="configs/piper_cogwam_jigsaw.yaml")
    parser.add_argument("--preflight-only", "--preflight", dest="preflight_only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    runtime = CogWAMPiperRuntime(config)
    if not args.preflight_only:
        asyncio.run(CogWAMServer(runtime, config).serve())


if __name__ == "__main__":
    main()
