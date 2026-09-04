#!/usr/bin/env python3
"""LingBot-VA to Piper adapters.

The official LingBot-VA server owns the VAE, text encoder, transformer and KV
cache.  This module keeps the robot-facing boundary small and testable: it
converts Piper observations to the official real-robot observation dictionary,
flattens the model's 2x16 action grid, and applies the physical limits before
the WebSocket layer can return an action.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import copy
from pathlib import Path
import sys
from typing import Any

import numpy as np


MODEL_ACTION_DIM = 14
MODEL_FRAME_CHUNK_SIZE = 2
MODEL_ACTION_PER_FRAME = 16
MODEL_ACTION_HORIZON = MODEL_FRAME_CHUNK_SIZE * MODEL_ACTION_PER_FRAME
CAMERA_NAMES = ("head", "left_wrist", "right_wrist")
OFFICIAL_CAMERA_KEYS = (
    "camera_observations.color_images.camera_head",
    "camera_observations.color_images.camera_wrist_left",
    "camera_observations.color_images.camera_wrist_right",
)
# Internal 30-channel LingBot-VA indices for the 14 real-robot values.
ACTION_CHANNELS = (14, 15, 16, 17, 18, 19, 28, 21, 22, 23, 24, 25, 26, 29)


def _finite_vector(value: Any, *, name: str, size: int) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain {size} finite numeric values") from exc
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain exactly {size} finite numeric values")
    return np.ascontiguousarray(vector, dtype=np.float32)


def _finite_actions(value: Any, *, name: str = "actions") -> np.ndarray:
    try:
        actions = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must have shape [T, 14]") from exc
    if actions.ndim != 2 or actions.shape[0] <= 0 or actions.shape[1] != MODEL_ACTION_DIM:
        raise ValueError(f"{name} must have shape [T, 14] with T > 0, got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(actions, dtype=np.float32)


@dataclass(frozen=True)
class PiperSafetyLimits:
    """Piper wire-order limits: [L1..L6, Lg, R1..R6, Rg]."""

    lower: np.ndarray
    upper: np.ndarray
    max_delta: np.ndarray

    def __post_init__(self) -> None:
        lower = _finite_vector(self.lower, name="limits.lower", size=MODEL_ACTION_DIM).copy()
        upper = _finite_vector(self.upper, name="limits.upper", size=MODEL_ACTION_DIM).copy()
        max_delta = _finite_vector(
            self.max_delta, name="limits.max_delta", size=MODEL_ACTION_DIM
        ).copy()
        if np.any(lower > upper):
            raise ValueError("limits.lower must be less than or equal to limits.upper")
        if np.any(max_delta <= 0):
            raise ValueError("limits.max_delta values must all be strictly positive")
        for values in (lower, upper, max_delta):
            values.setflags(write=False)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "max_delta", max_delta)


@dataclass(frozen=True)
class ActionProjectionSummary:
    projected_values: int
    max_projection: float


def project_and_limit_piper_actions(
    actions: Any,
    current_state: Any,
    limits: PiperSafetyLimits,
) -> tuple[np.ndarray, ActionProjectionSummary]:
    """Clamp a model trajectory to physical endpoints and per-step deltas."""

    if not isinstance(limits, PiperSafetyLimits):
        raise TypeError("limits must be a PiperSafetyLimits instance")
    trajectory = _finite_actions(actions)
    current = _finite_vector(current_state, name="current_state", size=MODEL_ACTION_DIM)
    projection = np.maximum(
        np.maximum(limits.lower - trajectory, 0.0),
        np.maximum(trajectory - limits.upper, 0.0),
    )
    bounded = np.clip(trajectory, limits.lower, limits.upper)

    accepted = np.empty_like(bounded)
    previous = current
    for index, prediction in enumerate(bounded):
        accepted[index] = np.clip(
            prediction,
            previous - limits.max_delta,
            previous + limits.max_delta,
        )
        previous = accepted[index]

    return np.ascontiguousarray(accepted), ActionProjectionSummary(
        projected_values=int(np.count_nonzero(projection)),
        max_projection=float(projection.max(initial=0.0)),
    )


def piper_state_to_vector(state: Mapping[str, Any]) -> np.ndarray:
    if not isinstance(state, Mapping):
        raise ValueError("state must contain left_arm and right_arm")
    left = _finite_vector(state.get("left_arm"), name="state.left_arm", size=7)
    right = _finite_vector(state.get("right_arm"), name="state.right_arm", size=7)
    return np.ascontiguousarray(np.concatenate((left, right)), dtype=np.float32)


def actions_to_piper_wire(actions: Any, *, dt: float) -> dict[str, Any]:
    trajectory = _finite_actions(actions)
    try:
        dt_value = float(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("dt must be a positive finite number") from exc
    if not np.isfinite(dt_value) or dt_value <= 0:
        raise ValueError("dt must be a positive finite number")
    return {
        "left_arm": trajectory[:, :7].tolist(),
        "right_arm": trajectory[:, 7:].tolist(),
        "time_list": [float(dt_value * (i + 1)) for i in range(len(trajectory))],
        "dt": dt_value,
    }


def flatten_lingbotva_actions(value: Any) -> np.ndarray:
    """Convert official output [14, 2, 16] to a [32, 14] trajectory."""

    try:
        actions = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("LingBot-VA action cannot be converted to a numeric array") from exc
    actions = np.squeeze(actions)
    if actions.ndim == 3 and actions.shape[0] == MODEL_ACTION_DIM:
        actions = np.transpose(actions, (1, 2, 0)).reshape(-1, MODEL_ACTION_DIM)
    elif actions.ndim == 2 and actions.shape[0] == MODEL_ACTION_DIM:
        actions = actions.T
    return _finite_actions(actions, name="LingBot-VA action")


def load_action_norm(path: str | Path) -> dict[str, list[float]]:
    import json

    norm_path = Path(path).expanduser().resolve()
    if not norm_path.is_file():
        raise FileNotFoundError(f"action_norm_path does not exist: {norm_path}")
    payload = json.loads(norm_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("action_norm.json must contain an object")
    q01 = _finite_vector(payload.get("q01"), name="action_norm.q01", size=MODEL_ACTION_DIM)
    q99 = _finite_vector(payload.get("q99"), name="action_norm.q99", size=MODEL_ACTION_DIM)
    if np.any(q01 >= q99):
        raise ValueError("action_norm.q01 must be less than q99 at every channel")
    action_order = payload.get("action_order")
    expected_order = ["left_arm", "left_gripper", "right_arm", "right_gripper"]
    if action_order is not None and list(action_order) != expected_order:
        raise ValueError(f"unsupported action_order: {action_order!r}")
    return {"q01": q01.tolist(), "q99": q99.tolist()}


def expand_action_norm(norm: Mapping[str, Any]) -> dict[str, list[float]]:
    """Map 14 raw quantiles into the official 30-channel model layout."""

    q01 = _finite_vector(norm.get("q01"), name="norm.q01", size=MODEL_ACTION_DIM)
    q99 = _finite_vector(norm.get("q99"), name="norm.q99", size=MODEL_ACTION_DIM)
    expanded_q01 = np.zeros(30, dtype=np.float32)
    expanded_q99 = np.ones(30, dtype=np.float32)
    for raw_index, internal_index in enumerate(ACTION_CHANNELS):
        expanded_q01[internal_index] = q01[raw_index]
        expanded_q99[internal_index] = q99[raw_index]
    return {"q01": expanded_q01.tolist(), "q99": expanded_q99.tolist()}


def _model_observation(
    image_frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    state: np.ndarray,
) -> dict[str, Any]:
    images = [
        dict(zip(OFFICIAL_CAMERA_KEYS, frame, strict=True)) for frame in image_frames
    ]
    # VA_Server.preprocess_action expects [C, F, H]. The state is either the
    # current 14D vector for the initial inference or the previous action grid.
    return {"obs": images, "state": state}


class LingBotVAPiperRuntime:
    """Lifecycle adapter around one official ``VA_Server`` instance."""

    def __init__(self, *, model: Any, config: Mapping[str, Any]) -> None:
        if model is None:
            raise ValueError("LingBot-VA runtime requires a model")
        self.model = model
        self.config = dict(config)
        self.initialized = False
        self.prompt = ""
        self._last_model_action: np.ndarray | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LingBotVAPiperRuntime":
        if not isinstance(config, Mapping):
            raise ValueError("LingBot-VA runtime config must be a mapping")
        config_copy = dict(config)
        official_root = Path(str(config_copy.get("lingbot_va_root", ""))).expanduser().resolve()
        if not (official_root / "wan_va").is_dir():
            raise FileNotFoundError(f"lingbot_va_root must contain wan_va/: {official_root}")
        model_path = Path(str(config_copy.get("model_path", ""))).expanduser().resolve()
        for required in ("transformer/config.json", "vae/config.json", "tokenizer", "text_encoder"):
            if not (model_path / required).exists():
                raise FileNotFoundError(f"model_path is missing {required}: {model_path}")
        action_norm_path = Path(
            str(config_copy.get("action_norm_path", model_path / "action_norm.json"))
        ).expanduser().resolve()
        norm_stat = expand_action_norm(load_action_norm(action_norm_path))

        if str(official_root) not in sys.path:
            sys.path.insert(0, str(official_root))
        try:
            import torch
            from wan_va.configs import VA_CONFIGS
            from wan_va.wan_va_server import VA_Server
        except ImportError as exc:
            raise RuntimeError(
                "LingBot-VA dependencies are unavailable; activate the official LingBot-VA environment"
            ) from exc

        device = str(config_copy.get("device", "cuda:0"))
        if not device.startswith("cuda"):
            raise ValueError("LingBot-VA deployment requires a CUDA device such as cuda:0")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")
        local_rank = int(device.rsplit(":", 1)[1]) if ":" in device else 0
        if device.startswith("cuda"):
            torch.cuda.set_device(local_rank)

        job_config = copy.deepcopy(VA_CONFIGS["realrobot"])
        job_config.wan22_pretrained_model_name_or_path = str(model_path)
        job_config.norm_stat = norm_stat
        job_config.local_rank = local_rank
        job_config.rank = 0
        job_config.world_size = 1
        job_config.save_root = str(config_copy.get("save_root", "/tmp/lingbotva-realrobot"))
        job_config.enable_offload = bool(config_copy.get("enable_offload", True))
        job_config.num_inference_steps = int(config_copy.get("num_inference_steps", 25))
        job_config.action_num_inference_steps = int(
            config_copy.get("action_num_inference_steps", 50)
        )
        job_config.guidance_scale = float(config_copy.get("guidance_scale", 5))
        job_config.action_guidance_scale = float(config_copy.get("action_guidance_scale", 1))

        # VA_Server.load_transformer passes attn_mode="torch" explicitly. No
        # mutation of the user-provided checkpoint config is required.
        model = VA_Server(job_config)
        return cls(model=model, config={**config_copy, "norm_stat": norm_stat})

    def reset(self, prompt: str) -> None:
        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError("instruction must be a non-empty string")
        self.model._reset(prompt=prompt)
        self.prompt = prompt
        self.initialized = False
        self._last_model_action = None

    def reset_connection(self) -> None:
        """Drop streaming state; the next request performs a fresh reset."""

        if self.initialized:
            clear_transformer = getattr(self.model.transformer, "clear_cache", None)
            if callable(clear_transformer):
                clear_transformer(self.model.cache_name)
            clear_vae = getattr(self.model.streaming_vae, "clear_cache", None)
            if callable(clear_vae):
                clear_vae()
            clear_half = getattr(self.model, "streaming_vae_half", None)
            clear_half = getattr(clear_half, "clear_cache", None)
            if callable(clear_half):
                clear_half()
        self.initialized = False
        self._last_model_action = None

    def expected_cache_frames(self) -> int:
        """Number of new key frames needed by the next cache update."""

        if not self.initialized:
            return 0
        return 1 if int(getattr(self.model, "frame_st_id", 0)) == 0 else MODEL_FRAME_CHUNK_SIZE

    def infer(
        self,
        *,
        head: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        state: Mapping[str, Any],
        history_images: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    ) -> np.ndarray:
        state_vector = piper_state_to_vector(state)
        current_frame = (head, left, right)
        if not self.initialized:
            observation = _model_observation([current_frame], state_vector[:, None, None])
            raw_actions, _ = self.model._infer(observation, frame_st_id=0)
            self.initialized = True
        else:
            if self._last_model_action is None:
                raise RuntimeError("LingBot-VA action history is unavailable")
            cache_frames = history_images or [current_frame]
            if len(cache_frames) > MODEL_FRAME_CHUNK_SIZE:
                raise ValueError(
                    f"LingBot-VA cache updates support at most {MODEL_FRAME_CHUNK_SIZE} key frames"
                )
            frame_st_id = int(getattr(self.model, "frame_st_id", 0))
            # The first cache update prepends the initial conditioning latent,
            # so official LingBot-VA expects the complete two-frame action.
            # Later updates pair the executed action frame(s) with key frames;
            # a one-frame truncated rollout therefore uses the first frame.
            previous_action = self._last_model_action
            if frame_st_id > 0:
                previous_action = previous_action[:, : len(cache_frames), :]
            observation = _model_observation(cache_frames, previous_action)
            # The vendored Wan streaming encoder requires at least three
            # temporal samples when its previous conv cache is present. Live
            # Piper rollouts provide one or two key frames per update, so
            # reset only the VAE conv cache before each independent chunk.
            # Transformer KV cache and the previous action conditioning remain
            # continuous across updates.
            clear_vae = getattr(self.model.streaming_vae, "clear_cache", None)
            if callable(clear_vae):
                clear_vae()
            self.model._compute_kv_cache(observation)
            raw_actions, _ = self.model._infer(
                observation, frame_st_id=int(self.model.frame_st_id)
            )
        raw_array = np.asarray(raw_actions, dtype=np.float32)
        self._last_model_action = raw_array.copy()
        actions = flatten_lingbotva_actions(raw_array)
        if actions.shape != (MODEL_ACTION_HORIZON, MODEL_ACTION_DIM):
            raise ValueError(
                f"LingBot-VA must return {(MODEL_ACTION_HORIZON, MODEL_ACTION_DIM)}, got {actions.shape}"
            )
        return actions
