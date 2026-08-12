#!/usr/bin/env python3
"""Piper JSON-WebSocket server for supported UVA-DiT Puzzle checkpoints.

The existing Piper client sends a JSON WebSocket request containing three JPEG
images and two 7-D arm vectors.  The supported Puzzle checkpoints use a
Pi0.7-style 32-D model head whose first 14 physical dimensions are laid out as
``[left_arm6, right_arm6, left_gripper, right_gripper]``.  This server is the
boundary adapter between those two contracts.

The module deliberately keeps its import surface dependency-light: pure layout
and configuration tests can run without CUDA, PyTorch, OpenCV, or websockets.
Those packages are imported only when the actual model server starts.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from typing import Any, Optional


PHYSICAL_ACTION_DIM = 14
MODEL_ACTION_DIM = 32
ACTION_PER_FRAME = 16
FUTURE_ACTION_STEPS = 32
DEFAULT_ACTION_DT = 1.0 / 10.0
DEFAULT_TASK = "拼图（圆形）"
ACTIVE_OUTPUT_DIMS = (6, 7, 8, 9, 10, 11, 13)
INACTIVE_OUTPUT_DIMS = (0, 1, 2, 3, 4, 5, 12)
ARM_JOINT_DIMS = tuple(range(12))
GRIPPER_OUTPUT_DIMS = (12, 13)
VISUAL_HISTORY_LABELS = {
    0: "image no-AR",
    2: "image AR2",
}

CAMERA_ALIASES = {
    "head": "cam_high",
    "camera_head": "cam_high",
    "top": "cam_high",
    "cam_high": "cam_high",
    "left_wrist": "cam_left_wrist",
    "camera_wrist_left": "cam_left_wrist",
    "camera_left": "cam_left_wrist",
    "cam_left_wrist": "cam_left_wrist",
    "right_wrist": "cam_right_wrist",
    "camera_wrist_right": "cam_right_wrist",
    "camera_right": "cam_right_wrist",
    "cam_right_wrist": "cam_right_wrist",
}
CANONICAL_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


class ContractError(ValueError):
    """Raised when a request or checkpoint does not match this deployment."""


def _is_vector_like(value: Any) -> bool:
    """Accept lists plus NumPy/Torch rows without importing either package."""
    return (
        not isinstance(value, (str, bytes, bytearray))
        and hasattr(value, "__len__")
        and hasattr(value, "__iter__")
    )


def _as_finite_vector(value: Any, length: int, field_name: str) -> list[float]:
    if not _is_vector_like(value):
        raise ContractError(f"{field_name} must be a {length}-value sequence")
    if len(value) != length:
        raise ContractError(
            f"{field_name} must contain exactly {length} values, got {len(value)}"
        )
    out: list[float] = []
    for index, item in enumerate(value):
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{field_name}[{index}] is not numeric: {item!r}") from exc
        if not math.isfinite(number):
            raise ContractError(f"{field_name}[{index}] must be finite, got {number!r}")
        out.append(number)
    return out


def piper_state_to_model(state: Mapping[str, Any]) -> list[float]:
    """Convert Piper [left6,Lg]/[right6,Rg] state to model arms-first order."""
    if not isinstance(state, Mapping):
        raise ContractError("state must be an object with left_arm and right_arm")
    left = _as_finite_vector(state.get("left_arm"), 7, "state.left_arm")
    right = _as_finite_vector(state.get("right_arm"), 7, "state.right_arm")
    return [*left[:6], *right[:6], left[6], right[6]]


def parse_piper_state_history(value: Any) -> Optional[list[list[float]]]:
    """Validate model-order measured Piper state history from a client request."""
    if value is None:
        return None
    if not _is_vector_like(value):
        raise ContractError("state_history must be a non-empty list of 14-D rows")
    if len(value) == 0:
        raise ContractError("state_history must not be empty")
    if len(value) > ACTION_PER_FRAME:
        raise ContractError(
            f"state_history may contain at most {ACTION_PER_FRAME} rows, got {len(value)}"
        )
    return [
        _as_finite_vector(row, PHYSICAL_ACTION_DIM, f"state_history[{index}]")
        for index, row in enumerate(value)
    ]


def model_row_to_piper(row: Sequence[Any]) -> tuple[list[float], list[float]]:
    """Convert one 14-D model-layout row to legacy left/right 7-D rows."""
    values = _as_finite_vector(row, PHYSICAL_ACTION_DIM, "model action")
    return [*values[:6], values[12]], [*values[6:12], values[13]]


def _rows_from_actions(actions: Any) -> list[list[float]]:
    if not _is_vector_like(actions):
        raise ContractError("model actions must be a non-empty [steps, 14] sequence")
    if len(actions) == 0:
        raise ContractError("model returned an empty action chunk")
    return [
        _as_finite_vector(row, PHYSICAL_ACTION_DIM, f"model actions[{index}]")
        for index, row in enumerate(actions)
    ]


def savgol_matrix(np: Any, length: int, window: int, poly: int) -> Any:
    """Zero-phase Savitzky-Golay smoother as an explicit ``[length, length]`` matrix.

    The whole model chunk is available at once, so the filter can be applied
    non-causally and costs no phase lag -- unlike an online EMA.  Interior
    samples use the centred least-squares polynomial; the first and last
    ``window // 2`` samples are read off the edge polynomial instead of being
    padded, which keeps the chunk endpoints anchored.
    """
    if window < 3 or window % 2 == 0:
        raise ContractError(f"smoothing window must be odd and >= 3, got {window}")
    if poly < 1 or poly >= window:
        raise ContractError(f"smoothing polyorder must be in [1, {window - 1}], got {poly}")
    if length < window:
        raise ContractError(
            f"cannot smooth a {length}-step chunk with a {window}-step window"
        )
    half = window // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    design = np.vander(offsets, poly + 1, increasing=True)
    pinv = np.linalg.pinv(design)
    matrix = np.zeros((length, length), dtype=np.float64)
    for index in range(length):
        centre = min(max(index, half), length - 1 - half)
        powers = np.power(float(index - centre), np.arange(poly + 1, dtype=np.float64))
        matrix[index, centre - half : centre + half + 1] = powers @ pinv
    return matrix


def finalize_model_actions(
    actions: Any,
    current_state: Sequence[Any],
    *,
    inactive_action_mode: str,
    max_joint_step: float,
) -> list[list[float]]:
    """Apply the server-side real-robot safety policy in model 14-D order."""
    if inactive_action_mode not in {"hold", "model"}:
        raise ContractError(
            f"unknown inactive_action_mode {inactive_action_mode!r}; expected hold or model"
        )
    if max_joint_step < 0:
        raise ContractError("max_joint_step must be >= 0")

    current = _as_finite_vector(current_state, PHYSICAL_ACTION_DIM, "current model state")
    final_rows = _rows_from_actions(actions)
    previous = list(current)
    for row in final_rows:
        if inactive_action_mode == "hold":
            for dim in INACTIVE_OUTPUT_DIMS:
                row[dim] = current[dim]
        if max_joint_step > 0:
            # B supervises only the right six joints plus right gripper.  The
            # gripper remains unclipped here because its physical scale is not
            # radians; it is still bounded by the checkpoint normalization.
            for dim in range(6, 12):
                row[dim] = max(
                    previous[dim] - max_joint_step,
                    min(previous[dim] + max_joint_step, row[dim]),
                )
        previous = list(row)
    return final_rows


def model_actions_to_piper_action(
    actions: Any,
    current_state: Sequence[Any],
    *,
    dt: float,
    inactive_action_mode: str,
    max_joint_step: float,
) -> tuple[dict[str, Any], list[list[float]]]:
    """Build the unchanged legacy Piper action object and final 14-D actions."""
    if not math.isfinite(dt) or dt <= 0:
        raise ContractError(f"action dt must be finite and > 0, got {dt!r}")
    final_rows = finalize_model_actions(
        actions,
        current_state,
        inactive_action_mode=inactive_action_mode,
        max_joint_step=max_joint_step,
    )
    left_rows: list[list[float]] = []
    right_rows: list[list[float]] = []
    for row in final_rows:
        left, right = model_row_to_piper(row)
        left_rows.append(left)
        right_rows.append(right)
    return (
        {
            "time_list": [float(dt * (index + 1)) for index in range(len(final_rows))],
            "dt": float(dt),
            "left_arm": left_rows,
            "right_arm": right_rows,
        },
        final_rows,
    )


def _find_checkpoint_config(checkpoint: str | Path) -> Path:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    for candidate in (checkpoint_path / "config.json", checkpoint_path.parent / "config.json"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"checkpoint config not found beside {checkpoint_path}; expected config.json"
    )


def _parse_dimension_list(value: Any, field_name: str) -> tuple[int, ...]:
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    elif isinstance(value, Sequence):
        pieces = list(value)
    else:
        raise ContractError(f"{field_name} must be a comma-separated dimension list")
    try:
        dims = tuple(int(piece) for piece in pieces)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{field_name} contains a non-integer dimension: {value!r}") from exc
    if len(set(dims)) != len(dims):
        raise ContractError(f"{field_name} contains duplicate dimensions: {dims}")
    return dims


def visual_history_label(value: Any) -> str:
    """Return the deployment label for a supported visual-history contract."""
    try:
        history_frames = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(
            "pi07_history_frames must be 0 (Puzzle C no-AR) or 2 (Puzzle B AR2), "
            f"got {value!r}"
        ) from exc
    try:
        return VISUAL_HISTORY_LABELS[history_frames]
    except KeyError as exc:
        raise ContractError(
            "pi07_history_frames must be 0 (Puzzle C no-AR) or 2 (Puzzle B AR2), "
            f"got {history_frames}"
        ) from exc


def validate_puzzle_checkpoint_config(checkpoint: str | Path) -> dict[str, Any]:
    """Validate the immutable training contract required by this adapter."""
    config_path = _find_checkpoint_config(checkpoint)
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ContractError(f"checkpoint config must be an object: {config_path}")

    expected = {
        "action_type": "joint_pos",
        "action_dim": MODEL_ACTION_DIM,
        "ur_physical_action_dim": PHYSICAL_ACTION_DIM,
        "pi07_context": True,
        "pi07_subtask_mode": "frame_aligned",
        "pi07_history_frame_skip": 4,
        "use_chunk_ar_train": True,
        "use_action_condition": True,
        "action_condition_mode": "causal_history",
        "action_condition_history_steps": ACTION_PER_FRAME,
        "ur_joint_condition_source": "puppet",
        "ur_joint_target_source": "puppet",
        "ur_joint_channel_layout": "arms_then_grippers",
        "fix_vae_temporal_action_alignment": True,
        "n_video_frames": 9,
        "action_per_frame": ACTION_PER_FRAME,
        "action_chunk_size": FUTURE_ACTION_STEPS,
        "frame_skip": 4,
    }
    problems = [
        f"{key}: expected {expected_value!r}, got {config.get(key)!r}"
        for key, expected_value in expected.items()
        if config.get(key) != expected_value
    ]
    active_dims = _parse_dimension_list(
        config.get("ur_action_loss_dims", ""), "ur_action_loss_dims"
    )
    if active_dims != ACTIVE_OUTPUT_DIMS:
        problems.append(
            "ur_action_loss_dims: expected Puzzle right-side mask "
            f"{ACTIVE_OUTPUT_DIMS}, got {active_dims}"
        )
    if config.get("use_state_tokens", False):
        problems.append("use_state_tokens must be false for the Piper Puzzle adapter")
    visual_history = ""
    try:
        visual_history = visual_history_label(config.get("pi07_history_frames"))
    except ContractError as exc:
        problems.append(str(exc))
    if problems:
        raise ContractError(
            "Puzzle checkpoint contract mismatch in "
            f"{config_path}:\n  - " + "\n  - ".join(problems)
        )
    config["_config_path"] = str(config_path)
    config["_piper_visual_history_label"] = visual_history
    return config


def check_runtime_artifacts(args: argparse.Namespace) -> dict[str, str]:
    """Fail before CUDA initialization when a required model artifact is missing."""
    if args.mock_policy:
        return {"mode": "mock"}
    config = validate_puzzle_checkpoint_config(args.checkpoint)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    required_files = {
        "checkpoint transformer": checkpoint / "transformer",
        "Qwen3.5 projection adapter": Path(args.qwen35_adapter).expanduser(),
        "normalization stats": Path(args.norm_stats).expanduser(),
        "UniDiT VAE": Path(args.pretrained).expanduser() / "vae",
        "UniDiT transformer": Path(args.pretrained).expanduser() / "transformer",
        "Qwen3.5": Path(args.qwen35).expanduser(),
    }
    missing = [f"{label}: {path}" for label, path in required_files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing inference artifacts:\n  - " + "\n  - ".join(missing))
    return {
        "config": str(config["_config_path"]),
        "checkpoint": str(checkpoint),
        "norm_stats": str(Path(args.norm_stats).expanduser().resolve()),
        "pretrained": str(Path(args.pretrained).expanduser().resolve()),
        "qwen35": str(Path(args.qwen35).expanduser().resolve()),
        "qwen35_adapter": str(Path(args.qwen35_adapter).expanduser().resolve()),
        "visual_history": str(config["_piper_visual_history_label"]),
    }


def _runtime_modules() -> tuple[Any, Any, Any]:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Piper UVA-DiT runtime needs opencv-python, numpy, and torch; "
            "activate the model environment before starting the server"
        ) from exc
    return cv2, np, torch


def _load_pi07_base() -> tuple[type[Any], Any]:
    here = Path(__file__).resolve()
    configured_root = os.environ.get("UVA_DIT_ROOT")
    candidates = [Path(configured_root).expanduser()] if configured_root else list(here.parents)
    repo_root = next(
        (
            candidate.resolve()
            for candidate in candidates
            if (
                candidate
                / "realmachine_deploy_v2"
                / "tianyi"
                / "infer_server_pi07_ar.py"
            ).is_file()
        ),
        None,
    )
    if repo_root is None:
        raise RuntimeError(
            "UVA-DiT source checkout not found; set UVA_DIT_ROOT to the UVA_dit "
            "repository containing realmachine_deploy_v2/tianyi/infer_server_pi07_ar.py"
        )

    tianyi_root = repo_root / "realmachine_deploy_v2" / "tianyi"
    for path in (str(tianyi_root), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        from infer_server_pi07_ar import Pi07InferenceEngine, logger  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised at GPU startup
        raise RuntimeError(
            "failed to import the UVA-DiT Pi07InferenceEngine from "
            f"{repo_root}; activate its CUDA environment and verify UVA_DIT_ROOT"
        ) from exc
    return Pi07InferenceEngine, logger


def _load_action_stats(stats_path: str, physical_dim: int) -> tuple[Any, Any]:
    _, _, torch = _runtime_modules()
    path = Path(stats_path).expanduser()
    try:
        stats = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Older torch versions do not expose weights_only.
        stats = torch.load(path, map_location="cpu")
    global_stats = stats.get("__global__", {}) if isinstance(stats, dict) else {}
    if "action_min" not in global_stats or "action_max" not in global_stats:
        raise ContractError(
            f"{path} must contain __global__/action_min and action_max for P→P inference"
        )
    q01 = torch.as_tensor(global_stats["action_min"]).reshape(-1, 1, 1).float()
    q99 = torch.as_tensor(global_stats["action_max"]).reshape(-1, 1, 1).float()
    if q01.numel() < physical_dim or q99.numel() < physical_dim:
        raise ContractError(
            f"{path} has fewer than {physical_dim} physical action statistics"
        )
    return q01, q99


class PiperPuzzleEngineMixin:
    """14D padded physical-action adapter layered onto Pi07InferenceEngine."""

    def __init__(self, args: argparse.Namespace, device: str) -> None:
        _, np, torch = _runtime_modules()
        self._np = np
        self._torch = torch
        self.infer_seed = int(args.seed)
        self.physical_action_dim = PHYSICAL_ACTION_DIM
        self.state_oob_mode = str(args.state_oob_mode)
        self.action_output_clip = bool(args.action_output_clip)
        self.action_smooth_window = int(getattr(args, "action_smooth_window", 0))
        self.action_smooth_poly = int(getattr(args, "action_smooth_poly", 2))
        self.action_smooth_grippers = bool(getattr(args, "action_smooth_grippers", False))
        self._smooth_matrix: Optional[Any] = None
        self._checkpoint_train_cfg = validate_puzzle_checkpoint_config(args.checkpoint)
        self.visual_history_label = str(
            self._checkpoint_train_cfg["_piper_visual_history_label"]
        )
        self.joint_condition_source = "puppet"
        self.joint_target_source = "puppet"
        self.joint_channel_layout = "arms_then_grippers"
        self.external_joint_channel_layout = "piper_left7_right7"
        self.training_action_type = "joint_pos"
        self.require_measured_history = True
        self._pending_history_count = 0
        super().__init__(args, device)

        if self.action_dim != MODEL_ACTION_DIM:
            raise ContractError(
                f"model action_dim must be {MODEL_ACTION_DIM}, got {self.action_dim}"
            )
        if self.use_state_tokens:
            raise ContractError("Piper Puzzle checkpoint was not trained with state tokens")
        if not self.use_action_condition:
            raise ContractError("Piper Puzzle checkpoint requires action conditioning")
        if self.action_condition_history_steps != ACTION_PER_FRAME:
            raise ContractError(
                "Piper Puzzle checkpoint requires causal action history16, got "
                f"{self.action_condition_history_steps}"
            )
        expected_history_frames = int(self._checkpoint_train_cfg["pi07_history_frames"])
        if self.history_frames != expected_history_frames:
            raise ContractError(
                "Piper Puzzle visual-history setup differs from checkpoint: "
                f"expected {expected_history_frames}, got {self.history_frames}"
            )

        self._condition_q01, self._condition_q99 = _load_action_stats(
            args.norm_stats, self.physical_action_dim
        )
        self._target_q01, self._target_q99 = _load_action_stats(
            args.norm_stats, self.physical_action_dim
        )
        self._q01 = self._condition_q01
        self._q99 = self._condition_q99
        self.action_mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self.device)
        self.action_mask[: self.physical_action_dim] = True
        self.out_action_dim = self.physical_action_dim
        self._latest_state_physical_model: Optional[Any] = None
        self._reset_rng()

    def _reset_rng(self) -> None:
        random.seed(self.infer_seed)
        self._np.random.seed(self.infer_seed)
        self._torch.manual_seed(self.infer_seed)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(self.infer_seed)

    def reset_runtime_state(self) -> None:
        super().reset_runtime_state()
        self._latest_state_physical_model = None
        self._pending_history_count = 0
        self._reset_rng()

    def _normalize_state_rows(self, rows: Any) -> Any:
        """Normalize physical 14D P→P values and append 18 zero head dims."""
        import torch.nn.functional as F  # type: ignore

        values = rows.float()
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.shape[-1] < self.physical_action_dim:
            raise ContractError(
                f"Piper state has {values.shape[-1]} dimensions; expected 14"
            )
        physical = values[:, : self.physical_action_dim]
        q01 = self._condition_q01.reshape(-1)[: self.physical_action_dim].to(
            device=physical.device, dtype=self._torch.float32
        )
        q99 = self._condition_q99.reshape(-1)[: self.physical_action_dim].to(
            device=physical.device, dtype=self._torch.float32
        )
        normalized = 2.0 * (physical - q01) / (q99 - q01 + 1e-6) - 1.0
        if self.state_oob_mode == "clip":
            normalized = normalized.clamp(-1.0, 1.0)
        elif self.state_oob_mode == "fail":
            if bool((normalized.abs() > 1.0).any()):
                bad = self._torch.nonzero((normalized.abs() > 1.0).any(dim=0)).reshape(-1)
                raise ContractError(
                    "Piper state lies outside checkpoint normalization range at "
                    f"dimensions {bad.tolist()}"
                )
        elif self.state_oob_mode != "extrapolate":
            raise ContractError(f"unknown state_oob_mode {self.state_oob_mode!r}")
        return F.pad(
            normalized,
            (0, self.action_dim - self.physical_action_dim),
            value=0.0,
        )

    def update_state(self, state: Sequence[float] | None) -> None:
        if state:
            values = self._torch.as_tensor(state, dtype=self._torch.float32).reshape(-1)
            if values.numel() < self.physical_action_dim:
                raise ContractError(
                    f"Piper state has {values.numel()} dimensions; expected 14"
                )
            self._latest_state_physical_model = (
                values[: self.physical_action_dim].detach().float().cpu()
            )
        super().update_state(list(state) if state is not None else None)

    def _build_cond_action(self) -> Any:
        condition = super()._build_cond_action()
        condition[:, self.physical_action_dim :] = 0.0
        return condition

    def _smooth_chunk(self, rows: Any) -> Any:
        """Zero-phase smooth the whole ``[steps, 14]`` chunk before it is sliced.

        Applied here rather than on the executed slice so that all four
        ``execute_steps`` slices served from one denoise stay consistent with
        each other and no step appears at a slice boundary.
        """
        if self.action_smooth_window <= 0:
            return rows
        np = self._np
        steps = int(rows.shape[0])
        if self._smooth_matrix is None or self._smooth_matrix.shape[0] != steps:
            self._smooth_matrix = savgol_matrix(
                np, steps, self.action_smooth_window, self.action_smooth_poly
            )
        dims = list(ARM_JOINT_DIMS)
        if self.action_smooth_grippers:
            dims += list(GRIPPER_OUTPUT_DIMS)
        smoothed = np.array(rows, dtype=np.float64, copy=True)
        smoothed[:, dims] = self._smooth_matrix @ smoothed[:, dims]
        return smoothed.astype(rows.dtype, copy=False)

    def _denormalize_actions(self, act_norm: Any) -> Any:
        physical = act_norm[: self.physical_action_dim].cpu().float()
        if self.action_output_clip:
            physical = physical.clamp(-1.0, 1.0)
        q01 = self._target_q01[: self.physical_action_dim]
        q99 = self._target_q99[: self.physical_action_dim]
        physical = (physical + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        return self._smooth_chunk(
            physical.reshape(self.physical_action_dim, -1).T.numpy()
        )

    def infer(
        self,
        obs: dict[str, Any],
        task: str,
        subtask: str,
        subgoal: Optional[dict[str, Any]],
        execute_steps: int,
    ) -> Any:
        """Infer from the measured history supplied by the current request."""
        actions = super().infer(obs, task, subtask, subgoal, execute_steps)
        # The base engine records predicted rows for its legacy exec_history
        # fallback.  This adapter requires measured history, so never retain
        # those predictions as a possible source for the next request.
        self._exec_action_hist.clear()
        self._pending_history_count = 0
        return actions

    def acknowledge_sent_actions(self, actions: Sequence[Sequence[float]]) -> None:
        # Kept for the existing WebSocket commit path.  Actual action
        # conditioning comes from request.state_history, never from actions.
        _rows_from_actions(actions)
        self._pending_history_count = 0

    def abort_pending_actions(self) -> None:
        self._pending_history_count = 0


def create_piper_puzzle_engine(args: argparse.Namespace, device: str) -> Any:
    """Create the dependency-heavy adapter only after runtime preflight passes."""
    pi07_base, logger = _load_pi07_base()

    class PiperPuzzleInferenceEngine(PiperPuzzleEngineMixin, pi07_base):
        pass

    engine = PiperPuzzleInferenceEngine(args, device)
    logger.info(
        "Piper Puzzle adapter ready: physical=14 model=32, P→P, "
        "measured_state_history16, %s, active_dims=%s",
        engine.visual_history_label,
        ACTIVE_OUTPUT_DIMS,
    )
    return engine


class MockPiperEngine:
    """Protocol-only engine used before the checkpoint/runtime is available."""

    def __init__(self) -> None:
        self.current_state = [0.0] * PHYSICAL_ACTION_DIM
        self.acknowledged: list[list[float]] = []
        self._last_timing: dict[str, Any] = {"mode": "mock"}

    def reset_runtime_state(self) -> None:
        self.acknowledged.clear()

    def update_state(self, state: Sequence[float] | None) -> None:
        if state is not None:
            self.current_state = _as_finite_vector(state, PHYSICAL_ACTION_DIM, "mock state")

    def infer(
        self,
        obs: dict[str, Any],
        task: str,
        subtask: str,
        subgoal: Optional[dict[str, Any]],
        execute_steps: int,
    ) -> list[list[float]]:
        del obs, task, subtask, subgoal
        self._last_timing = {"mode": "mock", "returned_steps": execute_steps}
        return [list(self.current_state) for _ in range(execute_steps)]

    def acknowledge_sent_actions(self, actions: Sequence[Sequence[float]]) -> None:
        self.acknowledged.extend(_rows_from_actions(actions))

    def abort_pending_actions(self) -> None:
        return None


def decode_legacy_image(payload: Any) -> Any:
    """Decode an old Piper `{encoding,data}` JPEG object into RGB HWC."""
    cv2, np, _ = _runtime_modules()
    if isinstance(payload, Mapping):
        encoding = str(payload.get("encoding", "jpeg")).lower()
        encoded = payload.get("data")
    elif isinstance(payload, str):
        encoding = "jpeg"
        encoded = payload
    else:
        raise ContractError("image payload must be a JPEG object or base64 string")
    if encoding not in {"jpeg", "jpg"}:
        raise ContractError(f"only JPEG image payloads are supported, got {encoding!r}")
    if not isinstance(encoded, str) or not encoded:
        raise ContractError("image payload is missing non-empty base64 data")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ContractError("image data is not valid base64") from exc
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ContractError("JPEG decode failed")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def legacy_images_to_observation(
    images: Any,
    *,
    device: str,
    allow_missing_images: bool,
) -> dict[str, Any]:
    """Convert old named image objects to canonical Pi07 torch observations."""
    if not isinstance(images, Mapping):
        raise ContractError("images must be an object")
    _, _, torch = _runtime_modules()
    observation: dict[str, Any] = {}
    for source_name, payload in images.items():
        model_name = CAMERA_ALIASES.get(str(source_name))
        if model_name is None:
            continue
        rgb = decode_legacy_image(payload)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        observation[model_name] = tensor.to(device)
    missing = [name for name in CANONICAL_CAMERAS if name not in observation]
    if missing and not allow_missing_images:
        raise ContractError(
            "missing required images: " + ", ".join(missing) + "; expected head,left_wrist,right_wrist"
        )
    if missing:
        fallback = observation.get("cam_high")
        if fallback is None:
            fallback = torch.zeros(3, 256, 320, device=device)
        for name in missing:
            observation[name] = fallback
    return observation


def _request_response(
    request_id: str, step: int, action: Optional[Mapping[str, Any]] = None, error: Optional[str] = None
) -> dict[str, Any]:
    return {
        "ok": error is None,
        "request_id": request_id,
        "step": int(step),
        "action": dict(action or {}),
        "error": error,
    }


@dataclass
class PreparedResponse:
    response: dict[str, Any]
    request_id: str
    sent_actions: Optional[list[list[float]]]
    raw_actions: Optional[list[list[float]]]
    should_cache: bool


class JsonlRecorder:
    """Small optional trace recorder; raw JPEG payloads stay out of the log."""

    def __init__(self, record_dir: str) -> None:
        root = Path(record_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"session_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

    def record(
        self,
        request: Mapping[str, Any],
        prepared: PreparedResponse,
        timing: Mapping[str, Any],
    ) -> None:
        entry = {
            "timestamp": time.time(),
            "request_id": prepared.request_id,
            "step": request.get("step"),
            "instruction": request.get("instruction"),
            "image_keys": sorted(request.get("images", {}).keys())
            if isinstance(request.get("images"), Mapping)
            else [],
            "state_history_len": len(request.get("state_history") or []),
            "raw_model_actions": prepared.raw_actions,
            "sent_model_actions": prepared.sent_actions,
            "response": prepared.response,
            "timing": dict(timing),
        }
        with self.path.open("a", encoding="utf-8") as record_file:
            record_file.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")


class PiperPolicySession:
    """One Pi07 history/session bound to one legacy WebSocket connection."""

    def __init__(
        self,
        args: argparse.Namespace,
        engine: Any,
        recorder: Optional[JsonlRecorder],
        raw_action_recorder: Optional[Any] = None,
    ) -> None:
        self.args = args
        self.engine = engine
        self.recorder = recorder
        self.raw_action_recorder = raw_action_recorder
        self._lock = asyncio.Lock()
        self._instruction: Optional[str] = None
        self._last_request_id: Optional[str] = None
        self._last_response: Optional[dict[str, Any]] = None

    async def reset(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self.engine.reset_runtime_state)
            self._instruction = None
            self._last_request_id = None
            self._last_response = None

    def _parse_request(self, message: Any) -> tuple[dict[str, Any], str, int]:
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        if not isinstance(message, str):
            raise ContractError("websocket message must be UTF-8 JSON")
        raw = json.loads(message)
        if not isinstance(raw, dict):
            raise ContractError("websocket message must be a JSON object")
        request_id = str(raw.get("request_id", "unknown"))
        step = int(raw.get("step", -1))
        if not request_id or request_id == "unknown":
            raise ContractError("request_id is required")
        if step < 0:
            raise ContractError("step must be a non-negative integer")
        return raw, request_id, step

    async def prepare(self, message: Any) -> PreparedResponse:
        request_id = "unknown"
        step = -1
        try:
            request, request_id, step = self._parse_request(message)
            async with self._lock:
                if request_id == self._last_request_id and self._last_response is not None:
                    return PreparedResponse(
                        response=self._last_response,
                        request_id=request_id,
                        sent_actions=None,
                        raw_actions=None,
                        should_cache=False,
                    )

                instruction = str(request.get("instruction", "")).strip()
                if not instruction:
                    raise ContractError("instruction must be a non-empty string")
                if (
                    self.args.expected_instruction
                    and not self.args.allow_instruction_override
                    and instruction != self.args.expected_instruction
                ):
                    raise ContractError(
                        "instruction mismatch: Puzzle checkpoint was trained with "
                        f"{self.args.expected_instruction!r}, got {instruction!r}; "
                        "pass --allow-instruction-override only for deliberate OOD testing"
                    )
                if step == 0 or self._instruction is None or instruction != self._instruction:
                    await asyncio.to_thread(self.engine.reset_runtime_state)
                    self._instruction = instruction
                    self._last_request_id = None
                    self._last_response = None

                model_state = piper_state_to_model(request.get("state"))
                measured_state_history = parse_piper_state_history(
                    request.get("state_history")
                )
                if (
                    getattr(self.engine, "require_measured_history", False)
                    and not measured_state_history
                ):
                    raise ContractError(
                        "state_history is required: send measured 14-D robot states "
                        "oldest-to-newest before Piper Puzzle inference"
                    )
                observation = legacy_images_to_observation(
                    request.get("images"),
                    device=self.args.device,
                    allow_missing_images=self.args.allow_missing_images,
                )
                await asyncio.to_thread(self.engine.update_state, model_state)
                update_state_history = getattr(self.engine, "update_state_history", None)
                if measured_state_history is not None and update_state_history is not None:
                    await asyncio.to_thread(
                        update_state_history, measured_state_history
                    )
                raw_actions_object = await asyncio.to_thread(
                    self.engine.infer,
                    observation,
                    instruction,
                    "",
                    None,
                    self.args.execute_steps,
                )
                raw_actions = _rows_from_actions(raw_actions_object)
                action, sent_actions = model_actions_to_piper_action(
                    raw_actions,
                    model_state,
                    dt=self.args.action_dt,
                    inactive_action_mode=self.args.inactive_action_mode,
                    max_joint_step=self.args.max_joint_step,
                )
                return PreparedResponse(
                    response=_request_response(request_id, step, action=action),
                    request_id=request_id,
                    sent_actions=sent_actions,
                    raw_actions=raw_actions,
                    should_cache=True,
                )
        except Exception as exc:
            traceback.print_exc()
            return PreparedResponse(
                response=_request_response(request_id, step, error=str(exc)),
                request_id=request_id,
                sent_actions=None,
                raw_actions=None,
                should_cache=False,
            )

    async def commit(self, request: Mapping[str, Any], prepared: PreparedResponse) -> None:
        if not prepared.should_cache:
            return
        async with self._lock:
            if prepared.sent_actions is not None:
                await asyncio.to_thread(self.engine.acknowledge_sent_actions, prepared.sent_actions)
            self._last_request_id = prepared.request_id
            self._last_response = prepared.response
            if self.raw_action_recorder is not None and prepared.raw_actions:
                try:
                    raw_action_steps = max(
                        1, int(getattr(self.args, "raw_action_steps", 8))
                    )
                    raw_rows = prepared.raw_actions[:raw_action_steps]
                    self.raw_action_recorder.record(
                        request_id=prepared.request_id,
                        step=int(request.get("step", -1)),
                        instruction=str(request.get("instruction", "")),
                        raw_actions=raw_rows,
                        current_state=piper_state_to_model(request.get("state")),
                        action_dt=float(self.args.action_dt),
                        timing=getattr(self.engine, "_last_timing", {}) or {},
                    )
                except Exception as exc:
                    print(f"[!] raw-action HDF5 record failed: {exc}", flush=True)
                    traceback.print_exc()
            if self.recorder is not None:
                timing = getattr(self.engine, "_last_timing", {}) or {}
                self.recorder.record(request, prepared, timing)

    async def abort(self, prepared: PreparedResponse) -> None:
        if prepared.sent_actions is None:
            return
        async with self._lock:
            await asyncio.to_thread(self.engine.abort_pending_actions)


class PiperWebSocketServer:
    def __init__(self, args: argparse.Namespace, engine: Any) -> None:
        self.args = args
        recorder = JsonlRecorder(args.record_dir) if args.record_dir else None
        raw_action_recorder = None
        raw_action_hdf5 = getattr(args, "raw_action_hdf5", "")
        if raw_action_hdf5:
            try:
                from raw_action_hdf5_recorder import RawModelActionHDF5Recorder
            except ImportError as exc:
                raise RuntimeError(
                    "cannot import server/raw_action_hdf5_recorder.py"
                ) from exc
            raw_action_recorder = RawModelActionHDF5Recorder(
                raw_action_hdf5,
                max_steps_per_request=int(getattr(args, "raw_action_steps", 8)),
            )
            print(
                f"Raw model action HDF5: {raw_action_recorder.path} "
                f"(prefix={raw_action_recorder.max_steps_per_request})",
                flush=True,
            )
        self.session = PiperPolicySession(args, engine, recorder, raw_action_recorder)
        self._connection_lock = asyncio.Lock()
        self._active_connection = False

    def close(self) -> None:
        raw_action_recorder = self.session.raw_action_recorder
        if raw_action_recorder is not None:
            raw_action_recorder.close()

    async def handler(self, websocket: Any) -> None:
        peer = getattr(websocket, "remote_address", None)
        async with self._connection_lock:
            if self._active_connection:
                response = _request_response(
                    "unknown", -1, error="another Piper policy client is already active"
                )
                await websocket.send(json.dumps(response, ensure_ascii=False))
                await websocket.close(code=1013, reason="single-client Pi07 session")
                return
            self._active_connection = True
        print(f"[+] Piper client connected: {peer}", flush=True)
        try:
            await self.session.reset()
            async for message in websocket:
                request: Mapping[str, Any] = {}
                try:
                    if isinstance(message, bytes):
                        message_for_log = message.decode("utf-8")
                    else:
                        message_for_log = message
                    parsed = json.loads(message_for_log)
                    if isinstance(parsed, Mapping):
                        request = parsed
                except Exception:
                    pass
                prepared = await self.session.prepare(message)
                try:
                    await websocket.send(json.dumps(prepared.response, ensure_ascii=False, allow_nan=False))
                    await self.session.commit(request, prepared)
                    if prepared.response.get("ok"):
                        timing = getattr(self.session.engine, "_last_timing", {}) or {}
                        print(
                            "[step {step}] request_id={request_id} actions={actions} mode={mode}".format(
                                step=prepared.response["step"],
                                request_id=prepared.request_id,
                                actions=len(prepared.sent_actions or []),
                                mode=timing.get("mode", "denoise"),
                            ),
                            flush=True,
                        )
                except Exception:
                    await self.session.abort(prepared)
                    raise
        except Exception as exc:
            print(f"[!] Piper WebSocket connection error from {peer}: {exc}", flush=True)
            traceback.print_exc()
        finally:
            async with self._connection_lock:
                self._active_connection = False
            print(f"[-] Piper client disconnected: {peer}", flush=True)


async def serve(args: argparse.Namespace) -> None:
    try:
        import websockets  # type: ignore
    except ImportError as exc:
        raise RuntimeError("install websockets in the UVA-DiT runtime environment") from exc

    if args.mock_policy:
        # The mock still decodes legacy JPEGs, but it does not need CUDA.
        args.device = "cpu"
        engine: Any = MockPiperEngine()
    else:
        _, _, torch = _runtime_modules()
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("--device requests CUDA but torch.cuda.is_available() is false")
        args.device = device
        engine = create_piper_puzzle_engine(args, device)

    server = PiperWebSocketServer(args, engine)
    try:
        async with websockets.serve(
            server.handler,
            args.host,
            args.port,
            max_size=args.max_size_mb * 1024 * 1024,
            ping_interval=20,
            ping_timeout=60,
        ):
            print(
                f"Piper UVA-DiT Puzzle server listening on ws://{args.host}:{args.port} "
                f"(execute_steps={args.execute_steps}, dt={args.action_dt:.9f})",
                flush=True,
            )
            await asyncio.Future()
    finally:
        server.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="", help="selected checkpoint_step_* directory")
    parser.add_argument("--pretrained", default="", help="UniDiT base model directory")
    parser.add_argument("--qwen35", default="", help="Qwen3.5-9B directory")
    parser.add_argument("--qwen35-adapter", default="", help="checkpoint qwen3vl_proj.pt")
    parser.add_argument("--norm-stats", default="", help="paired Puzzle train406 normalization stats")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7081)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute-steps", type=int, default=4)
    parser.add_argument("--action-dt", type=float, default=DEFAULT_ACTION_DT)
    parser.add_argument("--replan-every-call", action="store_true")
    parser.add_argument("--use-rtc-prefix", action="store_true")
    parser.add_argument("--video-steps", type=int, default=25)
    parser.add_argument("--action-steps", type=int, default=25)
    # Pi07InferenceEngine reads these directly.  Leave them unset by default
    # so the selected checkpoint's training config remains authoritative.
    parser.add_argument("--video-snr-shift", type=float, default=None)
    parser.add_argument("--action-snr-shift", type=float, default=None)
    parser.add_argument("--n-video-frames", type=int, default=None)
    parser.add_argument("--action-per-frame", type=int, default=None)
    parser.add_argument("--frame-skip", type=int, default=None)
    parser.add_argument("--history-frames", type=int, default=None)
    parser.add_argument("--history-frame-skip", type=int, default=None)
    parser.add_argument("--action-condition-history-steps", type=int, default=0)
    parser.add_argument(
        "--action-cond-mode",
        choices=("exec_history", "state_broadcast", "zero"),
        default="exec_history",
    )
    parser.add_argument(
        "--state-oob-mode", choices=("clip", "extrapolate", "fail"), default="clip"
    )
    parser.add_argument("--action-output-clip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--expected-instruction", default=DEFAULT_TASK)
    parser.add_argument("--allow-instruction-override", action="store_true")
    parser.add_argument("--inactive-action-mode", choices=("hold", "model"), default="hold")
    parser.add_argument("--max-joint-step", type=float, default=0.0)
    parser.add_argument(
        "--action-smooth-window",
        type=int,
        default=0,
        help="odd Savitzky-Golay window applied to the full model chunk; 0 disables",
    )
    parser.add_argument("--action-smooth-poly", type=int, default=2)
    parser.add_argument(
        "--action-smooth-grippers",
        action="store_true",
        help="also smooth the gripper dimensions; off by default so open/close timing is preserved",
    )
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--max-size-mb", type=int, default=64)
    parser.add_argument("--record-dir", default="")
    parser.add_argument(
        "--raw-action-hdf5",
        default="",
        help="HDF5 path for raw model actions; only the configured prefix is saved",
    )
    parser.add_argument(
        "--raw-action-steps",
        type=int,
        default=8,
        help="number of raw model actions to save from each successful response",
    )
    parser.add_argument("--mock-policy", action="store_true")
    parser.add_argument("--config-only", action="store_true")
    return parser


def _validate_cli(args: argparse.Namespace) -> None:
    if args.execute_steps <= 0 or args.execute_steps > FUTURE_ACTION_STEPS:
        raise ContractError(
            f"execute_steps must be in [1, {FUTURE_ACTION_STEPS}], got {args.execute_steps}"
        )
    if not math.isfinite(args.action_dt) or args.action_dt <= 0:
        raise ContractError("action_dt must be finite and > 0")
    if args.max_joint_step < 0:
        raise ContractError("max_joint_step must be >= 0")
    if args.max_size_mb <= 0:
        raise ContractError("max_size_mb must be positive")
    if args.raw_action_steps <= 0 or args.raw_action_steps > FUTURE_ACTION_STEPS:
        raise ContractError(
            f"raw_action_steps must be in [1, {FUTURE_ACTION_STEPS}]"
        )
    if args.action_smooth_window != 0:
        if args.action_smooth_window < 3 or args.action_smooth_window % 2 == 0:
            raise ContractError(
                "action_smooth_window must be 0 or an odd value >= 3, got "
                f"{args.action_smooth_window}"
            )
        if args.action_smooth_window > FUTURE_ACTION_STEPS:
            raise ContractError(
                f"action_smooth_window must be <= {FUTURE_ACTION_STEPS}"
            )
        if args.action_smooth_poly < 1 or args.action_smooth_poly >= args.action_smooth_window:
            raise ContractError(
                "action_smooth_poly must be in "
                f"[1, {args.action_smooth_window - 1}], got {args.action_smooth_poly}"
            )
    if not args.mock_policy:
        empty = [
            name
            for name in ("checkpoint", "pretrained", "qwen35", "qwen35_adapter", "norm_stats")
            if not getattr(args, name)
        ]
        if empty:
            raise ContractError("required model arguments are empty: " + ", ".join(empty))
        args.use_chunk_ar = True
        args.use_state_tokens = False
        args.subgoal_wm = ""


def main() -> None:
    args = build_arg_parser().parse_args()
    _validate_cli(args)
    artifacts = check_runtime_artifacts(args)
    if args.config_only:
        print(json.dumps(artifacts, ensure_ascii=False, indent=2))
        return
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
