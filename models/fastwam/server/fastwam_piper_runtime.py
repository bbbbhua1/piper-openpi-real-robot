#!/usr/bin/env python3
"""Pure FastWAM/Piper image, layout, and action-safety adapters.

This module intentionally has no FastWAM, CUDA, ROS, or WebSocket imports.  The
network/model layer can therefore depend on the same thoroughly tested boundary
that protects real robot commands, while these functions remain runnable on a
CPU-only development machine.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


_NUM_PIPER_VALUES = 14
_NUM_ARM_VALUES = 7

# Selecting FastWAM columns by this sequence yields Piper wire order:
# [L1..L6, Lg, R1..R6, Rg].
_PIPER_ORDER_FROM_FASTWAM = np.asarray(
    [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13], dtype=np.intp
)
# Selecting Piper columns by this sequence yields FastWAM order:
# [L1..L6, R1..R6, Lg, Rg].
_FASTWAM_ORDER_FROM_PIPER = np.asarray(
    [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13], dtype=np.intp
)
_PIPER_GRIPPER_INDICES = np.asarray([6, 13], dtype=np.intp)
_PIPER_ARM_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.intp)

LOG = logging.getLogger(__name__)

FASTWAM_ACTION_HORIZON = 32
FASTWAM_NUM_VIDEO_FRAMES = 9
PUZZLE_TASK_LABEL = "crimp"
DEFAULT_PROMPT_TEMPLATE = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)
PUZZLE_PROMPT = DEFAULT_PROMPT_TEMPLATE.format(task=PUZZLE_TASK_LABEL)


def _finite_vector(value: Any, *, name: str, size: int) -> np.ndarray:
    """Return an exactly-sized finite float32 vector or raise a clear error."""
    try:
        vector = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain {size} finite numeric values") from exc
    if vector.shape != (size,):
        raise ValueError(
            f"{name} must contain exactly {size} finite numeric values, got shape={vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain exactly {size} finite numeric values")
    return np.ascontiguousarray(vector, dtype=np.float32)


def _finite_actions(value: Any, *, name: str = "actions") -> np.ndarray:
    """Return a non-empty finite [T, 14] float32 trajectory."""
    try:
        actions = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must have shape [T, 14]") from exc
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != _NUM_PIPER_VALUES:
        raise ValueError(f"{name} must have shape [T, 14] with T > 0, got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(actions, dtype=np.float32)


def _fastwam_vector_to_piper(vector_fastwam: Any, *, name: str) -> np.ndarray:
    vector = _finite_vector(vector_fastwam, name=name, size=_NUM_PIPER_VALUES)
    return np.ascontiguousarray(vector[_PIPER_ORDER_FROM_FASTWAM], dtype=np.float32)


def _piper_vector_to_fastwam(vector_piper: Any, *, name: str) -> np.ndarray:
    vector = _finite_vector(vector_piper, name=name, size=_NUM_PIPER_VALUES)
    return np.ascontiguousarray(vector[_FASTWAM_ORDER_FROM_PIPER], dtype=np.float32)


@dataclass(frozen=True)
class PiperSafetyLimits:
    """Absolute and per-waypoint limits in Piper wire order.

    The order is ``[L1..L6, Lg, R1..R6, Rg]``.  Copies are made read-only so
    values validated at construction cannot be mutated accidentally by callers.
    """

    lower: np.ndarray
    upper: np.ndarray
    max_delta: np.ndarray

    def __post_init__(self) -> None:
        lower = _finite_vector(self.lower, name="limits.lower", size=_NUM_PIPER_VALUES).copy()
        upper = _finite_vector(self.upper, name="limits.upper", size=_NUM_PIPER_VALUES).copy()
        max_delta = _finite_vector(
            self.max_delta, name="limits.max_delta", size=_NUM_PIPER_VALUES
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
    """Metadata for finite model outputs corrected to a hardware boundary."""

    projected_values: int
    max_arm_projection: float
    max_gripper_projection: float


def piper_wire_to_fastwam(state: Mapping[str, Any]) -> np.ndarray:
    """Validate two 7D Piper vectors and return FastWAM-order float32 ``[14]``.

    Piper's JSON state has separate left/right arrays, with each gripper at the
    end of its arm vector.  FastWAM instead places both grippers after the two
    six-joint arm vectors.
    """
    if not isinstance(state, Mapping):
        raise ValueError("state must be a mapping containing left_arm and right_arm")
    try:
        left = _finite_vector(
            state.get("left_arm"), name="state.left_arm", size=_NUM_ARM_VALUES
        )
        right = _finite_vector(
            state.get("right_arm"), name="state.right_arm", size=_NUM_ARM_VALUES
        )
    except ValueError as exc:
        raise ValueError(
            "state.left_arm and state.right_arm must each contain exactly seven finite numeric values"
        ) from exc
    return np.ascontiguousarray(
        np.concatenate((left[:6], right[:6], left[6:7], right[6:7])), dtype=np.float32
    )


def fastwam_to_piper_wire(actions: np.ndarray, *, dt: float) -> dict[str, Any]:
    """Map ``[T, 14]`` FastWAM outputs to the existing Piper JSON action schema."""
    try:
        dt_value = float(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("dt must be a positive finite number") from exc
    if not np.isfinite(dt_value) or dt_value <= 0:
        raise ValueError("dt must be a positive finite number")

    fastwam_actions = _finite_actions(actions)
    piper_actions = np.ascontiguousarray(
        fastwam_actions[:, _PIPER_ORDER_FROM_FASTWAM], dtype=np.float32
    )
    num_waypoints = len(piper_actions)
    return {
        "left_arm": piper_actions[:, :_NUM_ARM_VALUES].tolist(),
        "right_arm": piper_actions[:, _NUM_ARM_VALUES:].tolist(),
        "time_list": [float(dt_value * (index + 1)) for index in range(num_waypoints)],
        "dt": dt_value,
    }


def _to_uint8_rgb_image(image: np.ndarray, *, name: str, input_color: str) -> np.ndarray:
    """Validate one HWC image and return a contiguous uint8 RGB image."""
    if input_color not in {"rgb", "bgr"}:
        raise ValueError("input_color must be either 'rgb' or 'bgr'")
    try:
        values = np.asarray(image, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an HWC image with three channels") from exc
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] != 3:
        raise ValueError(f"{name} must be an HWC image with three channels, got shape={values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} image values must be finite")
    if np.any(values < 0) or np.any(values > 255):
        raise ValueError(f"{name} image values must lie in [0, 255]")
    rgb = np.rint(values).astype(np.uint8)
    if input_color == "bgr":
        rgb = rgb[:, :, ::-1]
    return np.ascontiguousarray(rgb)


def _resize_image(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    """Resize an RGB image without importing a model or accelerator library."""
    from PIL import Image

    return np.asarray(
        Image.fromarray(image).resize((width, height), resample=Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def compose_fastwam_image(
    head: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    input_color: str = "rgb",
) -> np.ndarray:
    """Compose RobotWin tiles into float32 CHW ``[3, 384, 320]`` in ``[-1, 1]``.

    The caller chooses whether its decoded HWC images are RGB or BGR through
    ``input_color``.  The result always has RGB channel order: a 320x256 head
    view above 160x128 left/right wrist views.
    """
    head_rgb = _to_uint8_rgb_image(head, name="head", input_color=input_color)
    left_rgb = _to_uint8_rgb_image(left, name="left", input_color=input_color)
    right_rgb = _to_uint8_rgb_image(right, name="right", input_color=input_color)

    head_tile = _resize_image(head_rgb, width=320, height=256)
    left_tile = _resize_image(left_rgb, width=160, height=128)
    right_tile = _resize_image(right_rgb, width=160, height=128)
    bottom_row = np.concatenate((left_tile, right_tile), axis=1)
    image_hwc = np.concatenate((head_tile, bottom_row), axis=0)
    image_chw = np.transpose(image_hwc, (2, 0, 1)).astype(np.float32)
    return np.ascontiguousarray(image_chw / 127.5 - 1.0, dtype=np.float32)


def project_and_limit_fastwam_actions(
    actions_fastwam: np.ndarray,
    current_fastwam: np.ndarray,
    limits_piper: PiperSafetyLimits,
) -> tuple[np.ndarray, ActionProjectionSummary]:
    """Clamp finite model outputs to physical endpoints, then limit deltas.

    Absolute limits and deltas are intentionally expressed in Piper wire order,
    which keeps the deployment configuration aligned with the physical robot.
    Projection does not alter a verified hardware limit: it only replaces a
    finite raw prediction with that limit before sequential rate limiting.
    """
    if not isinstance(limits_piper, PiperSafetyLimits):
        raise TypeError("limits_piper must be a PiperSafetyLimits instance")
    actions = _finite_actions(actions_fastwam, name="actions_fastwam")
    current_piper = _fastwam_vector_to_piper(
        current_fastwam, name="current_fastwam"
    )
    actions_piper = np.ascontiguousarray(
        actions[:, _PIPER_ORDER_FROM_FASTWAM], dtype=np.float32
    )

    projection = np.maximum(
        np.maximum(limits_piper.lower - actions_piper, 0.0),
        np.maximum(actions_piper - limits_piper.upper, 0.0),
    )
    arm_projection = projection[:, _PIPER_ARM_INDICES]
    gripper_projection = projection[:, _PIPER_GRIPPER_INDICES]
    max_arm_projection = float(arm_projection.max(initial=0.0))
    max_gripper_projection = float(gripper_projection.max(initial=0.0))

    actions_piper = np.ascontiguousarray(
        np.clip(actions_piper, limits_piper.lower, limits_piper.upper), dtype=np.float32
    )

    accepted_piper = np.empty_like(actions_piper)
    previous = current_piper
    for timestep, prediction in enumerate(actions_piper):
        accepted = np.clip(
            prediction,
            previous - limits_piper.max_delta,
            previous + limits_piper.max_delta,
        )
        accepted_piper[timestep] = accepted
        previous = accepted

    return (
        np.ascontiguousarray(accepted_piper[:, _FASTWAM_ORDER_FROM_PIPER], dtype=np.float32),
        ActionProjectionSummary(
            projected_values=int(np.count_nonzero(projection)),
            max_arm_projection=max_arm_projection,
            max_gripper_projection=max_gripper_projection,
        ),
    )


def validate_and_limit_fastwam_actions(
    actions_fastwam: np.ndarray,
    current_fastwam: np.ndarray,
    limits_piper: PiperSafetyLimits,
) -> np.ndarray:
    """Compatibility wrapper returning only the safe FastWAM-order trajectory."""
    accepted, _summary = project_and_limit_fastwam_actions(
        actions_fastwam, current_fastwam, limits_piper
    )
    return accepted


def _positive_config_int(
    config: Mapping[str, Any], key: str, *, default: int | None = None
) -> int:
    """Read a strictly-positive, non-boolean integer deployment setting."""
    value = config.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a positive integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{key} must be a positive integer") from exc
    if integer <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return integer


def _optional_finite_float(config: Mapping[str, Any], key: str) -> float | None:
    """Read an optional finite float without accepting ambiguous values."""
    value = config.get(key)
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "null"}):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{key} must be a finite number or null") from exc
    if not np.isfinite(result):
        raise ValueError(f"{key} must be a finite number or null")
    return result


def _as_numpy(value: Any, *, name: str) -> np.ndarray:
    """Detach a torch-like value lazily, without importing torch for CPU tests."""
    if isinstance(value, np.ndarray):
        return value
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        value = numpy()
    try:
        return np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} cannot be converted to a numeric array") from exc


class FastWAMPiperRuntime:
    """Load and run the final Puzzle FastWAM policy behind a small pure boundary.

    Importing this module never imports FastWAM, Hydra, PyTorch, CUDA, ROS, or
    WebSocket libraries.  ``from_config`` performs the production-only imports;
    direct construction accepts fakes so the prompt, image composition, layout,
    and normalization contract stays CPU-testable.
    """

    def __init__(self, *, model: Any, processor: Any, config: Mapping[str, Any]) -> None:
        if not isinstance(config, Mapping):
            raise ValueError("FastWAM runtime config must be a mapping")
        if model is None or processor is None:
            raise ValueError("FastWAM runtime requires both a model and a processor")

        self.model = model
        self.processor = processor
        self.config = dict(config)

        task_label = str(self.config.get("task_label", PUZZLE_TASK_LABEL)).strip()
        if not task_label:
            raise ValueError("task_label must be a non-empty string")
        self.task_label = task_label
        prompt_template = str(self.config.get("prompt_template", DEFAULT_PROMPT_TEMPLATE))
        self.prompt = (
            prompt_template.format(task=task_label)
            if "{task}" in prompt_template
            else prompt_template
        )

        self.model_joint_order = str(self.config.get("model_joint_order", "fastwam")).strip().lower()
        if self.model_joint_order not in {"fastwam", "piper_wire"}:
            raise ValueError("model_joint_order must be 'fastwam' or 'piper_wire'")

        self.action_horizon = _positive_config_int(
            self.config, "model_action_horizon", default=FASTWAM_ACTION_HORIZON
        )
        if self.action_horizon % 4 != 0:
            raise ValueError("model_action_horizon must be divisible by 4")
        self.num_video_frames = _positive_config_int(
            self.config, "num_video_frames", default=FASTWAM_NUM_VIDEO_FRAMES
        )
        expected_video_frames = self.action_horizon // 4 + 1
        if self.num_video_frames != expected_video_frames:
            raise ValueError(
                "num_video_frames must equal model_action_horizon / 4 + 1 "
                f"({expected_video_frames} for horizon {self.action_horizon})"
            )
        self.num_inference_steps = _positive_config_int(
            self.config, "num_inference_steps", default=10
        )

        seed = self.config.get("seed")
        if seed is None or (isinstance(seed, str) and seed.strip().lower() in {"", "none", "null"}):
            self.seed: int | None = None
        elif isinstance(seed, bool):
            raise ValueError("seed must be an integer or null")
        else:
            try:
                self.seed = int(seed)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("seed must be an integer or null") from exc

        try:
            self.text_cfg_scale = float(self.config.get("text_cfg_scale", 1.0))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("text_cfg_scale must be finite") from exc
        if not np.isfinite(self.text_cfg_scale):
            raise ValueError("text_cfg_scale must be finite")
        self.negative_prompt = str(self.config.get("negative_prompt", ""))
        self.sigma_shift = _optional_finite_float(self.config, "sigma_shift")
        self.rand_device = str(self.config.get("rand_device", "cpu"))
        self.tiled = bool(self.config.get("tiled", False))

        # Test-only injection keeps this module usable in a minimal Python
        # environment. Production construction always supplies a real torch
        # module from ``from_config`` and leaves the factory unset.
        self._torch_module = self.config.get("_torch_module")
        tensor_factory = self.config.get("_tensor_factory")
        if tensor_factory is not None and not callable(tensor_factory):
            raise ValueError("_tensor_factory must be callable when supplied")
        self._tensor_factory = tensor_factory

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "FastWAMPiperRuntime":
        """Load the exact trained Puzzle policy, processor, and normalizer stats.

        This mirrors FastWAM's maintained RobotWin deployment loader rather than
        assuming a generic checkpoint layout.  It intentionally refuses a
        missing base project, checkpoint, or stats file before model allocation.
        """
        if not isinstance(config, Mapping):
            raise ValueError("FastWAM runtime config must be a mapping")
        config_copy = dict(config)
        root_value = config_copy.get("fastwam_root")
        if root_value is None:
            raise ValueError("fastwam_root is required")
        fastwam_root = Path(str(root_value)).expanduser().resolve()
        configs_root = fastwam_root / "configs"
        src_root = fastwam_root / "src"
        if not configs_root.is_dir() or not src_root.is_dir():
            raise FileNotFoundError(
                f"fastwam_root must contain configs/ and src/, got {fastwam_root}"
            )
        for import_root in (fastwam_root, src_root):
            if str(import_root) not in sys.path:
                sys.path.insert(0, str(import_root))

        checkpoint_path = cls._required_file(config_copy, "checkpoint_path")
        dataset_stats_path = cls._required_file(config_copy, "dataset_stats_path")
        model_base_path = Path(
            str(config_copy.get("model_base_path", "/bh/zbh_ckp/models/fastwam"))
        ).expanduser().resolve()
        if not model_base_path.is_dir():
            raise FileNotFoundError(
                "model_base_path must be the existing FastWAM base-model cache directory, got "
                f"{model_base_path}"
            )
        # FastWAM's ModelConfig otherwise defaults to ./checkpoints, which would
        # put multi-GB Wan base weights inside this lightweight deployment repo.
        # Set this before model construction so existing shared weights are
        # found and future downloads (if explicitly needed) use the model disk.
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(model_base_path)

        task_name = str(config_copy.get("task_name", "")).strip()
        supported_tasks = {
            "agilex_cobotmagic2_puzzle_joint_3cam_384_1e-4_baidu32",
            "realrobot_joint_3cam_384_1e-4",
            "realrobot_joint_3cam_384_chunk64_1e-4",
        }
        if task_name not in supported_tasks:
            raise ValueError(
                "unsupported FastWAM task; expected one of: "
                + ", ".join(sorted(supported_tasks))
            )
        config_name = cls._resolve_sim_config_name(config_copy, configs_root)

        try:
            import torch
            from hydra import compose, initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
            from hydra.utils import instantiate
            from omegaconf import OmegaConf
            from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        except ImportError as exc:  # pragma: no cover - exercised on GPU deployment
            raise RuntimeError(
                "FastWAM inference dependencies are unavailable; activate the fastwam environment"
            ) from exc

        device = str(config_copy.get("device", "cuda:0"))
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested for FastWAM inference but unavailable: {device}")
        model_dtype = cls._model_dtype(torch, config_copy.get("mixed_precision", "bf16"))

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
            cfg = compose(config_name=config_name, overrides=[f"task={task_name}"])

        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.load_text_encoder = True
        model_cfg.skip_dit_load_from_pretrain = True
        action_dit_path = str(model_cfg.get("action_dit_pretrained_path", ""))
        if action_dit_path and not Path(action_dit_path).is_absolute():
            candidates = [fastwam_root / action_dit_path, model_base_path / Path(action_dit_path).name]
            for candidate in candidates:
                if candidate.is_file():
                    model_cfg.action_dit_pretrained_path = str(candidate)
                    break
        model = instantiate(model_cfg, model_dtype=model_dtype, device=device)
        model.load_checkpoint(str(checkpoint_path))
        model = model.to(device).eval()

        processor = instantiate(cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(dataset_stats_path)))

        config_copy["_torch_module"] = torch
        return cls(model=model, processor=processor, config=config_copy)

    @staticmethod
    def _required_file(config: Mapping[str, Any], key: str) -> Path:
        value = config.get(key)
        if value is None or not str(value).strip():
            raise ValueError(f"{key} is required")
        path = Path(str(value)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{key} does not exist or is not a file: {path}")
        return path

    @staticmethod
    def _resolve_sim_config_name(config: Mapping[str, Any], configs_root: Path) -> str:
        explicit_path = config.get("sim_config_path")
        if explicit_path is not None and str(explicit_path).strip():
            candidate = Path(str(explicit_path)).expanduser().resolve()
        else:
            candidate = (configs_root / str(config.get("sim_config_name", "sim_robotwin.yaml"))).resolve()
        try:
            relative = candidate.relative_to(configs_root.resolve())
        except ValueError as exc:
            raise ValueError(f"sim config must be contained in {configs_root}, got {candidate}") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"sim config does not exist: {candidate}")
        return relative.as_posix()

    @staticmethod
    def _model_dtype(torch: Any, mixed_precision: Any) -> Any:
        mode = str(mixed_precision).strip().lower()
        dtype_by_precision = {
            "no": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if mode not in dtype_by_precision:
            raise ValueError("mixed_precision must be one of: no, fp16, bf16")
        return dtype_by_precision[mode]

    def _torch(self) -> Any:
        if self._torch_module is not None:
            return self._torch_module
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - guarded by fake test injection
            raise RuntimeError(
                "PyTorch is required for production FastWAM inference; use an injected tensor factory only in tests"
            ) from exc
        self._torch_module = torch
        return torch

    def _to_model_tensor(self, values: np.ndarray) -> Any:
        values = np.ascontiguousarray(values, dtype=np.float32)
        if self._tensor_factory is not None:
            return self._tensor_factory(values)
        return self._torch().as_tensor(values, dtype=self._torch().float32)

    def _no_grad_context(self) -> Any:
        if self._tensor_factory is not None and self._torch_module is None:
            return nullcontext()
        return self._torch().no_grad()

    def _normalize_state(self, state_fastwam: np.ndarray) -> Any:
        """Run FastWAM's real normalizer, or a deliberately injected test hook."""
        test_hook = getattr(self.processor, "normalize_state", None)
        if callable(test_hook):
            return test_hook(state_fastwam)

        torch = self._torch()
        try:
            state_meta = self.processor.shape_meta["state"]
            offset = 0
            state_fields: dict[str, Any] = {}
            for meta in state_meta:
                width = int(meta["raw_shape"])
                state_fields[meta["key"]] = torch.as_tensor(
                    state_fastwam[offset : offset + width], dtype=torch.float32
                ).unsqueeze(0)
                offset += width
            if offset != len(state_fastwam):
                raise ValueError(f"state metadata totals {offset}, expected {len(state_fastwam)}")
            batch = {"state": state_fields}
            transformed = self.processor.action_state_transform(batch)
            if transformed is None:
                transformed = batch
            normalized = self.processor.normalizer.forward(transformed)
            merger = getattr(self.processor, "action_state_merger", None)
            if merger is not None:
                normalized = merger.forward(normalized)
            return normalized["state"]
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("FastWAM processor failed to normalize the 14D Piper state") from exc

    def _denormalize_action(self, action: Any, normalized_state: Any | None = None) -> np.ndarray:
        """Undo the exact dataset normalizer before hardware safety validation."""
        test_hook = getattr(self.processor, "denormalize_action", None)
        if callable(test_hook):
            return _as_numpy(test_hook(action), name="denormalized action")

        torch = self._torch()
        try:
            action_tensor = action
            if not isinstance(action_tensor, torch.Tensor):
                action_tensor = torch.as_tensor(action_tensor, dtype=torch.float32)
            if action_tensor.ndim == 2:
                action_tensor = action_tensor.unsqueeze(0)
            if action_tensor.ndim != 3:
                raise ValueError(f"expected model action [T,D] or [B,T,D], got {tuple(action_tensor.shape)}")
            action_tensor = action_tensor.to(dtype=torch.float32, device="cpu")
            state_tensor = normalized_state
            if state_tensor is None:
                state_tensor = torch.zeros((action_tensor.shape[0], 1, action_tensor.shape[-1]), dtype=torch.float32)
            elif not isinstance(state_tensor, torch.Tensor):
                state_tensor = torch.as_tensor(state_tensor, dtype=torch.float32)
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0).unsqueeze(1)
            elif state_tensor.ndim == 2:
                state_tensor = state_tensor.unsqueeze(1)
            batch = {"action": action_tensor, "state": state_tensor}
            merger = getattr(self.processor, "action_state_merger", None)
            if merger is not None:
                batch = merger.backward(batch)
            batch = self.processor.normalizer.backward(batch)
            if merger is not None:
                action_meta = self.processor.shape_meta["action"]
                state_meta = self.processor.shape_meta["state"]
                merged_batch = {
                    "action": {
                        meta["key"]: batch["action"][meta["key"]].squeeze(0)
                        for meta in action_meta
                    },
                    "state": {
                        meta["key"]: batch["state"][meta["key"]].squeeze(0)
                        for meta in state_meta
                    },
                }
                merged_batch = merger.forward(merged_batch)
                batch["action"] = merged_batch["action"].unsqueeze(0)
            result = _as_numpy(batch["action"], name="denormalized action")
            if result.ndim != 3 or result.shape[0] != 1:
                raise ValueError(f"expected one action batch, got shape {result.shape}")
            return result[0]
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("FastWAM processor failed to denormalize the model action") from exc

    def infer(
        self,
        *,
        head: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
        state: Mapping[str, Any],
    ) -> np.ndarray:
        """Return one finite, unnormalized FastWAM action chunk."""
        state_fastwam = piper_wire_to_fastwam(state)
        if self.model_joint_order == "piper_wire":
            state_model = np.ascontiguousarray(state_fastwam[_PIPER_ORDER_FROM_FASTWAM], dtype=np.float32)
        else:
            state_model = state_fastwam
        image_chw = compose_fastwam_image(head, left, right, input_color="rgb")
        input_image = self._to_model_tensor(image_chw[np.newaxis, ...])
        proprio = self._normalize_state(state_model)
        infer_kwargs = {
            "prompt": self.prompt,
            "input_image": input_image,
            "action_horizon": self.action_horizon,
            "num_video_frames": self.num_video_frames,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        with self._no_grad_context():
            prediction = self.model.infer_action(**infer_kwargs)
        if not isinstance(prediction, Mapping) or "action" not in prediction:
            raise ValueError("FastWAM infer_action must return a mapping containing 'action'")
        actions_model = _finite_actions(
            self._denormalize_action(prediction["action"], proprio), name="FastWAM action"
        )
        if self.model_joint_order == "piper_wire":
            actions = np.ascontiguousarray(
                actions_model[:, _FASTWAM_ORDER_FROM_PIPER], dtype=np.float32
            )
        else:
            actions = actions_model
        if actions.shape != (self.action_horizon, _NUM_PIPER_VALUES):
            raise ValueError(
                "FastWAM action must have shape "
                f"[{self.action_horizon}, {_NUM_PIPER_VALUES}], got {actions.shape}"
            )
        return actions
