"""Local Qwen3-VL subtask planning and done checks for RoboTwin eval."""

import io
import json
import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

Q2_DONE_ONLY_SYSTEM_PROMPT = (
    "You are a robot execution status estimator. Given the global task, completed subtasks and "
    "the current subtask with their historical visual memory, the robot state values, and "
    "{observation_prompt}, predict whether the current subtask is done using the done query token. "
    "Do not generate a natural-language answer."
)


@dataclass
class VLMObservationBatch(Sequence):
    """Prompt frames plus cadence observations not yet acknowledged by the VLM."""

    prompt: List[tuple[int, Dict[str, Any]]]
    cache: List[tuple[int, Dict[str, Any]]]
    reset_memory: bool = False

    def __getitem__(self, index):
        return self.prompt[index]

    def __len__(self) -> int:
        return len(self.prompt)


def validate_dual_lora_checkpoint(
    checkpoint: str,
    *,
    expected_step: Optional[int] = None,
) -> Path:
    """Validate the strict Q1/Q2 LoRA artifact contract used by online eval."""
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    checkpoint_dir = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"VLM checkpoint not found: {checkpoint_path}")

    required = [
        checkpoint_dir / "pytorch_model.bin",
        checkpoint_dir / "robotwin_query_embeddings.bin",
        checkpoint_dir / "q1" / "adapter_config.json",
        checkpoint_dir / "q1" / "adapter_model.safetensors",
        checkpoint_dir / "q2" / "adapter_config.json",
        checkpoint_dir / "q2" / "adapter_model.safetensors",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "VLM checkpoint must contain separate Q1/Q2 LoRA adapters and query "
            f"embeddings; missing: {missing}"
        )

    adapter_configs = []
    for name in ("q1", "q2"):
        config_path = checkpoint_dir / name / "adapter_config.json"
        with config_path.open(encoding="utf-8") as handle:
            adapter_configs.append(json.load(handle))
    compatibility_keys = ("r", "lora_alpha", "task_type", "peft_type")
    mismatched = [
        key
        for key in compatibility_keys
        if adapter_configs[0].get(key) != adapter_configs[1].get(key)
    ]
    if set(adapter_configs[0].get("target_modules", ())) != set(
        adapter_configs[1].get("target_modules", ())
    ):
        mismatched.append("target_modules")
    if mismatched:
        raise ValueError(f"Q1/Q2 LoRA configs are incompatible for keys: {mismatched}")

    if expected_step is not None:
        state_path = checkpoint_dir / "trainer_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(
                f"trainer_state.json is required to verify VLM step {expected_step}: {state_path}"
            )
        with state_path.open(encoding="utf-8") as handle:
            actual_step = int(json.load(handle).get("global_step", -1))
        if actual_step != int(expected_step):
            raise ValueError(
                f"VLM checkpoint step mismatch: expected {expected_step}, got {actual_step}."
            )
    return checkpoint_dir


class VLMQ1InvalidOutputError(ValueError):
    def __init__(self, raw_output: str):
        self.raw_output = str(raw_output)
        super().__init__("VLM Q1 returned no valid remaining subtasks.")


def _extract_json_array(text: str) -> Optional[List[Dict[str, Any]]]:
    stripped = str(text).strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    candidates = [stripped]
    start, end = stripped.find("["), stripped.rfind("]")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
    return None


def normalize_subtasks(
    task_goal: str, subtasks: Optional[Sequence[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    normalized = []
    for index, item in enumerate(subtasks or []):
        if not isinstance(item, dict):
            continue
        goal = str(item.get("subtask_goal", "")).strip()
        if not goal:
            continue
        try:
            subtask_index = int(item.get("subtask_index", index))
        except (TypeError, ValueError):
            subtask_index = index
        normalized.append({"subtask_index": subtask_index, "subtask_goal": goal})
    return normalized or [{"subtask_index": 0, "subtask_goal": str(task_goal)}]


def parse_q1_subtasks_output(
    task_goal: str,
    raw_output: str,
    *,
    strict: bool,
    allow_empty: bool = False,
) -> List[Dict[str, Any]]:
    parsed = _extract_json_array(raw_output)
    if parsed == [] and allow_empty:
        return []
    valid = parsed is not None and any(
        isinstance(item, dict) and str(item.get("subtask_goal", "")).strip()
        for item in parsed
    )
    if not valid and strict:
        raise VLMQ1InvalidOutputError(raw_output)
    if not valid:
        logger.warning("VLM Q1 output is invalid; using the global task: %s", raw_output)
    return normalize_subtasks(task_goal, parsed)


def snapshot_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
    obs_data = observation["observation"]
    result = {
        "observation": {
            camera: {"rgb": np.array(obs_data[camera]["rgb"], copy=True)}
            for camera in ("head_camera", "left_camera", "right_camera")
        }
    }
    for key in ("endpose", "joint_action"):
        if key in observation:
            value = observation[key]
            result[key] = {
                name: np.array(item, copy=True) if isinstance(item, np.ndarray) else item
                for name, item in value.items()
            }
    return result


def _jpeg_rgb(array: np.ndarray) -> Image.Image:
    image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()


class RobotWinVLMSubtaskMonitor:
    """Loads one local Qwen3-VL checkpoint and serves Q1/Q2 synchronously."""

    def __init__(
        self,
        *,
        repo_path: str,
        base_model: str,
        checkpoint: str,
        device: str,
        dtype: str = "bf16",
        attn_implementation: str = "sdpa",
        model_max_length: int = 8192,
        done_threshold: float = 0.5,
        max_new_tokens: int = 512,
        plan_enabled: bool = True,
        min_pixels: int = 12544,
        max_pixels: int = 451584,
        views: str = "main,left_wrist,right_wrist",
        q2_frame_stride: int = 1,
        q2_state_enabled: bool = True,
        failure_query_enabled: bool = True,
        voting_done: bool = False,
        done_vote_count: int = 5,
        done_vote_head: int = 0,
        fixed_subtasks: Optional[Sequence[Dict[str, Any]]] = None,
        require_dual_lora: bool = True,
        q1_adapter_path: Optional[str] = None,
        load_q2_adapter: bool = True,
        expected_checkpoint_step: Optional[int] = None,
        expected_memory_frames: int = 5,
        memory_strategy: str = "subtask_cls_recent_images",
        memory_frame_stride: int = 4,
        completed_subtask_memory_stride: int = 16,
        current_subtask_memory_stride: int = 4,
    ) -> None:
        repo = Path(repo_path).expanduser().resolve()
        if (repo / "qwen-vl-finetune" / "qwenvl").is_dir():
            repo = repo / "qwen-vl-finetune"
        if not (repo / "qwenvl").is_dir():
            raise FileNotFoundError(f"Could not find qwen-vl-finetune/qwenvl under {repo_path}")
        for name, value in (("vlm_base_model", base_model), ("vlm_checkpoint", checkpoint)):
            if not str(value).strip():
                raise ValueError(f"{name} is required when VLM evaluation is enabled.")
        if require_dual_lora:
            validate_dual_lora_checkpoint(
                checkpoint,
                expected_step=expected_checkpoint_step,
            )
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        try:
            from qwenvl.data.data_processor import IGNORE_INDEX, get_rope_index_3
            from qwenvl.data.robotwin_processor import (
                DEFAULT_ROBOTWIN_VIEWS,
                MEMORY_TOKEN,
                Q1_SYSTEM_PROMPT,
                Q2_SYSTEM_PROMPT,
                QUERY_TOKENS,
                ROBOTWIN_MAIN_IMAGE_SIZE,
                ROBOTWIN_MEMORY_VIEWS,
                _group_memory_frames_for_prompt,
                _messages_for_sample,
                _subtask_memory_frame_indices,
                _system_prompt,
                _user_content,
                parse_robotwin_views,
            )
            from tools.utils import robotwin_eval as robotwin_eval_utils
        except ImportError as exc:
            raise ImportError(
                "Failed to import the Qwen3-VL RoboTwin evaluator. Use the VLM eval "
                "environment (recent transformers and qwen_vl_utils are required)."
            ) from exc

        self.device = str(device)
        self.done_threshold = float(done_threshold)
        self.max_new_tokens = int(max_new_tokens)
        self.plan_enabled = bool(plan_enabled)
        self.q2_state_enabled = bool(q2_state_enabled)
        self.failure_query_enabled = bool(failure_query_enabled)
        self.voting_done = bool(voting_done)
        self.done_vote_count = int(done_vote_count)
        self.done_vote_head = int(done_vote_head)
        self.expected_memory_frames = int(expected_memory_frames)
        if self.expected_memory_frames < 0:
            raise ValueError("expected_memory_frames must be non-negative.")
        self.memory_strategy = str(memory_strategy).strip().lower()
        if self.memory_strategy not in {"image_window", "subtask_cls_recent_images"}:
            raise ValueError(
                "memory_strategy must be 'image_window' or 'subtask_cls_recent_images', "
                f"got {memory_strategy!r}."
            )
        self.memory_frame_stride = max(1, int(memory_frame_stride))
        self.completed_subtask_memory_stride = max(
            1, int(completed_subtask_memory_stride)
        )
        self.current_subtask_memory_stride = max(
            1, int(current_subtask_memory_stride)
        )
        if (
            self.memory_strategy == "subtask_cls_recent_images"
            and self.memory_frame_stride != self.current_subtask_memory_stride
        ):
            raise ValueError(
                "subtask_cls_recent_images requires memory_frame_stride to match "
                "current_subtask_memory_stride."
            )
        self.fixed_subtasks = (
            normalize_subtasks("", fixed_subtasks) if fixed_subtasks else None
        )
        self.last_q1_raw_output = None
        self._lock = threading.RLock()
        self._ignore_index = IGNORE_INDEX
        self._get_rope_index = get_rope_index_3
        self._views = tuple(parse_robotwin_views(views)) if views else tuple(DEFAULT_ROBOTWIN_VIEWS)
        self._memory_views = tuple(ROBOTWIN_MEMORY_VIEWS)
        if not set(self._memory_views).issubset(self._views):
            raise ValueError(
                "VLM memory views must be included in the observation views: "
                f"memory={self._memory_views}, observation={self._views}."
            )
        self._q1_system_prompt = Q1_SYSTEM_PROMPT
        self._q2_system_prompt = (
            Q2_SYSTEM_PROMPT
            if self.failure_query_enabled
            else Q2_DONE_ONLY_SYSTEM_PROMPT
        )
        self._done_query_token = QUERY_TOKENS["done"]
        self._messages_for_sample = _messages_for_sample
        self._group_memory_frames_for_prompt = _group_memory_frames_for_prompt
        self._subtask_memory_frame_indices = _subtask_memory_frame_indices
        self._system_prompt = _system_prompt
        self._user_content = _user_content
        self._move_to_device = robotwin_eval_utils.move_to_device
        self._memory_token = MEMORY_TOKEN
        self._main_image_size = tuple(ROBOTWIN_MAIN_IMAGE_SIZE)
        self._min_pixels = int(min_pixels)
        self._max_pixels = int(max_pixels)

        args = SimpleNamespace(
            base_model=str(base_model), checkpoint=str(checkpoint), data_root="",
            q1_adapter_path=q1_adapter_path,
            load_q2_adapter=bool(load_q2_adapter),
            test_ratio=0.05, split_seed=0, q2_frame_stride=int(q2_frame_stride),
            model_max_length=int(model_max_length), device=self.device, dtype=str(dtype),
            attn_implementation=str(attn_implementation), min_pixels=int(min_pixels),
            max_pixels=int(max_pixels), video_min_pixels=28 * 28 * 144,
            video_max_pixels=28 * 28 * 576, video_min_frames=4, video_max_frames=8,
            video_fps=2.0, voting_done=self.voting_done,
            done_vote_count=self.done_vote_count,
        )
        self._install_tokenizer_fast_fallback(robotwin_eval_utils)
        context = robotwin_eval_utils.load_eval_context(
            args, prefer_checkpoint_processor=False
        )
        self.model = context["model"]
        self.processor = context["processor"]
        self.tokenizer = context["tokenizer"]
        self.collator = context["collator"]
        self.query_token_ids = context["query_token_ids"]
        self.merge_size = context["merge_size"]
        self._memory_token_id = int(
            self.tokenizer.convert_tokens_to_ids(self._memory_token)
        )
        self.collator.memory_token_id = self._memory_token_id
        self._online_cls_memory: Dict[int, torch.Tensor] = {}
        self._online_subtask_starts: List[int] = []
        self._online_step_images: Dict[int, Dict[str, Image.Image]] = {}
        self._online_memory_last_step: Optional[int] = None
        self._online_memory_cached_step: Optional[int] = None

    @staticmethod
    def _install_tokenizer_fast_fallback(robotwin_eval_utils) -> None:
        if getattr(robotwin_eval_utils, "_fastwam_fast_tokenizer_fallback", False):
            return
        original_loader = robotwin_eval_utils.load_robotwin_tokenizer

        def load_robotwin_tokenizer(args):
            try:
                return original_loader(args)
            except TypeError as exc:
                if "os.PathLike" not in str(exc) and "vocab_file" not in str(exc):
                    raise
            from transformers import AutoTokenizer

            kwargs = {
                "model_max_length": args.model_max_length,
                "padding_side": "right",
                "use_fast": True,
            }
            source = args.checkpoint if Path(args.checkpoint).exists() else args.base_model
            try:
                tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(args.base_model, **kwargs)
            special_tokens = robotwin_eval_utils.robotwin_special_tokens(
                voting_done=bool(getattr(args, "voting_done", False)),
                done_vote_count=int(getattr(args, "done_vote_count", 5)),
            )
            config_path = Path(args.checkpoint) / "tokenizer_config.json"
            if config_path.exists():
                with config_path.open("r", encoding="utf-8") as handle:
                    tokenizer_config = json.load(handle)
                special_tokens = tokenizer_config.get(
                    "extra_special_tokens", special_tokens
                )
                if getattr(args, "voting_done", False):
                    for token in robotwin_eval_utils.robotwin_special_tokens(
                        voting_done=True,
                        done_vote_count=int(getattr(args, "done_vote_count", 5)),
                    ):
                        if token not in special_tokens:
                            special_tokens.append(token)
            tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            return tokenizer

        robotwin_eval_utils.load_robotwin_tokenizer = load_robotwin_tokenizer
        robotwin_eval_utils._fastwam_fast_tokenizer_fallback = True

    def _images(self, observation: Dict[str, Any]) -> Dict[str, Image.Image]:
        obs = observation["observation"]
        main = _jpeg_rgb(obs["head_camera"]["rgb"])
        if main.size != self._main_image_size:
            main = main.resize(self._main_image_size, Image.Resampling.BICUBIC)
        wrist_size = (max(1, main.width // 2), max(1, main.height // 2))
        return {
            "main": main,
            "left_wrist": _jpeg_rgb(obs["left_camera"]["rgb"]).resize(
                wrist_size, Image.Resampling.BICUBIC
            ),
            "right_wrist": _jpeg_rgb(obs["right_camera"]["rgb"]).resize(
                wrist_size, Image.Resampling.BICUBIC
            ),
        }

    def _image_content(self, observations):
        if isinstance(observations, dict):
            return self._images(observations)
        expected_count = self.expected_memory_frames + 1
        if self.memory_strategy == "image_window" and len(observations) != expected_count:
            raise ValueError(
                "VLM history length does not match training: "
                f"expected {self.expected_memory_frames} memory frames plus current "
                f"({expected_count} total), got {len(observations)}."
            )
        if self.memory_strategy == "subtask_cls_recent_images" and not (
            1 <= len(observations) <= expected_count
        ):
            raise ValueError(
                "VLM recent-memory history must contain current plus at most "
                f"{self.expected_memory_frames} history frames, got {len(observations)}."
            )
        result = []
        for index, (step, observation) in enumerate(observations):
            images = self._images(observation)
            if index != len(observations) - 1:
                images = {view: images[view] for view in self._memory_views}
            result.append((int(step), images))
        if not result:
            raise ValueError("VLM observation history is empty.")
        return result

    @staticmethod
    def _observation_sequence(observations):
        if isinstance(observations, VLMObservationBatch):
            return [(int(step), observation) for step, observation in observations.prompt]
        if isinstance(observations, dict):
            return [(0, observations)]
        return [(int(step), observation) for step, observation in observations]

    @staticmethod
    def _cache_observation_sequence(observations):
        if isinstance(observations, VLMObservationBatch):
            return [(int(step), observation) for step, observation in observations.cache]
        return RobotWinVLMSubtaskMonitor._observation_sequence(observations)

    def _reset_online_memory(self) -> None:
        self._online_cls_memory.clear()
        self._online_subtask_starts.clear()
        self._online_step_images.clear()
        self._online_memory_last_step = None
        self._online_memory_cached_step = None

    @torch.inference_mode()
    def _cache_online_cls_memory(self, observations) -> None:
        if self.memory_strategy != "subtask_cls_recent_images":
            return
        sequence = self._cache_observation_sequence(observations)
        unique = {}
        for step, observation in sequence:
            unique[int(step)] = observation
        pending = [
            (step, observation)
            for step, observation in unique.items()
            if step not in self._online_cls_memory
        ]
        if not pending:
            return

        for start in range(0, len(pending), 8):
            frame_batch = pending[start : start + 8]
            flat_images = []
            for _, observation in frame_batch:
                images = self._images(observation)
                flat_images.extend(images[view] for view in self._memory_views)
            processed = self.processor.image_processor(
                images=flat_images,
                return_tensors="pt",
                min_pixels=self._min_pixels,
                max_pixels=self._max_pixels,
            )
            pixel_values = processed["pixel_values"].to(self.device)
            image_grid_thw = processed["image_grid_thw"].to(self.device)
            image_embeds, _ = self.model.get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
            pooled = [
                torch.nn.functional.layer_norm(
                    embed.mean(dim=0), (embed.shape[-1],)
                )
                for embed in image_embeds
            ]
            tokens = torch.stack(pooled).reshape(
                len(frame_batch), len(self._memory_views), -1
            )
            tokens = tokens.detach().to(device="cpu", dtype=torch.bfloat16)
            for row, (step, _) in enumerate(frame_batch):
                self._online_cls_memory[int(step)] = tokens[row].contiguous()

    @staticmethod
    def _prompt_subtask(item: Dict[str, Any], fallback_index: int) -> Dict[str, Any]:
        return {
            "subtask_index": int(item.get("subtask_index", fallback_index)),
            "subtask_goal": str(item.get("subtask_goal", "")),
        }

    def _online_memory_prompt(
        self,
        observations,
        completed_subtasks,
        current_subtask: Optional[Dict[str, Any]] = None,
        *,
        reset_memory: bool = False,
    ) -> Dict[str, Any]:
        images = self._image_content(observations)
        sequence = self._observation_sequence(observations)
        if not sequence:
            raise ValueError("VLM observation history is empty.")
        current_step = int(sequence[-1][0])
        reference_step = current_step - (
            current_step % self.current_subtask_memory_stride
        )
        if reset_memory:
            self._reset_online_memory()
        self._cache_online_cls_memory(observations)
        cache_sequence = self._cache_observation_sequence(observations)
        if cache_sequence:
            cached_step = max(step for step, _ in cache_sequence)
            self._online_memory_cached_step = max(
                cached_step,
                self._online_memory_cached_step
                if self._online_memory_cached_step is not None
                else cached_step,
            )
        self._online_memory_last_step = max(
            current_step,
            self._online_memory_last_step
            if self._online_memory_last_step is not None
            else current_step,
        )

        result = {
            "images": images,
            "step_images": [],
            "step_memory_frame_indices": [],
            "memory_frame_indices": None,
            "memory_embeddings": None,
            "completed_memory_frame_counts": {},
            "active_memory_frame_count": 0,
            "image_memory_use_offsets": self.memory_strategy
            == "subtask_cls_recent_images",
            "current_frame_index": current_step,
        }
        if self.memory_strategy != "subtask_cls_recent_images":
            return result

        completed = [
            self._prompt_subtask(item, index)
            for index, item in enumerate(completed_subtasks)
        ]
        active = self._prompt_subtask(
            current_subtask or {},
            len(completed),
        )
        visible_completed_count = len(completed)
        required_starts = visible_completed_count + 1
        if not self._online_subtask_starts:
            self._online_subtask_starts.append(0)
            initial = next(
                (observation for step, observation in sequence if int(step) == 0),
                None,
            )
            if initial is not None:
                images = self._images(initial)
                self._online_step_images[0] = {
                    view: images[view] for view in self._memory_views
                }
        while len(self._online_subtask_starts) < required_starts:
            self._online_subtask_starts.append(reference_step)
            source = next(
                (
                    observation
                    for step, observation in reversed(
                        [*self._cache_observation_sequence(observations), *sequence]
                    )
                    if int(step) == reference_step
                ),
                None,
            )
            if source is None:
                raise ValueError(
                    "Online VLM memory cannot find the subtask-start observation "
                    f"at frame {reference_step}."
                )
            images = self._images(source)
            self._online_step_images[reference_step] = {
                view: images[view] for view in self._memory_views
            }
        # Q1 and Q2 share this monotonic memory state. An older asynchronous Q2
        # snapshot may finish after Q1 has advanced it; read that snapshot's
        # prefix without rolling back the current episode state.
        visible_subtask_starts = self._online_subtask_starts[:required_starts]

        prompt_subtasks = [*completed, active]
        timed_subtasks = []
        for index, subtask in enumerate(prompt_subtasks):
            start = int(visible_subtask_starts[index])
            end = (
                int(visible_subtask_starts[index + 1]) - 1
                if index + 1 < len(visible_subtask_starts)
                else reference_step
            )
            timed_subtasks.append({**subtask, "start_frame": start, "end_frame": end})

        (
            memory_frame_indices,
            step_memory_frame_indices,
            _,
        ) = self._subtask_memory_frame_indices(
            reference_step,
            visible_completed_count,
            timed_subtasks,
            self.completed_subtask_memory_stride,
            self.current_subtask_memory_stride,
            self.expected_memory_frames,
        )
        missing = [
            frame
            for frame in memory_frame_indices
            if frame not in self._online_cls_memory
        ]
        if missing:
            logger.warning(
                "Online VLM CLS memory is missing sampled frames %s at step %d; "
                "using the available subset.",
                missing,
                current_step,
            )
        memory_frame_indices = [
            frame
            for frame in memory_frame_indices
            if frame in self._online_cls_memory
        ]
        if memory_frame_indices:
            result["memory_embeddings"] = torch.cat(
                [self._online_cls_memory[frame] for frame in memory_frame_indices],
                dim=0,
            )
            completed_counts, active_count = self._group_memory_frames_for_prompt(
                memory_frame_indices,
                timed_subtasks,
                visible_completed_count,
            )
            result["memory_frame_indices"] = memory_frame_indices
            result["completed_memory_frame_counts"] = completed_counts
            result["active_memory_frame_count"] = active_count
        missing_steps = [
            frame
            for frame in step_memory_frame_indices
            if frame not in self._online_step_images
        ]
        if missing_steps:
            raise ValueError(
                "Online VLM step memory is missing full-image frames: "
                f"{missing_steps}"
            )
        result["step_memory_frame_indices"] = step_memory_frame_indices
        result["step_images"] = [
            (frame, self._online_step_images[frame])
            for frame in step_memory_frame_indices
        ]
        return result

    def _attach_generation_memory(
        self,
        prompt: Dict[str, Any],
        memory_embeddings: Optional[torch.Tensor],
    ) -> None:
        if memory_embeddings is None:
            return
        positions = torch.nonzero(
            prompt["input_ids"][0].eq(self._memory_token_id),
            as_tuple=False,
        ).flatten()
        if positions.numel() != memory_embeddings.shape[0]:
            raise ValueError(
                "VLM Q1 memory placeholder count does not match online CLS embeddings: "
                f"{positions.numel()} vs {memory_embeddings.shape[0]}."
            )
        prompt["robotwin_memory_token_pos"] = positions.unsqueeze(0)
        prompt["robotwin_memory_embeddings"] = memory_embeddings.unsqueeze(0)

    @staticmethod
    def _latest(observations):
        return observations if isinstance(observations, dict) else observations[-1][1]

    @staticmethod
    def _state_values(observation: Dict[str, Any]):
        vector = (observation.get("joint_action") or {}).get("vector")
        if vector is not None:
            state = np.asarray(vector, dtype=np.float32).reshape(-1)
            if state.shape[0] == 14:
                return tuple(float(value) for value in state)
        return None

    @staticmethod
    def _normalize_messages(messages):
        result = []
        for message in messages:
            item = dict(message)
            if isinstance(item.get("content"), str):
                item["content"] = [{"type": "text", "text": item["content"]}]
            result.append(item)
        return result

    def plan_subtasks(
        self, task_goal: str, observations, completed_subtasks=(), *, strict=False,
        allow_empty=False,
    ):
        with self._lock:
            reset_memory = bool(
                isinstance(observations, VLMObservationBatch)
                and observations.reset_memory
            )
            if self.fixed_subtasks is not None:
                if reset_memory:
                    self._reset_online_memory()
                return normalize_subtasks(task_goal, self.fixed_subtasks)
            if not self.plan_enabled:
                if reset_memory:
                    self._reset_online_memory()
                return normalize_subtasks(task_goal, None)
            completed = [
                self._prompt_subtask(item, index)
                for index, item in enumerate(completed_subtasks)
            ]
            memory = self._online_memory_prompt(
                observations,
                completed,
                reset_memory=(
                    reset_memory
                    or (
                        not completed
                        and self._observation_sequence(observations)[-1][0] == 0
                    )
                ),
            )
            content = self._user_content(
                task_goal,
                completed,
                memory["images"],
                self._views,
                memory_views=self._memory_views,
                memory_frame_indices=memory["memory_frame_indices"],
                completed_memory_frame_counts=memory["completed_memory_frame_counts"],
                active_memory_frame_count=memory["active_memory_frame_count"],
                image_memory_use_offsets=memory["image_memory_use_offsets"],
                current_frame_index=memory["current_frame_index"],
                step_images=memory["step_images"],
            )
            messages = self._normalize_messages(self._messages_for_sample(
                self._system_prompt(self._q1_system_prompt, self._views), content
            ))
            prompt = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            )
            self._attach_generation_memory(
                prompt,
                memory["memory_embeddings"],
            )
            prompt = self._move_to_device(prompt, self.device)
            input_len = prompt["input_ids"].shape[1]
            with torch.inference_mode():
                generated = self.model.generate(
                    **prompt, max_new_tokens=self.max_new_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    robotwin_mode="q1",
                )
            raw = self.tokenizer.decode(
                generated[0, input_len:], skip_special_tokens=True
            ).strip()
            self.last_q1_raw_output = raw
            return parse_q1_subtasks_output(
                task_goal, raw, strict=strict, allow_empty=allow_empty
            )

    def _q2_batch(self, task_goal, completed_subtasks, current_subtask, observations):
        completed = [
            self._prompt_subtask(item, index)
            for index, item in enumerate(completed_subtasks)
        ]
        current = self._prompt_subtask(current_subtask, len(completed))
        memory = self._online_memory_prompt(
            observations,
            completed,
            current,
        )
        content = self._user_content(
            task_goal, completed, memory["images"], self._views,
            memory_views=self._memory_views,
            current_goal=current["subtask_goal"],
            include_query_tokens=self.failure_query_enabled,
            current_subtask_index=current["subtask_index"],
            state_values=(
                self._state_values(self._latest(observations))
                if self.q2_state_enabled
                else None
            ),
            memory_frame_indices=memory["memory_frame_indices"],
            completed_memory_frame_counts=memory["completed_memory_frame_counts"],
            active_memory_frame_count=memory["active_memory_frame_count"],
            image_memory_use_offsets=memory["image_memory_use_offsets"],
            current_frame_index=memory["current_frame_index"],
            step_images=memory["step_images"],
        )
        if not self.failure_query_enabled:
            content.append({"type": "text", "text": self._done_query_token})
        messages = self._normalize_messages(self._messages_for_sample(
            self._system_prompt(self._q2_system_prompt, self._views), content
        ))
        item = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
        input_ids = item["input_ids"]
        item["labels"] = torch.full_like(input_ids, self._ignore_index)
        grids = item.get("image_grid_thw")
        if grids is not None and not isinstance(grids, (list, tuple)):
            grids = [grids]
        item["position_ids"], _ = self._get_rope_index(
            self.merge_size, input_ids,
            image_grid_thw=torch.cat(grids, dim=0) if grids else None,
        )
        for key in ("robotwin_current_done", "robotwin_current_failure"):
            item[key] = torch.tensor(-100.0, dtype=torch.float32)
        if memory["memory_embeddings"] is not None:
            item["robotwin_memory_embeddings"] = memory["memory_embeddings"]
        batch = self.collator([item])
        if batch["robotwin_done_query_pos"].lt(0).any():
            raise ValueError("VLM Q2 prompt was truncated; increase vlm_model_max_length.")
        if not self.failure_query_enabled:
            batch.pop("robotwin_failure_query_pos", None)
            batch.pop("robotwin_current_failure", None)
        if self.voting_done:
            vote_id = int(self.query_token_ids[f"done_vote_{self.done_vote_head}"])
            positions = batch["robotwin_done_query_pos"]
            valid = positions.ge(0)
            rows = torch.arange(batch["input_ids"].shape[0])[valid]
            batch["input_ids"][rows, positions[valid].long()] = vote_id
            batch["robotwin_done_vote_index"] = torch.full(
                (batch["input_ids"].shape[0],), self.done_vote_head, dtype=torch.long
            )
        return self._move_to_device(batch, self.device)

    def check_current_done(
        self, task_goal: str, completed_subtasks, current_subtask, observation
    ) -> Dict[str, Any]:
        with self._lock, torch.inference_mode():
            outputs = self.model(**self._q2_batch(
                task_goal, completed_subtasks, current_subtask, observation
            ))
        probability = torch.sigmoid(
            outputs.robotwin_logits["current_done"]
        ).detach().float().cpu()[0].item()
        return {"done": probability >= self.done_threshold,
                "done_prob": float(probability),
                "threshold": self.done_threshold,
                "memory_cached_step": self._online_memory_cached_step}
