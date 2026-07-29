#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JSON websocket server that adapts the fixed Piper client to an OpenPI policy.

The provided Piper client sends JSON requests and expects JSON responses shaped as:
    {"ok": true, "request_id": "...", "step": 0, "action": {...}}

OpenPI's official server uses a different msgpack websocket protocol, so this wrapper
keeps the Piper client unchanged while loading a local OpenPI checkpoint directly.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from collections.abc import Mapping
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
import websockets


DEFAULT_OPENPI_ROOT = "/media/section/iclr_proj/zbh/openpi"
DEFAULT_CONFIG = "pi05_s40_3"
DEFAULT_CHECKPOINT = (
    "/media/section/iclr_proj/zbh/openpi/training_runs/pi05_s40_3/checkpoints/"
    "pi05_s40_3/s40_3_pi05_8gpu_50k_fixed_indices/49999"
)

OPENPI_IMAGE_KEYS = ("image", "wrist_image_left", "wrist_image_right")
DEFAULT_IMAGE_ALIASES = {
    "image": (
        "image",
        "cam_high",
        "head",
        "head_camera",
        "camera_head",
        "top",
        "top_camera",
        "front",
        "front_camera",
        "base",
        "base_0_rgb",
    ),
    "wrist_image_left": (
        "wrist_image_left",
        "cam_left_wrist",
        "left_wrist",
        "left_wrist_camera",
        "camera_left_wrist",
        "left",
        "left_wrist_0_rgb",
    ),
    "wrist_image_right": (
        "wrist_image_right",
        "cam_right_wrist",
        "right_wrist",
        "right_wrist_camera",
        "camera_right_wrist",
        "right",
        "right_wrist_0_rgb",
    ),
}


def build_policy_response(request_id: str, step: int, action: dict[str, Any], error: str | None = None) -> dict[str, Any]:
    response: dict[str, Any] = {
        "ok": error is None,
        "request_id": request_id,
        "step": step,
        "action": action,
    }
    if error is not None:
        response["error"] = error
    return response


def _load_binary_blob(value: Any) -> bytes:
    if isinstance(value, str):
        data = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
        return base64.b64decode(data)
    if isinstance(value, list):
        return bytes(value)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    raise TypeError(f"cannot decode binary blob from {type(value).__name__}")


def _decode_compressed_image(raw: bytes, *, input_color: str) -> np.ndarray:
    try:
        import cv2

        arr = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cv2.imdecode returned None")
        if input_color == "bgr":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception:
        from PIL import Image

        image = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        if input_color == "bgr":
            # PIL already returns RGB for compressed images.
            return image
        return image


def decode_image_payload(payload: Any, *, input_color: str) -> np.ndarray:
    """Decode a flexible JSON image payload into HWC uint8 RGB."""
    already_rgb = False
    if isinstance(payload, Mapping):
        if "image" in payload and "data" not in payload:
            return decode_image_payload(payload["image"], input_color=input_color)
        if "array" in payload:
            image = np.asarray(payload["array"])
        elif "data" in payload:
            encoding = str(payload.get("encoding", payload.get("format", "jpeg"))).lower()
            raw = _load_binary_blob(payload["data"])
            if encoding in {"jpeg", "jpg", "png", "compressed", "bgr8; jpeg compressed bgr8"}:
                image = _decode_compressed_image(raw, input_color=input_color)
                already_rgb = True
            elif "shape" in payload:
                dtype = np.dtype(payload.get("dtype", "uint8"))
                image = np.frombuffer(raw, dtype=dtype).reshape(tuple(payload["shape"]))
            else:
                image = _decode_compressed_image(raw, input_color=input_color)
                already_rgb = True
        else:
            raise ValueError(f"unsupported image object keys: {sorted(payload.keys())}")
    elif isinstance(payload, str):
        image = _decode_compressed_image(_load_binary_blob(payload), input_color=input_color)
        already_rgb = True
    else:
        image = np.asarray(payload)

    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"expected HWC image with 3 or 4 channels, got shape={image.shape}")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if input_color == "bgr" and not already_rgb:
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image)


def hwc_rgb_to_chw_uint8(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(image, (2, 0, 1)).astype(np.uint8, copy=False))


def parse_image_map(items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--image-map must look like openpi_key=client_key")
        openpi_key, client_key = item.split("=", 1)
        if openpi_key not in OPENPI_IMAGE_KEYS:
            raise ValueError(f"unsupported OpenPI image key {openpi_key!r}; expected one of {OPENPI_IMAGE_KEYS}")
        mapping[openpi_key] = client_key
    return mapping


def find_client_image(images: Mapping[str, Any], openpi_key: str, explicit_map: Mapping[str, str]) -> Any | None:
    if openpi_key in explicit_map:
        return images.get(explicit_map[openpi_key])
    for alias in DEFAULT_IMAGE_ALIASES[openpi_key]:
        if alias in images:
            return images[alias]
    return None


def normalize_state(raw_state: Mapping[str, Any], *, left_dim: int, right_dim: int) -> np.ndarray:
    def arm_vector(name: str, dim: int) -> list[float]:
        values = raw_state.get(name)
        if values is None:
            raise ValueError(f"state.{name} is required")
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"state.{name} must be a list, got {type(values).__name__}")
        vector = [float(value) for value in values]
        if len(vector) == dim:
            return vector
        gripper_name = name.replace("_arm", "_gripper")
        if len(vector) == dim - 1 and gripper_name in raw_state:
            return vector + [float(raw_state[gripper_name])]
        raise ValueError(f"state.{name} must have {dim} values, got {len(vector)}")

    left = arm_vector("left_arm", left_dim)
    right = arm_vector("right_arm", right_dim)
    state = np.asarray(left + right, dtype=np.float32)
    if state.shape != (left_dim + right_dim,):
        raise ValueError(f"expected state shape {(left_dim + right_dim,)}, got {state.shape}")
    return state


def request_to_openpi_observation(
    request: Mapping[str, Any],
    *,
    image_map: Mapping[str, str],
    input_color: str,
    allow_missing_images: bool,
    state_left_dim: int,
    state_right_dim: int,
    decoded_images: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    raw_images = request.get("images", {})
    if not isinstance(raw_images, Mapping):
        raise ValueError("request.images must be an object")

    obs: dict[str, Any] = {
        "state": normalize_state(request.get("state", {}), left_dim=state_left_dim, right_dim=state_right_dim),
    }

    for openpi_key in OPENPI_IMAGE_KEYS:
        payload = find_client_image(raw_images, openpi_key, image_map)
        if payload is None:
            if not allow_missing_images:
                raise ValueError(
                    f"missing image for {openpi_key}; got client image keys={sorted(raw_images.keys())}. "
                    "Use --image-map openpi_key=client_key if the names differ."
                )
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = decode_image_payload(payload, input_color=input_color)
        if decoded_images is not None:
            decoded_images[openpi_key] = image
        obs[openpi_key] = hwc_rgb_to_chw_uint8(image)

    instruction = request.get("instruction")
    if instruction:
        obs["prompt"] = np.asarray(str(instruction))
    return obs


def _safe_path_component(value: str, *, max_len: int = 80) -> str:
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in value)
    return (safe or "unknown")[:max_len]


def save_received_images(
    *,
    save_dir: str,
    step: int,
    request_id: str,
    request: Mapping[str, Any],
    obs: Mapping[str, Any],
    decoded_images: Mapping[str, np.ndarray],
) -> Path:
    from PIL import Image

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    step_dir = (
        Path(save_dir)
        / f"step_{step:06d}_{timestamp}_{_safe_path_component(request_id)}"
    )
    step_dir.mkdir(parents=True, exist_ok=True)

    image_shapes: dict[str, list[int]] = {}
    for name, image in decoded_images.items():
        image_uint8 = np.asarray(image)
        if image_uint8.dtype != np.uint8:
            image_uint8 = np.clip(image_uint8, 0, 255).astype(np.uint8)
        image_shapes[name] = list(image_uint8.shape)
        Image.fromarray(image_uint8).save(step_dir / f"{name}.png")

    raw_images = request.get("images", {})
    metadata = {
        "request_id": request_id,
        "step": step,
        "saved_at_unix": time.time(),
        "client_image_keys": sorted(raw_images.keys()) if isinstance(raw_images, Mapping) else [],
        "openpi_image_keys": sorted(decoded_images.keys()),
        "openpi_image_shapes": image_shapes,
        "state_shape": list(np.asarray(obs["state"]).shape),
        "instruction": request.get("instruction"),
    }
    (step_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return step_dir


def clip_action_chunk(
    actions: np.ndarray,
    current_state: np.ndarray,
    *,
    max_joint_step: float,
) -> np.ndarray:
    if max_joint_step <= 0:
        return actions
    clipped = np.array(actions, dtype=np.float32, copy=True)
    joint_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    previous = current_state.astype(np.float32, copy=True)
    for row in clipped:
        low = previous[joint_indices] - max_joint_step
        high = previous[joint_indices] + max_joint_step
        row[joint_indices] = np.clip(row[joint_indices], low, high)
        previous = row
    return clipped


def openpi_actions_to_client_action(
    actions: np.ndarray,
    *,
    dt: float,
    horizon: int,
    left_arm_dim: int,
    right_arm_dim: int,
) -> dict[str, Any]:
    if actions.ndim != 2 or actions.shape[1] < 14:
        raise ValueError(f"expected OpenPI actions with shape [T, >=14], got {actions.shape}")
    chunk = np.asarray(actions[:horizon, :14], dtype=np.float32)
    if len(chunk) == 0:
        raise ValueError("policy returned an empty action chunk")

    action: dict[str, Any] = {
        "time_list": [float(dt * (index + 1)) for index in range(len(chunk))],
        "dt": float(dt),
    }

    if left_arm_dim == 7 and right_arm_dim == 7:
        action["left_arm"] = chunk[:, :7].tolist()
        action["right_arm"] = chunk[:, 7:14].tolist()
    elif left_arm_dim == 6 and right_arm_dim == 6:
        action["left_arm"] = chunk[:, :6].tolist()
        action["right_arm"] = chunk[:, 7:13].tolist()
        action["left_gripper"] = chunk[:, 6].tolist()
        action["right_gripper"] = chunk[:, 13].tolist()
    else:
        raise ValueError("only action dims (7,7) or (6,6) are supported")
    return action


class MockPolicy:
    def infer(self, obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
        state = np.asarray(obs["state"], dtype=np.float32)
        return {"actions": np.repeat(state[None, :], 10, axis=0)}


class OpenPIPolicy:
    def __init__(
        self,
        *,
        openpi_root: str,
        config_name: str,
        checkpoint_dir: str,
        num_infer_steps: int,
    ) -> None:
        root = Path(openpi_root).resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from openpi import transforms
        from openpi.policies import policy_config
        from openpi.training import config as openpi_config

        inference_repack = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "image",
                            "cam_left_wrist": "wrist_image_left",
                            "cam_right_wrist": "wrist_image_right",
                        },
                        "state": "state",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        train_config = openpi_config.get_config(config_name)
        self.policy = policy_config.create_trained_policy(
            train_config,
            checkpoint_dir,
            repack_transforms=inference_repack,
            sample_kwargs={"num_steps": num_infer_steps},
        )

    def infer(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        return self.policy.infer(dict(obs))


class PiperOpenPIServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.image_map = parse_image_map(args.image_map)
        if args.mock_policy:
            self.policy: Any = MockPolicy()
        else:
            self.policy = OpenPIPolicy(
                openpi_root=args.openpi_root,
                config_name=args.config,
                checkpoint_dir=args.checkpoint,
                num_infer_steps=args.num_infer_steps,
            )

    async def handler(self, websocket: Any) -> None:
        peer = getattr(websocket, "remote_address", None)
        print(f"[+] client connected: {peer}", flush=True)
        try:
            async for message in websocket:
                response = await self.handle_message(message)
                await websocket.send(json.dumps(response, ensure_ascii=False))
        except websockets.exceptions.ConnectionClosed:
            print(f"[-] client disconnected: {peer}", flush=True)
        except Exception as exc:
            print(f"[!] connection error with {peer}: {exc}", flush=True)
            traceback.print_exc()

    async def handle_message(self, message: str) -> dict[str, Any]:
        request_id = "unknown"
        step = -1
        try:
            raw_request = json.loads(message)
            if not isinstance(raw_request, Mapping):
                raise ValueError("websocket message must be a JSON object")
            request_id = str(raw_request.get("request_id", request_id))
            step = int(raw_request.get("step", step))

            start = time.monotonic()
            decoded_images: dict[str, np.ndarray] = {}
            obs = request_to_openpi_observation(
                raw_request,
                image_map=self.image_map,
                input_color=self.args.input_color,
                allow_missing_images=self.args.allow_missing_images,
                state_left_dim=self.args.state_left_dim,
                state_right_dim=self.args.state_right_dim,
                decoded_images=decoded_images,
            )
            saved_image_dir = None
            if self.args.save_received_images and step % self.args.save_image_every == 0:
                saved_image_dir = save_received_images(
                    save_dir=self.args.save_image_dir,
                    step=step,
                    request_id=request_id,
                    request=raw_request,
                    obs=obs,
                    decoded_images=decoded_images,
                )
            policy_result = self.policy.infer(obs)
            actions = np.asarray(policy_result["actions"], dtype=np.float32)
            actions = clip_action_chunk(actions, obs["state"], max_joint_step=self.args.max_joint_step)
            action = openpi_actions_to_client_action(
                actions,
                dt=self.args.action_dt,
                horizon=self.args.horizon,
                left_arm_dim=self.args.left_arm_dim,
                right_arm_dim=self.args.right_arm_dim,
            )

            elapsed_ms = (time.monotonic() - start) * 1000
            print(
                "[step {}] request_id={} infer_total_ms={:.1f} images={} state_shape={} saved_images={}".format(
                    step,
                    request_id,
                    elapsed_ms,
                    sorted(raw_request.get("images", {}).keys()) if isinstance(raw_request.get("images"), Mapping) else [],
                    tuple(obs["state"].shape),
                    saved_image_dir,
                ),
                flush=True,
            )
            return build_policy_response(request_id=request_id, step=step, action=action)
        except Exception as exc:
            print(f"[!] failed to process request {request_id} step={step}: {exc}", flush=True)
            traceback.print_exc()
            return build_policy_response(request_id=request_id, step=step, action={}, error=str(exc))


async def serve(args: argparse.Namespace) -> None:
    server = PiperOpenPIServer(args)
    async with websockets.serve(
        server.handler,
        args.host,
        args.port,
        max_size=args.max_size_mb * 1024 * 1024,
        ping_interval=20,
        ping_timeout=60,
    ):
        print(f"piper OpenPI policy server listening on ws://{args.host}:{args.port}", flush=True)
        await asyncio.Future()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Piper JSON websocket server backed by an OpenPI checkpoint")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7081)
    parser.add_argument("--openpi-root", default=os.environ.get("OPENPI_ROOT", DEFAULT_OPENPI_ROOT))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-infer-steps", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=10, help="number of returned waypoints")
    parser.add_argument("--action-dt", type=float, default=0.1, help="seconds between returned waypoints")
    parser.add_argument("--state-left-dim", type=int, default=7)
    parser.add_argument("--state-right-dim", type=int, default=7)
    parser.add_argument("--left-arm-dim", type=int, default=7, help="client response left_arm vector length")
    parser.add_argument("--right-arm-dim", type=int, default=7, help="client response right_arm vector length")
    parser.add_argument(
        "--image-map",
        action="append",
        default=[],
        help="repeatable openpi_key=client_key; openpi_key is image, wrist_image_left, or wrist_image_right",
    )
    parser.add_argument(
        "--input-color",
        choices=["bgr", "rgb"],
        default="bgr",
        help="color order for decoded client images before conversion to OpenPI RGB",
    )
    parser.add_argument("--allow-missing-images", action="store_true", help="fill missing images with black frames")
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=0.0,
        help="optional max radians per waypoint for joint dims; <=0 disables clipping",
    )
    parser.add_argument("--max-size-mb", type=int, default=64)
    parser.add_argument("--mock-policy", action="store_true", help="echo current state as actions without OpenPI")
    parser.add_argument(
        "--save-received-images",
        action="store_true",
        help="save decoded received images for each selected request",
    )
    parser.add_argument(
        "--save-image-dir",
        default="/media/section/iclr_proj/zbh/openpi/received_images",
        help="directory for --save-received-images output",
    )
    parser.add_argument(
        "--save-image-every",
        type=int,
        default=1,
        help="save one image set every N request steps when --save-received-images is enabled",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.action_dt <= 0:
        raise ValueError("--action-dt must be positive")
    if args.save_image_every <= 0:
        raise ValueError("--save-image-every must be positive")
    asyncio.run(serve(args))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\npolicy server stopped")
