#!/usr/bin/env python3
"""JSON WebSocket server for the LingBot-VA Piper real-robot policy."""

from __future__ import annotations

import argparse
import asyncio
import base64
from collections.abc import Mapping
import io
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

from lingbotva_piper_runtime import (
    LingBotVAPiperRuntime,
    MODEL_ACTION_HORIZON,
    PiperSafetyLimits,
    actions_to_piper_wire,
    piper_state_to_vector,
    project_and_limit_piper_actions,
)


LOG = logging.getLogger(__name__)
REQUIRED_CAMERA_NAMES = ("head", "left_wrist", "right_wrist")


def build_policy_response(
    request_id: str, step: int, action: Mapping[str, Any], error: str | None = None
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "ok": error is None,
        "request_id": request_id,
        "step": int(step),
        "action": dict(action),
    }
    if error is not None:
        response["error"] = str(error)
    return response


def _load_binary_blob(value: Any) -> bytes:
    if isinstance(value, str):
        encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
        try:
            return base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("image data is not valid base64") from exc
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, list):
        try:
            return bytes(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("binary image byte list is invalid") from exc
    raise ValueError(f"cannot decode image data from {type(value).__name__}")


def _decode_compressed_rgb(raw: bytes) -> np.ndarray:
    if not raw:
        raise ValueError("compressed image data is empty")
    try:
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as image:
            return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
    except Exception as exc:
        raise ValueError("failed to decode compressed RGB image") from exc


def decode_rgb_image(payload: Any, *, input_color: str) -> np.ndarray:
    """Decode a JSON image payload into an HWC uint8 RGB array."""

    if input_color not in {"rgb", "bgr"}:
        raise ValueError("input_color must be 'rgb' or 'bgr'")
    compressed = False
    if isinstance(payload, Mapping):
        if "image" in payload and not any(key in payload for key in ("data", "array")):
            return decode_rgb_image(payload["image"], input_color=input_color)
        if "array" in payload:
            image = np.asarray(payload["array"])
        elif "data" in payload:
            encoding = str(payload.get("encoding", "jpeg")).lower()
            if encoding not in {"jpeg", "jpg", "png", "compressed"}:
                raise ValueError(f"unsupported image encoding {encoding!r}")
            image = _decode_compressed_rgb(_load_binary_blob(payload["data"]))
            compressed = True
        else:
            raise ValueError("image object must contain 'data' or 'array'")
    elif isinstance(payload, str):
        image = _decode_compressed_rgb(_load_binary_blob(payload))
        compressed = True
    elif isinstance(payload, list):
        image = np.asarray(payload)
    else:
        raise ValueError(f"unsupported image payload type {type(payload).__name__}")

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"expected HWC image with 3 or 4 channels, got shape={image.shape}")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("image dimensions must be positive")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if not np.issubdtype(image.dtype, np.number) or not np.all(np.isfinite(image)):
        raise ValueError("image pixels must be finite numeric values")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if input_color == "bgr" and not compressed:
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image)


def decode_required_rgb_images(
    images: Any, *, input_color: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(images, Mapping):
        raise ValueError("request.images must be a JSON object")
    missing = [name for name in REQUIRED_CAMERA_NAMES if name not in images]
    if missing:
        raise ValueError("missing required image(s): " + ", ".join(missing))
    return tuple(
        decode_rgb_image(images[name], input_color=input_color)
        for name in REQUIRED_CAMERA_NAMES
    )  # type: ignore[return-value]


def decode_history_images(
    payload: Any, *, input_color: str
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("history_images must be a non-empty list")
    return [
        decode_required_rgb_images(frame, input_color=input_color) for frame in payload
    ]


def _positive_int(config: Mapping[str, Any], key: str, *, default: int | None = None) -> int:
    value = config.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a positive integer")
    try:
        value = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{key} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def safety_limits_from_config(config: Mapping[str, Any]) -> PiperSafetyLimits:
    def vector(key: str) -> np.ndarray:
        values = config.get(key)
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"{key} must contain 14 verified values")
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (14,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{key} must contain exactly 14 finite numeric values")
        return result

    lower = vector("joint_lower")
    upper = vector("joint_upper")
    max_delta = vector("max_delta")
    if not np.all(lower < upper):
        raise ValueError("joint_lower must be strictly less than joint_upper")
    if not np.all(max_delta > 0):
        raise ValueError("max_delta must be strictly positive")
    return PiperSafetyLimits(lower=lower, upper=upper, max_delta=max_delta)


class PiperLingBotVAServer:
    """Protocol and safety adapter around one stateful LingBot-VA runtime."""

    def __init__(self, *, runtime: Any, config: Mapping[str, Any]) -> None:
        self.runtime = runtime
        self.config = dict(config)
        self.input_color = str(self.config.get("input_color", "bgr")).lower()
        self.action_dt = float(self.config.get("action_dt", 1 / 30))
        self.max_inference_latency_s = float(
            self.config.get("max_inference_latency_s", 20.0)
        )
        if self.input_color not in {"rgb", "bgr"}:
            raise ValueError("input_color must be 'rgb' or 'bgr'")
        if not np.isfinite(self.action_dt) or self.action_dt <= 0:
            raise ValueError("action_dt must be a positive finite number")
        if not np.isfinite(self.max_inference_latency_s) or self.max_inference_latency_s <= 0:
            raise ValueError("max_inference_latency_s must be a positive finite number")
        self.execution_horizon = _positive_int(
            self.config, "execution_horizon", default=MODEL_ACTION_HORIZON
        )
        if self.execution_horizon != MODEL_ACTION_HORIZON:
            raise ValueError(
                "server execution_horizon is fixed at 32; use the robot client's "
                "--execution-horizon for action truncation"
            )
        self.safety_limits = safety_limits_from_config(self.config)
        self._lock = asyncio.Lock()

    async def handler(self, websocket: Any, *_unused: Any) -> None:
        peer = getattr(websocket, "remote_address", None)
        LOG.info("LingBot-VA client connected: %s", peer)
        try:
            async for message in websocket:
                response = await self.handle_message(message)
                await websocket.send(json.dumps(response, ensure_ascii=False))
        except Exception:
            LOG.exception("connection error with %s", peer)
        finally:
            reset = getattr(self.runtime, "reset_connection", None)
            if callable(reset):
                try:
                    reset()
                except Exception:
                    LOG.exception("failed to reset LingBot-VA runtime after disconnect")
            LOG.info("LingBot-VA client disconnected: %s", peer)

    async def handle_message(self, message: str) -> dict[str, Any]:
        request_id = "unknown"
        step = -1
        started = time.monotonic()
        try:
            request = json.loads(message)
            if not isinstance(request, Mapping):
                raise ValueError("websocket message must be a JSON object")
            request_id = str(request.get("request_id", request_id))
            step = int(request.get("step", step))
            instruction = str(request.get("instruction", self.config.get("instruction", ""))).strip()
            if not instruction:
                raise ValueError("instruction must be a non-empty string")

            async with self._lock:
                initial_request = bool(request.get("reset", False)) or not bool(
                    getattr(self.runtime, "initialized", False)
                )
                if initial_request:
                    self.runtime.reset(instruction)
                head, left, right = decode_required_rgb_images(
                    request.get("images"), input_color=self.input_color
                )
                state = request.get("state")
                current = piper_state_to_vector(state)
                history_payload = request.get("history_images")
                if history_payload is not None:
                    history_images = decode_history_images(
                        history_payload, input_color=self.input_color
                    )
                else:
                    expected = getattr(self.runtime, "expected_cache_frames", lambda: 0)()
                    history_images = [
                        (head, left, right) for _ in range(int(expected))
                    ] or None
                prediction = self.runtime.infer(
                    head=head,
                    left=left,
                    right=right,
                    state=state,
                    history_images=history_images,
                )
                response_prediction = prediction[: self.execution_horizon]
                if initial_request and len(response_prediction) > 16:
                    # The first 16 points are the model's conditioning frame;
                    # the robot client deliberately skips them. Start safety
                    # rate limiting at the actual current robot state for the
                    # executed second frame.
                    safe_tail, summary = project_and_limit_piper_actions(
                        response_prediction[16:], current, self.safety_limits
                    )
                    safe_prediction = np.concatenate(
                        (np.repeat(current[None, :], 16, axis=0), safe_tail), axis=0
                    )
                else:
                    safe_prediction, summary = project_and_limit_piper_actions(
                        response_prediction, current, self.safety_limits
                    )
                if summary.projected_values:
                    LOG.warning(
                        "projected %d LingBot-VA values; max_projection=%.6f",
                        summary.projected_values,
                        summary.max_projection,
                    )
                elapsed_s = time.monotonic() - started
                if elapsed_s > self.max_inference_latency_s:
                    raise TimeoutError(
                        "inference latency {:.3f}s exceeded configured limit {:.3f}s".format(
                            elapsed_s, self.max_inference_latency_s
                        )
                    )
                action = actions_to_piper_wire(safe_prediction, dt=self.action_dt)
                LOG.info(
                    "step=%s request_id=%s latency_ms=%.1f action_horizon=%s",
                    step,
                    request_id,
                    elapsed_s * 1000,
                    self.execution_horizon,
                )
                return build_policy_response(request_id, step, action)
        except Exception as exc:
            LOG.exception("failed to process request_id=%s step=%s", request_id, step)
            return build_policy_response(request_id, step, {}, error=str(exc))


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load LingBot-VA config") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ValueError(f"config {path} must contain a YAML object")
    return dict(config)


async def serve(server: PiperLingBotVAServer) -> None:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("websockets is required to serve LingBot-VA requests") from exc
    host = str(server.config.get("host", "127.0.0.1"))
    port = _positive_int(server.config, "port", default=7084)
    max_size_mb = _positive_int(server.config, "max_message_size_mb", default=64)
    async with websockets.serve(
        server.handler,
        host,
        port,
        max_size=max_size_mb * 1024 * 1024,
        ping_interval=None,
        ping_timeout=None,
    ):
        LOG.info("LingBot-VA Piper policy server listening on ws://%s:%s", host, port)
        await asyncio.Future()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LingBot-VA Piper real-robot policy server")
    parser.add_argument("--config", required=True)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="load the checkpoint and exit without opening a WebSocket listener",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    for key in ("host", "port", "device"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    runtime = LingBotVAPiperRuntime.from_config(config)
    if args.preflight_only:
        LOG.info("LingBot-VA checkpoint preflight succeeded")
        return
    asyncio.run(serve(PiperLingBotVAServer(runtime=runtime, config=config)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.info("LingBot-VA Piper policy server stopped")
