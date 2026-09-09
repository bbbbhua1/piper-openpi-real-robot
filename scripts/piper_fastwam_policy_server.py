#!/usr/bin/env python3
"""Fail-closed JSON WebSocket policy server for the FastWAM Piper policy.

The robot-side client already speaks a small JSON protocol and expects Piper
``left_arm``/``right_arm`` action chunks.  FastWAM uses another state/action
ordering, so this wrapper deliberately owns only protocol decoding, safety, and
the call into :class:`FastWAMPiperRuntime`; it does not import ROS or publish a
single hardware command itself.
"""

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

from fastwam_piper_runtime import (
    PiperSafetyLimits,
    fastwam_to_piper_wire,
    piper_wire_to_fastwam,
    project_and_limit_fastwam_actions,
)
from motion_postprocessor import MotionPostprocessor

try:
    # The request adapter stays CPU-testable even while the model-loading
    # extension is unavailable. Real CLI startup below explicitly refuses in
    # that case, so this never weakens physical-inference safety.
    from fastwam_piper_runtime import FastWAMPiperRuntime
except ImportError:  # pragma: no cover - exercised by a later model-runtime task
    FastWAMPiperRuntime = None  # type: ignore[misc, assignment]


LOG = logging.getLogger(__name__)
REQUIRED_CAMERA_NAMES = ("head", "left_wrist", "right_wrist")
FASTWAM_ACTION_HORIZON = 32


def build_policy_response(
    request_id: str, step: int, action: Mapping[str, Any], error: str | None = None
) -> dict[str, Any]:
    """Preserve the existing Piper client response envelope.

    The original server only includes ``error`` for failed requests, which
    keeps successful responses byte-for-byte compatible with the existing
    client parser.
    """

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
    raise ValueError(f"cannot decode binary image data from {type(value).__name__}")


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
    """Decode a JSON image payload to a contiguous HWC uint8 RGB image.

    Compressed JPEG/PNG payloads are decoded by Pillow, whose ``convert('RGB')``
    result is already RGB.  ``input_color`` applies only to uncompressed arrays
    sent by a BGR ROS/OpenCV producer.
    """

    if input_color not in {"rgb", "bgr"}:
        raise ValueError("input_color must be 'rgb' or 'bgr'")

    is_compressed = False
    if isinstance(payload, Mapping):
        if "image" in payload and "data" not in payload and "array" not in payload:
            return decode_rgb_image(payload["image"], input_color=input_color)
        if "array" in payload:
            image = np.asarray(payload["array"])
        elif "data" in payload:
            encoding = str(payload.get("encoding", payload.get("format", "jpeg"))).lower()
            if encoding not in {
                "jpeg",
                "jpg",
                "png",
                "compressed",
                "bgr8; jpeg compressed bgr8",
                "rgb8; jpeg compressed rgb8",
            }:
                raise ValueError(f"unsupported compressed image encoding {encoding!r}")
            image = _decode_compressed_rgb(_load_binary_blob(payload["data"]))
            is_compressed = True
        else:
            raise ValueError("image object must contain 'data' or 'array'")
    elif isinstance(payload, str):
        image = _decode_compressed_rgb(_load_binary_blob(payload))
        is_compressed = True
    elif isinstance(payload, list):
        image = np.asarray(payload)
    else:
        raise ValueError(f"unsupported image payload type {type(payload).__name__}")

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"expected HWC image with 3 or 4 channels, got shape={image.shape}")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"image dimensions must be positive, got shape={image.shape}")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if not np.issubdtype(image.dtype, np.number) or not np.all(np.isfinite(image)):
        raise ValueError("image pixels must be finite numeric values")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if input_color == "bgr" and not is_compressed:
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image)


def decode_required_rgb_images(
    images: Any, *, input_color: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact three client camera streams, or reject the request."""

    if not isinstance(images, Mapping):
        raise ValueError("request.images must be a JSON object")
    missing = [name for name in REQUIRED_CAMERA_NAMES if name not in images]
    if missing:
        raise ValueError(
            "missing required image(s): {}; received: {}".format(
                ", ".join(missing), ", ".join(sorted(str(name) for name in images))
            )
        )
    return tuple(
        decode_rgb_image(images[name], input_color=input_color) for name in REQUIRED_CAMERA_NAMES
    )  # type: ignore[return-value]


def _safety_vector(config: Mapping[str, Any], key: str) -> np.ndarray:
    values = config.get(key)
    if not isinstance(values, (list, tuple)):
        raise ValueError(
            f"{key} is required for physical inference and must contain 14 verified values"
        )
    try:
        vector = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must contain 14 finite numeric values") from exc
    if vector.shape != (14,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{key} must contain exactly 14 finite numeric values")
    return vector


def safety_limits_from_config(config: Mapping[str, Any]) -> PiperSafetyLimits:
    """Construct validated limits; missing placeholder config always fails closed."""

    lower = _safety_vector(config, "joint_lower")
    upper = _safety_vector(config, "joint_upper")
    max_delta = _safety_vector(config, "max_delta")
    if not np.all(lower < upper):
        raise ValueError("joint_lower must be strictly less than joint_upper at all 14 positions")
    if not np.all(max_delta > 0.0):
        raise ValueError("max_delta must be strictly positive at all 14 positions")
    return PiperSafetyLimits(
        lower=lower,
        upper=upper,
        max_delta=max_delta,
    )


def _positive_int(config: Mapping[str, Any], key: str, *, default: int | None = None) -> int:
    raw = config.get(key, default)
    if isinstance(raw, bool):
        raise ValueError(f"{key} must be a positive integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


class PiperFastWAMServer:
    """Small, dependency-injectable adapter around one initialized FastWAM runtime."""

    def __init__(self, *, runtime: Any, config: Mapping[str, Any]) -> None:
        self.runtime = runtime
        self.config = dict(config)
        self.input_color = str(self.config.get("input_color", "bgr")).lower()
        if self.input_color not in {"rgb", "bgr"}:
            raise ValueError("input_color must be 'rgb' or 'bgr'")

        self.model_action_horizon = _positive_int(
            self.config, "model_action_horizon", default=FASTWAM_ACTION_HORIZON
        )
        if self.model_action_horizon % 4 != 0:
            raise ValueError("model_action_horizon must be divisible by 4")
        self.execution_horizon = _positive_int(self.config, "execution_horizon")
        if self.execution_horizon > self.model_action_horizon:
            raise ValueError("execution_horizon must be <= model_action_horizon")

        try:
            self.action_dt = float(self.config.get("action_dt"))
            self.max_inference_latency_s = float(self.config.get("max_inference_latency_s", 5.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("action_dt and max_inference_latency_s must be finite positive numbers") from exc
        if (
            not np.isfinite(self.action_dt)
            or self.action_dt <= 0.0
            or not np.isfinite(self.max_inference_latency_s)
            or self.max_inference_latency_s <= 0.0
        ):
            raise ValueError("action_dt and max_inference_latency_s must be finite positive numbers")

        # This runs before a listener is opened. A config with placeholders can
        # still load in --preflight-only mode, but it can never serve actions.
        self.safety_limits = safety_limits_from_config(self.config)
        self.motion_postprocessor = MotionPostprocessor(
            self.config, default_horizon=self.execution_horizon
        )

    async def handler(self, websocket: Any, *_unused: Any) -> None:
        peer = getattr(websocket, "remote_address", None)
        LOG.info("client connected: %s", peer)
        try:
            async for message in websocket:
                response = await self.handle_message(message)
                await websocket.send(json.dumps(response, ensure_ascii=False))
        except Exception:
            LOG.exception("connection error with %s", peer)
        finally:
            LOG.info("client disconnected: %s", peer)

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

            head, left, right = decode_required_rgb_images(
                request.get("images"), input_color=self.input_color
            )
            state = request.get("state")
            if not isinstance(state, Mapping):
                raise ValueError("request.state must be a JSON object")
            state_fastwam = piper_wire_to_fastwam(state)

            prediction = np.asarray(
                self.runtime.infer(head=head, left=left, right=right, state=state), dtype=np.float32
            )
            if prediction.shape != (self.model_action_horizon, 14):
                raise ValueError(
                    "FastWAM runtime must return finite actions with shape ({}, 14), got {}".format(
                        self.model_action_horizon, prediction.shape
                    )
                )
            if not np.all(np.isfinite(prediction)):
                raise ValueError("FastWAM runtime returned non-finite actions")

            # FastWAM predicts a candidate trajectory. Validate every point
            # that can actually be published in this response; the next
            # observation triggers a fresh rolling replan.
            execution_prediction = prediction[: self.execution_horizon]
            execution_prediction = self.motion_postprocessor.process(
                execution_prediction, str(request.get("motion_mode", "smooth"))
            )
            safe_prediction, projection = project_and_limit_fastwam_actions(
                execution_prediction, state_fastwam, self.safety_limits
            )
            if projection.projected_values:
                LOG.warning(
                    "projected %s FastWAM output value(s) to verified Piper limits; "
                    "max_arm_projection=%.6f rad max_gripper_projection=%.6f m",
                    projection.projected_values,
                    projection.max_arm_projection,
                    projection.max_gripper_projection,
                )
            elapsed_s = time.monotonic() - started
            if elapsed_s > self.max_inference_latency_s:
                raise TimeoutError(
                    "inference latency {:.3f}s exceeded configured limit {:.3f}s".format(
                        elapsed_s, self.max_inference_latency_s
                    )
                )
            action = fastwam_to_piper_wire(safe_prediction, dt=self.action_dt)
            LOG.info(
                "step=%s request_id=%s latency_ms=%.1f action_horizon=%s",
                step, request_id, elapsed_s * 1000, self.execution_horizon,
            )
            LOG.info(
                "motion_mode=%s server_adapters=%s",
                request.get("motion_mode", "smooth"), self.motion_postprocessor.enabled,
            )
            return build_policy_response(request_id=request_id, step=step, action=action)
        except Exception as exc:
            LOG.exception("failed to process request_id=%s step=%s", request_id, step)
            return build_policy_response(request_id=request_id, step=step, action={}, error=str(exc))


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load the FastWAM Piper config") from exc
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise ValueError(f"config {config_path} must contain a YAML object")
    return dict(config)


async def serve(server: PiperFastWAMServer) -> None:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("websockets is required to serve FastWAM policy requests") from exc

    host = str(server.config.get("host", "127.0.0.1"))
    port = _positive_int(server.config, "port", default=7081)
    max_size_mb = _positive_int(server.config, "max_message_size_mb", default=64)
    ping_interval_s = _positive_int(server.config, "ping_interval_s", default=20)
    ping_timeout_s = _positive_int(server.config, "ping_timeout_s", default=60)
    async with websockets.serve(
        server.handler,
        host,
        port,
        max_size=max_size_mb * 1024 * 1024,
        ping_interval=ping_interval_s,
        ping_timeout=ping_timeout_s,
    ):
        LOG.info("FastWAM Piper policy server listening on ws://%s:%s", host, port)
        await asyncio.Future()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed JSON WebSocket server for the FastWAM Piper Puzzle policy"
    )
    parser.add_argument("--config", required=True, help="path to piper_fastwam_puzzle.yaml")
    parser.add_argument("--host", help="override the host from the YAML config")
    parser.add_argument("--port", type=int, help="override the port from the YAML config")
    parser.add_argument("--device", help="override the model device from the YAML config")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="load the FastWAM model/checkpoint/stats without opening a listener; safety limits may stay blank",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    if args.host is not None:
        config["host"] = args.host
    if args.port is not None:
        config["port"] = args.port
    if args.device is not None:
        config["device"] = args.device
    if FastWAMPiperRuntime is None:
        raise RuntimeError(
            "FastWAMPiperRuntime model loading is unavailable; deploy the FastWAM runtime adapter "
            "before starting physical inference"
        )
    runtime = FastWAMPiperRuntime.from_config(config)
    if args.preflight_only:
        LOG.info(
            "FastWAM preflight succeeded: device=%s model_action_horizon=%s num_video_frames=%s",
            config.get("device"),
            config.get("model_action_horizon"),
            config.get("num_video_frames"),
        )
        return
    server = PiperFastWAMServer(runtime=runtime, config=config)
    asyncio.run(serve(server))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.info("FastWAM Piper policy server stopped")
