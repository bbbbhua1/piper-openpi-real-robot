#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JSON websocket protocol helpers for remote policy inference."""

from __future__ import annotations

import base64
import time
import uuid
from typing import Any, Dict, Mapping, Optional


CAMERA_NAMES = ("head", "left_wrist", "right_wrist")


def _try_encode_numpy_image(image: Any, jpeg_quality: int, resize_scale: float) -> bytes:
    import cv2

    if resize_scale != 1.0:
        image = cv2.resize(image, None, fx=resize_scale, fy=resize_scale)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    ok, encoded = cv2.imencode(".jpg", image, encode_param)
    if not ok:
        raise RuntimeError("failed to JPEG encode image")
    return encoded.tobytes()


def encode_image(image: Any, jpeg_quality: int = 80, resize_scale: float = 1.0) -> Dict[str, Any]:
    """Encode a camera frame for JSON transport.

    `image` may be:
      - a numpy HWC image, encoded to JPEG with OpenCV,
      - raw JPEG bytes,
      - a dict containing {"encoding": "jpeg", "data": bytes}.
    """

    if isinstance(image, dict):
        encoding = image.get("encoding", "jpeg")
        data = image.get("data", b"")
        width = image.get("width")
        height = image.get("height")
    elif isinstance(image, (bytes, bytearray)):
        encoding = "jpeg"
        data = bytes(image)
        width = None
        height = None
    else:
        encoding = "jpeg"
        data = _try_encode_numpy_image(image, jpeg_quality, resize_scale)
        height, width = image.shape[:2]

    return {
        "encoding": encoding,
        "data": base64.b64encode(data).decode("ascii"),
        "width": width,
        "height": height,
    }


def decode_image(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "encoding": payload.get("encoding", "jpeg"),
        "data": base64.b64decode(payload.get("data", "")),
        "width": payload.get("width"),
        "height": payload.get("height"),
    }


def build_policy_request(
    step: int,
    instruction: str,
    state: Mapping[str, Any],
    images: Optional[Mapping[str, Any]] = None,
    jpeg_quality: int = 80,
    resize_scale: float = 1.0,
) -> Dict[str, Any]:
    encoded_images = {}
    for name, image in (images or {}).items():
        encoded_images[name] = encode_image(
            image,
            jpeg_quality=jpeg_quality,
            resize_scale=resize_scale,
        )

    return {
        "request_id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "step": int(step),
        "instruction": instruction,
        "state": dict(state),
        "images": encoded_images,
    }


def decode_policy_request(payload: Mapping[str, Any]) -> Dict[str, Any]:
    images = {}
    for name, image_payload in payload.get("images", {}).items():
        images[name] = decode_image(image_payload)

    return {
        "request_id": payload["request_id"],
        "timestamp": payload.get("timestamp"),
        "step": int(payload.get("step", 0)),
        "instruction": payload.get("instruction", ""),
        "state": payload.get("state", {}),
        "images": images,
        "raw": dict(payload),
    }


def build_policy_response(
    request_id: str,
    step: int,
    action: Mapping[str, Any],
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "ok": error is None,
        "request_id": request_id,
        "step": int(step),
        "action": dict(action),
        "error": error,
    }
