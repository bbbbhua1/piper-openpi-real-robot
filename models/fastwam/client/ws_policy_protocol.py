#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JSON websocket protocol helpers for remote policy inference."""

from __future__ import annotations

import base64
import time
import uuid
from typing import Any, Dict, Mapping, Optional


CAMERA_NAMES = ("head", "left_wrist", "right_wrist")
# These are the per-camera tiles consumed by the FastWAM composition step.
# Sending JPEGs at these dimensions avoids carrying full-resolution camera
# frames over the WebSocket, without changing the model's image geometry.
FASTWAM_TRANSPORT_IMAGE_SIZES = {
    "head": (320, 256),
    "left_wrist": (160, 128),
    "right_wrist": (160, 128),
}
# LingBot-VA consumes three equally sized 320x256 camera tiles.
LINGBOTVA_TRANSPORT_IMAGE_SIZES = {
    "head": (320, 256),
    "left_wrist": (320, 256),
    "right_wrist": (320, 256),
}


def _validated_target_size(target_size: Optional[tuple[int, int]]) -> Optional[tuple[int, int]]:
    if target_size is None:
        return None
    if len(target_size) != 2:
        raise ValueError("target_size must be a (width, height) pair")
    width, height = (int(target_size[0]), int(target_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError("target_size width and height must be positive")
    return width, height


def _encode_jpeg_bgr(image: Any, jpeg_quality: int) -> bytes:
    import cv2

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    ok, encoded = cv2.imencode(".jpg", image, encode_param)
    if not ok:
        raise RuntimeError("failed to JPEG encode image")
    return encoded.tobytes()


def _try_encode_numpy_image(
    image: Any,
    jpeg_quality: int,
    resize_scale: float,
    target_size: Optional[tuple[int, int]],
) -> tuple[bytes, int, int]:
    import cv2

    if resize_scale != 1.0:
        image = cv2.resize(image, None, fx=resize_scale, fy=resize_scale)
    if target_size is not None:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    height, width = image.shape[:2]
    return _encode_jpeg_bgr(image, jpeg_quality), width, height


def _resize_compressed_jpeg(
    data: bytes, jpeg_quality: int, target_size: tuple[int, int]
) -> tuple[bytes, int, int]:
    """Decode, resize, and re-encode a ROS compressed camera image once.

    This work is deliberately performed at request time, not in the ROS image
    callback, so an idle robot does not spend CPU continually transcoding every
    incoming camera frame.
    """

    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode compressed JPEG for transport resize")
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return _encode_jpeg_bgr(image, jpeg_quality), target_size[0], target_size[1]


def encode_image(
    image: Any,
    jpeg_quality: int = 80,
    resize_scale: float = 1.0,
    target_size: Optional[tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Encode a camera frame for JSON transport.

    `image` may be:
      - a numpy HWC image, encoded to JPEG with OpenCV,
      - raw JPEG bytes,
      - a dict containing {"encoding": "jpeg", "data": bytes}.

    With ``target_size``, compressed JPEGs are decoded, resized, and encoded at
    request time. Without it, compressed data remains byte-for-byte unchanged.
    """

    target_size = _validated_target_size(target_size)

    if isinstance(image, dict):
        encoding = image.get("encoding", "jpeg")
        data = bytes(image.get("data", b""))
        width = image.get("width")
        height = image.get("height")
        if target_size is not None:
            data, width, height = _resize_compressed_jpeg(data, jpeg_quality, target_size)
    elif isinstance(image, (bytes, bytearray)):
        encoding = "jpeg"
        data = bytes(image)
        width = None
        height = None
        if target_size is not None:
            data, width, height = _resize_compressed_jpeg(data, jpeg_quality, target_size)
    else:
        encoding = "jpeg"
        data, width, height = _try_encode_numpy_image(
            image, jpeg_quality, resize_scale, target_size
        )

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
    transport_image_sizes: Optional[Mapping[str, tuple[int, int]]] = None,
) -> Dict[str, Any]:
    encoded_images = {}
    for name, image in (images or {}).items():
        encoded_images[name] = encode_image(
            image,
            jpeg_quality=jpeg_quality,
            resize_scale=resize_scale,
            target_size=(transport_image_sizes or {}).get(name),
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
