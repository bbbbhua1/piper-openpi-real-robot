#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example remote policy.

Replace `predict_action` with real model inference on the server host.
The default implementation returns a short hold-position chunk, which is useful
for testing the websocket path without moving the robot unexpectedly.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _as_float_list(values: Any, fallback_len: int = 7) -> List[float]:
    if not isinstance(values, (list, tuple)):
        return [0.0] * fallback_len
    return [float(value) for value in values]


def predict_action(request: Dict[str, Any]) -> Dict[str, Any]:
    state = request.get("state", {})
    left_arm = _as_float_list(state.get("left_arm"))
    right_arm = _as_float_list(state.get("right_arm"))

    chunk_size = 3
    dt = 0.1
    return {
        "left_arm": [left_arm for _ in range(chunk_size)],
        "right_arm": [right_arm for _ in range(chunk_size)],
        "dt": dt,
    }
