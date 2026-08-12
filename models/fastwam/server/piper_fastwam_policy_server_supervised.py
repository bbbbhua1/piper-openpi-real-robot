#!/usr/bin/env python3
"""Human-supervised raw-action FastWAM JSON WebSocket server for Piper.

This is intentionally separate from ``piper_fastwam_policy_server.py``.  It
preserves that server's request protocol and FastWAM preprocessing, but sends
finite model waypoints without software joint projection or sequential delta
limiting.  The robot's ROS/CAN driver and firmware remain responsible for
their own hard protections.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

from fastwam_piper_runtime import fastwam_to_piper_wire, piper_wire_to_fastwam
from piper_fastwam_policy_server import (
    FASTWAM_ACTION_HORIZON,
    FastWAMPiperRuntime,
    _positive_int,
    build_policy_response,
    decode_required_rgb_images,
    load_config,
)


LOG = logging.getLogger(__name__)


class SupervisedRawPiperFastWAMServer:
    """WebSocket adapter that forwards finite FastWAM actions unchanged."""

    def __init__(self, *, runtime: Any, config: Mapping[str, Any]) -> None:
        self.runtime = runtime
        self.config = dict(config)
        self.input_color = str(self.config.get("input_color", "bgr")).lower()
        if self.input_color not in {"rgb", "bgr"}:
            raise ValueError("input_color must be 'rgb' or 'bgr'")

        self.model_action_horizon = _positive_int(
            self.config, "model_action_horizon", default=FASTWAM_ACTION_HORIZON
        )
        if self.model_action_horizon != FASTWAM_ACTION_HORIZON:
            raise ValueError("model_action_horizon is fixed at 32 for the trained Puzzle checkpoint")
        self.execution_horizon = _positive_int(self.config, "execution_horizon")
        if self.execution_horizon > FASTWAM_ACTION_HORIZON:
            raise ValueError("execution_horizon must be <= 32")

        try:
            self.action_dt = float(self.config.get("action_dt"))
        except (TypeError, ValueError) as exc:
            raise ValueError("action_dt must be a finite positive number") from exc
        if not np.isfinite(self.action_dt) or self.action_dt <= 0.0:
            raise ValueError("action_dt must be a finite positive number")

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
            piper_wire_to_fastwam(state)

            prediction = np.asarray(
                self.runtime.infer(head=head, left=left, right=right, state=state),
                dtype=np.float32,
            )
            if prediction.shape != (FASTWAM_ACTION_HORIZON, 14):
                raise ValueError(
                    "FastWAM runtime must return finite actions with shape (32, 14), got {}".format(
                        prediction.shape
                    )
                )
            if not np.all(np.isfinite(prediction)):
                raise ValueError("FastWAM runtime returned non-finite actions")

            raw_prediction = prediction[: self.execution_horizon]
            action = fastwam_to_piper_wire(raw_prediction, dt=self.action_dt)
            LOG.warning(
                "SUPERVISED RAW ACTION MODE: step=%s request_id=%s latency_ms=%.1f "
                "action_horizon=%s; no software clamp or delta limit applied",
                step,
                request_id,
                (time.monotonic() - started) * 1000,
                self.execution_horizon,
            )
            return build_policy_response(request_id=request_id, step=step, action=action)
        except Exception as exc:
            LOG.exception("failed to process request_id=%s step=%s", request_id, step)
            return build_policy_response(request_id=request_id, step=step, action={}, error=str(exc))


async def serve(server: SupervisedRawPiperFastWAMServer) -> None:
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
        LOG.warning("SUPERVISED RAW ACTION MODE enabled; listener on ws://%s:%s", host, port)
        await asyncio.Future()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Supervised raw-action JSON WebSocket server for FastWAM Piper Puzzle policy"
    )
    parser.add_argument("--config", required=True, help="path to supervised raw FastWAM YAML config")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="load the FastWAM model/checkpoint/stats without opening a listener",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args()
    config = load_config(Path(args.config))
    if FastWAMPiperRuntime is None:
        raise RuntimeError(
            "FastWAMPiperRuntime model loading is unavailable; deploy the FastWAM runtime adapter "
            "before starting physical inference"
        )
    runtime = FastWAMPiperRuntime.from_config(config)
    if args.preflight_only:
        LOG.warning(
            "SUPERVISED RAW ACTION MODE preflight succeeded: device=%s "
            "model_action_horizon=%s num_video_frames=%s",
            config.get("device"),
            config.get("model_action_horizon"),
            config.get("num_video_frames"),
        )
        return
    server = SupervisedRawPiperFastWAMServer(runtime=runtime, config=config)
    asyncio.run(serve(server))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.info("supervised raw FastWAM Piper policy server stopped")
