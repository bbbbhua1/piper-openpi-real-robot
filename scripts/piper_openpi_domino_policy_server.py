#!/usr/bin/env python3
"""Pi05 Domino JSON WebSocket server for the Piper robot client."""

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
import websockets

from piper_openpi_policy_server import (
    OpenPIPolicy,
    build_policy_response,
    openpi_actions_to_client_action,
    parse_image_map,
    request_to_openpi_observation,
    save_received_images,
)


LOG = logging.getLogger(__name__)


def _vector(config: Mapping[str, Any], key: str) -> np.ndarray:
    values = config.get(key)
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (14,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{key} must contain exactly 14 finite values")
    return result


def _limits(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower, upper, delta = (_vector(config, key) for key in ("joint_lower", "joint_upper", "max_delta"))
    if not np.all(lower < upper):
        raise ValueError("joint_lower must be strictly less than joint_upper")
    if not np.all(delta > 0):
        raise ValueError("max_delta must be strictly positive")
    return lower, upper, delta


def limit_actions(actions: Any, current_state: Any, config: Mapping[str, Any]) -> np.ndarray:
    trajectory = np.asarray(actions, dtype=np.float32)
    current = np.asarray(current_state, dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[1] < 14 or len(trajectory) == 0:
        raise ValueError(f"expected actions with shape [T, >=14], got {trajectory.shape}")
    if current.shape != (14,) or not np.all(np.isfinite(current)):
        raise ValueError("current state must contain 14 finite values")
    trajectory = trajectory[:, :14]
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("policy returned non-finite actions")
    lower, upper, max_delta = _limits(config)
    bounded = np.clip(trajectory, lower, upper)
    accepted = np.empty_like(bounded)
    previous = current
    for index, prediction in enumerate(bounded):
        accepted[index] = np.clip(prediction, previous - max_delta, previous + max_delta)
        previous = accepted[index]
    return accepted


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("config must contain a YAML object")
    return dict(payload)


class DominoOpenPIServer:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.image_map = parse_image_map(list(self.config.get("image_map", [])))
        self.policy = OpenPIPolicy(
            openpi_root=str(self.config["openpi_root"]),
            config_name=str(self.config.get("config", "pi05_realrobot")),
            checkpoint_dir=str(self.config["checkpoint"]),
            num_infer_steps=int(self.config.get("num_infer_steps", 10)),
        )

    async def handler(self, websocket: Any) -> None:
        peer = getattr(websocket, "remote_address", None)
        LOG.info("Domino Pi05 client connected: %s", peer)
        try:
            async for message in websocket:
                response = await self.handle_message(message)
                await websocket.send(json.dumps(response, ensure_ascii=False))
        except websockets.exceptions.ConnectionClosed:
            LOG.info("Domino Pi05 client disconnected: %s", peer)
        except Exception:
            LOG.exception("Domino Pi05 connection error: %s", peer)

    async def handle_message(self, message: str) -> dict[str, Any]:
        request_id, step = "unknown", -1
        try:
            request = json.loads(message)
            if not isinstance(request, Mapping):
                raise ValueError("websocket message must be a JSON object")
            request_id = str(request.get("request_id", request_id))
            step = int(request.get("step", step))
            decoded_images: dict[str, np.ndarray] = {}
            obs = request_to_openpi_observation(
                request,
                image_map=self.image_map,
                input_color=str(self.config.get("input_color", "bgr")),
                allow_missing_images=False,
                state_left_dim=int(self.config.get("state_left_dim", 7)),
                state_right_dim=int(self.config.get("state_right_dim", 7)),
                decoded_images=decoded_images,
            )
            # Keep inference aligned with the Domino training data even when
            # an older client sends a shorthand or missing instruction.
            obs["prompt"] = str(self.config["task_label"])
            result = self.policy.infer(obs)
            state = np.asarray(obs["state"], dtype=np.float32)
            actions = limit_actions(result["actions"], state, self.config)
            action = openpi_actions_to_client_action(
                actions,
                dt=float(self.config.get("action_dt", 0.1)),
                horizon=int(self.config.get("horizon", 50)),
                left_arm_dim=int(self.config.get("left_arm_dim", 7)),
                right_arm_dim=int(self.config.get("right_arm_dim", 7)),
            )
            LOG.info("step=%s request_id=%s action_horizon=%s", step, request_id, len(actions))
            return build_policy_response(request_id, step, action)
        except Exception as exc:
            LOG.exception("failed request_id=%s step=%s", request_id, step)
            return build_policy_response(request_id, step, {}, error=str(exc))


async def serve(server: DominoOpenPIServer) -> None:
    host = str(server.config.get("host", "127.0.0.1"))
    port = int(server.config.get("port", 7085))
    async with websockets.serve(
        server.handler,
        host,
        port,
        max_size=int(server.config.get("max_size_mb", 64)) * 1024 * 1024,
        ping_interval=20,
        ping_timeout=60,
    ):
        LOG.info("Pi05 Domino policy server listening on ws://%s:%s", host, port)
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pi05 Domino Piper policy server")
    parser.add_argument("--config", default="configs/piper_openpi_domino.yaml")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.host is not None:
        config["host"] = args.host
    if args.port is not None:
        config["port"] = args.port
    asyncio.run(serve(DominoOpenPIServer(config)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        main()
    except KeyboardInterrupt:
        LOG.info("Pi05 Domino policy server stopped")
