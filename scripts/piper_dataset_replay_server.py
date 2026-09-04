#!/usr/bin/env python3
"""One-shot, exact-action replay server for the recorded Piper Puzzle episode.

This program is deliberately separate from FastWAM inference.  It never loads
a model and it never changes a target produced by the recorded successful
episode.  Before accepting a client, it validates every source target against
the verified absolute Piper limits in the episode's start-pose configuration.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from piper_start_pose import StartPoseConfig, load_start_pose_config
from ws_policy_protocol import build_policy_response


LOG = logging.getLogger(__name__)

DEFAULT_DATASET_ROOT = Path(
    "/bh/zbh_ckp/datasets/fastwam/"
    "agilex_cobotmagic2_puzzle_20260801_v21_puppet_joint"
)
DEFAULT_EPISODE_INDEX = 159
DEFAULT_START_POSE_CONFIG = Path(
    "/bh/zbh_self/projects/piper-openpi-real-robot/configs/"
    "piper_fastwam_puzzle_start_pose.json"
)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7082
SOURCE_FPS = 30.0
SOURCE_DT = 1.0 / SOURCE_FPS
CADENCE_TOLERANCE_S = 5e-5
PIPER_ORDER_FROM_FASTWAM = np.asarray(
    [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13], dtype=np.intp
)


@dataclass(frozen=True)
class ReplayEpisode:
    """Validated one-shot replay targets in the physical Piper wire order."""

    episode_index: int
    source_path: Path
    piper_actions: np.ndarray
    source_timestamps: np.ndarray

    @property
    def num_waypoints(self) -> int:
        return int(self.piper_actions.shape[0])

    @property
    def source_duration_s(self) -> float:
        return float(self.source_timestamps[-1] - self.source_timestamps[0])

    def as_piper_action(self) -> dict[str, Any]:
        """Return the existing client's 7D-left/7D-right action envelope.

        The existing client requires every arrival time to be positive.  The
        source action at timestamp zero is the verified start pose, so it is
        issued one source tick after the client begins execution.  All action
        values and every relative source timestamp difference are unchanged.
        """

        relative_times = self.source_timestamps - self.source_timestamps[0]
        return {
            "left_arm": self.piper_actions[:, :7].tolist(),
            "right_arm": self.piper_actions[:, 7:].tolist(),
            "time_list": (relative_times + SOURCE_DT).tolist(),
            "dt": SOURCE_DT,
        }


def _episode_path(dataset_root: Path, episode_index: int) -> Path:
    if isinstance(episode_index, bool) or not isinstance(episode_index, int):
        raise ValueError("episode_index must be an integer")
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    return (
        dataset_root
        / "data"
        / "chunk-{:03d}".format(episode_index // 1000)
        / "episode_{:06d}.parquet".format(episode_index)
    )


def _as_float64(values: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("{} must be a finite numeric array with shape {}".format(name, shape)) from exc
    if array.shape != shape:
        raise ValueError("{} must have shape {}, got {}".format(name, shape, array.shape))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain only finite values".format(name))
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_episode(
    fastwam_actions: np.ndarray,
    timestamps: np.ndarray,
    config: StartPoseConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if fastwam_actions.ndim != 2 or fastwam_actions.shape[0] < 2:
        raise ValueError("action must have shape [T, 14] with T >= 2")
    actions = _as_float64(
        fastwam_actions,
        name="action",
        shape=(fastwam_actions.shape[0], 14),
    )
    times = _as_float64(timestamps, name="timestamp", shape=(actions.shape[0],))

    intervals = np.diff(times)
    if np.any(intervals <= 0.0):
        raise ValueError("timestamp must be strictly increasing")
    if np.any(np.abs(intervals - SOURCE_DT) > CADENCE_TOLERANCE_S):
        bad = int(np.flatnonzero(np.abs(intervals - SOURCE_DT) > CADENCE_TOLERANCE_S)[0])
        raise ValueError(
            "timestamp cadence at frame {} is {:.9f}s, expected {:.9f}s".format(
                bad + 1, intervals[bad], SOURCE_DT
            )
        )

    piper_actions = np.ascontiguousarray(actions[:, PIPER_ORDER_FROM_FASTWAM])
    lower = np.asarray(config.joint_lower, dtype=np.float64)
    upper = np.asarray(config.joint_upper, dtype=np.float64)
    violations = np.argwhere((piper_actions < lower) | (piper_actions > upper))
    if violations.size:
        frame_index, piper_index = (int(item) for item in violations[0])
        value = float(piper_actions[frame_index, piper_index])
        raise ValueError(
            "action outside verified absolute bounds at frame={}, piper_index={}: "
            "value={} not in [{}, {}]".format(
                frame_index,
                piper_index,
                value,
                float(lower[piper_index]),
                float(upper[piper_index]),
            )
        )

    configured_start = np.asarray(config.left_arm + config.right_arm, dtype=np.float64)
    if not np.allclose(piper_actions[0], configured_start, atol=1e-6, rtol=0.0):
        max_error = float(np.max(np.abs(piper_actions[0] - configured_start)))
        raise ValueError(
            "first replay action does not match configured start pose; max_error={:.9f}".format(
                max_error
            )
        )
    piper_actions.setflags(write=False)
    times.setflags(write=False)
    return piper_actions, times


def load_episode(
    dataset_root: Path, episode_index: int, config: StartPoseConfig
) -> ReplayEpisode:
    """Load the fixed replay data without modifying values or source files."""

    if episode_index != DEFAULT_EPISODE_INDEX:
        raise ValueError("only verified replay episode {} is permitted".format(DEFAULT_EPISODE_INDEX))
    source_path = _episode_path(dataset_root, episode_index)
    if not source_path.is_file():
        raise FileNotFoundError("replay parquet does not exist: {}".format(source_path))
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("PyArrow is required to read the replay parquet") from exc

    table = pq.read_table(source_path, columns=["timestamp", "action"])
    timestamps = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64)
    fastwam_actions = np.asarray(table.column("action").to_pylist(), dtype=np.float64)
    piper_actions, timestamps = _validate_episode(fastwam_actions, timestamps, config)
    return ReplayEpisode(
        episode_index=episode_index,
        source_path=source_path,
        piper_actions=piper_actions,
        source_timestamps=timestamps,
    )


class DatasetReplayServer:
    """Serve one immutable episode response over the existing JSON protocol."""

    def __init__(self, episode: ReplayEpisode) -> None:
        self.episode = episode
        self._served = False
        self._served_lock = asyncio.Lock()

    async def handler(self, websocket: Any, *_unused: Any) -> None:
        peer = getattr(websocket, "remote_address", None)
        LOG.info("replay client connected: %s", peer)
        try:
            async for message in websocket:
                response = await self.handle_message(message)
                await websocket.send(json.dumps(response, ensure_ascii=False))
        except Exception:
            LOG.exception("replay connection error with %s", peer)
        finally:
            LOG.info("replay client disconnected: %s", peer)

    async def handle_message(self, message: str) -> dict[str, Any]:
        request_id = "unknown"
        step = -1
        try:
            request = json.loads(message)
            if not isinstance(request, Mapping):
                raise ValueError("websocket message must be a JSON object")
            request_id = str(request.get("request_id", request_id))
            step = int(request.get("step", step))
            if not isinstance(request.get("state"), Mapping):
                raise ValueError("request.state must be a JSON object")
            async with self._served_lock:
                if self._served:
                    raise RuntimeError(
                        "episode replay already served; restart the replay server for another run"
                    )
                self._served = True
            action = self.episode.as_piper_action()
            LOG.info(
                "serving exact replay: episode=%s request_id=%s frames=%s duration_s=%.6f",
                self.episode.episode_index,
                request_id,
                self.episode.num_waypoints,
                self.episode.source_duration_s,
            )
            return build_policy_response(request_id=request_id, step=step, action=action)
        except Exception as exc:
            LOG.warning("replay request rejected request_id=%s step=%s: %s", request_id, step, exc)
            return build_policy_response(
                request_id=request_id,
                step=step,
                action={},
                error=str(exc),
            )


async def serve(server: DatasetReplayServer, *, host: str, port: int) -> None:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("websockets is required to run the replay server") from exc
    async with websockets.serve(
        server.handler,
        host,
        port,
        max_size=4 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=60,
    ):
        LOG.info("Piper dataset replay server listening on ws://%s:%s", host, port)
        await asyncio.Future()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve exact recorded Puzzle episode 159 actions; no FastWAM model is loaded"
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--episode-index", type=int, default=DEFAULT_EPISODE_INDEX)
    parser.add_argument("--start-pose-config", type=Path, default=DEFAULT_START_POSE_CONFIG)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_arg_parser().parse_args()
    if args.dataset_root.resolve() != DEFAULT_DATASET_ROOT.resolve():
        raise ValueError("dataset-root must remain fixed at {}".format(DEFAULT_DATASET_ROOT))
    if args.port <= 0 or args.port > 65535:
        raise ValueError("port must be in [1, 65535]")
    config = load_start_pose_config(args.start_pose_config)
    episode = load_episode(args.dataset_root, args.episode_index, config)
    LOG.info(
        "replay preflight passed: episode=%s frames=%s source_duration_s=%.6f "
        "cadence_hz=%.1f source=%s; no model loaded",
        episode.episode_index,
        episode.num_waypoints,
        episode.source_duration_s,
        SOURCE_FPS,
        episode.source_path,
    )
    if args.preflight_only:
        return
    asyncio.run(serve(DatasetReplayServer(episode), host=args.host, port=args.port))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.info("Piper dataset replay server stopped")
