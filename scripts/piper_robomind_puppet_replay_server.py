#!/usr/bin/env python3
"""One-shot exact replay server for RoboMIND/Puppet episode 382.

The server intentionally does not load a policy model.  It reads the fixed
Puppet trajectory, validates it in Piper wire order, and returns the complete
453-waypoint action chunk through the existing JSON WebSocket protocol.
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
    "/bh/media/unify/special_project/puzzle/"
    "agilex_cobotmagic2_dualArm-gripper-3cameras_34/"
    "agilex_cobotmagic2_dualArm-gripper-3cameras_34_puzzle_20260805/"
    "success/lerobot_RoboMIND"
)
DEFAULT_EPISODE_INDEX = 382
DEFAULT_START_POSE_CONFIG = Path(
    "/bh/zbh_self/projects/piper-openpi-real-robot/configs/"
    "piper_robomind_puppet_episode_382_start_pose.json"
)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7082
SOURCE_FPS = 30.0
SOURCE_DT = 1.0 / SOURCE_FPS
CADENCE_TOLERANCE_S = 5e-5
VALUE_TOLERANCE = 1e-6
EXPECTED_WAYPOINTS = 453
PUPPET_COLUMNS = (
    "episode_index",
    "timestamp",
    "puppet.arm_left_position_align.data",
    "puppet.end_effector_left_position_align.data",
    "puppet.arm_right_position_align.data",
    "puppet.end_effector_right_position_align.data",
)


@dataclass(frozen=True)
class ReplayEpisode:
    """Validated immutable action waypoints in Piper wire order."""

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
        relative_times = self.source_timestamps - self.source_timestamps[0]
        return {
            "left_arm": self.piper_actions[:, :7].tolist(),
            "right_arm": self.piper_actions[:, 7:].tolist(),
            "time_list": (relative_times + SOURCE_DT).tolist(),
            "dt": SOURCE_DT,
        }


def _episode_path(dataset_root: Path) -> Path:
    return dataset_root / "data" / "chunk-000" / "file-000.parquet"


def _as_float64(values: Any, *, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("{} must be a finite numeric array".format(name)) from exc
    if shape is not None and array.shape != shape:
        raise ValueError("{} must have shape {}, got {}".format(name, shape, array.shape))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain only finite values".format(name))
    return np.ascontiguousarray(array, dtype=np.float64)


def _load_puppet_columns(path: Path) -> dict[str, np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("PyArrow is required to read the RoboMIND replay parquet") from exc

    table = pq.read_table(path, columns=list(PUPPET_COLUMNS))
    columns = table.to_pydict()
    return {
        "episode_index": _as_float64(columns["episode_index"], name="episode_index"),
        "timestamp": _as_float64(columns["timestamp"], name="timestamp"),
        "left_arm": _as_float64(
            columns["puppet.arm_left_position_align.data"],
            name="puppet.arm_left_position_align.data",
        ),
        "left_gripper": _as_float64(
            columns["puppet.end_effector_left_position_align.data"],
            name="puppet.end_effector_left_position_align.data",
        ),
        "right_arm": _as_float64(
            columns["puppet.arm_right_position_align.data"],
            name="puppet.arm_right_position_align.data",
        ),
        "right_gripper": _as_float64(
            columns["puppet.end_effector_right_position_align.data"],
            name="puppet.end_effector_right_position_align.data",
        ),
    }


def _validate_episode(
    columns: dict[str, np.ndarray],
    episode_index: int,
    config: StartPoseConfig,
) -> tuple[np.ndarray, np.ndarray]:
    episode_ids = columns["episode_index"]
    selection = episode_ids == float(episode_index)
    if int(np.count_nonzero(selection)) < 2:
        raise ValueError("episode {} must contain at least two rows".format(episode_index))

    timestamps = columns["timestamp"][selection]
    left_arm = columns["left_arm"][selection]
    left_gripper = columns["left_gripper"][selection]
    right_arm = columns["right_arm"][selection]
    right_gripper = columns["right_gripper"][selection]

    num_rows = int(timestamps.shape[0])
    if num_rows != EXPECTED_WAYPOINTS:
        raise ValueError(
            "episode {} must contain {} waypoints, got {}".format(
                episode_index, EXPECTED_WAYPOINTS, num_rows
            )
        )
    for name, array, shape in (
        ("timestamp", timestamps, (num_rows,)),
        ("left_arm", left_arm, (num_rows, 6)),
        ("left_gripper", left_gripper, (num_rows,)),
        ("right_arm", right_arm, (num_rows, 6)),
        ("right_gripper", right_gripper, (num_rows,)),
    ):
        _as_float64(array, name=name, shape=shape)

    intervals = np.diff(timestamps)
    if np.any(intervals <= 0.0):
        raise ValueError("episode timestamps must be strictly increasing")
    cadence_error = np.abs(intervals - SOURCE_DT)
    if np.any(cadence_error > CADENCE_TOLERANCE_S):
        bad = int(np.flatnonzero(cadence_error > CADENCE_TOLERANCE_S)[0])
        raise ValueError(
            "timestamp cadence at frame {} is {:.9f}s, expected {:.9f}s".format(
                bad + 1, intervals[bad], SOURCE_DT
            )
        )

    left = np.column_stack((left_arm, left_gripper))
    right = np.column_stack((right_arm, right_gripper))
    if not np.allclose(left, left[0], atol=VALUE_TOLERANCE, rtol=0.0):
        raise ValueError("Puppet left observation pose is not constant")

    configured_start = np.asarray(config.left_arm + config.right_arm, dtype=np.float64)
    source_start = np.concatenate((left[0], right[0]))
    if not np.allclose(source_start, configured_start, atol=VALUE_TOLERANCE, rtol=0.0):
        max_error = float(np.max(np.abs(source_start - configured_start)))
        raise ValueError(
            "episode first Puppet frame does not match start-pose config; "
            "max_error={:.9f}".format(max_error)
        )

    piper_actions = np.ascontiguousarray(np.column_stack((left, right)), dtype=np.float64)
    lower = np.asarray(config.joint_lower, dtype=np.float64)
    upper = np.asarray(config.joint_upper, dtype=np.float64)
    violations = np.argwhere((piper_actions < lower) | (piper_actions > upper))
    if violations.size:
        frame_index, piper_index = (int(item) for item in violations[0])
        value = float(piper_actions[frame_index, piper_index])
        raise ValueError(
            "Puppet action outside Piper bounds at frame={}, piper_index={}: "
            "value={} not in [{}, {}]".format(
                frame_index,
                piper_index,
                value,
                float(lower[piper_index]),
                float(upper[piper_index]),
            )
        )

    piper_actions.setflags(write=False)
    timestamps = np.ascontiguousarray(timestamps, dtype=np.float64)
    timestamps.setflags(write=False)
    return piper_actions, timestamps


def load_episode(
    dataset_root: Path, episode_index: int, config: StartPoseConfig
) -> ReplayEpisode:
    if episode_index != DEFAULT_EPISODE_INDEX:
        raise ValueError("only verified Puppet episode {} is permitted".format(DEFAULT_EPISODE_INDEX))
    source_path = _episode_path(dataset_root)
    if not source_path.is_file():
        raise FileNotFoundError("RoboMIND replay parquet does not exist: {}".format(source_path))
    columns = _load_puppet_columns(source_path)
    piper_actions, timestamps = _validate_episode(columns, episode_index, config)
    return ReplayEpisode(
        episode_index=episode_index,
        source_path=source_path,
        piper_actions=piper_actions,
        source_timestamps=timestamps,
    )


class PuppetReplayServer:
    """Serve one immutable episode response over the existing JSON protocol."""

    def __init__(self, episode: ReplayEpisode) -> None:
        self.episode = episode
        self._served = False
        self._served_lock = asyncio.Lock()

    async def handler(self, websocket: Any, *_unused: Any) -> None:
        peer = getattr(websocket, "remote_address", None)
        LOG.info("Puppet replay client connected: %s", peer)
        try:
            async for message in websocket:
                response = await self.handle_message(message)
                await websocket.send(json.dumps(response, ensure_ascii=False))
        except Exception:
            LOG.exception("Puppet replay connection error with %s", peer)
        finally:
            LOG.info("Puppet replay client disconnected: %s", peer)

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
            LOG.info(
                "serving exact Puppet replay: episode=%s request_id=%s frames=%s duration_s=%.6f",
                self.episode.episode_index,
                request_id,
                self.episode.num_waypoints,
                self.episode.source_duration_s,
            )
            return build_policy_response(
                request_id=request_id,
                step=step,
                action=self.episode.as_piper_action(),
            )
        except Exception as exc:
            LOG.warning(
                "Puppet replay request rejected request_id=%s step=%s: %s",
                request_id,
                step,
                exc,
            )
            return build_policy_response(
                request_id=request_id,
                step=step,
                action={},
                error=str(exc),
            )


async def serve(server: PuppetReplayServer, *, host: str, port: int) -> None:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("websockets is required to run the Puppet replay server") from exc
    async with websockets.serve(
        server.handler,
        host,
        port,
        max_size=4 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=60,
    ):
        LOG.info("Piper RoboMIND Puppet replay server listening on ws://%s:%s", host, port)
        await asyncio.Future()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve exact RoboMIND/Puppet episode 382 actions; no model is loaded"
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
    if args.episode_index != DEFAULT_EPISODE_INDEX:
        raise ValueError("episode-index must remain {}".format(DEFAULT_EPISODE_INDEX))
    if args.port <= 0 or args.port > 65535:
        raise ValueError("port must be in [1, 65535]")

    config = load_start_pose_config(args.start_pose_config)
    episode = load_episode(args.dataset_root, args.episode_index, config)
    left = episode.piper_actions[:, :7]
    LOG.info(
        "Puppet preflight passed: episode=%s frames=%s duration_s=%.6f cadence_hz=%.1f "
        "left_constant=%s source=%s",
        episode.episode_index,
        episode.num_waypoints,
        episode.source_duration_s,
        SOURCE_FPS,
        bool(np.allclose(left, left[0], atol=VALUE_TOLERANCE, rtol=0.0)),
        episode.source_path,
    )
    if args.preflight_only:
        return
    asyncio.run(serve(PuppetReplayServer(episode), host=args.host, port=args.port))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.info("Piper RoboMIND Puppet replay server stopped")
