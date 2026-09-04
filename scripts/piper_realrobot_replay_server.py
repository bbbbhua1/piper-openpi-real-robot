#!/usr/bin/env python3
"""Replay one episode from the real-robot RoboMIND dataset.

The server is deliberately model-free.  It reads one episode from the
LeRobot/RoboMIND Parquet file and returns the complete 14-dimensional Piper
joint trajectory through the JSON WebSocket protocol used by
``websocket_policy_client.py``.  The validated episode remains in memory and
can be requested repeatedly without restarting the process.

The default action source is ``puppet`` because that is the source used by the
existing RoboMIND replay server.  ``--action-source master`` is available for
datasets where the master stream is the desired command trajectory.
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

from ws_policy_protocol import build_policy_response


LOG = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path("/bh/media/pku-data/realrobot")
DEFAULT_EPISODE_INDEX = 0
DEFAULT_ACTION_SOURCE = "puppet"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7082
SOURCE_FPS = 30.0
SOURCE_DT = 1.0 / SOURCE_FPS
CADENCE_TOLERANCE_S = 5e-5
GRIPPER_ZERO_TOLERANCE = 0.001

# Piper limits used by the existing robot-side client.  Gripper limits remain
# [0.0, 0.1]; the validator separately tolerates the encoder's tiny signed
# mechanical-zero drift in this dataset without changing source values.
PIPER_JOINT_LOWER = np.asarray(
    [
        -2.617993878,
        0.0,
        -2.967059728,
        -1.745329252,
        -1.221730476,
        -3.141592654,
        0.0,
        -2.617993878,
        0.0,
        -2.967059728,
        -1.745329252,
        -1.221730476,
        -3.141592654,
        0.0,
    ],
    dtype=np.float64,
)
PIPER_JOINT_UPPER = np.asarray(
    [
        2.617993878,
        3.141592654,
        0.0,
        1.745329252,
        1.221730476,
        3.141592654,
        0.1,
        2.617993878,
        3.141592654,
        0.0,
        1.745329252,
        1.221730476,
        3.141592654,
        0.1,
    ],
    dtype=np.float64,
)

_SOURCE_COLUMNS = {
    "puppet": (
        "puppet.arm_left_position_align.data",
        "puppet.end_effector_left_position_align.data",
        "puppet.arm_right_position_align.data",
        "puppet.end_effector_right_position_align.data",
    ),
    "master": (
        "master.arm_left_position_align.data",
        "master.end_effector_left_position_align.data",
        "master.arm_right_position_align.data",
        "master.end_effector_right_position_align.data",
    ),
}
_REQUIRED_COLUMNS = ("episode_index", "timestamp", "frame_index")


@dataclass(frozen=True)
class ReplayEpisode:
    """An immutable, validated trajectory in Piper wire order."""

    episode_index: int
    source_name: str
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
        """Return the action shape consumed by the existing Piper client."""

        relative_times = self.source_timestamps - self.source_timestamps[0]
        return {
            "left_arm": self.piper_actions[:, :7].tolist(),
            "right_arm": self.piper_actions[:, 7:].tolist(),
            # The client requires positive arrival times.  The first source
            # frame is the recorded start pose, so issue it one source tick in.
            "time_list": (relative_times + SOURCE_DT).tolist(),
            "dt": SOURCE_DT,
        }


def resolve_dataset_root(data_root: Path) -> Path:
    """Resolve either a dataset directory or the shared realrobot directory."""

    direct = data_root / "data" / "chunk-000" / "file-000.parquet"
    if direct.is_file():
        return data_root

    candidates = sorted(
        path.parent.parent.parent
        for path in data_root.glob("**/data/chunk-*/file-*.parquet")
        if path.is_file()
    )
    unique = list(dict.fromkeys(candidates))
    if not unique:
        raise FileNotFoundError(
            "no LeRobot dataset found below {}; expected data/chunk-*/file-*.parquet".format(
                data_root
            )
        )
    if len(unique) > 1:
        raise ValueError(
            "multiple datasets found below {}; pass --dataset-root explicitly: {}".format(
                data_root, ", ".join(str(path) for path in unique)
            )
        )
    return unique[0]


def list_episode_indices(data_root: Path) -> list[int]:
    """List episode ids from semantic annotations without importing PyArrow."""

    dataset_root = resolve_dataset_root(data_root)
    annotation_dir = dataset_root / "anno_semantic"
    indices = []
    for path in annotation_dir.glob("episode_*.json"):
        try:
            indices.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    if indices:
        return sorted(set(indices))
    raise FileNotFoundError("no episode annotations found in {}".format(annotation_dir))


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


def _parquet_files(dataset_root: Path, episode_index: int) -> list[Path]:
    chunk = dataset_root / "data" / "chunk-{:03d}".format(episode_index // 1000)
    files = sorted(chunk.glob("file-*.parquet"))
    if not files:
        raise FileNotFoundError("no Parquet file found in {}".format(chunk))
    return files


def _validate_episode(
    table_columns: Mapping[str, Any],
    episode_index: int,
    action_source: str,
) -> tuple[np.ndarray, np.ndarray]:
    ids = _as_float64(table_columns["episode_index"], name="episode_index")
    selection = ids == float(episode_index)
    count = int(np.count_nonzero(selection))
    if count < 2:
        raise ValueError("episode {} must contain at least two rows".format(episode_index))

    frame_index = _as_float64(table_columns["frame_index"], name="frame_index")[selection]
    timestamps = _as_float64(table_columns["timestamp"], name="timestamp")[selection]
    left_arm_name, left_gripper_name, right_arm_name, right_gripper_name = _SOURCE_COLUMNS[
        action_source
    ]
    left_arm = _as_float64(table_columns[left_arm_name], name=left_arm_name)[selection]
    left_gripper = _as_float64(
        table_columns[left_gripper_name], name=left_gripper_name
    )[selection]
    right_arm = _as_float64(table_columns[right_arm_name], name=right_arm_name)[selection]
    right_gripper = _as_float64(
        table_columns[right_gripper_name], name=right_gripper_name
    )[selection]

    rows = count
    for name, array, shape in (
        ("frame_index", frame_index, (rows,)),
        ("timestamp", timestamps, (rows,)),
        (left_arm_name, left_arm, (rows, 6)),
        (left_gripper_name, left_gripper, (rows,)),
        (right_arm_name, right_arm, (rows, 6)),
        (right_gripper_name, right_gripper, (rows,)),
    ):
        _as_float64(array, name=name, shape=shape)

    # Keep the recorded order, but tolerate a Parquet writer that interleaves
    # rows by sorting on the per-episode frame index.  A malformed or duplicate
    # frame sequence is rejected rather than silently replayed.
    order = np.argsort(frame_index, kind="stable")
    frame_index = frame_index[order]
    timestamps = timestamps[order]
    left_arm = left_arm[order]
    left_gripper = left_gripper[order]
    right_arm = right_arm[order]
    right_gripper = right_gripper[order]
    expected_frames = np.arange(rows, dtype=np.float64)
    if not np.array_equal(frame_index, expected_frames):
        raise ValueError(
            "episode frame_index must contain each frame 0..{} exactly once".format(rows - 1)
        )

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

    piper_actions = np.ascontiguousarray(
        np.column_stack((left_arm, left_gripper, right_arm, right_gripper)), dtype=np.float64
    )
    out_of_bounds = (piper_actions < PIPER_JOINT_LOWER) | (
        piper_actions > PIPER_JOINT_UPPER
    )
    # Piper's gripper encoder can report a small signed value at its mechanical
    # zero.  Keep the configured physical range at [0.0, 0.1], while accepting
    # only this known <=1 mm representation drift; larger negative values fail.
    for gripper_index in (6, 13):
        out_of_bounds[:, gripper_index] = (
            (piper_actions[:, gripper_index] < -GRIPPER_ZERO_TOLERANCE)
            | (piper_actions[:, gripper_index] > PIPER_JOINT_UPPER[gripper_index])
        )
    violations = np.argwhere(out_of_bounds)
    if violations.size:
        frame, joint = (int(value) for value in violations[0])
        value = float(piper_actions[frame, joint])
        raise ValueError(
            "{} action outside Piper bounds at frame={}, piper_index={}: "
            "value={} not in [{}, {}]".format(
                action_source,
                frame,
                joint,
                value,
                float(PIPER_JOINT_LOWER[joint]),
                float(PIPER_JOINT_UPPER[joint]),
            )
        )

    piper_actions.setflags(write=False)
    timestamps = np.ascontiguousarray(timestamps, dtype=np.float64)
    timestamps.setflags(write=False)
    return piper_actions, timestamps


def load_episode(
    data_root: Path,
    episode_index: int,
    action_source: str = DEFAULT_ACTION_SOURCE,
) -> ReplayEpisode:
    """Load and validate exactly one episode from the shared dataset."""

    if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
        raise ValueError("episode_index must be a non-negative integer")
    if action_source not in _SOURCE_COLUMNS:
        raise ValueError("action_source must be one of {}".format(sorted(_SOURCE_COLUMNS)))

    dataset_root = resolve_dataset_root(data_root)
    columns = _REQUIRED_COLUMNS + _SOURCE_COLUMNS[action_source]
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("PyArrow is required to read the replay Parquet file") from exc

    for source_path in _parquet_files(dataset_root, episode_index):
        table = pq.read_table(source_path, columns=list(columns))
        table_columns = table.to_pydict()
        ids = np.asarray(table_columns["episode_index"])
        if not np.any(ids == episode_index):
            continue
        piper_actions, timestamps = _validate_episode(
            table_columns, episode_index, action_source
        )
        return ReplayEpisode(
            episode_index=episode_index,
            source_name=action_source,
            source_path=source_path,
            piper_actions=piper_actions,
            source_timestamps=timestamps,
        )

    raise FileNotFoundError(
        "episode {} was not found in {}".format(episode_index, dataset_root / "data")
    )


class RealRobotReplayServer:
    """Serve one immutable episode repeatedly over the JSON WebSocket protocol."""

    def __init__(self, episode: ReplayEpisode) -> None:
        self.episode = episode

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
            LOG.info(
                "serving replay: episode=%s source=%s request_id=%s frames=%s duration_s=%.6f",
                self.episode.episode_index,
                self.episode.source_name,
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
            LOG.warning("replay request rejected request_id=%s step=%s: %s", request_id, step, exc)
            return build_policy_response(
                request_id=request_id,
                step=step,
                action={},
                error=str(exc),
            )


async def serve(server: RealRobotReplayServer, *, host: str, port: int) -> None:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("websockets is required to run the replay server") from exc
    async with websockets.serve(
        server.handler,
        host,
        port,
        max_size=16 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=60,
    ):
        LOG.info("Piper realrobot replay server listening on ws://%s:%s", host, port)
        await asyncio.Future()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay one realrobot RoboMIND episode; no policy model is loaded"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="dataset directory or a parent directory containing one LeRobot dataset",
    )
    parser.add_argument("--episode-index", type=int, default=DEFAULT_EPISODE_INDEX)
    parser.add_argument(
        "--action-source",
        choices=sorted(_SOURCE_COLUMNS),
        default=DEFAULT_ACTION_SOURCE,
        help="trajectory columns to replay (default: puppet)",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--list-episodes", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_arg_parser().parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root)
    if args.list_episodes:
        for episode_index in list_episode_indices(dataset_root):
            print(episode_index)
        return
    if args.port <= 0 or args.port > 65535:
        raise ValueError("port must be in [1, 65535]")

    episode = load_episode(dataset_root, args.episode_index, args.action_source)
    LOG.info(
        "preflight passed: episode=%s source=%s frames=%s duration_s=%.6f cadence_hz=%.1f path=%s",
        episode.episode_index,
        episode.source_name,
        episode.num_waypoints,
        episode.source_duration_s,
        SOURCE_FPS,
        episode.source_path,
    )
    if args.preflight_only:
        return
    asyncio.run(serve(RealRobotReplayServer(episode), host=args.host, port=args.port))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.info("Piper realrobot replay server stopped")
