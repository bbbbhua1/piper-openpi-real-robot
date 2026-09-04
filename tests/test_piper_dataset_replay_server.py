from __future__ import annotations

import asyncio
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from piper_dataset_replay_server import (  # noqa: E402
    DatasetReplayServer,
    ReplayEpisode,
    SOURCE_DT,
    load_episode,
)
from piper_start_pose import PIPER_WIRE_ORDER, StartPoseConfig  # noqa: E402


def make_config() -> StartPoseConfig:
    return StartPoseConfig(
        name="episode159",
        source_dataset="unused",
        source_episode=159,
        piper_wire_order=PIPER_WIRE_ORDER,
        left_arm=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.01),
        right_arm=(7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.02),
        joint_lower=(-20.0,) * 14,
        joint_upper=(20.0,) * 14,
        move_hz=20.0,
        max_joint_speed=0.15,
        min_duration_s=2.0,
        max_duration_s=20.0,
        feedback_timeout_s=5.0,
        arm_tolerance=0.01,
        gripper_tolerance=0.001,
    )


def source_row(*, first: float = 1.0) -> list[float]:
    return [first, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.01, 0.02]


def write_episode(root: Path, actions: list[list[float]], timestamps: list[float]) -> None:
    path = root / "data" / "chunk-000"
    path.mkdir(parents=True)
    table = pa.table({"timestamp": pa.array(timestamps, type=pa.float64()), "action": actions})
    pq.write_table(table, path / "episode_000159.parquet")


class DatasetReplayLoaderTest(unittest.TestCase):
    def test_loader_maps_fastwam_source_to_piper_wire_order_without_value_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [source_row(), source_row(), source_row()]
            write_episode(root, rows, [0.0, SOURCE_DT, 2.0 * SOURCE_DT])

            episode = load_episode(root, 159, make_config())
            action = episode.as_piper_action()

        self.assertEqual(action["left_arm"][0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.01])
        self.assertEqual(action["right_arm"][0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.02])
        np.testing.assert_allclose(action["time_list"], [SOURCE_DT, 2.0 * SOURCE_DT, 3.0 * SOURCE_DT])
        self.assertTrue(episode.piper_actions.flags.writeable is False)
        self.assertTrue(episode.source_timestamps.flags.writeable is False)

    def test_loader_rejects_value_outside_absolute_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_episode(root, [source_row(), source_row(first=21.0)], [0.0, SOURCE_DT])

            with self.assertRaisesRegex(ValueError, "outside verified absolute bounds"):
                load_episode(root, 159, make_config())

    def test_loader_rejects_wrong_cadence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_episode(root, [source_row(), source_row()], [0.0, 0.05])

            with self.assertRaisesRegex(ValueError, "timestamp cadence"):
                load_episode(root, 159, make_config())


class DatasetReplayServerTest(unittest.IsolatedAsyncioTestCase):
    def _episode(self) -> ReplayEpisode:
        actions = np.asarray(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.01, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.02]],
            dtype=np.float64,
        )
        timestamps = np.asarray([0.0], dtype=np.float64)
        actions.setflags(write=False)
        timestamps.setflags(write=False)
        return ReplayEpisode(159, Path("/episode_000159.parquet"), actions, timestamps)

    async def test_first_request_returns_exact_chunk_and_second_request_is_rejected(self) -> None:
        server = DatasetReplayServer(self._episode())
        request = {"request_id": "one", "step": 0, "state": {"left_arm": [0.0] * 7, "right_arm": [0.0] * 7}}

        first = await server.handle_message(json.dumps(request))
        second = await server.handle_message(json.dumps({**request, "request_id": "two", "step": 1}))

        self.assertTrue(first["ok"])
        self.assertEqual(first["action"], self._episode().as_piper_action())
        self.assertFalse(second["ok"])
        self.assertIn("already served", second["error"])

    async def test_request_without_state_is_rejected_without_consuming_replay(self) -> None:
        server = DatasetReplayServer(self._episode())

        invalid = await server.handle_message(json.dumps({"request_id": "bad", "step": 0}))
        valid = await server.handle_message(json.dumps({"request_id": "good", "step": 0, "state": {}}))

        self.assertFalse(invalid["ok"])
        self.assertTrue(valid["ok"])


if __name__ == "__main__":
    unittest.main()
