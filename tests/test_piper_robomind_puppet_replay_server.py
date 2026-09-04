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

from piper_robomind_puppet_replay_server import (  # noqa: E402
    EXPECTED_WAYPOINTS,
    PuppetReplayServer,
    ReplayEpisode,
    SOURCE_DT,
    load_episode,
)
from piper_start_pose import PIPER_WIRE_ORDER, StartPoseConfig  # noqa: E402


LEFT = [0.15348975360393524, 1.580007791519165, -1.0552573204040527, -0.9023781418800354, 1.180993676185608, 2.010979175567627]
RIGHT = [-0.0006628719856962562, 0.006367059890180826, -0.004988984204828739, 0.012629455886781216, 0.08828408271074295, 0.029096592217683792]


def make_config() -> StartPoseConfig:
    return StartPoseConfig(
        name="episode382",
        source_dataset="unused",
        source_episode=382,
        piper_wire_order=PIPER_WIRE_ORDER,
        left_arm=tuple(LEFT + [0.0005]),
        right_arm=tuple(RIGHT + [0.0008]),
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


def write_episode(root: Path, *, episode: int = 382, rows: int = EXPECTED_WAYPOINTS) -> None:
    path = root / "data" / "chunk-000"
    path.mkdir(parents=True)
    left = [LEFT] * rows
    left_gripper = [0.0005] * rows
    right = [RIGHT] * rows
    right_gripper = [0.0008] * rows
    table = pa.table(
        {
            "episode_index": [episode] * rows,
            "timestamp": [index * SOURCE_DT for index in range(rows)],
            "puppet.arm_left_position_align.data": left,
            "puppet.end_effector_left_position_align.data": left_gripper,
            "puppet.arm_right_position_align.data": right,
            "puppet.end_effector_right_position_align.data": right_gripper,
        }
    )
    pq.write_table(table, path / "file-000.parquet")


class PuppetLoaderTest(unittest.TestCase):
    def test_loader_builds_constant_left_and_right_puppet_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_episode(root)
            episode = load_episode(root, 382, make_config())

        self.assertEqual(episode.num_waypoints, EXPECTED_WAYPOINTS)
        np.testing.assert_allclose(episode.piper_actions[0, :7], LEFT + [0.0005])
        np.testing.assert_allclose(episode.piper_actions[0, 7:], RIGHT + [0.0008])
        np.testing.assert_allclose(
            episode.piper_actions[:, :7],
            np.tile(episode.piper_actions[0, :7], (EXPECTED_WAYPOINTS, 1)),
        )
        np.testing.assert_allclose(
            episode.as_piper_action()["time_list"][:3],
            [SOURCE_DT, 2.0 * SOURCE_DT, 3.0 * SOURCE_DT],
        )
        self.assertFalse(episode.piper_actions.flags.writeable)

    def test_loader_rejects_another_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "only verified Puppet episode 382"):
                load_episode(Path(directory), 381, make_config())

    def test_loader_rejects_wrong_waypoint_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_episode(root, rows=2)
            with self.assertRaisesRegex(ValueError, "must contain 453 waypoints"):
                load_episode(root, 382, make_config())

    def test_loader_rejects_value_outside_piper_limits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_episode(root)
            path = root / "data" / "chunk-000" / "file-000.parquet"
            table = pq.read_table(path)
            right = [list(value) for value in table.column("puppet.arm_right_position_align.data").to_pylist()]
            right[12][0] = 21.0
            table = table.set_column(
                table.schema.get_field_index("puppet.arm_right_position_align.data"),
                "puppet.arm_right_position_align.data",
                pa.array(right),
            )
            pq.write_table(table, path)
            with self.assertRaisesRegex(ValueError, "outside Piper bounds"):
                load_episode(root, 382, make_config())


class PuppetServerTest(unittest.IsolatedAsyncioTestCase):
    def _episode(self) -> ReplayEpisode:
        actions = np.zeros((2, 14), dtype=np.float64)
        timestamps = np.asarray([0.0, SOURCE_DT], dtype=np.float64)
        actions.setflags(write=False)
        timestamps.setflags(write=False)
        return ReplayEpisode(382, Path("/episode_382.parquet"), actions, timestamps)

    async def test_first_request_returns_exact_chunk_and_second_is_rejected(self) -> None:
        server = PuppetReplayServer(self._episode())
        request = {"request_id": "one", "step": 0, "state": {}}

        first = await server.handle_message(json.dumps(request))
        second = await server.handle_message(json.dumps({**request, "request_id": "two", "step": 1}))

        self.assertTrue(first["ok"])
        self.assertEqual(first["action"], self._episode().as_piper_action())
        self.assertFalse(second["ok"])
        self.assertIn("already served", second["error"])


if __name__ == "__main__":
    unittest.main()
