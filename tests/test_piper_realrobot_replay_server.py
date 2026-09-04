from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from piper_realrobot_replay_server import (  # noqa: E402
    DEFAULT_ACTION_SOURCE,
    RealRobotReplayServer,
    SOURCE_DT,
    load_episode,
    list_episode_indices,
)


def write_dataset(root: Path, *, episode: int = 0, rows: int = 3) -> None:
    path = root / "data" / "chunk-000"
    path.mkdir(parents=True)
    left = [[0.01, 0.02, -0.03, 0.04, 0.05, -0.06]] * rows
    right = [[0.11, 0.12, -0.13, 0.14, 0.15, -0.16]] * rows
    table = pa.table(
        {
            "episode_index": [episode] * rows,
            "frame_index": list(range(rows)),
            "timestamp": [index * SOURCE_DT for index in range(rows)],
            "puppet.arm_left_position_align.data": left,
            "puppet.end_effector_left_position_align.data": [-0.0002] * rows,
            "puppet.arm_right_position_align.data": right,
            "puppet.end_effector_right_position_align.data": [-0.0001] * rows,
            "master.arm_left_position_align.data": left,
            "master.end_effector_left_position_align.data": [0.0005] * rows,
            "master.arm_right_position_align.data": right,
            "master.end_effector_right_position_align.data": [0.0008] * rows,
        }
    )
    pq.write_table(table, path / "file-000.parquet")
    annotations = root / "anno_semantic"
    annotations.mkdir()
    (annotations / "episode_{:06d}.json".format(episode)).write_text("{}")


class RealRobotLoaderTest(unittest.TestCase):
    def test_loads_selected_episode_and_keeps_source_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root, episode=7)
            episode = load_episode(root, 7)

        self.assertEqual(episode.episode_index, 7)
        self.assertEqual(episode.source_name, DEFAULT_ACTION_SOURCE)
        self.assertEqual(episode.num_waypoints, 3)
        np.testing.assert_allclose(
            episode.piper_actions[0],
            [
                0.01,
                0.02,
                -0.03,
                0.04,
                0.05,
                -0.06,
                -0.0002,
                0.11,
                0.12,
                -0.13,
                0.14,
                0.15,
                -0.16,
                -0.0001,
            ],
        )
        self.assertFalse(episode.piper_actions.flags.writeable)
        self.assertEqual(
            episode.as_piper_action()["time_list"],
            [SOURCE_DT, 2 * SOURCE_DT, 3 * SOURCE_DT],
        )

    def test_master_source_is_selectable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root, episode=3)
            episode = load_episode(root, 3, action_source="master")
        self.assertEqual(episode.source_name, "master")
        self.assertEqual(episode.piper_actions[0, 6], 0.0005)
        self.assertEqual(episode.piper_actions[0, 13], 0.0008)

    def test_loader_rejects_action_outside_piper_limits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root)
            path = root / "data" / "chunk-000" / "file-000.parquet"
            table = pq.read_table(path)
            values = [
                list(value)
                for value in table.column("puppet.arm_left_position_align.data").to_pylist()
            ]
            values[1][0] = 3.0
            table = table.set_column(
                table.schema.get_field_index("puppet.arm_left_position_align.data"),
                "puppet.arm_left_position_align.data",
                pa.array(values),
            )
            pq.write_table(table, path)
            with self.assertRaisesRegex(ValueError, "outside Piper bounds"):
                load_episode(root, 0)

    def test_loader_rejects_large_negative_gripper_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root)
            path = root / "data" / "chunk-000" / "file-000.parquet"
            table = pq.read_table(path)
            values = [-0.01, -0.01, -0.01]
            table = table.set_column(
                table.schema.get_field_index("puppet.end_effector_left_position_align.data"),
                "puppet.end_effector_left_position_align.data",
                pa.array(values),
            )
            pq.write_table(table, path)
            with self.assertRaisesRegex(ValueError, "outside Piper bounds"):
                load_episode(root, 0)

    def test_lists_annotation_episode_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root, episode=12)
            (root / "anno_semantic" / "episode_000013.json").write_text("{}")
            self.assertEqual(list_episode_indices(root), [12, 13])

    def test_server_can_serve_repeated_requests(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_dataset(root)
                server = RealRobotReplayServer(load_episode(root, 0))
                request = {"request_id": "one", "step": 0, "state": {}}
                first = await server.handle_message(json.dumps(request))
                second = await server.handle_message(json.dumps({**request, "request_id": "two"}))
                self.assertTrue(first["ok"])
                self.assertEqual(len(first["action"]["left_arm"]), 3)
                self.assertTrue(second["ok"])
                self.assertEqual(first["action"], second["action"])

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
