#!/usr/bin/env python3
"""Dependency-light tests for raw-action right-arm extraction."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("replay_raw_action_hdf5.py")
SPEC = importlib.util.spec_from_file_location("replay_raw_action_hdf5", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
REPLAY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPLAY
SPEC.loader.exec_module(REPLAY)

try:
    import h5py  # type: ignore
except ImportError:  # pragma: no cover
    h5py = None


class ReplayMappingTests(unittest.TestCase):
    def test_model_row_extracts_right_arm_and_right_gripper(self) -> None:
        self.assertEqual(
            REPLAY.model_action_to_right_wire(list(range(14))),
            [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 13.0],
        )

    def test_invalid_width_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 14"):
            REPLAY.model_action_to_right_wire([0.0] * 13)


@unittest.skipIf(h5py is None, "h5py is required for HDF5 loading tests")
class ReplayHDF5Tests(unittest.TestCase):
    def test_loader_selects_rows_and_request_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "raw.hdf5"
            with h5py.File(path, "w") as h5_file:
                h5_file.create_dataset("action", data=[list(range(14)), [20 + i for i in range(14)]])
                h5_file.create_dataset("request_index", data=[3, 4])
                h5_file.attrs["raw_model_output"] = True
                h5_file.attrs["model_action_layout"] = (
                    "left_arm6,right_arm6,left_gripper,right_gripper"
                )
            data = REPLAY.load_replay_data(path, start_index=1, max_steps=1)
            self.assertEqual(data.right_actions, [[26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 33.0]])
            self.assertEqual(data.request_indices, [4])
            self.assertEqual(data.metadata["selected_rows"], "1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
