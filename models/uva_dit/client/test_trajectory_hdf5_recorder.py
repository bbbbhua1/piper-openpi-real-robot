#!/usr/bin/env python3
"""Tests for robot-local replayable Piper HDF5 trajectories."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest


try:
    import h5py  # noqa: F401
except ImportError:
    h5py = None

from trajectory_hdf5_recorder import TrajectoryHDF5Recorder, piper_state_to_wire_row


@unittest.skipUnless(h5py is not None, "h5py is required for HDF5 recorder tests")
class TrajectoryHDF5RecorderTests(unittest.TestCase):
    def test_wire_order_and_reopenable_schema(self) -> None:
        left = [float(index) for index in range(7)]
        right = [float(10 + index) for index in range(7)]
        state = {"left_arm": left, "right_arm": right}
        with tempfile.TemporaryDirectory() as tmp:
            recorder = TrajectoryHDF5Recorder(
                tmp,
                session_name="test_session",
                instruction="拼图（圆形）",
            )
            recorder.set_initial_state(state)
            recorder.append(
                {"left_arm": left, "right_arm": right},
                state,
                request_index=3,
                waypoint_index=0,
            )
            recorder.append(
                {"left_arm": [value + 1.0 for value in left], "right_arm": right},
                state,
                request_index=3,
                waypoint_index=1,
            )
            path = recorder.path
            recorder.close()

            with h5py.File(str(path), "r") as h5:
                self.assertEqual(h5["action"].shape, (2, 14))
                self.assertEqual(h5["state"].shape, (2, 14))
                self.assertEqual(
                    h5["action"][0].tolist(),
                    piper_state_to_wire_row(state),
                )
                self.assertEqual(
                    h5["initial_state"][...].tolist(),
                    piper_state_to_wire_row(state),
                )
                self.assertEqual(h5["request_index"][...].tolist(), [3, 3])
                self.assertEqual(h5["waypoint_index"][...].tolist(), [0, 1])
                self.assertEqual(h5.attrs["action_type"], "absolute_joint_position")
                self.assertEqual(h5.attrs["layout"], "[L1..L6,Lg,R1..R6,Rg]")
                self.assertEqual(h5.attrs["num_rows"], 2)
                self.assertTrue(bool(h5.attrs["initial_state_recorded"]))

    def test_rejects_invalid_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = TrajectoryHDF5Recorder(tmp, session_name="invalid")
            with self.assertRaises(ValueError):
                recorder.set_initial_state(
                    {"left_arm": [0.0] * 6, "right_arm": [0.0] * 7}
                )
            recorder.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
