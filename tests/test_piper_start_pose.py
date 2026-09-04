from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
CONFIG_PATH = REPO_ROOT / "configs" / "piper_fastwam_puzzle_start_pose.json"
sys.path.insert(0, str(SCRIPTS_DIR))

from piper_start_pose import (  # noqa: E402
    load_start_pose_config,
    pose_errors,
    pose_is_reached,
    validate_state_within_bounds,
)


class PiperStartPoseConfigTest(unittest.TestCase):
    def _valid_payload(self) -> dict[str, object]:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def _load_payload(self, payload: dict[str, object]):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pose.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return load_start_pose_config(path)

    def test_loads_runtime_screened_episode_159_pose(self) -> None:
        config = load_start_pose_config(CONFIG_PATH)

        self.assertEqual(config.source_episode, 159)
        self.assertEqual(
            config.left_arm,
            (
                0.018856963,
                0.015263500,
                -0.003558576,
                0.035690423,
                0.091668218,
                -0.025258912,
                0.0005,
            ),
        )
        self.assertEqual(config.right_arm[-1], 0.0008)
        self.assertLess(config.max_joint_speed, 0.3)

    def test_rejects_wrong_dimensions_nonfinite_and_out_of_bounds_target(self) -> None:
        cases = []
        wrong_dimensions = self._valid_payload()
        wrong_dimensions["left_arm"] = [0.0] * 6
        cases.append((wrong_dimensions, "left_arm"))

        nonfinite = self._valid_payload()
        nonfinite["right_arm"][0] = math.inf
        cases.append((nonfinite, "right_arm"))

        outside_bounds = self._valid_payload()
        outside_bounds["left_arm"][1] = -0.01
        cases.append((outside_bounds, "outside verified bounds"))

        for payload, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self._load_payload(payload)

    def test_rejects_invalid_motion_and_tolerance_settings(self) -> None:
        for key, value, message in (
            ("move_hz", 0.0, "move_hz"),
            ("max_joint_speed", 0.3, "hardware speed limit"),
            ("arm_tolerance", -0.1, "arm_tolerance"),
            ("feedback_timeout_s", False, "feedback_timeout_s"),
        ):
            payload = self._valid_payload()
            payload[key] = value
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, message):
                    self._load_payload(payload)

    def test_pose_reached_uses_separate_arm_and_gripper_tolerances(self) -> None:
        config = load_start_pose_config(CONFIG_PATH)
        state = {
            "left_arm": list(config.left_arm),
            "right_arm": list(config.right_arm),
        }
        state["left_arm"][0] += 0.009
        state["right_arm"][6] += 0.0009

        arm_error, gripper_error = pose_errors(state, config)

        self.assertAlmostEqual(arm_error, 0.009)
        self.assertAlmostEqual(gripper_error, 0.0009)
        self.assertTrue(pose_is_reached(state, config))

        state["left_arm"][0] += 0.002
        self.assertFalse(pose_is_reached(state, config))

    def test_current_state_must_be_finite_and_within_verified_bounds(self) -> None:
        config = load_start_pose_config(CONFIG_PATH)
        state = {
            "left_arm": list(config.left_arm),
            "right_arm": list(config.right_arm),
        }
        validate_state_within_bounds(state, config)

        state["right_arm"][1] = -0.01
        with self.assertRaisesRegex(ValueError, "current state is outside verified bounds"):
            validate_state_within_bounds(state, config)

        state["right_arm"][1] = math.nan
        with self.assertRaisesRegex(ValueError, "state.right_arm"):
            validate_state_within_bounds(state, config)


if __name__ == "__main__":
    unittest.main()
