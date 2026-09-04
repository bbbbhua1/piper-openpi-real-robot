from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest


CLIENT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = CLIENT_DIR / "piper_fastwam_puzzle_start_pose.json"
sys.path.insert(0, str(CLIENT_DIR))

if "websockets" not in sys.modules:
    sys.modules["websockets"] = types.ModuleType("websockets")

import websocket_policy_client as policy_client  # noqa: E402
from piper_start_pose import load_start_pose_config  # noqa: E402


class _FakeSource:
    def __init__(self, state):
        self.state = state
        self.safety_checks = 0

    def get_state(self):
        return self.state

    def raise_if_safety_tripped(self):
        self.safety_checks += 1


class _FakeClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, duration):
        self.value += duration


class WebsocketPolicyClientStartPoseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_start_pose_config(CONFIG_PATH)

    def _required_args(self):
        return ["--uri", "ws://127.0.0.1:7083", "--instruction", "crimp"]

    def test_conflicting_start_pose_and_no_home_are_rejected(self) -> None:
        args = policy_client.build_arg_parser().parse_args(
            self._required_args()
            + ["--start-pose-config", str(CONFIG_PATH), "--no-home-on-start"]
        )

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            policy_client.validate_initialization_args(args)

    def test_feedback_gate_accepts_pose_within_tolerances(self) -> None:
        state = {
            "left_arm": list(self.config.left_arm),
            "right_arm": list(self.config.right_arm),
        }
        source = _FakeSource(state)
        clock = _FakeClock()

        result = policy_client.wait_until_start_pose(
            source,
            self.config,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

        self.assertEqual(result, state)
        self.assertEqual(source.safety_checks, 1)

    def test_feedback_gate_times_out_for_wrong_pose(self) -> None:
        source = _FakeSource({"left_arm": [0.0] * 7, "right_arm": [0.0] * 7})
        clock = _FakeClock()

        with self.assertRaisesRegex(RuntimeError, "feedback did not reach"):
            policy_client.wait_until_start_pose(
                source,
                self.config,
                sleep=clock.sleep,
                monotonic=clock.monotonic,
            )

        self.assertGreater(source.safety_checks, 0)


if __name__ == "__main__":
    unittest.main()
