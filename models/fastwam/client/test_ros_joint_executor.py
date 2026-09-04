from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest
from unittest import mock


CLIENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CLIENT_DIR))

if "websockets" not in sys.modules:
    sys.modules["websockets"] = types.ModuleType("websockets")
import websocket_policy_client as policy_client  # noqa: E402


class _Publisher:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def publish(self, message: object) -> None:
        self.messages.append(message)


class RosJointExecutorHoldTest(unittest.TestCase):
    def make_executor(self) -> policy_client.RosJointExecutor:
        executor = policy_client.RosJointExecutor.__new__(policy_client.RosJointExecutor)
        executor.left_pub = _Publisher()
        executor.right_pub = _Publisher()
        executor.left_arm_dim = 7
        executor.right_arm_dim = 7
        executor._last_published_target = None
        executor._build_joint_msg = lambda positions: tuple(positions)
        return executor

    def test_publish_pair_remembers_exact_last_command(self) -> None:
        executor = self.make_executor()

        executor._publish_pair([1, 2], [3, 4])

        self.assertEqual(executor.left_pub.messages, [(1.0, 2.0)])
        self.assertEqual(executor.right_pub.messages, [(3.0, 4.0)])
        self.assertEqual(executor._last_published_target, ([1.0, 2.0], [3.0, 4.0]))

    def test_hold_uses_last_command_instead_of_stale_observation(self) -> None:
        executor = self.make_executor()
        executor._last_published_target = ([0.11] * 7, [-0.22] * 7)
        executor.set_enable = mock.Mock()
        executor._publish_pair = mock.Mock()
        stale_state = {"left_arm": [0.0] * 7, "right_arm": [0.0] * 7}

        with mock.patch.object(policy_client.time, "sleep", side_effect=KeyboardInterrupt):
            executor.hold_position(stale_state, hold_hz=20.0)

        executor.set_enable.assert_called_once_with(True, "hold")
        executor._publish_pair.assert_called_once_with([0.11] * 7, [-0.22] * 7)

    def test_hold_falls_back_to_observed_state_before_any_command(self) -> None:
        executor = self.make_executor()
        executor.set_enable = mock.Mock()
        executor._publish_pair = mock.Mock()
        state = {"left_arm": [0.3] * 7, "right_arm": [-0.4] * 7}

        with mock.patch.object(policy_client.time, "sleep", side_effect=KeyboardInterrupt):
            executor.hold_position(state, hold_hz=20.0)

        executor._publish_pair.assert_called_once_with([0.3] * 7, [-0.4] * 7)

    def test_move_to_pose_publishes_exact_target_and_checks_safety(self) -> None:
        executor = self.make_executor()
        executor.set_enable = mock.Mock()
        executor._raise_if_shutdown = mock.Mock()
        safety_checks = []
        target_left = [0.02, 0.01, -0.01, 0.03, 0.14, -0.02, 0.0005]
        target_right = [0.01, 0.005, -0.001, -0.01, 0.15, 0.0, 0.0008]
        state = {"left_arm": [0.0] * 7, "right_arm": [0.0] * 7}

        with mock.patch.object(policy_client.time, "sleep"):
            duration = executor.move_to_pose(
                state,
                target_left,
                target_right,
                move_hz=20.0,
                min_duration=0.1,
                max_duration=2.0,
                joint_speed=0.15,
                safety_check=lambda: safety_checks.append(True),
                label="FastWAM Puzzle training start",
            )

        self.assertGreaterEqual(duration, 0.1)
        self.assertEqual(executor._last_published_target, (target_left, target_right))
        self.assertGreater(len(safety_checks), 0)
        executor.set_enable.assert_called_once_with(True, "FastWAM Puzzle training start")

    def test_move_to_pose_refuses_to_exceed_configured_speed(self) -> None:
        executor = self.make_executor()
        executor.set_enable = mock.Mock()
        executor._raise_if_shutdown = mock.Mock()
        state = {"left_arm": [0.0] * 7, "right_arm": [0.0] * 7}

        with self.assertRaisesRegex(ValueError, "max_duration"):
            executor.move_to_pose(
                state,
                [1.0] + [0.0] * 6,
                [0.0] * 7,
                move_hz=20.0,
                min_duration=0.1,
                max_duration=2.0,
                joint_speed=0.15,
                safety_check=lambda: None,
                label="test",
            )

        executor.set_enable.assert_not_called()


if __name__ == "__main__":
    unittest.main()
