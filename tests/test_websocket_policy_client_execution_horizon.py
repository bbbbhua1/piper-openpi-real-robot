from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

if "websockets" not in sys.modules:
    sys.modules["websockets"] = types.ModuleType("websockets")

import websocket_policy_client as policy_client  # noqa: E402


class ExecutionHorizonArgumentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.required_args = [
            "--uri",
            "ws://127.0.0.1:18003",
            "--instruction",
            "crimp",
        ]

    def test_execution_horizon_sets_max_waypoints(self) -> None:
        args = policy_client.build_arg_parser().parse_args(
            self.required_args + ["--execution-horizon", "64"]
        )

        self.assertEqual(args.max_waypoints, 64)

    def test_legacy_max_waypoints_alias_is_preserved(self) -> None:
        args = policy_client.build_arg_parser().parse_args(
            self.required_args + ["--max-waypoints", "40"]
        )

        self.assertEqual(args.max_waypoints, 40)

    def test_execution_horizon_truncates_synchronized_action_fields(self) -> None:
        action = {
            "left_arm": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "right_arm": [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            "left_gripper": [0.1, 0.2, 0.3],
            "right_gripper": [0.4, 0.5, 0.6],
            "time_list": [0.1, 0.2, 0.3],
        }

        chunk = policy_client.parse_chunked_action(
            action,
            default_dt=0.05,
            left_arm_dim=2,
            right_arm_dim=2,
            max_waypoints=2,
        )

        self.assertEqual(chunk["num_waypoints"], 2)
        self.assertEqual(chunk["source_num_waypoints"], 3)
        self.assertEqual(chunk["left_arm"], action["left_arm"][:2])
        self.assertEqual(chunk["right_arm"], action["right_arm"][:2])
        self.assertEqual(chunk["left_gripper"], [[0.1], [0.2]])
        self.assertEqual(chunk["right_gripper"], [[0.4], [0.5]])
        self.assertEqual(chunk["time_list"], action["time_list"][:2])
        self.assertEqual(chunk["executed_waypoints"], 2)
        self.assertEqual(chunk["interp_factor"], 1)

    def test_action_interp_factor_argument_is_parsed(self) -> None:
        args = policy_client.build_arg_parser().parse_args(
            self.required_args + ["--action-interp-factor", "5"]
        )

        self.assertEqual(args.interp_factor, 5)

    def test_interp_factor_one_keeps_original_waypoints(self) -> None:
        action = {
            "left_arm": [[0.0, 0.0], [1.0, 2.0]],
            "right_arm": [[3.0, 4.0], [5.0, 6.0]],
            "time_list": [0.1, 0.2],
        }

        chunk = policy_client.parse_chunked_action(
            action,
            default_dt=0.1,
            left_arm_dim=2,
            right_arm_dim=2,
            interp_factor=1,
        )

        self.assertEqual(chunk["num_waypoints"], 2)
        self.assertEqual(chunk["left_arm"], action["left_arm"])
        self.assertEqual(chunk["time_list"], action["time_list"])

    def test_interp_factor_inserts_linear_midpoints_and_keeps_duration(self) -> None:
        action = {
            "left_arm": [[0.0, 0.0], [1.0, 2.0]],
            "right_arm": [[3.0, 4.0], [5.0, 6.0]],
            "left_gripper": [0.0, 1.0],
            "right_gripper": [0.2, 0.4],
            "time_list": [0.1, 0.2],
        }

        chunk = policy_client.parse_chunked_action(
            action,
            default_dt=0.1,
            left_arm_dim=2,
            right_arm_dim=2,
            interp_factor=2,
            smooth_gripper=True,
        )

        self.assertEqual(chunk["num_waypoints"], 3)
        self.assertEqual(chunk["executed_waypoints"], 2)
        self.assertEqual(chunk["left_arm"], [[0.0, 0.0], [0.5, 1.0], [1.0, 2.0]])
        self.assertEqual(chunk["right_arm"], [[3.0, 4.0], [4.0, 5.0], [5.0, 6.0]])
        self.assertEqual(chunk["left_gripper"], [[0.0], [0.5], [1.0]])
        self.assertEqual(len(chunk["right_gripper"]), 3)
        self.assertAlmostEqual(chunk["right_gripper"][1][0], 0.3)
        self.assertEqual(chunk["right_gripper"][0], [0.2])
        self.assertEqual(chunk["right_gripper"][2], [0.4])
        self.assertEqual(len(chunk["time_list"]), 3)
        self.assertAlmostEqual(chunk["time_list"][0], 0.1)
        self.assertAlmostEqual(chunk["time_list"][1], 0.15)
        self.assertAlmostEqual(chunk["time_list"][2], 0.2)

    def test_horizon_is_applied_before_interpolation(self) -> None:
        action = {
            "left_arm": [[0.0], [1.0], [10.0]],
            "right_arm": [[2.0], [3.0], [30.0]],
            "time_list": [0.1, 0.2, 0.3],
        }

        chunk = policy_client.parse_chunked_action(
            action,
            default_dt=0.1,
            left_arm_dim=1,
            right_arm_dim=1,
            max_waypoints=2,
            interp_factor=2,
        )

        self.assertEqual(chunk["source_num_waypoints"], 3)
        self.assertEqual(chunk["executed_waypoints"], 2)
        self.assertEqual(chunk["num_waypoints"], 3)
        self.assertEqual(chunk["left_arm"], [[0.0], [0.5], [1.0]])
        self.assertNotIn([10.0], chunk["left_arm"])

    def test_action_chunk_blend_steps_argument_is_parsed(self) -> None:
        args = policy_client.build_arg_parser().parse_args(
            self.required_args + ["--action-chunk-blend-steps", "5"]
        )

        self.assertEqual(args.blend_steps, 5)

    def test_blend_steps_zero_keeps_chunk(self) -> None:
        chunk = policy_client.parse_chunked_action(
            {
                "left_arm": [[1.0], [2.0]],
                "right_arm": [[3.0], [4.0]],
                "time_list": [0.1, 0.2],
            },
            default_dt=0.1,
            left_arm_dim=1,
            right_arm_dim=1,
        )
        blended = policy_client.blend_action_chunks(
            chunk,
            previous_left=[0.0],
            previous_right=[0.0],
            blend_steps=0,
            default_dt=0.1,
            left_arm_dim=1,
            right_arm_dim=1,
        )

        self.assertEqual(blended["left_arm"], [[1.0], [2.0]])
        self.assertEqual(blended["blend_steps"], 0)

    def test_blend_steps_inserts_linear_bridge_from_previous_chunk(self) -> None:
        chunk = policy_client.parse_chunked_action(
            {
                "left_arm": [[1.0], [2.0]],
                "right_arm": [[3.0], [4.0]],
                "time_list": [0.1, 0.2],
            },
            default_dt=0.1,
            left_arm_dim=1,
            right_arm_dim=1,
        )
        blended = policy_client.blend_action_chunks(
            chunk,
            previous_left=[0.0],
            previous_right=[1.0],
            blend_steps=2,
            default_dt=0.1,
            left_arm_dim=1,
            right_arm_dim=1,
        )

        self.assertEqual(blended["blend_steps"], 2)
        self.assertEqual(blended["num_waypoints"], 3)
        self.assertEqual(blended["left_arm"], [[0.5], [1.0], [2.0]])
        self.assertEqual(blended["right_arm"], [[2.0], [3.0], [4.0]])
        self.assertEqual(len(blended["time_list"]), 3)
        self.assertAlmostEqual(blended["time_list"][0], 0.1)
        self.assertAlmostEqual(blended["time_list"][1], 0.2)
        self.assertAlmostEqual(blended["time_list"][2], 0.3)

    def test_prepare_skips_blend_without_previous_pose(self) -> None:
        chunk = policy_client.prepare_executed_chunk(
            {
                "left_arm": [[1.0], [2.0]],
                "right_arm": [[3.0], [4.0]],
                "time_list": [0.1, 0.2],
            },
            default_dt=0.1,
            left_arm_dim=1,
            right_arm_dim=1,
            blend_steps=5,
        )

        self.assertEqual(chunk["num_waypoints"], 2)
        self.assertEqual(chunk["blend_steps"], 0)
        self.assertEqual(chunk["left_arm"], [[1.0], [2.0]])

    def test_gripper_is_stepwise_when_smoothing_joints_only(self) -> None:
        chunk = policy_client.parse_chunked_action(
            {
                "left_arm": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.08]],
                "right_arm": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.06]],
                "time_list": [0.1, 0.2],
            },
            default_dt=0.1,
            left_arm_dim=7,
            right_arm_dim=7,
            interp_factor=2,
            smooth_gripper=False,
        )

        self.assertEqual(chunk["left_arm"][1][:6], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        self.assertEqual(chunk["left_arm"][1][6], 0.0)
        self.assertEqual(chunk["left_arm"][2][6], 0.08)
        self.assertEqual(chunk["right_arm"][1][6], 0.0)
        self.assertEqual(chunk["right_arm"][2][6], 0.06)

    def test_separate_gripper_field_is_stepwise_by_default(self) -> None:
        chunk = policy_client.parse_chunked_action(
            {
                "left_arm": [[0.0, 0.0], [1.0, 2.0]],
                "right_arm": [[3.0, 4.0], [5.0, 6.0]],
                "left_gripper": [0.0, 1.0],
                "right_gripper": [0.2, 0.4],
                "time_list": [0.1, 0.2],
            },
            default_dt=0.1,
            left_arm_dim=2,
            right_arm_dim=2,
            interp_factor=2,
        )

        self.assertEqual(chunk["left_arm"], [[0.0, 0.0], [0.5, 1.0], [1.0, 2.0]])
        self.assertEqual(chunk["left_gripper"], [[0.0], [0.0], [1.0]])
        self.assertEqual(chunk["right_gripper"], [[0.2], [0.2], [0.4]])

    def test_blend_keeps_previous_gripper_until_last_step(self) -> None:
        chunk = policy_client.parse_chunked_action(
            {
                "left_arm": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05]],
                "right_arm": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04]],
                "time_list": [0.1],
            },
            default_dt=0.1,
            left_arm_dim=7,
            right_arm_dim=7,
        )
        blended = policy_client.blend_action_chunks(
            chunk,
            previous_left=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01],
            previous_right=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02],
            blend_steps=2,
            default_dt=0.1,
            left_arm_dim=7,
            right_arm_dim=7,
            smooth_gripper=False,
        )

        self.assertAlmostEqual(blended["left_arm"][0][0], 0.5)
        self.assertAlmostEqual(blended["left_arm"][0][6], 0.01)
        self.assertAlmostEqual(blended["left_arm"][1][6], 0.05)

    def test_smooth_gripper_flag_is_parsed(self) -> None:
        on_args = policy_client.build_arg_parser().parse_args(
            self.required_args + ["--smooth-gripper"]
        )
        off_args = policy_client.build_arg_parser().parse_args(
            self.required_args + ["--no-smooth-gripper"]
        )

        self.assertTrue(on_args.smooth_gripper)
        self.assertFalse(off_args.smooth_gripper)


if __name__ == "__main__":
    unittest.main()
