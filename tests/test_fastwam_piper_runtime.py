"""CPU-only tests for the FastWAM/Piper runtime boundary.

The runtime module deliberately has no FastWAM, CUDA, ROS, or WebSocket imports so
these safety-critical layout and image transformations can be exercised anywhere.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import fastwam_piper_runtime as runtime  # noqa: E402


class PiperFastWAMLayoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = {
            "left_arm": [1, 2, 3, 4, 5, 6, 7],
            "right_arm": [8, 9, 10, 11, 12, 13, 14],
        }

    def test_piper_wire_to_fastwam_reorders_grippers(self) -> None:
        self.assertEqual(
            runtime.piper_wire_to_fastwam(self.state).tolist(),
            [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 7, 14],
        )

    def test_fastwam_to_piper_wire_is_exact_inverse(self) -> None:
        action = np.asarray(
            [[1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 7, 14]],
            dtype=np.float32,
        )

        result = runtime.fastwam_to_piper_wire(action, dt=1 / 30)

        self.assertEqual(result["left_arm"], [[1, 2, 3, 4, 5, 6, 7]])
        self.assertEqual(result["right_arm"], [[8, 9, 10, 11, 12, 13, 14]])
        self.assertEqual(result["time_list"], [1 / 30])
        self.assertEqual(result["dt"], 1 / 30)

    def test_nonfinite_or_wrong_dimension_state_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "seven finite"):
            runtime.piper_wire_to_fastwam(
                {"left_arm": [0] * 6, "right_arm": [0] * 7}
            )
        with self.assertRaisesRegex(ValueError, "seven finite"):
            runtime.piper_wire_to_fastwam(
                {"left_arm": [0] * 6 + [float("nan")], "right_arm": [0] * 7}
            )

    def test_nonfinite_or_wrong_dimension_actions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[T, 14\]"):
            runtime.fastwam_to_piper_wire(np.zeros((1, 13), dtype=np.float32), dt=0.1)
        with self.assertRaisesRegex(ValueError, "finite"):
            runtime.fastwam_to_piper_wire(
                np.full((1, 14), np.inf, dtype=np.float32), dt=0.1
            )
        with self.assertRaisesRegex(ValueError, "positive finite"):
            runtime.fastwam_to_piper_wire(np.zeros((1, 14), dtype=np.float32), dt=0.0)


class PiperFastWAMImageTest(unittest.TestCase):
    def test_compose_fastwam_image_has_exact_robotwin_geometry(self) -> None:
        head = np.zeros((24, 32, 3), dtype=np.uint8)
        left = np.full((24, 32, 3), 127, dtype=np.uint8)
        right = np.full((24, 32, 3), 255, dtype=np.uint8)

        image = runtime.compose_fastwam_image(head, left, right)

        self.assertEqual(image.shape, (3, 384, 320))
        self.assertEqual(image.dtype, np.float32)
        self.assertGreaterEqual(float(image.min()), -1.0)
        self.assertLessEqual(float(image.max()), 1.0)
        self.assertTrue(np.allclose(image[:, :256, :], -1.0))
        self.assertTrue(np.allclose(image[:, 256:, :160], 127 / 127.5 - 1.0))
        self.assertTrue(np.allclose(image[:, 256:, 160:], 1.0))

    def test_compose_fastwam_image_supports_caller_selected_bgr_conversion(self) -> None:
        blue_bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        blue_bgr[:, :, 0] = 255
        black = np.zeros((2, 2, 3), dtype=np.uint8)

        image = runtime.compose_fastwam_image(
            blue_bgr, black, black, input_color="bgr"
        )

        self.assertTrue(np.allclose(image[2, :256, :], 1.0))
        self.assertTrue(np.allclose(image[:2, :256, :], -1.0))

    def test_compose_fastwam_image_rejects_non_rgb_or_nonfinite_inputs(self) -> None:
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "HWC.*three channels"):
            runtime.compose_fastwam_image(np.zeros((2, 2), dtype=np.uint8), rgb, rgb)
        with self.assertRaisesRegex(ValueError, "finite"):
            runtime.compose_fastwam_image(
                np.full((2, 2, 3), np.nan, dtype=np.float32), rgb, rgb
            )


class PiperFastWAMSafetyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = runtime.PiperSafetyLimits(
            lower=np.full(14, -1.0, dtype=np.float32),
            upper=np.full(14, 1.0, dtype=np.float32),
            max_delta=np.full(14, 0.25, dtype=np.float32),
        )
        self.current_fastwam = np.zeros(14, dtype=np.float32)

    def test_joint_limits_and_per_step_delta_clamp_finite_predictions(self) -> None:
        prediction = np.full((3, 14), 0.9, dtype=np.float32)

        limited = runtime.validate_and_limit_fastwam_actions(
            prediction, self.current_fastwam, self.limits
        )

        self.assertTrue(np.all(limited[0] <= 0.25))
        self.assertTrue(np.all(limited[1] <= 0.50))
        self.assertTrue(np.all(limited[2] <= 0.75))
        outside_limits = np.full((1, 14), 1.01, dtype=np.float32)
        limited_outside = runtime.validate_and_limit_fastwam_actions(
            outside_limits, self.current_fastwam, self.limits
        )
        np.testing.assert_allclose(limited_outside, 0.25)

    def test_limits_are_interpreted_in_piper_wire_order(self) -> None:
        # Piper index 6 is L_gripper, which is FastWAM index 12.
        lower = np.full(14, -1.0, dtype=np.float32)
        upper = np.full(14, 1.0, dtype=np.float32)
        upper[6] = 0.2
        limits = runtime.PiperSafetyLimits(
            lower=lower,
            upper=upper,
            max_delta=np.ones(14, dtype=np.float32),
        )
        action = np.zeros((1, 14), dtype=np.float32)
        action[0, 12] = 0.3

        limited = runtime.validate_and_limit_fastwam_actions(
            action, self.current_fastwam, limits
        )
        self.assertAlmostEqual(float(limited[0, 12]), 0.2, places=6)

    def test_small_gripper_overshoot_is_saturated_at_physical_boundary(self) -> None:
        lower = np.full(14, -1.0, dtype=np.float32)
        upper = np.full(14, 1.0, dtype=np.float32)
        lower[[6, 13]] = 0.0
        upper[[6, 13]] = 0.08
        limits = runtime.PiperSafetyLimits(
            lower=lower,
            upper=upper,
            max_delta=np.ones(14, dtype=np.float32),
        )
        action = np.zeros((1, 14), dtype=np.float32)
        # FastWAM index 12 maps to Piper index 6 (left gripper).
        action[0, 12] = -0.002818341

        limited = runtime.validate_and_limit_fastwam_actions(
            action, self.current_fastwam, limits
        )

        self.assertEqual(float(limited[0, 12]), 0.0)

    def test_finite_arm_overshoot_is_projected_before_delta_limit(self) -> None:
        lower = np.full(14, -1.0, dtype=np.float32)
        upper = np.full(14, 1.0, dtype=np.float32)
        # Piper/FastWAM index 1 is left arm joint 2, whose real lower bound is 0.
        lower[1] = 0.0
        max_delta = np.ones(14, dtype=np.float32)
        max_delta[1] = 0.01
        limits = runtime.PiperSafetyLimits(
            lower=lower,
            upper=upper,
            max_delta=max_delta,
        )
        current = self.current_fastwam.copy()
        current[1] = 0.02

        for overshoot in (0.049, 0.05, 0.089842878, 0.1273711):
            with self.subTest(overshoot=overshoot):
                action = np.zeros((1, 14), dtype=np.float32)
                action[0, 1] = -overshoot

                limited = runtime.validate_and_limit_fastwam_actions(
                    action, current, limits
                )

                # The raw value is first projected to the physical 0.0-rad
                # boundary, then limited from the current 0.02 by 0.01 rad.
                self.assertAlmostEqual(float(limited[0, 1]), 0.01, places=6)

    def test_projection_summary_and_large_finite_overshoots_are_clamped(self) -> None:
        lower = np.full(14, -1.0, dtype=np.float32)
        upper = np.full(14, 1.0, dtype=np.float32)
        lower[[6, 13]] = 0.0
        upper[[6, 13]] = 0.08
        upper[2] = 0.0
        limits = runtime.PiperSafetyLimits(
            lower=lower,
            upper=upper,
            max_delta=np.ones(14, dtype=np.float32),
        )
        recoverable = np.zeros((1, 14), dtype=np.float32)
        recoverable[0, 2] = 0.089842878  # Piper L3 upper limit is 0.0.
        recoverable[0, 12] = -0.5  # Piper L gripper lower limit is 0.0.
        recoverable[0, 0] = 1.101  # Piper L1 upper limit is 1.0.
        accepted, summary = runtime.project_and_limit_fastwam_actions(
            recoverable, self.current_fastwam, limits
        )
        self.assertEqual(float(accepted[0, 2]), 0.0)
        self.assertEqual(float(accepted[0, 12]), 0.0)
        self.assertEqual(float(accepted[0, 0]), 1.0)
        self.assertEqual(summary.projected_values, 3)
        self.assertAlmostEqual(summary.max_arm_projection, 0.101, places=6)
        self.assertAlmostEqual(summary.max_gripper_projection, 0.5, places=6)

    def test_invalid_limits_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            runtime.PiperSafetyLimits(
                lower=np.zeros(14), upper=np.ones(14), max_delta=np.zeros(14)
            )
        with self.assertRaisesRegex(ValueError, "lower.*upper"):
            runtime.PiperSafetyLimits(
                lower=np.ones(14), upper=np.zeros(14), max_delta=np.ones(14)
            )


class _FakeModel:
    """A CPU-only inference double; production models return the same mapping."""

    def __init__(self, action: np.ndarray | None = None) -> None:
        self.kwargs: dict[str, object] = {}
        self.action = (
            np.tile(np.arange(14, dtype=np.float32), (32, 1))
            if action is None
            else np.asarray(action, dtype=np.float32)
        )

    def infer_action(self, **kwargs: object) -> dict[str, np.ndarray]:
        self.kwargs = kwargs
        return {"action": self.action.copy()}


class _FakeProcessor:
    def __init__(self, *, action_offset: float = 0.0) -> None:
        self.action_offset = action_offset
        self.normalized_state: np.ndarray | None = None

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        self.normalized_state = np.asarray(state, dtype=np.float32)[np.newaxis, ...] + 1.0
        return self.normalized_state

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return np.asarray(action, dtype=np.float32) + self.action_offset


class FastWAMPiperRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.head = np.zeros((8, 12, 3), dtype=np.uint8)
        self.left = np.full((8, 12, 3), 64, dtype=np.uint8)
        self.right = np.full((8, 12, 3), 192, dtype=np.uint8)
        self.state = {
            "left_arm": [0, 1, 2, 3, 4, 5, 6],
            "right_arm": [7, 8, 9, 10, 11, 12, 13],
        }
        # The injected identity factory proves the test path has no torch,
        # FastWAM, CUDA, ROS, or WebSocket dependency.
        self.config: dict[str, object] = {
            "task_label": "crimp",
            "model_action_horizon": 32,
            "num_video_frames": 9,
            "num_inference_steps": 10,
            "seed": 17,
            "text_cfg_scale": 1.0,
            "negative_prompt": "",
            "rand_device": "cpu",
            "tiled": False,
            "_tensor_factory": lambda values: values,
        }

    def test_runtime_uses_training_prompt_and_fixed_temporal_shapes(self) -> None:
        model = _FakeModel()
        processor = _FakeProcessor()
        policy = runtime.FastWAMPiperRuntime(
            model=model, processor=processor, config=self.config
        )

        actions = policy.infer(
            head=self.head, left=self.left, right=self.right, state=self.state
        )

        self.assertEqual(
            model.kwargs["prompt"],
            "A video recorded from a robot's point of view executing the following instruction: crimp",
        )
        self.assertEqual(model.kwargs["action_horizon"], 32)
        self.assertEqual(model.kwargs["num_video_frames"], 9)
        self.assertEqual(actions.shape, (32, 14))
        self.assertEqual(model.kwargs["input_image"].shape, (1, 3, 384, 320))
        np.testing.assert_array_equal(
            model.kwargs["proprio"][0],
            np.asarray([1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 7, 14], dtype=np.float32),
        )

    def test_runtime_denormalizes_before_returning_actions(self) -> None:
        policy = runtime.FastWAMPiperRuntime(
            model=_FakeModel(),
            processor=_FakeProcessor(action_offset=10.0),
            config=self.config,
        )

        actions = policy.infer(
            head=self.head, left=self.left, right=self.right, state=self.state
        )

        np.testing.assert_allclose(actions[0], np.arange(14, dtype=np.float32) + 10.0)

    def test_realrobot_order_keeps_grippers_in_training_positions(self) -> None:
        config = dict(self.config)
        config.update(
            {
                "task_label": "Assemble the puzzle pieces on the table to complete the puzzle",
                "model_joint_order": "piper_wire",
            }
        )
        model = _FakeModel()
        processor = _FakeProcessor()
        policy = runtime.FastWAMPiperRuntime(model=model, processor=processor, config=config)

        actions = policy.infer(
            head=self.head, left=self.left, right=self.right, state=self.state
        )

        self.assertEqual(
            model.kwargs["prompt"],
            "A video recorded from a robot's point of view executing the following instruction: "
            "Assemble the puzzle pieces on the table to complete the puzzle",
        )
        # The model was given the realrobot training order and its output is
        # converted back to the canonical FastWAM order for the safety layer.
        np.testing.assert_array_equal(
            processor.normalized_state[0],
            np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            actions[0], np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
