from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


SERVER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SERVER_DIR))

import piper_fastwam_policy_server_supervised as raw_module  # noqa: E402
from test_piper_fastwam_policy_server import FakeRuntime, make_request  # noqa: E402


def make_raw_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "input_color": "bgr",
        "execution_horizon": 2,
        "action_dt": 1 / 30,
    }
    config.update(overrides)
    return config


class SupervisedRawPiperFastWAMServerTest(unittest.IsolatedAsyncioTestCase):
    async def test_finite_out_of_range_action_is_forwarded_without_clamp_or_delta_limit(
        self,
    ) -> None:
        prediction = np.zeros((32, 14), dtype=np.float32)
        prediction[0, 1] = -0.127371117
        prediction[0, 2] = 0.089842878
        prediction[1, 1] = 2.5
        server = raw_module.SupervisedRawPiperFastWAMServer(
            runtime=FakeRuntime(prediction), config=make_raw_config()
        )

        response = await server.handle_message(json.dumps(make_request()))

        self.assertTrue(response["ok"])
        self.assertAlmostEqual(response["action"]["left_arm"][0][1], -0.127371117)
        self.assertAlmostEqual(response["action"]["left_arm"][0][2], 0.089842878)
        self.assertAlmostEqual(response["action"]["left_arm"][1][1], 2.5)

    async def test_non_finite_prediction_returns_existing_error_envelope(self) -> None:
        prediction = np.zeros((32, 14), dtype=np.float32)
        prediction[0, 0] = np.nan
        server = raw_module.SupervisedRawPiperFastWAMServer(
            runtime=FakeRuntime(prediction), config=make_raw_config()
        )

        response = await server.handle_message(json.dumps(make_request()))

        self.assertFalse(response["ok"])
        self.assertEqual(response["action"], {})
        self.assertIn("non-finite", response["error"])

    def test_config_does_not_require_standard_safety_vectors(self) -> None:
        raw_module.SupervisedRawPiperFastWAMServer(
            runtime=FakeRuntime(np.zeros((32, 14), dtype=np.float32)),
            config=make_raw_config(),
        )

    def test_raw_deployment_config_has_no_software_motion_limit_keys(self) -> None:
        config = raw_module.load_config(
            SERVER_DIR / "piper_fastwam_puzzle_supervised_raw.yaml"
        )

        self.assertEqual(
            config["task_label"],
            "Assemble the puzzle pieces on the table to complete the puzzle",
        )
        self.assertEqual(config["execution_horizon"], 64)
        self.assertNotIn("joint_lower", config)
        self.assertNotIn("joint_upper", config)
        self.assertNotIn("max_delta", config)


if __name__ == "__main__":
    unittest.main()
