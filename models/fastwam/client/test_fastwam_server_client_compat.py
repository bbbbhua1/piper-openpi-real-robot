from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


CLIENT_DIR = Path(__file__).resolve().parent
SERVER_DIR = CLIENT_DIR.parent / "server"
sys.path.insert(0, str(CLIENT_DIR))
sys.path.insert(0, str(SERVER_DIR))

from piper_fastwam_policy_server import PiperFastWAMServer  # noqa: E402
from ws_policy_protocol import (  # noqa: E402
    FASTWAM_TRANSPORT_IMAGE_SIZES,
    build_policy_request,
)


class _ZeroRuntime:
    def infer(self, **kwargs: object) -> np.ndarray:
        self.last_inputs = kwargs
        return np.zeros((32, 14), dtype=np.float32)


def _server_config() -> dict[str, object]:
    return {
        "input_color": "bgr",
        "model_action_horizon": 32,
        "execution_horizon": 2,
        "action_dt": 1.0 / 30.0,
        "max_inference_latency_s": 5.0,
        "joint_lower": [-1.0] * 14,
        "joint_upper": [1.0] * 14,
        "max_delta": [0.2] * 14,
    }


class FastWAMClientServerCompatibilityTest(unittest.IsolatedAsyncioTestCase):
    async def test_client_request_is_accepted_by_fastwam_server_adapter(self) -> None:
        # Use distinct channel values so an accidental color swap is visible to
        # the server-side decoded HWC input as well as in the transport payload.
        images = {
            "head": np.full((480, 640, 3), [255, 0, 0], dtype=np.uint8),
            "left_wrist": np.full((240, 320, 3), [0, 255, 0], dtype=np.uint8),
            "right_wrist": np.full((240, 320, 3), [0, 0, 255], dtype=np.uint8),
        }
        request = build_policy_request(
            step=4,
            instruction="crimp",
            state={"left_arm": [0.0] * 7, "right_arm": [0.0] * 7},
            images=images,
            transport_image_sizes=FASTWAM_TRANSPORT_IMAGE_SIZES,
        )
        runtime = _ZeroRuntime()
        server = PiperFastWAMServer(runtime=runtime, config=_server_config())

        response = await server.handle_message(json.dumps(request))

        self.assertTrue(response["ok"])
        self.assertEqual(response["request_id"], request["request_id"])
        self.assertEqual(response["step"], 4)
        self.assertEqual(len(response["action"]["left_arm"]), 2)
        self.assertEqual(len(response["action"]["right_arm"]), 2)
        self.assertEqual(response["action"]["time_list"], [1.0 / 30.0, 2.0 / 30.0])

        self.assertEqual(runtime.last_inputs["head"].shape, (256, 320, 3))
        self.assertEqual(runtime.last_inputs["left"].shape, (128, 160, 3))
        self.assertEqual(runtime.last_inputs["right"].shape, (128, 160, 3))
        self.assertGreater(int(runtime.last_inputs["head"][0, 0, 2]), 200)
        self.assertGreater(int(runtime.last_inputs["left"][0, 0, 1]), 200)
        self.assertGreater(int(runtime.last_inputs["right"][0, 0, 0]), 200)
        self.assertEqual(runtime.last_inputs["state"]["left_arm"], [0.0] * 7)
        self.assertEqual(runtime.last_inputs["state"]["right_arm"], [0.0] * 7)


if __name__ == "__main__":
    unittest.main()
