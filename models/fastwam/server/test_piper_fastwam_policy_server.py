from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
from PIL import Image


SERVER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SERVER_DIR))

import piper_fastwam_policy_server as server_module  # noqa: E402
from piper_fastwam_policy_server import PiperFastWAMServer  # noqa: E402


def encode_jpeg(image: np.ndarray) -> dict[str, str]:
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="JPEG", quality=100)
    return {
        "encoding": "jpeg",
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def make_request(*, request_id: str = "request-1", step: int = 3) -> dict[str, object]:
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    payload = encode_jpeg(image)
    return {
        "request_id": request_id,
        "step": step,
        "instruction": "ignored by FastWAM; the server uses the fixed crimp task label",
        "state": {"left_arm": [0.0] * 7, "right_arm": [0.0] * 7},
        "images": {
            "head": payload,
            "left_wrist": payload,
            "right_wrist": payload,
        },
    }


def make_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "input_color": "bgr",
        "execution_horizon": 2,
        "action_dt": 1 / 30,
        "max_inference_latency_s": 5.0,
        "joint_lower": [-1.0] * 14,
        "joint_upper": [1.0] * 14,
        "max_delta": [0.2] * 14,
    }
    config.update(overrides)
    return config


class PiperFastWAMDeploymentConfigTest(unittest.TestCase):
    def test_pi05_style_clamping_preserves_physical_limits(self) -> None:
        config = server_module.load_config(
            SERVER_DIR / "piper_fastwam_puzzle.yaml"
        )

        self.assertEqual(int(config["seed"]), 42)
        self.assertNotIn("arm_projection_limit", config)
        self.assertNotIn("gripper_projection_limit", config)
        self.assertNotIn("large_arm_projection_threshold", config)
        self.assertNotIn("max_consecutive_large_projection_responses", config)
        self.assertEqual(config["max_delta"][:6], [0.01] * 6)


class FakeRuntime:
    def __init__(self, prediction: np.ndarray) -> None:
        self.prediction = prediction
        self.calls: list[dict[str, object]] = []

    def infer(self, **kwargs: object) -> np.ndarray:
        self.calls.append(kwargs)
        return self.prediction.copy()


class FailingRuntime:
    def infer(self, **_kwargs: object) -> np.ndarray:
        raise RuntimeError("model unavailable")


class PiperFastWAMServerTest(unittest.IsolatedAsyncioTestCase):
    async def test_valid_base64_jpeg_request_returns_existing_piper_action_schema(self) -> None:
        runtime = FakeRuntime(np.zeros((32, 14), dtype=np.float32))
        config = make_config(execution_horizon=2)
        server = PiperFastWAMServer(runtime=runtime, config=config)
        request = make_request()

        response = await server.handle_message(json.dumps(request))

        self.assertTrue(response["ok"])
        self.assertEqual(response["request_id"], request["request_id"])
        self.assertEqual(response["step"], request["step"])
        self.assertEqual(len(response["action"]["left_arm"]), config["execution_horizon"])
        self.assertEqual(len(response["action"]["right_arm"]), config["execution_horizon"])
        self.assertEqual(len(response["action"]["left_arm"][0]), 7)
        self.assertEqual(len(response["action"]["right_arm"][0]), 7)
        self.assertEqual(response["action"]["time_list"], [1 / 30, 2 / 30])
        self.assertNotIn("error", response)
        self.assertEqual(len(runtime.calls), 1)

        # The client sends BGR camera frames but a compressed JPEG has already
        # been decoded to RGB by Pillow. The server must never swap it again.
        decoded_pixel = runtime.calls[0]["head"][0, 0]
        self.assertGreater(int(decoded_pixel[0]), 240)
        self.assertLess(int(decoded_pixel[1]), 5)
        self.assertLess(int(decoded_pixel[2]), 5)

    async def test_bgr_raw_array_is_converted_to_rgb_before_runtime(self) -> None:
        runtime = FakeRuntime(np.zeros((32, 14), dtype=np.float32))
        server = PiperFastWAMServer(runtime=runtime, config=make_config())
        request = make_request()
        bgr = np.zeros((4, 5, 3), dtype=np.uint8)
        bgr[:, :, 2] = 128
        request["images"] = {
            name: {"array": bgr.tolist()} for name in ("head", "left_wrist", "right_wrist")
        }

        response = await server.handle_message(json.dumps(request))

        self.assertTrue(response["ok"])
        np.testing.assert_array_equal(runtime.calls[0]["left"][0, 0], [128, 0, 0])

    async def test_missing_required_camera_fails_closed(self) -> None:
        server = PiperFastWAMServer(
            runtime=FakeRuntime(np.zeros((32, 14), dtype=np.float32)), config=make_config()
        )
        request = make_request()
        del request["images"]["right_wrist"]

        response = await server.handle_message(json.dumps(request))

        self.assertEqual(response["ok"], False)
        self.assertEqual(response["request_id"], request["request_id"])
        self.assertEqual(response["step"], request["step"])
        self.assertEqual(response["action"], {})
        self.assertIn("right_wrist", response["error"])

    async def test_model_or_safety_error_returns_no_action(self) -> None:
        server = PiperFastWAMServer(runtime=FailingRuntime(), config=make_config())
        request = make_request()

        response = await server.handle_message(json.dumps(request))

        self.assertEqual(
            response,
            {
                "ok": False,
                "request_id": request["request_id"],
                "step": request["step"],
                "action": {},
                "error": mock.ANY,
            },
        )

    async def test_absolute_limit_violation_is_clamped_and_rate_limited(self) -> None:
        server = PiperFastWAMServer(
            runtime=FakeRuntime(np.full((32, 14), 2.0, dtype=np.float32)),
            config=make_config(),
        )
        request = make_request()

        response = await server.handle_message(json.dumps(request))

        self.assertTrue(response["ok"])
        np.testing.assert_allclose(response["action"]["left_arm"][0][:6], [0.2] * 6)
        np.testing.assert_allclose(response["action"]["left_arm"][1][:6], [0.4] * 6)

    async def test_repeated_large_projection_returns_rate_limited_actions(self) -> None:
        prediction = np.zeros((32, 14), dtype=np.float32)
        # FastWAM index 2 maps to Piper left J3, whose upper bound is 0.0.
        prediction[:, 2] = 0.089842878
        config = make_config(
            joint_upper=[1.0, 1.0, 0.0] + [1.0] * 11,
        )
        server = PiperFastWAMServer(runtime=FakeRuntime(prediction), config=config)

        first = await server.handle_message(json.dumps(make_request(request_id="one")))
        second = await server.handle_message(json.dumps(make_request(request_id="two")))
        third = await server.handle_message(json.dumps(make_request(request_id="three")))

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertTrue(third["ok"])
        self.assertEqual(first["action"]["left_arm"][0][2], 0.0)
        self.assertEqual(second["action"]["left_arm"][0][2], 0.0)
        self.assertEqual(third["action"]["left_arm"][0][2], 0.0)

    async def test_unpublished_tail_violation_does_not_block_safe_replan_chunk(self) -> None:
        """Only the rolling chunk can be published and must be validated.

        The remaining model candidates are superseded by the next observation
        and replan, so rejecting a safe chunk because of an un-published tail
        would neither improve safety nor permit the intended FastWAM rollout.
        """
        prediction = np.zeros((32, 14), dtype=np.float32)
        prediction[2:, 1] = 2.0  # Piper left joint 2 is outside [-1, 1].
        server = PiperFastWAMServer(
            runtime=FakeRuntime(prediction), config=make_config(execution_horizon=2)
        )

        response = await server.handle_message(json.dumps(make_request()))

        self.assertTrue(response["ok"])
        self.assertEqual(len(response["action"]["left_arm"]), 2)

    async def test_latency_limit_rejects_action_after_inference(self) -> None:
        server = PiperFastWAMServer(
            runtime=FakeRuntime(np.zeros((32, 14), dtype=np.float32)),
            config=make_config(max_inference_latency_s=0.01),
        )

        with mock.patch.object(server_module.time, "monotonic", side_effect=(1.0, 1.02)):
            response = await server.handle_message(json.dumps(make_request()))

        self.assertFalse(response["ok"])
        self.assertEqual(response["action"], {})
        self.assertIn("latency", response["error"])

    def test_execution_horizon_cannot_exceed_model_horizon(self) -> None:
        with self.assertRaisesRegex(ValueError, "execution_horizon.*32"):
            PiperFastWAMServer(
                runtime=FakeRuntime(np.zeros((32, 14), dtype=np.float32)),
                config=make_config(execution_horizon=33),
            )

    def test_physical_server_refuses_unverified_safety_vectors(self) -> None:
        with self.assertRaisesRegex(ValueError, "joint_lower"):
            PiperFastWAMServer(
                runtime=FakeRuntime(np.zeros((32, 14), dtype=np.float32)),
                config=make_config(joint_lower=[]),
            )


if __name__ == "__main__":
    unittest.main()
