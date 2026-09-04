from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "models" / "lingbotva" / "server"
CLIENT_DIR = REPO_ROOT / "models" / "lingbotva" / "client"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for directory in (str(SERVER_DIR), str(CLIENT_DIR), str(SCRIPTS_DIR)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from lingbotva_piper_policy_server import PiperLingBotVAServer  # noqa: E402
from lingbotva_piper_runtime import (  # noqa: E402
    ACTION_CHANNELS,
    MODEL_ACTION_HORIZON,
    LingBotVAPiperRuntime,
    PiperSafetyLimits,
    actions_to_piper_wire,
    expand_action_norm,
    flatten_lingbotva_actions,
    load_action_norm,
    project_and_limit_piper_actions,
)
from lingbotva_piper_robot_client import truncate_action  # noqa: E402


def jpeg_payload(image: np.ndarray) -> dict[str, str]:
    output = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(output, format="JPEG", quality=100)
    return {"encoding": "jpeg", "data": base64.b64encode(output.getvalue()).decode("ascii")}


def request() -> dict[str, object]:
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    payload = jpeg_payload(image)
    return {
        "request_id": "lingbot-request",
        "step": 0,
        "instruction": "Complete the jigsaw puzzle on the table.",
        "reset": True,
        "state": {"left_arm": [0.0] * 7, "right_arm": [0.0] * 7},
        "images": {name: payload for name in ("head", "left_wrist", "right_wrist")},
    }


def config(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "input_color": "bgr",
        "execution_horizon": 32,
        "action_dt": 1 / 30,
        "max_inference_latency_s": 5.0,
        "instruction": "Complete the jigsaw puzzle on the table.",
        "joint_lower": [-1.0] * 14,
        "joint_upper": [1.0] * 14,
        "max_delta": [0.2] * 14,
    }
    value.update(overrides)
    return value


class LingBotVAAdapterTest(unittest.TestCase):
    def test_official_action_grid_is_flattened_in_time_major_order(self) -> None:
        values = np.zeros((14, 2, 16), dtype=np.float32)
        for channel in range(14):
            for frame in range(2):
                for slot in range(16):
                    values[channel, frame, slot] = channel * 100 + frame * 16 + slot
        flattened = flatten_lingbotva_actions(values)
        self.assertEqual(flattened.shape, (32, 14))
        np.testing.assert_array_equal(flattened[0], np.arange(14, dtype=np.float32) * 100)
        np.testing.assert_array_equal(
            flattened[16], np.arange(14, dtype=np.float32) * 100 + 16
        )

    def test_checkpoint_action_norm_maps_to_30_channels(self) -> None:
        norm_path = Path(
            "/bh/media/ZBH/lingbot-va-realrobot-ft-step17000-inference-20260831/action_norm.json"
        )
        norm = load_action_norm(norm_path)
        expanded = expand_action_norm(norm)
        self.assertEqual(len(expanded["q01"]), 30)
        self.assertEqual(len(expanded["q99"]), 30)
        for raw_index, internal_index in enumerate(ACTION_CHANNELS):
            self.assertEqual(expanded["q01"][internal_index], norm["q01"][raw_index])
            self.assertEqual(expanded["q99"][internal_index], norm["q99"][raw_index])

    def test_safety_projection_is_in_piper_wire_order(self) -> None:
        limits = PiperSafetyLimits(
            lower=np.full(14, -1.0, dtype=np.float32),
            upper=np.full(14, 1.0, dtype=np.float32),
            max_delta=np.full(14, 0.25, dtype=np.float32),
        )
        action = np.full((2, 14), 2.0, dtype=np.float32)
        accepted, summary = project_and_limit_piper_actions(action, np.zeros(14), limits)
        self.assertEqual(summary.projected_values, 28)
        np.testing.assert_allclose(accepted[0], 0.25)
        np.testing.assert_allclose(accepted[1], 0.5)

    def test_action_wire_schema_contains_32_synchronized_waypoints(self) -> None:
        action = actions_to_piper_wire(np.zeros((32, 14), dtype=np.float32), dt=1 / 30)
        self.assertEqual(len(action["left_arm"]), 32)
        self.assertEqual(len(action["right_arm"]), 32)
        self.assertEqual(len(action["time_list"]), 32)
        self.assertEqual(len(action["left_arm"][0]), 7)
        self.assertAlmostEqual(action["time_list"][-1], 32 / 30)


class FakeLingBotModel:
    def __init__(self) -> None:
        self.frame_st_id = 0
        self.reset_prompts: list[str] = []
        self.infer_calls: list[tuple[int, dict[str, object]]] = []
        self.cache_calls = 0
        self.streaming_vae = FakeStreamingVAE()

    def _reset(self, *, prompt: str) -> None:
        self.reset_prompts.append(prompt)
        self.frame_st_id = 0

    def _infer(self, observation: dict[str, object], *, frame_st_id: int):
        self.infer_calls.append((frame_st_id, observation))
        return np.zeros((14, 2, 16), dtype=np.float32), None

    def _compute_kv_cache(self, observation: dict[str, object]) -> None:
        self.cache_calls += 1
        self.frame_st_id += len(observation["obs"]) + (1 if self.frame_st_id == 0 else 0)


class FakeStreamingVAE:
    def __init__(self) -> None:
        self.clear_calls = 0

    def clear_cache(self) -> None:
        self.clear_calls += 1


class FakeRuntime:
    def __init__(self) -> None:
        self.model = FakeLingBotModel()
        self.initialized = False
        self.runtime = LingBotVAPiperRuntime(model=self.model, config={})

    def reset(self, instruction: str) -> None:
        self.runtime.reset(instruction)
        self.initialized = self.runtime.initialized

    def infer(self, **kwargs: object) -> np.ndarray:
        result = self.runtime.infer(**kwargs)
        self.initialized = self.runtime.initialized
        return result

    def expected_cache_frames(self) -> int:
        return self.runtime.expected_cache_frames()


class LingBotVAServerTest(unittest.IsolatedAsyncioTestCase):
    async def test_first_request_initializes_and_second_request_updates_kv_cache(self) -> None:
        fake = FakeRuntime()
        server = PiperLingBotVAServer(runtime=fake, config=config())
        first = await server.handle_message(json.dumps(request()))
        self.assertTrue(first["ok"])
        self.assertEqual(len(first["action"]["left_arm"]), 32)
        self.assertEqual(fake.model.infer_calls[0][0], 0)
        self.assertEqual(fake.model.cache_calls, 0)

        second_request = request()
        second_request["step"] = 1
        second_request["reset"] = False
        second_request["history_images"] = [second_request["images"]]
        second = await server.handle_message(json.dumps(second_request))
        self.assertTrue(second["ok"])
        self.assertEqual(fake.model.cache_calls, 1)
        self.assertEqual(fake.model.streaming_vae.clear_calls, 1)
        self.assertEqual(fake.model.infer_calls[1][0], 2)
        self.assertEqual(len(fake.model.infer_calls[1][1]["obs"]), 1)
        self.assertEqual(fake.model.infer_calls[1][1]["state"].shape, (14, 2, 16))
        self.assertEqual(len(fake.model.reset_prompts), 1)

        third_request = request()
        third_request["step"] = 2
        third_request["reset"] = False
        third_request["history_images"] = [third_request["images"], third_request["images"]]
        third = await server.handle_message(json.dumps(third_request))
        self.assertTrue(third["ok"])
        self.assertEqual(fake.model.cache_calls, 2)
        self.assertEqual(len(fake.model.infer_calls[2][1]["obs"]), 2)
        self.assertEqual(fake.model.streaming_vae.clear_calls, 2)

    async def test_invalid_request_fails_closed(self) -> None:
        fake = FakeRuntime()
        server = PiperLingBotVAServer(runtime=fake, config=config())
        payload = request()
        del payload["images"]["right_wrist"]
        response = await server.handle_message(json.dumps(payload))
        self.assertFalse(response["ok"])
        self.assertEqual(response["action"], {})
        self.assertIn("right_wrist", response["error"])


class LingBotVAClientTest(unittest.TestCase):
    def test_client_truncates_all_action_fields_together(self) -> None:
        action = {
            "left_arm": [[float(i)] * 7 for i in range(4)],
            "right_arm": [[float(i)] * 7 for i in range(4)],
            "left_gripper": [[float(i)] for i in range(4)],
            "right_gripper": [[float(i)] for i in range(4)],
            "time_list": [0.1, 0.2, 0.3, 0.4],
            "dt": 0.1,
        }
        result = truncate_action(action, 2)
        self.assertEqual(len(result["left_arm"]), 2)
        self.assertEqual(len(result["right_arm"]), 2)
        self.assertEqual(len(result["left_gripper"]), 2)
        self.assertEqual(result["time_list"], [0.1, 0.2])

    def test_client_skips_the_initial_conditioning_frame(self) -> None:
        action = {
            "left_arm": [[float(i)] * 7 for i in range(32)],
            "right_arm": [[float(i)] * 7 for i in range(32)],
            "time_list": [float(i + 1) / 30 for i in range(32)],
            "dt": 1 / 30,
        }
        result = truncate_action(action, 32, skip_initial_frame=True)
        self.assertEqual(len(result["left_arm"]), 16)
        self.assertEqual(result["left_arm"][0], [16.0] * 7)
        self.assertAlmostEqual(result["time_list"][0], 1 / 30)
        self.assertLess(result["time_list"][0], result["time_list"][1])


if __name__ == "__main__":
    unittest.main()
