import asyncio
import base64
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cogwam_piper_policy_server import (  # noqa: E402
    CogWAMServer,
    _to_client_action,
    load_config,
    project_cogwam_actions,
    request_to_robotwin,
)


def _jpeg() -> str:
    import io

    buffer = io.BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _request(**extra):
    payload = {
        "request_id": "test",
        "step": 0,
        "instruction": "task",
        "state": {"left_arm": [0] * 7, "right_arm": [0] * 7},
        "images": {"head": _jpeg(), "left_wrist": _jpeg(), "right_wrist": _jpeg()},
    }
    payload.update(extra)
    return payload


def test_cogwam_config_is_distinct_and_memory_contract_is_strict():
    config = load_config(ROOT / "configs/piper_cogwam_jigsaw.yaml")
    assert config["wan_memory_attention_mode"] == "static_kv"
    assert config["wan_memory_enabled"] is True
    assert config["execution_horizon"] == 32
    assert "vlm_replan_on_subtask_done" not in config
    assert config["vlm_enabled"] is True
    assert config["vlm_q2_state_enabled"] is False
    assert config["vlm_failure_query_enabled"] is False
    assert abs(config["action_dt"] - (1.0 / 30.0)) < 1e-9
    assert config["vlm_base_model"] == "/bh/zbh_ckp/models/Qwen3-VL-2B-Instruct"
    assert len(config["joint_lower"]) == len(config["joint_upper"]) == len(config["max_delta"]) == 14
    assert config["vlm_memory_strategy"] == "subtask_cls_recent_images"
    assert config["vlm_memory_frames"] == 5
    assert config["vlm_memory_frame_stride"] == 4
    assert config["vlm_completed_subtask_memory_stride"] == 16
    assert config["vlm_current_subtask_memory_stride"] == 4


def test_request_maps_piper_state_and_three_views_to_robotwin_contract():
    observation, state = request_to_robotwin(_request(), input_color="bgr")
    assert state.shape == (14,)
    assert observation["joint_action"]["vector"].shape == (14,)
    assert observation["observation"]["head_camera"]["rgb"].shape == (8, 8, 3)


def test_action_schema_is_14d_piper_joint_chunk():
    action = _to_client_action(np.arange(70, dtype=np.float32).reshape(5, 14), horizon=4, dt=0.1)
    assert len(action["left_arm"]) == len(action["right_arm"]) == 4
    assert len(action["left_arm"][0]) == len(action["right_arm"][0]) == 7
    assert action["time_list"][-1] == 0.4


def test_cogwam_action_projection_clips_bounds_and_step_delta():
    config = {
        "joint_lower": [-1.0] * 14,
        "joint_upper": [1.0] * 14,
        "max_delta": [0.1] * 14,
    }
    projected = project_cogwam_actions(
        np.asarray([[2.0] * 14, [-2.0] * 14], dtype=np.float32),
        np.zeros(14, dtype=np.float32),
        config,
    )
    np.testing.assert_allclose(projected[0], 0.1)
    np.testing.assert_allclose(projected[1], 0.0)


class _FakeRuntime:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def plan_q1_once(self, observation):
        self.planned = True

    def infer(self, observation, request_step):
        return np.zeros((64, 14), dtype=np.float32)


def test_final_task_done_is_external_and_q2_status_is_reported():
    config = {
        "execution_horizon": 64,
        "action_dt": 0.1,
        "input_color": "bgr",
        "joint_lower": [-10.0] * 14,
        "joint_upper": [10.0] * 14,
        "max_delta": [100.0] * 14,
    }
    server = CogWAMServer(_FakeRuntime(), config)
    response = asyncio.run(server.handle(json.dumps(_request(final_task_done=True))))
    assert response["ok"] is True
    assert response["final_task_done"] is True
    assert response["q2_enabled"] is False
    assert response["q2_status"] is None
    assert response["wan_memory_attention_mode"] == "static_kv"


def test_reset_request_clears_runtime_without_inference():
    config = {"execution_horizon": 64}
    runtime = _FakeRuntime()
    server = CogWAMServer(runtime, config)
    response = asyncio.run(server.handle(json.dumps({"request_id": "reset", "type": "reset"})))
    assert response["ok"] is True
    assert response["reset"] is True
    assert runtime.reset_count == 1
