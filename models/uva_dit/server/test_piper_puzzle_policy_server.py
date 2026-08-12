#!/usr/bin/env python3
"""Dependency-light checks for the Piper UVA-DiT adapter boundary."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

try:
    import h5py  # type: ignore
except ImportError:  # pragma: no cover - exercised on minimal dev environments
    h5py = None


MODULE_PATH = Path(__file__).with_name("piper_puzzle_policy_server.py")
SPEC = importlib.util.spec_from_file_location("piper_puzzle_policy_server", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
SERVER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SERVER
SPEC.loader.exec_module(SERVER)


def valid_puzzle_config() -> dict:
    return {
        "action_type": "joint_pos",
        "action_dim": 32,
        "ur_physical_action_dim": 14,
        "pi07_context": True,
        "pi07_subtask_mode": "frame_aligned",
        "pi07_history_frames": 2,
        "pi07_history_frame_skip": 4,
        "use_chunk_ar_train": True,
        "use_action_condition": True,
        "action_condition_mode": "causal_history",
        "action_condition_history_steps": 16,
        "ur_joint_condition_source": "puppet",
        "ur_joint_target_source": "puppet",
        "ur_joint_channel_layout": "arms_then_grippers",
        "fix_vae_temporal_action_alignment": True,
        "n_video_frames": 9,
        "action_per_frame": 16,
        "action_chunk_size": 32,
        "frame_skip": 4,
        "ur_action_loss_dims": "6,7,8,9,10,11,13",
        "use_state_tokens": False,
    }


def c_noar_puzzle_config() -> dict:
    config = valid_puzzle_config()
    config["pi07_history_frames"] = 0
    return config


class PiperProtocolTests(unittest.TestCase):
    def test_pi07_scheduler_arguments_are_exposed(self) -> None:
        args = SERVER.build_arg_parser().parse_args([])
        self.assertIsNone(args.video_snr_shift)
        self.assertIsNone(args.action_snr_shift)

    def test_piper_state_maps_to_arms_then_grippers(self) -> None:
        state = {
            "left_arm": [0, 1, 2, 3, 4, 5, 6],
            "right_arm": [7, 8, 9, 10, 11, 12, 13],
        }
        self.assertEqual(
            SERVER.piper_state_to_model(state),
            list(range(6)) + list(range(7, 13)) + [6.0, 13.0],
        )

    def test_model_row_maps_back_to_legacy_arms(self) -> None:
        row = list(range(14))
        left, right = SERVER.model_row_to_piper(row)
        self.assertEqual(left, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 12.0])
        self.assertEqual(right, [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 13.0])

    def test_hold_mode_freezes_unsupervised_left_side(self) -> None:
        current = list(range(14))
        raw = [[99.0] * 14 for _ in range(4)]
        action, sent = SERVER.model_actions_to_piper_action(
            raw,
            current,
            dt=1 / 30,
            inactive_action_mode="hold",
            max_joint_step=0.0,
        )
        self.assertEqual(sent[0][:6], current[:6])
        self.assertEqual(sent[0][12], current[12])
        self.assertEqual(action["left_arm"][0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 12.0])
        self.assertEqual(action["right_arm"][0], [99.0] * 7)
        self.assertEqual(action["time_list"], [1 / 30, 2 / 30, 3 / 30, 4 / 30])

    def test_right_arm_slew_limit_cascades_over_waypoints(self) -> None:
        current = [0.0] * 14
        raw = [[0.0] * 14, [0.0] * 14]
        raw[0][6] = 5.0
        raw[1][6] = 5.0
        sent = SERVER.finalize_model_actions(
            raw,
            current,
            inactive_action_mode="hold",
            max_joint_step=0.25,
        )
        self.assertEqual(sent[0][6], 0.25)
        self.assertEqual(sent[1][6], 0.5)

    def test_invalid_piper_state_is_rejected(self) -> None:
        with self.assertRaisesRegex(SERVER.ContractError, "exactly 7"):
            SERVER.piper_state_to_model({"left_arm": [0] * 6, "right_arm": [0] * 7})

    def test_measured_state_history_is_validated_in_model_order(self) -> None:
        rows = SERVER.parse_piper_state_history([list(range(14))])
        self.assertEqual(rows, [list(range(14))])

    def test_measured_state_history_rejects_empty_overlong_and_nonfinite(self) -> None:
        with self.assertRaisesRegex(SERVER.ContractError, "must not be empty"):
            SERVER.parse_piper_state_history([])
        with self.assertRaisesRegex(SERVER.ContractError, "at most 16"):
            SERVER.parse_piper_state_history([[0.0] * 14 for _ in range(17)])
        with self.assertRaisesRegex(SERVER.ContractError, "finite"):
            SERVER.parse_piper_state_history([[float("nan")] + [0.0] * 13])


@unittest.skipIf(h5py is None, "h5py is required for the raw-action HDF5 test")
class RawActionHDF5Tests(unittest.TestCase):
    def test_saves_raw_prefix_and_piper_wire_mapping(self) -> None:
        from raw_action_hdf5_recorder import RawModelActionHDF5Recorder

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "raw_model_actions.hdf5"
            recorder = RawModelActionHDF5Recorder(path, max_steps_per_request=2)
            raw = [list(range(14)), [100 + index for index in range(14)], [200] * 14]
            written = recorder.record(
                request_id="request-1",
                step=3,
                instruction="拼图（圆形）",
                raw_actions=raw,
                current_state=list(range(14)),
                action_dt=0.067,
                timing={"mode": "denoise"},
            )
            recorder.close()

            self.assertEqual(written, 2)
            with h5py.File(recorder.path, "r") as h5_file:
                self.assertEqual(h5_file["action"].shape, (2, 14))
                self.assertEqual(h5_file["action"][0].tolist(), list(range(14)))
                self.assertEqual(
                    h5_file["action_piper_wire"][0].tolist(),
                    [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13],
                )
                self.assertEqual(h5_file["current_state"][0].tolist(), list(range(14)))
                self.assertEqual(h5_file["waypoint_index"][:].tolist(), [0, 1])
                self.assertEqual(h5_file.attrs["saved_prefix_steps"], 2)
                self.assertTrue(h5_file.attrs["raw_model_output"])

    def test_existing_path_is_not_overwritten(self) -> None:
        from raw_action_hdf5_recorder import RawModelActionHDF5Recorder

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "raw_model_actions.hdf5"
            first = RawModelActionHDF5Recorder(path)
            first.close()
            second = RawModelActionHDF5Recorder(path)
            second.close()
            self.assertNotEqual(first.path, second.path)


class CheckpointContractTests(unittest.TestCase):
    def _write_config(self, config: dict) -> Path:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        checkpoint = root / "checkpoint_step_00016000"
        checkpoint.mkdir()
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return checkpoint

    def tearDown(self) -> None:
        temp_dir = getattr(self, "temp_dir", None)
        if temp_dir is not None:
            temp_dir.cleanup()

    def test_expected_puzzle_config_is_accepted(self) -> None:
        checkpoint = self._write_config(valid_puzzle_config())
        loaded = SERVER.validate_puzzle_checkpoint_config(checkpoint)
        self.assertEqual(loaded["ur_physical_action_dim"], 14)
        self.assertEqual(loaded["ur_action_loss_dims"], "6,7,8,9,10,11,13")
        self.assertEqual(loaded["_piper_visual_history_label"], "image AR2")

    def test_c_noar_puzzle_config_is_accepted(self) -> None:
        checkpoint = self._write_config(c_noar_puzzle_config())
        loaded = SERVER.validate_puzzle_checkpoint_config(checkpoint)
        self.assertEqual(loaded["_piper_visual_history_label"], "image no-AR")

    def test_tianyi_physical_width_is_rejected(self) -> None:
        config = valid_puzzle_config()
        config["ur_physical_action_dim"] = 16
        checkpoint = self._write_config(config)
        with self.assertRaisesRegex(SERVER.ContractError, "ur_physical_action_dim"):
            SERVER.validate_puzzle_checkpoint_config(checkpoint)

    def test_wrong_loss_mask_is_rejected(self) -> None:
        config = valid_puzzle_config()
        config["ur_action_loss_dims"] = "0,1,2"
        checkpoint = self._write_config(config)
        with self.assertRaisesRegex(SERVER.ContractError, "ur_action_loss_dims"):
            SERVER.validate_puzzle_checkpoint_config(checkpoint)

    def test_unknown_visual_history_is_rejected(self) -> None:
        config = valid_puzzle_config()
        config["pi07_history_frames"] = 1
        checkpoint = self._write_config(config)
        with self.assertRaisesRegex(SERVER.ContractError, "pi07_history_frames"):
            SERVER.validate_puzzle_checkpoint_config(checkpoint)


class MockEngineTests(unittest.TestCase):
    def test_mock_engine_records_final_actions_only_after_ack(self) -> None:
        engine = SERVER.MockPiperEngine()
        engine.update_state(list(range(14)))
        raw = engine.infer({}, "task", "", None, 4)
        self.assertEqual(engine.acknowledged, [])
        engine.acknowledge_sent_actions(raw)
        self.assertEqual(len(engine.acknowledged), 4)
        self.assertEqual(engine.acknowledged[0], list(range(14)))


class MeasuredHistoryMockEngine(SERVER.MockPiperEngine):
    require_measured_history = True

    def __init__(self) -> None:
        super().__init__()
        self.measured_history = None

    def update_state_history(self, rows) -> None:
        self.measured_history = rows


class RawOutputMockEngine(SERVER.MockPiperEngine):
    def infer(self, obs, task, subtask, subgoal, execute_steps):
        del obs, task, subtask, subgoal
        self._last_timing = {"mode": "raw-output-mock"}
        return [[99.0] * 14 for _ in range(execute_steps)]


class SessionSchedulingTests(unittest.IsolatedAsyncioTestCase):
    async def test_four_steps_are_acknowledged_only_after_response_commit(self) -> None:
        args = SimpleNamespace(
            expected_instruction="拼图（圆形）",
            allow_instruction_override=False,
            device="cpu",
            allow_missing_images=False,
            execute_steps=4,
            action_dt=1 / 30,
            inactive_action_mode="hold",
            max_joint_step=0.0,
        )
        engine = SERVER.MockPiperEngine()
        session = SERVER.PiperPolicySession(args, engine, recorder=None)
        request = {
            "request_id": "request-1",
            "step": 0,
            "instruction": "拼图（圆形）",
            "state": {"left_arm": [0] * 7, "right_arm": [0] * 7},
            "images": {"head": {}, "left_wrist": {}, "right_wrist": {}},
        }
        original_converter = SERVER.legacy_images_to_observation
        SERVER.legacy_images_to_observation = lambda *args, **kwargs: {}
        try:
            prepared = await session.prepare(json.dumps(request))
            self.assertTrue(prepared.response["ok"])
            self.assertEqual(len(prepared.sent_actions or []), 4)
            self.assertEqual(engine.acknowledged, [])
            await session.commit(request, prepared)
            self.assertEqual(len(engine.acknowledged), 4)

            duplicate = await session.prepare(json.dumps(request))
            self.assertFalse(duplicate.should_cache)
            self.assertEqual(duplicate.response, prepared.response)
            self.assertEqual(len(engine.acknowledged), 4)
        finally:
            SERVER.legacy_images_to_observation = original_converter

    async def test_measured_history_is_passed_before_inference(self) -> None:
        args = SimpleNamespace(
            expected_instruction="拼图（圆形）",
            allow_instruction_override=False,
            device="cpu",
            allow_missing_images=False,
            execute_steps=2,
            action_dt=1 / 30,
            inactive_action_mode="hold",
            max_joint_step=0.0,
        )
        engine = MeasuredHistoryMockEngine()
        session = SERVER.PiperPolicySession(args, engine, recorder=None)
        request = {
            "request_id": "request-history",
            "step": 0,
            "instruction": "拼图（圆形）",
            "state": {"left_arm": [0] * 7, "right_arm": [0] * 7},
            "state_history": [list(range(14))],
            "images": {"head": {}, "left_wrist": {}, "right_wrist": {}},
        }
        original_converter = SERVER.legacy_images_to_observation
        SERVER.legacy_images_to_observation = lambda *args, **kwargs: {}
        try:
            prepared = await session.prepare(json.dumps(request))
            self.assertTrue(prepared.response["ok"])
            self.assertEqual(engine.measured_history, [list(range(14))])
        finally:
            SERVER.legacy_images_to_observation = original_converter

    async def test_real_history_requirement_rejects_missing_history(self) -> None:
        args = SimpleNamespace(
            expected_instruction="拼图（圆形）",
            allow_instruction_override=False,
            device="cpu",
            allow_missing_images=False,
            execute_steps=2,
            action_dt=1 / 30,
            inactive_action_mode="hold",
            max_joint_step=0.0,
        )
        session = SERVER.PiperPolicySession(args, MeasuredHistoryMockEngine(), recorder=None)
        request = {
            "request_id": "request-no-history",
            "step": 0,
            "instruction": "拼图（圆形）",
            "state": {"left_arm": [0] * 7, "right_arm": [0] * 7},
            "images": {"head": {}, "left_wrist": {}, "right_wrist": {}},
        }
        original_converter = SERVER.legacy_images_to_observation
        SERVER.legacy_images_to_observation = lambda *args, **kwargs: {}
        try:
            prepared = await session.prepare(json.dumps(request))
            self.assertFalse(prepared.response["ok"])
            self.assertIn("state_history is required", prepared.response["error"])
        finally:
            SERVER.legacy_images_to_observation = original_converter

    @unittest.skipIf(h5py is None, "h5py is required for the raw-action HDF5 test")
    async def test_hdf5_keeps_raw_output_before_left_hold(self) -> None:
        from raw_action_hdf5_recorder import RawModelActionHDF5Recorder

        args = SimpleNamespace(
            expected_instruction="拼图（圆形）",
            allow_instruction_override=False,
            device="cpu",
            allow_missing_images=False,
            execute_steps=4,
            raw_action_steps=2,
            action_dt=1 / 30,
            inactive_action_mode="hold",
            max_joint_step=0.0,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_recorder = RawModelActionHDF5Recorder(
                Path(temp_dir) / "raw_model_actions.hdf5",
                max_steps_per_request=args.raw_action_steps,
            )
            session = SERVER.PiperPolicySession(
                args, RawOutputMockEngine(), recorder=None, raw_action_recorder=raw_recorder
            )
            request = {
                "request_id": "raw-output-request",
                "step": 0,
                "instruction": "拼图（圆形）",
                "state": {"left_arm": [0] * 7, "right_arm": [0] * 7},
                "images": {"head": {}, "left_wrist": {}, "right_wrist": {}},
            }
            original_converter = SERVER.legacy_images_to_observation
            SERVER.legacy_images_to_observation = lambda *args, **kwargs: {}
            try:
                prepared = await session.prepare(json.dumps(request, ensure_ascii=False))
                self.assertEqual(prepared.sent_actions[0][:6], [0.0] * 6)
                self.assertEqual(prepared.sent_actions[0][6:12], [99.0] * 6)
                await session.commit(request, prepared)
            finally:
                SERVER.legacy_images_to_observation = original_converter
                raw_recorder.close()

            with h5py.File(raw_recorder.path, "r") as h5_file:
                self.assertEqual(h5_file["action"].shape, (2, 14))
                self.assertEqual(h5_file["action"][0].tolist(), [99.0] * 14)


if __name__ == "__main__":
    unittest.main(verbosity=2)
