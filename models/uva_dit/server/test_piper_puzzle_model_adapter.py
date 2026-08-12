#!/usr/bin/env python3
"""Torch-level tests for the 14D padded Pi07 adapter, without model weights."""

from __future__ import annotations

from collections import deque
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


MODULE_PATH = Path(__file__).with_name("piper_puzzle_policy_server.py")
SPEC = importlib.util.spec_from_file_location("piper_puzzle_policy_server", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
SERVER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SERVER
SPEC.loader.exec_module(SERVER)


def make_adapter_shell():
    engine = SERVER.PiperPuzzleEngineMixin.__new__(SERVER.PiperPuzzleEngineMixin)
    engine._torch = torch
    engine._np = np
    engine.physical_action_dim = 14
    engine.action_dim = 32
    engine.state_oob_mode = "clip"
    engine.action_output_clip = False
    engine._condition_q01 = torch.arange(14, dtype=torch.float32).reshape(-1, 1, 1)
    engine._condition_q99 = engine._condition_q01 + 2.0
    engine._target_q01 = engine._condition_q01.clone()
    engine._target_q99 = engine._condition_q99.clone()
    engine._exec_action_hist = deque(maxlen=16)
    engine._pending_history_count = 0
    return engine


class PaddingAndHistoryTests(unittest.TestCase):
    def test_normalize_physical14_then_pad18_zero_dimensions(self) -> None:
        engine = make_adapter_shell()
        midpoint = torch.arange(14, dtype=torch.float32) + 1.0
        normalized = engine._normalize_state_rows(midpoint)
        self.assertEqual(tuple(normalized.shape), (1, 32))
        torch.testing.assert_close(normalized[:, :14], torch.zeros(1, 14))
        torch.testing.assert_close(normalized[:, 14:], torch.zeros(1, 18))

    def test_denormalize_returns_time_major_physical14_only(self) -> None:
        engine = make_adapter_shell()
        actions = engine._denormalize_actions(torch.zeros(32, 2, 3))
        self.assertEqual(actions.shape, (6, 14))
        expected = np.broadcast_to(np.arange(14, dtype=np.float32) + 1.0, (6, 14))
        np.testing.assert_allclose(actions, expected, atol=2e-6)

    def test_acknowledged_actions_do_not_build_history(self) -> None:
        engine = make_adapter_shell()
        engine._pending_history_count = 2
        sent = [
            (torch.arange(14, dtype=torch.float32) + 1.0).tolist(),
            (torch.arange(14, dtype=torch.float32) + 1.5).tolist(),
        ]
        engine.acknowledge_sent_actions(sent)
        self.assertEqual(engine._pending_history_count, 0)
        self.assertEqual(len(engine._exec_action_hist), 0)

    def test_measured_rows_are_the_supported_history_input(self) -> None:
        engine = make_adapter_shell()
        measured = torch.stack(
            [
                torch.arange(14, dtype=torch.float32) + 1.0,
                torch.arange(14, dtype=torch.float32) + 1.5,
            ]
        )
        engine._measured_state_hist = engine._normalize_state_rows(measured)
        self.assertEqual(tuple(engine._measured_state_hist.shape), (2, 32))
        torch.testing.assert_close(
            engine._measured_state_hist[0, :14], torch.zeros(14)
        )
        torch.testing.assert_close(
            engine._measured_state_hist[1, :14], torch.full((14,), 0.5)
        )
        torch.testing.assert_close(
            engine._measured_state_hist[:, 14:], torch.zeros((2, 18))
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
