#!/usr/bin/env python3
"""Client-side tests for recording actual waypoint commands."""

from __future__ import annotations

import unittest

from websocket_policy_client import MockExecutor


class WaypointCallbackTests(unittest.TestCase):
    def test_callback_receives_exact_merged_commands(self) -> None:
        executor = MockExecutor(1.0 / 30.0, 7, 7)
        events = []
        action = {
            "left_arm": [[1.0] * 7, [2.0] * 7],
            "right_arm": [[3.0] * 7, [4.0] * 7],
            "time_list": [1.0 / 30.0, 2.0 / 30.0],
        }
        executor.execute_action(
            action,
            on_waypoint=lambda left, right, index: events.append(
                (left, right, index)
            ),
        )
        self.assertEqual(
            events,
            [
                ([1.0] * 7, [3.0] * 7, 0),
                ([2.0] * 7, [4.0] * 7, 1),
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
