# FastWAM Bounded Action Projection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make finite boundary overshoots recoverable while preserving Piper's absolute and per-waypoint physical limits.

**Architecture:** Keep the pure action adapter responsible for projection, hard rejection, and rate limiting.  Let the WebSocket server consume projection metadata to log and trip only repeated large corrections.  Pin the diffusion seed in deployment configuration.

**Tech Stack:** Python 3.10, NumPy, unittest, PyYAML, FastWAM runtime.

## Global Constraints

- Never alter the verified `joint_lower`, `joint_upper`, or `max_delta` vectors.
- Never publish a robot command as part of testing or deployment.
- Arm correction ceiling is `0.10 rad`; gripper correction ceiling is `0.005 m`.
- Large-arm threshold is `0.05 rad`; trip after three consecutive responses.

---

### Task 1: Return projection metadata from the pure safety adapter

**Files:**
- Modify: `scripts/fastwam_piper_runtime.py`
- Test: `tests/test_fastwam_piper_runtime.py`

**Interfaces:**
- Produces: `project_and_limit_fastwam_actions(actions_fastwam, current_fastwam, limits_piper) -> (np.ndarray, ActionProjectionSummary)`.
- Preserves: `validate_and_limit_fastwam_actions(...) -> np.ndarray` for existing callers/tests.

- [ ] Add a failing test for a `+0.089842878 rad` L3 prediction that is projected to its `0.0` upper endpoint and then rate-limited.
- [ ] Add a failing test for an arm overshoot larger than `0.10 rad` and a gripper overshoot larger than `0.005 m`; each must raise `ValueError`.
- [ ] Implement projection to physical endpoints, calculate count/max corrections, apply existing sequential `max_delta`, and return immutable projection metadata.
- [ ] Run `python -m unittest tests/test_fastwam_piper_runtime.py -v` and require all tests to pass.

### Task 2: Add server-side repeated-large-projection trip logic

**Files:**
- Modify: `scripts/piper_fastwam_policy_server.py`
- Test: `tests/test_piper_fastwam_policy_server.py`

**Interfaces:**
- Consumes: `ActionProjectionSummary.max_arm_projection` from Task 1.
- Produces: a normal action response for the first two large projected requests and a no-action error response on the third consecutive request.

- [ ] Add a failing async server test using a finite `1.09` prediction against an upper limit of `1.0`, with a `0.10` correction ceiling and a three-response trip.
- [ ] Parse and validate `large_arm_projection_threshold` and `max_consecutive_large_projection_responses` from config.
- [ ] Replace the adapter call with the metadata-returning API, log corrections, reset the streak after a non-large response, and return no action when the streak reaches the configured limit.
- [ ] Run `python -m unittest tests/test_piper_fastwam_policy_server.py -v` and require all tests to pass.

### Task 3: Pin deployment configuration and verify a no-actuation rollout

**Files:**
- Modify: `configs/piper_fastwam_puzzle.yaml`
- Modify: `README.md`
- Test: `tests/test_piper_fastwam_policy_server.py`

- [ ] Replace tolerance-only fields with the bounded-projection settings, set `seed: 42`, and add a test for the exact values.
- [ ] Document that a server restart is mandatory after syncing and that only `--executor mock` may be used for the first live check.
- [ ] Run the full CPU suite with `python -m unittest discover -s tests -v`.
- [ ] Sync only changed server/config files to `/bh/zbh_self/projects/piper-openpi-real-robot`, run `--preflight-only`, then request an operator restart before the mock test.  Do not invoke `--execute-actions`.
