# FastWAM Pi05-Style Action Clamping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Return rate-limited, physically bounded FastWAM actions for all finite predictions without projection-ceiling failures.

**Architecture:** Keep the pure runtime adapter responsible for absolute and sequential clamping. Keep the WebSocket server responsible for request validation and diagnostic logging only; it no longer maintains a projection trip state.

**Tech Stack:** Python 3.10, NumPy, unittest, PyYAML, FastWAM runtime.

## Global Constraints

- Never alter the verified `joint_lower`, `joint_upper`, or `max_delta` vectors.
- Do not publish a robot command during implementation or verification.
- Reject only malformed, non-finite, unavailable-model, or timeout responses; finite joint values are clamped.

---

### Task 1: Make the pure adapter clamp every finite value

**Files:**
- Modify: `scripts/fastwam_piper_runtime.py`
- Modify: `tests/test_fastwam_piper_runtime.py`

**Interfaces:**
- Produces: `project_and_limit_fastwam_actions(actions_fastwam, current_fastwam, limits_piper) -> (np.ndarray, ActionProjectionSummary)`.
- Preserves: `validate_and_limit_fastwam_actions(...) -> np.ndarray`.

- [ ] Replace the tests that expect projection-ceiling `ValueError` with a test that sends `-0.1273711` at a zero lower bound and expects the sequentially limited value.
- [ ] Add a test that a large finite gripper overshoot is clamped to the physical endpoint and then its configured delta.
- [ ] Remove projection-ceiling fields and rejection branches while retaining finite validation, endpoint clamp, sequential `max_delta`, and projection summary.
- [ ] Run `python -m unittest tests/test_fastwam_piper_runtime.py -v` and require success.

### Task 2: Remove server projection-trip state

**Files:**
- Modify: `scripts/piper_fastwam_policy_server.py`
- Modify: `tests/test_piper_fastwam_policy_server.py`
- Modify: `configs/piper_fastwam_puzzle.yaml`

**Interfaces:**
- Consumes: `ActionProjectionSummary` only for logging.
- Produces: a normal action response for repeated finite out-of-range predictions.

- [ ] Replace the no-action and three-response-trip tests with finite large-overshoot responses that remain `ok` and respect `max_delta`.
- [ ] Remove projection-threshold configuration parsing and streak state, retaining a warning log for corrected finite values.
- [ ] Remove unused projection-threshold keys from the deployment YAML without changing physical ranges or deltas.
- [ ] Run `python -m unittest tests/test_piper_fastwam_policy_server.py -v` and require success.

### Task 3: Deploy and perform a no-actuation check

**Files:**
- Modify: `README.md`

- [ ] Document that finite FastWAM predictions are endpoint-clamped and rate-limited, while non-finite values still fail closed.
- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Sync only the changed files to `/bh/zbh_self/projects/piper-openpi-real-robot` and run `CUDA_VISIBLE_DEVICES=0 bash scripts/run_piper_fastwam_server.sh --preflight-only` in `/bh/zbh_self/envs/fastwam`.
- [ ] Ask the operator to restart terminal 1 and run terminal 3 with `--executor mock`; do not use `--execute-actions` during verification.
