# FastWAM Arm Boundary Tolerance 0.05 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise only the FastWAM arm prediction clamp tolerance from `0.01 rad` to `0.05 rad` while preserving verified physical bounds and the `0.01 rad` waypoint delta limit.

**Architecture:** Reuse the existing `PiperSafetyLimits` and `validate_and_limit_fastwam_actions` implementation unchanged. Pin the deployment value in YAML, strengthen its exact boundary tests, update the operator-facing explanation, and deploy only the server-side repository files.

**Tech Stack:** Python 3.10, NumPy, PyYAML, `unittest`, YAML, SSH.

## Global Constraints

- `joint_lower` and `joint_upper` remain unchanged verified physical limits.
- `max_delta` remains `0.01 rad` for every arm joint at 30 Hz.
- `gripper_boundary_tolerance` remains `0.005 m`.
- Arm predictions outside a physical bound by at most `0.05 rad` are clamped to that bound; overshoots greater than `0.05 rad` reject the response.
- No model, checkpoint, dataset, normalization, image, ROS, start-pose, hold, or robot-side file changes.
- Deployment does not restart the running server or send robot commands.
- Preserve the unrelated untracked `scripts/diagnose_fastwam_validation_boundaries.py` file.

---

### Task 1: Pin and Test the 0.05 Clamp Boundary

**Files:**
- Modify: `tests/test_fastwam_piper_runtime.py`
- Modify: `tests/test_piper_fastwam_policy_server.py`
- Modify: `configs/piper_fastwam_puzzle.yaml`
- Modify: `README.md`

**Interfaces:**
- Consumes: `PiperSafetyLimits` and `validate_and_limit_fastwam_actions(actions_fastwam, current_fastwam, limits_piper)`.
- Produces: deployment config with `arm_boundary_tolerance == 0.05`; no Python interface changes.

- [ ] **Step 1: Add a failing deployment-value test**

Add this test to `tests/test_piper_fastwam_policy_server.py`:

```python
class PiperFastWAMDeploymentConfigTest(unittest.TestCase):
    def test_arm_boundary_tolerance_is_005_without_changing_hard_limits(self) -> None:
        config = server_module.load_config(REPO_ROOT / "configs/piper_fastwam_puzzle.yaml")
        self.assertEqual(float(config["arm_boundary_tolerance"]), 0.05)
        self.assertEqual(float(config["gripper_boundary_tolerance"]), 0.005)
        self.assertEqual(config["max_delta"][:6], [0.01] * 6)
```

- [ ] **Step 2: Run the deployment-value test before changing YAML**

Run:

```bash
python -m unittest tests.test_piper_fastwam_policy_server.PiperFastWAMDeploymentConfigTest -v
```

Expected: FAIL with `0.01 != 0.05`.

- [ ] **Step 3: Strengthen exact recovery tests**

Replace the existing arm recovery test with subtests using overshoots `0.049` and `0.05`. Use `max_delta[1] = 0.01`, current left joint 2 at `0.02`, and a lower bound of `0.0`; assert the first returned left-joint-2 value is `0.01`, proving clamp-to-zero occurs before the delta limiter. Update the hard rejection test to use a `0.051` overshoot with `arm_boundary_tolerance=0.05` and assert `ValueError`.

- [ ] **Step 4: Change the single deployment value and explanation**

In `configs/piper_fastwam_puzzle.yaml`, set:

```yaml
arm_boundary_tolerance: 0.05
```

Update the adjacent comment and `README.md` to say that arm overshoots up to `0.05 rad` are clamped to the unchanged physical bound, while `max_delta` remains `0.01 rad` per waypoint and larger overshoots reject.

- [ ] **Step 5: Run focused and complete tests**

Run:

```bash
python -m unittest \
  tests.test_fastwam_piper_runtime.PiperFastWAMSafetyTest \
  tests.test_piper_fastwam_policy_server.PiperFastWAMDeploymentConfigTest -v
python -m unittest discover -s tests -v
```

Expected: all available tests pass; the two OpenCV transport tests may remain skipped on the local host.

- [ ] **Step 6: Commit the implementation**

Run:

```bash
git add configs/piper_fastwam_puzzle.yaml README.md tests/test_fastwam_piper_runtime.py tests/test_piper_fastwam_policy_server.py docs/superpowers/plans/2026-08-04-fastwam-arm-boundary-tolerance-005.md
git commit -m "fix: widen FastWAM arm clamp tolerance"
```

### Task 2: Development-Machine Deployment

**Files:**
- Deploy: `configs/piper_fastwam_puzzle.yaml`
- Deploy: `README.md`
- Deploy: `tests/test_fastwam_piper_runtime.py`
- Deploy: `tests/test_piper_fastwam_policy_server.py`

**Interfaces:**
- Consumes: committed Task 1 files.
- Produces: the same config and tests under `/bh/zbh_self/projects/piper-openpi-real-robot`.

- [ ] **Step 1: Copy only the four changed runtime-facing files**

Use `scp -P 3763` to copy the config, documentation, and tests to the matching paths under `/bh/zbh_self/projects/piper-openpi-real-robot`. Do not copy or modify any file below `/bh/media`.

- [ ] **Step 2: Run development-machine validation**

Run:

```bash
cd /bh/zbh_self/projects/piper-openpi-real-robot
/bh/zbh_self/envs/fastwam/bin/python -m unittest \
  tests.test_fastwam_piper_runtime.PiperFastWAMSafetyTest \
  tests.test_piper_fastwam_policy_server.PiperFastWAMDeploymentConfigTest -v
/bh/zbh_self/envs/fastwam/bin/python -m unittest discover -s tests -v
```

Expected: all available tests pass and the deployment config test reports `0.05`.

- [ ] **Step 3: Verify exact deployment and report restart requirement**

Read the deployed YAML through `load_config` and print:

```text
ARM_BOUNDARY_TOLERANCE_OK 0.05 MAX_DELTA_ARM 0.01 GRIPPER_TOLERANCE 0.005
```

Compare local and development-machine SHA-256 hashes for the YAML. Do not stop PID `194303` or launch another server. Report that the foreground FastWAM server must be manually restarted before the new value takes effect.

## Self-Review

- The plan changes one runtime value and only the tests/documentation needed to make its safety semantics explicit.
- Verified hard bounds, delta limits, and gripper tolerance are invariant.
- No placeholder, undefined interface, robot-side deployment, or automatic process action is included.
