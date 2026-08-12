# FastWAM Piper Episode 159 Start Pose Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the episode-242 FastWAM Puzzle start pose with the runtime-screened real episode-159 start pose and deploy the configuration without moving the robot.

**Architecture:** Keep the existing start-pose loader, motion controller, feedback gate, model, and safety limits unchanged. Update only the traceable JSON data and its exact-value tests/documentation, validate locally and on the development machine, then copy the JSON to the robot and compare hashes.

**Tech Stack:** JSON, Python 3.9/3.10, `unittest`, SSH, SHA-256.

## Global Constraints

- Source data remains read-only at `/bh/zbh_ckp/datasets/fastwam/agilex_cobotmagic2_puzzle_20260801_v21_puppet_joint`.
- The replacement is the real frame-zero state from episode 159 in Piper order `[L1..L6,Lg,R1..R6,Rg]`.
- Do not change joint bounds, action-delta limits, boundary tolerances, checkpoint, inference horizon, image processing, or ROS execution behavior.
- Deployment must not start `websocket_policy_client.py`, publish ROS commands, enable the robot, or move either arm.
- Preserve unrelated working-tree changes.

---

### Task 1: Pin the Episode 159 Configuration Contract

**Files:**
- Modify: `tests/test_piper_start_pose.py`
- Modify: `configs/piper_fastwam_puzzle_start_pose.json`
- Modify: `README.md`

**Interfaces:**
- Consumes: existing `load_start_pose_config(path)` and `StartPoseConfig`.
- Produces: the unchanged config-file interface with `source_episode == 159` and the exact recorded left/right targets.

- [ ] **Step 1: Change the exact-value test first**

Replace the episode-242 assertion with:

```python
def test_loads_runtime_screened_episode_159_pose(self) -> None:
    config = load_start_pose_config(CONFIG_PATH)
    self.assertEqual(config.source_episode, 159)
    self.assertEqual(
        config.left_arm,
        (
            0.018856963,
            0.015263500,
            -0.003558576,
            0.035690423,
            0.091668218,
            -0.025258912,
            0.0005,
        ),
    )
    self.assertEqual(config.right_arm[-1], 0.0008)
    self.assertLess(config.max_joint_speed, 0.3)
```

- [ ] **Step 2: Verify that the contract fails against episode 242**

Run:

```bash
python -m unittest tests.test_piper_start_pose.PiperStartPoseConfigTest.test_loads_runtime_screened_episode_159_pose -v
```

Expected: FAIL because the checked-in JSON still reports episode 242.

- [ ] **Step 3: Replace only the traceable pose values**

Set these JSON fields exactly:

```json
{
  "name": "fastwam_puzzle_episode_159_start",
  "source_episode": 159,
  "left_arm": [0.018856963, 0.015263500, -0.003558576, 0.035690423, 0.091668218, -0.025258912, 0.0005],
  "right_arm": [0.012245688, 0.004709880, -0.000837312, -0.010954832, 0.154728279, 0.0, 0.0008]
}
```

Leave the dataset path, bounds, motion rate, speed, timeouts, and tolerances byte-for-byte unchanged.

- [ ] **Step 4: Update operator documentation**

Replace the medoid/episode-242 description with the runtime-screened episode-159 rationale. State that the replacement adds left-joint-2 margin, was screened for three stochastic runs, and still requires live-scene validation because recorded images cannot guarantee the live prediction.

- [ ] **Step 5: Run focused tests**

Run:

```bash
python -m unittest tests.test_piper_start_pose tests.test_websocket_policy_client_start_pose -v
```

Expected: all tests pass and the exact target values are episode 159.

### Task 2: Regression Validation and Non-Actuating Deployment

**Files:**
- Deploy: `configs/piper_fastwam_puzzle_start_pose.json`
- Validate without modifying: `scripts/piper_start_pose.py`, `scripts/websocket_policy_client.py`

**Interfaces:**
- Consumes: the episode-159 JSON from Task 1.
- Produces: identical JSON content at the local repository, development machine, and robot project.

- [ ] **Step 1: Run the repository regression suite**

Run in the local repository and `/bh/zbh_self/projects/piper-openpi-real-robot`:

```bash
python -m unittest discover -s tests -v
```

Expected: all available tests pass; environment-specific optional dependency skips are acceptable only when reported explicitly.

- [ ] **Step 2: Commit the reviewed change**

Run:

```bash
git add configs/piper_fastwam_puzzle_start_pose.json tests/test_piper_start_pose.py README.md docs/superpowers/specs/2026-08-04-fastwam-piper-training-start-pose-design.md docs/superpowers/plans/2026-08-04-fastwam-piper-episode-159-start-pose.md
git commit -m "fix: use runtime-screened FastWAM start pose"
```

- [ ] **Step 3: Copy only non-actuating files**

Copy the JSON and updated test/documentation repository state to `/bh/zbh_self/projects/piper-openpi-real-robot`, then copy only the JSON to:

```text
/home/agilex/piper-openpi-real-robot/configs/piper_fastwam_puzzle_start_pose.json
```

Do not launch a client or ROS command during either copy.

- [ ] **Step 4: Validate both remote copies**

On the development machine, import the config through `load_start_pose_config` and print:

```text
START_POSE_CONFIG_OK episode=159 left_j2=0.015263500
```

On the robot, use the xrocs environment to run the same read-only import and print the same line.

- [ ] **Step 5: Compare exact file hashes**

Run SHA-256 on the local, development-machine, and robot JSON files.

Expected: all three hashes are identical. Report the exact modified paths, environments used, and that no robot process was started.

## Self-Review

- The plan covers the approved episode-159 substitution, exact-value test, documentation, regression tests, deployment, import validation, and three-way hash comparison.
- No model, limit, tolerance, ROS, image, dataset, or checkpoint change is included.
- No placeholder steps or undefined interfaces remain.
