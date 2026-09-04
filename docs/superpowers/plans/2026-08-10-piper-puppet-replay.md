# Piper Puppet Episode 382 Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replay the verified RoboMIND/Puppet episode 382 on the Piper robot using the existing WebSocket client protocol, with the left arm held at its fixed observation pose and only the right-arm Puppet trajectory changing.

**Architecture:** Add a standalone replay server that loads only episode 382 from the shared RoboMIND Parquet file, converts Puppet's `[right six joints, right gripper]` and fixed left pose to the existing Piper `[L1,L2,L3,L4,L5,L6,Lg,R1,R2,R3,R4,R5,R6,Rg]` envelope, and returns one immutable 453-waypoint action chunk. Add a dedicated start-pose JSON so the existing client moves to the first Puppet frame before requesting the chunk. Keep the existing UVA-DiT model server, existing client, and existing FastWAM replay server unchanged.

**Tech Stack:** Python 3.10, NumPy, PyArrow, websockets, existing `ws_policy_protocol.py`, existing ROS `websocket_policy_client.py`.

## Global Constraints

- Source data is fixed to `/bh/media/unify/special_project/puzzle/agilex_cobotmagic2_dualArm-gripper-3cameras_34/agilex_cobotmagic2_dualArm-gripper-3cameras_34_puzzle_20260805/success/lerobot_RoboMIND/data/chunk-000/file-000.parquet`.
- Only `episode_index=382` is accepted; the server must reject other episodes.
- Only `puppet.*_align.data` columns are used; no `master` or model output is used.
- Episode 382 must validate as 453 rows at approximately 30 Hz, with strictly increasing timestamps.
- Left waypoints are constant and contain the verified Puppet left observation pose; right waypoints contain the Puppet right arm and right gripper values without smoothing or clipping.
- Validate finite values and the existing Piper absolute joint limits before serving; validation is a preflight guard and must not modify trajectory values.
- The replay server uses a separate WebSocket port (7082) and the client uses `--max-steps 1`, so the one-shot replay does not request a second chunk.
- Do not push, overwrite the UVA-DiT server, or start physical motion during offline validation.

---

### Task 1: Add the episode 382 start-pose configuration

**Files:**
- Create: `/bh/zbh_self/projects/piper-openpi-real-robot/configs/piper_robomind_puppet_episode_382_start_pose.json`
- Reference: `/bh/zbh_self/projects/piper-openpi-real-robot/scripts/piper_start_pose.py`

**Interfaces:**
- Consumes: the verified fixed left Puppet pose and the first right Puppet frame.
- Produces: a `StartPoseConfig` accepted by `load_start_pose_config`, with source episode 382 and Piper wire order `[L1,L2,L3,L4,L5,L6,Lg,R1,R2,R3,R4,R5,R6,Rg]`.

- [ ] **Step 1: Write the JSON with the exact values.**

  Set `left_arm` to `[0.15348975360393524, 1.580007791519165, -1.0552573204040527, -0.9023781418800354, 1.180993676185608, 2.010979175567627, 0.0005000000237487257]` and `right_arm` to `[-0.0006628719856962562, 0.006367059890180826, -0.004988984204828739, 0.012629455886781216, 0.08828408271074295, 0.029096592217683792, 0.0007999999797903001]`.

  Use the existing verified Piper limits, `move_hz=20.0`, `max_joint_speed=0.15`, `min_duration_s=2.0`, `max_duration_s=20.0`, `feedback_timeout_s=5.0`, `arm_tolerance=0.01`, and `gripper_tolerance=0.001`.

- [ ] **Step 2: Validate the config without ROS.**

  Run:

  ```bash
  cd /bh/zbh_self/projects/piper-openpi-real-robot
  /bh/zbh_self/envs/fastwam/bin/python -c 'from scripts.piper_start_pose import load_start_pose_config; print(load_start_pose_config("configs/piper_robomind_puppet_episode_382_start_pose.json"))'
  ```

  Expected: a printed `StartPoseConfig` with `source_episode=382` and no exception.

---

### Task 2: Add the independent RoboMIND Puppet replay server

**Files:**
- Create: `/bh/zbh_self/projects/piper-openpi-real-robot/scripts/piper_robomind_puppet_replay_server.py`
- Create: `/bh/zbh_self/projects/piper-openpi-real-robot/tests/test_piper_robomind_puppet_replay_server.py`
- Reference without modification: `scripts/piper_dataset_replay_server.py`, `scripts/ws_policy_protocol.py`

**Interfaces:**
- `load_episode(dataset_root: Path, episode_index: int, config: StartPoseConfig) -> ReplayEpisode`
- `ReplayEpisode.as_piper_action() -> dict[str, Any]`
- `PuppetReplayServer.handle_message(message: str) -> dict[str, Any]`
- CLI defaults: fixed dataset root, episode 382, host `127.0.0.1`, port `7082`, and `--preflight-only`.

- [ ] **Step 1: Write failing loader tests.**

  Cover the exact column names, conversion to 14D Piper order, constant 7D left waypoints, 453-frame shape, 30-Hz cadence, rejection of another episode index, rejection of non-finite values, and rejection of a right-arm value outside the start-pose limits.

- [ ] **Step 2: Implement the minimal loader.**

  Read only `episode_index`, `timestamp`, `puppet.arm_left_position_align.data`, `puppet.end_effector_left_position_align.data`, `puppet.arm_right_position_align.data`, and `puppet.end_effector_right_position_align.data` with PyArrow. Filter rows where `episode_index == 382`, sort by source row order already present in the Parquet file, stack left six joints plus left gripper and right six joints plus right gripper, validate 14D bounds and `timestamp[i+1]-timestamp[i]` against `1/30` with tolerance `5e-5`, and mark returned NumPy arrays read-only.

- [ ] **Step 3: Implement one-shot WebSocket serving.**

  Require a JSON object containing `request_id`, `step`, and a mapping `state`. Return `build_policy_response` with the exact full action chunk and `time_list = [dt, 2*dt, ..., 453*dt]`. Serve only the first valid request; return an error response for later requests. Do not import or load a model.

- [ ] **Step 4: Run the focused tests.**

  Run:

  ```bash
  cd /bh/zbh_self/projects/piper-openpi-real-robot
  /bh/zbh_self/envs/fastwam/bin/python -m unittest -v tests/test_piper_robomind_puppet_replay_server.py
  ```

  Expected: all loader and one-shot protocol tests pass.

---

### Task 3: Run real-data offline preflight

**Files:**
- Read only: the fixed RoboMIND Parquet file and the two newly created files.
- Output: no checkpoint or dataset copy; write only a short preflight log under `/bh/zbh_self/logs/` if needed.

**Interfaces:**
- Consumes: the start-pose config and replay server CLI.
- Produces: validated episode summary and a local WebSocket request/response smoke test with `--preflight-only` first.

- [ ] **Step 1: Run real Parquet preflight.**

  ```bash
  cd /bh/zbh_self/projects/piper-openpi-real-robot
  /bh/zbh_self/envs/fastwam/bin/python scripts/piper_robomind_puppet_replay_server.py --preflight-only
  ```

  Expected: episode 382, 453 waypoints, about 15.066667 seconds, 30 Hz, constant left pose, and no limit/cadence error.

- [ ] **Step 2: Run a local mock protocol smoke test.**

  Start the server on an unused local port, send one valid JSON policy request using the existing protocol, assert `ok=true`, 453 left/right waypoints, and exact first/last right Puppet values, then stop the server. No ROS, model server, or physical robot is involved.

---

### Task 4: Prepare the real-robot command sequence

**Files:**
- Create: `/bh/zbh_self/projects/piper-openpi-real-robot/docs/piper_robomind_puppet_episode_382_replay.md`
- Do not modify the existing model-server runbook.

**Interfaces:**
- Terminal 1: development-machine replay server plus SSH local tunnel `17082 -> 7082`.
- Terminal 2: robot ROS/Piper/camera status verification only.
- Terminal 3: existing client with ROS source, ROS executor, the new start-pose config, `--max-steps 1`, and `--execute-actions`.

- [ ] **Step 1: Verify the robot has one ROS master, one Piper launch, both CAN ports up, and the observation topics ready.**
- [ ] **Step 2: Start the replay server and tunnel only after offline preflight passes.**
- [ ] **Step 3: Run the client once.**

  The client first moves to the episode start pose at the configured low speed, then requests and publishes the one 453-waypoint chunk. Since the response contains a constant left arm, the left side receives hold commands while the right side follows Puppet. The client will hold on exit unless interrupted.

- [ ] **Step 4: Stop the replay server and tunnel after the run; no git push.**

## Self-review

- The plan leaves the existing UVA-DiT server, existing FastWAM replay server, and general client unchanged.
- The plan covers all six Puppet columns needed for right-only replay plus the fixed left hold trajectory.
- The plan has explicit tests for dimensions, cadence, bounds, protocol, and real-data preflight.
- No model checkpoint, dataset, or large generated artifact is copied; the shared dataset is read in place and the `fastwam` environment is reused.
