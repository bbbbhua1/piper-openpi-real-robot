# FastWAM Piper Real-Robot Deployment Design

## Goal

Run the completed Puzzle FastWAM checkpoint on the dual-arm Piper robot through the existing WebSocket and ROS joint-control workflow. The target is the task represented by the training label `crimp`, not the prior OpenPI domino task.

## Scope and constraints

- Deploy inference on `ssh -p 3763 root@10.40.1.215`, which has two NVIDIA A800 80 GB GPUs.
- Preserve the existing Pi0.5/OpenPI server as a working baseline. Do not replace or modify it.
- Preserve the existing SSH tunnel, JSON WebSocket protocol, robot-side ROS client, camera recording, and fault-stop behavior wherever possible.
- Use the final checkpoint and its matching normalization statistics:

  ```text
  /bh/zbh_ckp/runs/fastwam/agilex_cobotmagic2_puzzle_joint_3cam_384_1e-4_baidu32/CogWAM_fastWAM-copy5_retry4/checkpoints/weights/step_001695.pt
  /bh/zbh_ckp/datasets/fastwam/agilex_cobotmagic2_puzzle_20260801_v21_puppet_joint/normalization/dataset_stats.json
  ```

- Use the existing FastWAM project and environment:

  ```text
  /bh/zbh_self/projects/fastwam
  /bh/zbh_self/envs/fastwam
  ```

- Do not modify source data under `/bh/media/unify` or `/bh/media/unify-fileset`.
- Physical execution is in scope, but only after the software and safety gates in this design pass.

## Selected architecture

Add a sibling FastWAM backend rather than changing `scripts/piper_openpi_policy_server.py`.

```text
Piper ROS client                 Local computer                 Development server
----------------                 --------------                 ------------------
websocket_policy_client.py  ->   SSH tunnel                ->    piper_fastwam_policy_server.py
3 camera JPEGs + 14D state                                       FastWAM + Wan inference
14D joint action chunk      <-                               <-  normalize, predict, denormalize
```

The current robot client keeps its JSON request and response format. The new server owns all FastWAM-specific loading, image composition, normalization, action layout conversion, and output safety validation. This maintains a reversible fallback to the known OpenPI workflow.

## Server responsibilities

Create `scripts/piper_fastwam_policy_server.py` in this repository. It must:

1. Import FastWAM from a configurable `--fastwam-root` (defaulting to `/bh/zbh_self/projects/fastwam`) and load the model with the same Hydra task configuration used for training.
2. Load `step_001695.pt`, set the model to evaluation mode, and load `dataset_stats.json` into `FastWAMProcessor` before accepting requests.
3. Decode the existing `head`, `left_wrist`, and `right_wrist` JPEG payloads as RGB, then compose the exact RobotWin-style FastWAM image:
   - head: 320 x 256;
   - each wrist: 160 x 128;
   - concatenate wrist cameras horizontally below the head;
   - final image: RGB 384 x 320, CHW, scaled from `[0, 255]` to `[-1, 1]`.
4. Use the training prompt template exactly:

   ```text
   A video recorded from a robot's point of view executing the following instruction: crimp
   ```

5. Invoke FastWAM with `action_horizon=32` and `num_video_frames=9`, matching `num_frames=33` and `action_video_freq_ratio=4` used in training.
6. Normalize the current state before inference and denormalize predicted actions afterwards using the loaded dataset statistics.
7. Return the existing JSON response shape with a bounded initial segment of the predicted 32-step action trajectory. `--execution-horizon` controls that segment separately from the model action horizon.

## Required joint-layout adapter

The current Piper wire layout is assumed to be:

```text
[L1,L2,L3,L4,L5,L6,L_gripper, R1,R2,R3,R4,R5,R6,R_gripper]
```

The Puzzle FastWAM data layout is:

```text
[L1,L2,L3,L4,L5,L6, R1,R2,R3,R4,R5,R6, L_gripper,R_gripper]
```

The new server must make this conversion explicit and test it in both directions. It must reject an incoming state unless both arm vectors have exactly seven finite values. It must validate this assumption against a live ROS observation before any motion is enabled.

Before returning a FastWAM action, the server converts it back to the Piper wire layout. The existing client can therefore retain its two seven-value `left_arm` / `right_arm` response fields and its current publishers.

## Safety and control behavior

- The server must reject NaN/Inf actions, malformed requests, missing required images, and actions outside configured absolute Piper joint limits.
- Clamp per-waypoint joint deltas against the measured current Piper state. Gripper limits are configured independently of arm joint limits.
- A server error must return `ok: false` and no action object. The client must not publish any action from an unsuccessful response.
- Add request-to-action latency logging and a configurable maximum inference latency. Requests exceeding it fail closed.
- Keep execution disabled for offline and hardware dry-run tests. The existing robot status fault monitor and `/enable_flag=false` stop behavior remain authoritative.
- After dry-run approval, start with a single waypoint and conservative delta limits. Only then raise `execution-horizon` for the complete `crimp` task.
- `action_dt` is configurable. Its initial value must be validated against the 30 FPS action labels rather than copied from the Pi0.5 command.
- Each physical request uses the existing smooth home sequence before policy execution. The client then holds the final **published** waypoint, not a possibly delayed ROS observation; `Ctrl-C` exits that hold loop while leaving the arm enabled at the held pose.

## Deployment configuration

Add a dedicated checked-in deployment configuration containing only runtime paths and non-secret defaults: FastWAM root, checkpoint path, stats path, model task configuration, prompt task label, GPU selection, model horizon, execution horizon, image mappings, and safety limits.

The launch command activates `/bh/zbh_self/envs/fastwam`, uses one selected A800, points at the config, and listens only on `127.0.0.1:7081`. The existing SSH tunnel and robot-side client keep their current roles. No permanent proxy configuration is written.

## Validation plan

1. Unit-test image decode/composition, state-layout conversion, action reverse conversion, finite-value validation, limits, and successful/failed JSON responses without CUDA or ROS.
2. Start the FastWAM server with a mock model and run the existing robot client in mock source/mock executor mode.
3. Load the actual model on one A800 and replay a saved three-camera state request. Record VRAM use, action shape, action range, and inference latency.
4. On the robot, run the actual client without `--execute-actions`; save received server images and verify their keys, dimensions, prompt, normalized state range, and returned action values.
5. With the task scene physically prepared and an operator at the emergency stop, run one limited waypoint, then a bounded short chunk, then the full `crimp` attempt. Stop immediately on any safety rejection, ROS fault, timeout, or action-limit violation.

## Non-goals

- Converting the policy to EEF control.
- Deleting, changing, or replacing the OpenPI backend.
- Making the current single-task `crimp` checkpoint a general language-conditioned policy.
- Changing robot calibration, hardware limits, or camera mounts from software alone.
