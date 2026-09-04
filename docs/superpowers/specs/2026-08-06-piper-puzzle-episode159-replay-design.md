# Piper Puzzle Episode 159 Replay Design

## Goal

Replay one verified FastWAM Puzzle training episode on the real Piper robot at
the recorded 30 Hz rate, without model inference, to determine whether the
recorded joint trajectory can be physically reproduced through the deployed
server, tunnel, and robot-client path.

## Selected source

- Dataset: `/bh/zbh_ckp/datasets/fastwam/agilex_cobotmagic2_puzzle_20260801_v21_puppet_joint`
- Episode: `159`
- Source action order: `[L1,L2,L3,L4,L5,L6,R1,R2,R3,R4,R5,R6,Lg,Rg]`
- Piper wire order: `[L1,L2,L3,L4,L5,L6,Lg,R1,R2,R3,R4,R5,R6,Rg]`
- Frames: 610 at 30 Hz (20.30 seconds)
- Head reference: `videos/chunk-000/observation.images.cam_high/episode_000159.mp4`

Episode 159 is selected because its first observation/action exactly matches
the existing verified Puzzle start-pose configuration. The robot can therefore
move to a known recorded starting pose before replay starts.

## Components and data flow

1. A new server-side replay program loads only the selected episode parquet
   file with PyArrow, verifies finite 14-D actions, verified absolute Piper
   joint bounds, and the expected 30 Hz timestamps, converts FastWAM order to
   Piper wire order, and serves one fixed action stream over the existing
   WebSocket request/response protocol.
2. The existing SSH tunnel exposes that server to the robot network without
   changing the FastWAM policy server.
3. The existing robot WebSocket client first moves to the configured start
   pose, verifies feedback, then asks for the replay sequence and publishes it
   through the current ROS executor.
4. The robot client records the live three camera feeds. The selected dataset
   head video is separately copied to the operator's local computer for
   visual comparison.

## Safety and stop behavior

- The replay does not loosen verified absolute joint limits.
- The recorded action values and their 30 Hz cadence are preserved exactly:
  no interpolation, projection, per-waypoint rate limiting, smoothing, or
  time rescaling is permitted. This exception is limited to this one immutable
  successful historical episode; it does not change FastWAM model-output
  safety handling.
- Startup rejects the replay if any recorded target is non-finite or outside
  a verified absolute Piper limit.
- Start-pose feedback must be within the current arm and gripper tolerances
  before action publication begins.
- The replay is exactly 30 Hz; it is not sped up.
- A first Ctrl-C stops further waypoints and holds the latest command. A
  second Ctrl-C exits the client.
- Replay is limited to one episode and one client connection, so it cannot
  silently loop or serve a different training sample.

## Validation

Before deployment, tests will confirm the episode's length, first-vector
match to the configured start pose, order conversion, exact timestamp cadence,
all absolute bounds, byte-for-byte preservation of each source action after
order conversion, and that the final WebSocket action chunk is correctly
formed. A ROS-free mock client test will exercise the protocol. On the robot,
the operator will confirm start-pose completion before allowing the 20.30 s
replay to begin.

## Outputs

- Server/client code remains under `/bh/zbh_self/projects/piper-openpi-real-robot`.
- Replay logs are written under `/bh/zbh_self/logs/fastwam`.
- The robot's live replay recording stays under
  `/home/agilex/piper-openpi-real-robot/recordings` and uses the existing
  optional upload mechanism.
- The selected reference head video is copied to the local Mac under
  `/Users/hua/Downloads/fastwam-replay-reference/`.
