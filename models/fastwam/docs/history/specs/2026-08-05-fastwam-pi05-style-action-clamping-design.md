# FastWAM Pi05-Style Action Clamping Design

## Objective

Make FastWAM handling of finite action predictions match the established Pi05
deployment behavior: never reject a request merely because a finite model
prediction lies far outside the robot workspace.

## Selected design

For each of the 24 executable FastWAM waypoints, the server will first clamp
each finite arm and gripper position to the verified Piper absolute joint
range, then sequentially clamp it to the verified per-waypoint `max_delta`
from the current/previous accepted command. This is the Pi05-style sequential
rate-clamp behavior, with the additional verified hardware endpoint clamp
retained for the real Piper robot.

The projection magnitude remains logged for diagnostics, but it will not
reject an otherwise finite response and there will be no consecutive-
projection trip. NaN, Inf, malformed input, invalid configuration, and model
timeouts continue to return no action.

## Invariants

- `joint_lower`, `joint_upper`, and `max_delta` do not change.
- Returned values always stay inside verified physical ranges.
- Every returned point differs from the preceding accepted point by at most
  its configured `max_delta`.
- No test or deployment action publishes a robot command.

## Verification

CPU tests will cover large finite arm and gripper overshoots, absolute-range
clamping, sequential rate clamping, non-finite rejection, and a server
response that continues after repeated large projections. The changed files
will be deployed to the inference server, preflighted on GPU, then checked
with a robot-side `--executor mock` request only.
