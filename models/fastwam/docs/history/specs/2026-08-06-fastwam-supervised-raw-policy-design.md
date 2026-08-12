# FastWAM supervised raw-action policy server

## Purpose

Provide an opt-in FastWAM server for a continuously supervised real-robot
experiment. The current `piper_fastwam_policy_server.py` remains the default,
safe deployment. The new variant allows the user to determine whether the
existing software action projection and sequential waypoint delta clamp are
responsible for poor execution.

## Scope

The new `piper_fastwam_policy_server_supervised.py` will use the same
WebSocket protocol, camera decoding, FastWAM prompt, image preprocessing,
checkpoint loading, action horizon, and Piper/FastWAM action-column mapping as
the current FastWAM server.

For a finite `[T, 14]` model output, it will publish the configured execution
horizon without:

- clipping predicted joints to the configured `joint_lower`/`joint_upper`;
- rejecting output based on a projection magnitude; or
- applying a delta limit between successive predicted waypoints.

The existing robot client already forwards accepted action waypoints directly
to the ROS command topics. It needs no action-execution change for this
experiment.

## Boundaries retained

This is not a bypass of the robot driver's or firmware's protection. The
following remain unchanged:

- model-output shape and finite-number validation;
- request schema and three-camera validation;
- ROS/CAN driver and firmware hard limits;
- reported hardware fault handling;
- the existing `Ctrl-C` hold/exit behavior, maximum client step count, and
  recording/upload behavior; and
- the existing optional controlled start-pose motion.

The server will log `SUPERVISED RAW ACTION MODE` at startup and before a
returned action so accidental use is visible in the terminal log.

## Configuration and invocation

A separate `configs/piper_fastwam_puzzle_supervised_raw.yaml` will carry the
same model and transport settings as the standard configuration, but will not
require software joint or per-waypoint action-limit fields. The standard
configuration is unchanged. The launcher will accept the alternate config,
making the mode change explicit in Terminal 1 only.

## Verification

Unit tests will cover that a finite FastWAM trajectory is mapped to Piper
order unchanged, and that malformed/non-finite output is still rejected.
Existing standard-server tests will continue to cover the protected server.
Before deployment, both server files will be syntax checked and the test suite
will be run. The new server and its config will then be copied to the selected
development machine; the robot client remains unchanged.
