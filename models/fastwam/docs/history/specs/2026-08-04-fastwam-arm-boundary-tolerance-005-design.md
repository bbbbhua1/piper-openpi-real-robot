# FastWAM Arm Boundary Tolerance 0.05 Design

## Objective

Allow the FastWAM Puzzle policy server to recover from arm predictions that
overshoot a verified physical joint endpoint by at most `0.05 rad`, while
continuing to prohibit every command outside the robot's verified physical
joint range.

This change addresses live-scene predictions such as left joint 2 at
`-0.023954365 rad` when that joint's verified lower bound is `0.0 rad`.

## Selected Approach

Change only the server-side deployment value:

```yaml
arm_boundary_tolerance: 0.05
```

The existing validation algorithm remains unchanged:

1. Convert FastWAM actions into Piper wire order.
2. Reject the entire response if any arm prediction exceeds a verified lower
   or upper bound by more than `0.05 rad`.
3. Clamp every arm prediction that is outside a verified bound by at most
   `0.05 rad` back to that exact verified bound.
4. Apply the existing waypoint-by-waypoint `max_delta: 0.01 rad` limiter to the
   clamped trajectory before returning it to the robot client.
5. Log a warning whenever an arm prediction is clamped.

For example, a raw left-joint-2 prediction of `-0.023954365 rad` is recoverable
and becomes `0.0 rad`. A raw prediction below `-0.05 rad` remains a hard error
and produces no action response.

## Invariants

The following values and behaviors must not change:

- `joint_lower` and `joint_upper` remain the verified physical Piper limits.
- No value outside those limits may appear in the action returned to the robot.
- Arm `max_delta` remains `0.01 rad` per 30 Hz waypoint.
- `gripper_boundary_tolerance` remains `0.005 m`.
- Gripper bounds, checkpoint, normalization statistics, execution horizon,
  image processing, start-pose logic, ROS topics, hold behavior, and fault
  handling remain unchanged.
- A model arm overshoot greater than `0.05 rad` rejects the full response.

The operator's ability to stop the robot is additional supervision and is not
used as a substitute for any software or hardware safety boundary.

## Alternatives Rejected

### Increase the tolerance only to `0.03 rad`

This would cover the observed `-0.023954365 rad` prediction with less recovery
range, but it does not implement the explicitly selected `0.05 rad` threshold.

### Expand `joint_lower` or `joint_upper`

This would misrepresent verified physical limits and could permit an actual
out-of-range command. It is not allowed.

### Disable absolute validation

This removes the fail-closed boundary and is not allowed.

## Tests

The existing safety tests must continue to pass. Add explicit boundary tests
in Piper wire order that demonstrate:

- a `0.049 rad` arm overshoot is clamped to the physical endpoint;
- a `0.050 rad` arm overshoot is accepted at the inclusive threshold;
- a `0.051 rad` arm overshoot raises `ValueError` and returns no action;
- clamped actions still obey the existing `0.01 rad` per-waypoint delta limit;
- gripper recovery behavior remains unchanged at `0.005 m`.

## Deployment and Rollback

Update the checked-in config and its documentation/tests, run the complete
local and development-machine test suites, then deploy the config to:

```text
/bh/zbh_self/projects/piper-openpi-real-robot/configs/piper_fastwam_puzzle.yaml
```

The currently running FastWAM server has already loaded the old value and must
be stopped and restarted by the operator after deployment. File deployment
must not start a server or publish any robot command.

Rollback consists of restoring:

```yaml
arm_boundary_tolerance: 0.01
```

and restarting the FastWAM server. No robot-side file changes are required.
