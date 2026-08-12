# FastWAM Bounded Action Projection Design

## Objective

Prevent finite FastWAM predictions just outside a verified Piper joint range
from aborting a real-robot request, without ever returning a command outside
the physical range or above the verified per-waypoint motion cap.

## Selected design

The server will project raw, finite FastWAM arm and gripper positions onto the
verified Piper range before applying the existing sequential `max_delta`
limiter.  Arm projections greater than `0.10 rad` and gripper projections
greater than `0.005 m` still fail closed.  The `0.10 rad` arm ceiling covers
the observed `+0.089842878 rad` J3 overshoot but remains a diagnostic guard,
not a widened hardware range.

The server records the count and greatest projection in each response.  A
projection of at least `0.05 rad` is a large projection; after three
consecutive responses containing one, it returns no action and requires a
fresh observation/restart.  Small boundary projections do not trip this
counter.  The runtime seed is fixed at `42` so the same observation does not
receive a new diffusion-noise draw on every request.

## Invariants

- Piper `joint_lower`, `joint_upper`, and `max_delta` remain unchanged.
- Returned arm and gripper values are always within the verified hardware
  range, and every waypoint remains within `max_delta` of the preceding
  accepted waypoint.
- NaN/Inf values, malformed requests, and projections larger than the stated
  ceilings return no action.
- The server remains ROS-free and cannot publish a robot command itself.

## Verification

CPU tests cover boundary projection, hard rejection, rate limiting, and the
consecutive-large-projection trip.  Deployment then runs the existing
preflight and a robot-side mock request only; no physical action is published
during verification.
