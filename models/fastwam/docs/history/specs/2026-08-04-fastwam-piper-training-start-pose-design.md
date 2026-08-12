# FastWAM Piper Training Start Pose Design

## Objective

Start the real Piper Puzzle/`crimp` rollout from a representative pose that
actually occurs at the beginning of the fine-tuning demonstrations. The client
must reach and verify that pose before sending the first observation to
FastWAM. Existing hardware joint limits, action delta limits, fault monitoring,
and final-position hold behavior remain enabled.

This change does not alter training data, model weights, camera composition,
action normalization, or the OpenPI initialization path.

## Source Pose Selection

The source dataset is read-only:

```text
/bh/zbh_ckp/datasets/fastwam/agilex_cobotmagic2_puzzle_20260801_v21_puppet_joint
```

All 300 episode-start `observation.state` vectors were compared after scaling
each dimension by its interquartile range, with a small floor for dimensions
whose IQR is zero. Episode 242 was initially selected as the dataset medoid,
but a live-scene inference from that pose predicted left joint 2 at
`-0.023954365 rad`, beyond its verified `0.0 rad` lower bound. Because the
live image changes the prediction, selecting solely by state-space centrality
is insufficient.

The replacement must remain a real recorded start state and provide more
left-joint-2 margin without becoming an extreme training outlier. Nine real
candidate starts were screened with the deployed checkpoint, their own three
recorded camera frames, the matching state, three stochastic inference runs,
and the 24-waypoint execution horizon. Episode 159 is selected: its left joint
2 starts at `0.015263500 rad`, and the three predicted minima were
`0.010517888`, `0.010517888`, and `0.003623437 rad`. None of the screened
episode-159 trajectories had a hard-limit violation. Episode 58 was rejected
despite its larger initial margin because it is a strong state-space outlier
and one repeat still predicted a negative value.

FastWAM state order is `[L1..L6,R1..R6,Lg,Rg]`. The robot-side Piper wire order
is `[L1..L6,Lg,R1..R6,Rg]`. The selected Piper targets are:

```text
left_arm:
[0.018856963, 0.015263500, -0.003558576,
 0.035690423, 0.091668218, -0.025258912, 0.000500000]

right_arm:
[0.012245688, 0.004709880, -0.000837312,
 -0.010954832, 0.154728279, 0.000000000, 0.000800000]
```

## Configuration and Interface

Add a robot-side JSON start-pose file under `configs/`. It records:

- dataset path and episode index;
- state ordering;
- seven finite target values for each arm;
- the verified Piper lower and upper bounds used to validate the target;
- interpolation frequency and maximum joint speed;
- arm and gripper feedback tolerances;
- movement and feedback timeouts.

Add `--start-pose-config PATH` to `websocket_policy_client.py`. Initialization
selection is deterministic:

1. With `--start-pose-config`, move to and verify the configured task pose.
2. Without it, preserve the existing home/default behavior.
3. Reject a command that combines `--start-pose-config` with
   `--no-home-on-start`, because those instructions conflict.

The FastWAM real-robot command will use `--start-pose-config` and will no longer
use `--no-home-on-start`. OpenPI commands remain unchanged.

## Motion and Verification

The client validates the JSON schema, vector dimensions, finite values, and
absolute hardware bounds before enabling or publishing. It then moves both
arms from the latest observed state to the configured target using the existing
smoothstep interpolation pattern. The configured maximum speed must remain
below the verified hardware speed limit. Movement is interruptible and checks
the existing ROS arm-fault monitor throughout.

After publishing the target, the client waits for live joint feedback. It may
connect to the policy server only when all six arm joints on each side are
within `0.01 rad` and both grippers are within `0.001 m` of the selected start.
The successful measured state, target, maximum error, and source episode are
printed for the experiment log.

## Failure Behavior

Malformed configuration, out-of-range targets, movement timeout, feedback
timeout, ROS shutdown, arm fault, or operator interruption prevents policy
inference. No request is sent to FastWAM in those cases. Cleanup retains the
existing fail-closed status handling; where the status monitor permits it, the
client holds the last published waypoint and leaves the robot enabled.

The absolute joint limits are not widened or disabled. Model predictions that
cross those limits continue to be rejected by the policy server.

## Tests and Deployment

Unit tests will cover:

- loading the validated episode-159 start-pose configuration;
- rejecting malformed, non-finite, or out-of-bounds targets;
- smooth interpolation ending at the exact configured target;
- successful and failed feedback verification;
- rejecting conflicting initialization arguments;
- preserving default home and no-home behavior when no start pose is supplied;
- interrupting initialization on a simulated safety trip.

After tests pass in the FastWAM development environment, deploy the generic
client change and the JSON start pose to the robot project at:

```text
/home/agilex/piper-openpi-real-robot
```

Deployment itself must not start the robot client or publish a physical
command. Offline screening is necessary but not sufficient because the live
camera scene differs from a recorded training frame. The first on-site use of
the replacement remains a supervised initialization-only check, followed by a
live-image inference check before full action execution.
