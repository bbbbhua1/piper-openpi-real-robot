# FastWAM Piper Training Start Pose Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the real Piper smoothly to the representative episode-242 training start pose, verify live feedback, and only then permit FastWAM inference.

**Architecture:** Store the traceable pose and safety parameters in a robot-side JSON file. A new pure-Python loader validates that file; the existing ROS executor performs interruptible smoothstep motion, while the client gates WebSocket connection on live feedback reaching the target. Default OpenPI home behavior remains unchanged.

**Tech Stack:** Python 3.9, JSON, NumPy-free validation helpers, ROS Noetic publishers/subscribers, `unittest`.

## Global Constraints

- Source pose is the real initial observation from episode 242 of `/bh/zbh_ckp/datasets/fastwam/agilex_cobotmagic2_puzzle_20260801_v21_puppet_joint`.
- Piper order is `[L1..L6,Lg,R1..R6,Rg]`.
- Arm feedback tolerance is `0.01 rad`; gripper feedback tolerance is `0.001 m`.
- Configured motion speed must stay below the verified `0.3 rad/s` hardware limit.
- Existing absolute limits, action delta limits, ROS fault handling, hold behavior, image transport, OpenPI initialization, training data, and checkpoints must not be weakened or changed.
- Deployment must not start the robot client or publish a physical command.

---

### Task 1: Traceable Start-Pose Configuration and Pure Validation

**Files:**
- Create: `configs/piper_fastwam_puzzle_start_pose.json`
- Create: `scripts/piper_start_pose.py`
- Create: `tests/test_piper_start_pose.py`

**Interfaces:**
- Produces: `StartPoseConfig`, `load_start_pose_config(path)`, `pose_errors(state, config)`, and `pose_is_reached(state, config)`.
- Consumes: JSON values in Piper left/right seven-value order.

- [ ] **Step 1: Write failing loader and feedback tests**

```python
def test_loads_episode_242_pose():
    config = load_start_pose_config(CONFIG_PATH)
    assert config.source_episode == 242
    assert config.left_arm == (
        0.037469711, 0.006087956, -0.003157364,
        0.040016536, 0.143180355, -0.027735960, 0.0005,
    )
    assert config.right_arm[-1] == 0.0008

def test_rejects_target_outside_verified_bounds(self):
    payload = json.loads(CONFIG_PATH.read_text())
    payload["left_arm"][1] = -0.01
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "invalid.json"
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "outside verified bounds"):
            load_start_pose_config(path)

def test_pose_reached_uses_separate_arm_and_gripper_tolerances():
    config = load_start_pose_config(CONFIG_PATH)
    state = {"left_arm": list(config.left_arm), "right_arm": list(config.right_arm)}
    state["left_arm"][0] += 0.009
    state["right_arm"][6] += 0.0009
    assert pose_is_reached(state, config)
    state["left_arm"][0] += 0.002
    assert not pose_is_reached(state, config)
```

- [ ] **Step 2: Run tests and verify they fail before implementation**

Run:

```bash
python -m unittest tests.test_piper_start_pose -v
```

Expected: import failure because `piper_start_pose.py` does not yet exist.

- [ ] **Step 3: Add the exact episode-242 JSON configuration**

```json
{
  "name": "fastwam_puzzle_episode_242_start",
  "source_dataset": "/bh/zbh_ckp/datasets/fastwam/agilex_cobotmagic2_puzzle_20260801_v21_puppet_joint",
  "source_episode": 242,
  "piper_wire_order": "[L1,L2,L3,L4,L5,L6,Lg,R1,R2,R3,R4,R5,R6,Rg]",
  "left_arm": [0.037469711, 0.006087956, -0.003157364, 0.040016536, 0.143180355, -0.027735960, 0.0005],
  "right_arm": [0.012245688, 0.004709880, -0.000837312, -0.010954832, 0.154728279, 0.0, 0.0008],
  "joint_lower": [-2.617993878, 0.0, -2.967059728, -1.745329252, -1.221730476, -3.141592654, 0.0, -2.617993878, 0.0, -2.967059728, -1.745329252, -1.221730476, -3.141592654, 0.0],
  "joint_upper": [2.617993878, 3.141592654, 0.0, 1.745329252, 1.221730476, 3.141592654, 0.08, 2.617993878, 3.141592654, 0.0, 1.745329252, 1.221730476, 3.141592654, 0.08],
  "move_hz": 20.0,
  "max_joint_speed": 0.15,
  "min_duration_s": 2.0,
  "max_duration_s": 20.0,
  "feedback_timeout_s": 5.0,
  "arm_tolerance": 0.01,
  "gripper_tolerance": 0.001
}
```

- [ ] **Step 4: Implement strict parsing and feedback comparison**

```python
@dataclass(frozen=True)
class StartPoseConfig:
    name: str
    source_dataset: str
    source_episode: int
    left_arm: tuple[float, ...]
    right_arm: tuple[float, ...]
    joint_lower: tuple[float, ...]
    joint_upper: tuple[float, ...]
    move_hz: float
    max_joint_speed: float
    min_duration_s: float
    max_duration_s: float
    feedback_timeout_s: float
    arm_tolerance: float
    gripper_tolerance: float

def pose_errors(state, config):
    left = finite_vector(state.get("left_arm"), "state.left_arm", 7)
    right = finite_vector(state.get("right_arm"), "state.right_arm", 7)
    arm_error = max(abs(left[i] - config.left_arm[i]) for i in range(6))
    arm_error = max(arm_error, *(abs(right[i] - config.right_arm[i]) for i in range(6)))
    gripper_error = max(abs(left[6] - config.left_arm[6]), abs(right[6] - config.right_arm[6]))
    return arm_error, gripper_error

def pose_is_reached(state, config):
    arm_error, gripper_error = pose_errors(state, config)
    return arm_error <= config.arm_tolerance and gripper_error <= config.gripper_tolerance
```

`load_start_pose_config` must reject non-object JSON, wrong vector sizes, booleans/non-finite numbers, `lower >= upper`, targets outside bounds, non-positive rates/tolerances/timeouts, `max_joint_speed >= 0.3`, and `max_duration_s < min_duration_s`.

- [ ] **Step 5: Run Task 1 tests and commit**

Run:

```bash
python -m unittest tests.test_piper_start_pose -v
```

Expected: all Task 1 tests pass.

Commit:

```bash
git add configs/piper_fastwam_puzzle_start_pose.json scripts/piper_start_pose.py tests/test_piper_start_pose.py
git commit -m "feat: add verified FastWAM Puzzle start pose"
```

---

### Task 2: Interruptible Move and Feedback-Gated Inference

**Files:**
- Modify: `scripts/websocket_policy_client.py`
- Modify: `tests/test_ros_joint_executor.py`
- Create: `tests/test_websocket_policy_client_start_pose.py`

**Interfaces:**
- Consumes: `StartPoseConfig`, `load_start_pose_config`, `pose_errors`, and `pose_is_reached` from Task 1.
- Produces: `RosJointExecutor.move_to_pose(...)` and `wait_until_start_pose(...)`.

- [ ] **Step 1: Write failing motion and gating tests**

```python
def test_move_to_pose_publishes_exact_target_and_checks_safety():
    executor = make_executor_without_ros()
    checks = []
    duration = executor.move_to_pose(
        current_state(), TARGET_LEFT, TARGET_RIGHT,
        move_hz=20.0, min_duration=0.1, max_duration=2.0,
        joint_speed=0.15, safety_check=lambda: checks.append(True),
        label="FastWAM Puzzle training start",
    )
    assert duration >= 0.1
    assert executor._last_published_target == (TARGET_LEFT, TARGET_RIGHT)
    assert checks

def test_wait_until_start_pose_times_out_without_policy_connection(self):
    source = FakeSource(always_state=far_from_target())
    with self.assertRaisesRegex(RuntimeError, "feedback did not reach"):
        wait_until_start_pose(source, CONFIG, sleep=lambda _: None, monotonic=fake_clock())

def test_start_pose_config_conflicts_with_no_home(self):
    args = build_arg_parser().parse_args(required_args() + [
        "--start-pose-config", "pose.json", "--no-home-on-start"
    ])
    with self.assertRaisesRegex(ValueError, "cannot be combined"):
        validate_initialization_args(args)
```

- [ ] **Step 2: Run the new tests and verify failure**

Run:

```bash
python -m unittest tests.test_ros_joint_executor tests.test_websocket_policy_client_start_pose -v
```

Expected: failures because the new motion and gating functions are absent.

- [ ] **Step 3: Generalize the existing smooth home movement**

Add this executor interface while preserving `move_home` as a wrapper:

```python
def move_to_pose(
    self, state, target_left, target_right, *, move_hz, min_duration,
    max_duration, joint_speed, safety_check, label,
):
    start_left, start_right = self._state_pair(state)
    target_left = _normalize_numeric_vector(target_left, "target.left_arm", self.left_arm_dim)
    target_right = _normalize_numeric_vector(target_right, "target.right_arm", self.right_arm_dim)
    max_delta = max(abs(a - b) for a, b in zip(start_left + start_right, target_left + target_right))
    duration = max(min_duration, min(max_duration, max_delta / joint_speed))
    steps = max(2, int(duration * move_hz))
    self.set_enable(True, label)
    for step in range(steps + 1):
        self._raise_if_shutdown()
        safety_check()
        alpha = _smoothstep(step / steps)
        left = [(1 - alpha) * a + alpha * b for a, b in zip(start_left, target_left)]
        right = [(1 - alpha) * a + alpha * b for a, b in zip(start_right, target_right)]
        self._publish_pair(left, right)
        time.sleep(1.0 / move_hz)
    return duration
```

`move_home` calls `move_to_pose` with two zero vectors, preserving existing CLI behavior and tests.

- [ ] **Step 4: Gate policy connection on live feedback**

```python
def wait_until_start_pose(source, config, *, sleep=time.sleep, monotonic=time.monotonic):
    deadline = monotonic() + config.feedback_timeout_s
    while monotonic() < deadline:
        raise_if_safety_tripped(source)
        state = get_current_state(source)
        arm_error, gripper_error = pose_errors(state, config)
        if pose_is_reached(state, config):
            print("[start-pose] reached source_episode={} arm_error={:.6f} gripper_error={:.6f}".format(
                config.source_episode, arm_error, gripper_error
            ))
            return state
        sleep(0.05)
    raise RuntimeError("start-pose feedback did not reach configured tolerances")
```

In `run_loop`, load and validate the config before creating the WebSocket. When `--execute-actions --start-pose-config PATH` is active, call `move_to_pose`, then `wait_until_start_pose`, and only then enter `websockets.connect`. A start-pose config without `--execute-actions` is validation-only and does not publish.

- [ ] **Step 5: Add and validate the CLI argument**

```python
parser.add_argument(
    "--start-pose-config",
    default=None,
    help="move to and verify a task-specific pose before physical policy execution",
)

def validate_initialization_args(args):
    if args.start_pose_config and not args.home_on_start:
        raise ValueError("--start-pose-config cannot be combined with --no-home-on-start")
```

When a start-pose config is present, it replaces the default zero-home target; no zero observation is sent to FastWAM.

- [ ] **Step 6: Run focused and regression tests, then commit**

Run:

```bash
python -m unittest \
  tests.test_piper_start_pose \
  tests.test_ros_joint_executor \
  tests.test_websocket_policy_client_start_pose \
  tests.test_fastwam_piper_runtime \
  tests.test_piper_fastwam_policy_server \
  tests.test_ws_policy_protocol -v
```

Expected: all available tests pass; OpenCV-only protocol tests may skip in the server environment.

Commit:

```bash
git add scripts/websocket_policy_client.py tests/test_ros_joint_executor.py tests/test_websocket_policy_client_start_pose.py
git commit -m "feat: initialize FastWAM from its training start pose"
```

---

### Task 3: Documentation, Deployment, and Non-Physical Verification

**Files:**
- Modify: `README.md`
- Deploy: `scripts/websocket_policy_client.py`, `scripts/piper_start_pose.py`, and `configs/piper_fastwam_puzzle_start_pose.json`

**Interfaces:**
- Consumes: the tested Task 1 and Task 2 client interface.
- Produces: the final supervised FastWAM robot command.

- [ ] **Step 1: Update the FastWAM command and initialization explanation**

Replace `--no-home-on-start` with:

```bash
--start-pose-config /home/agilex/piper-openpi-real-robot/configs/piper_fastwam_puzzle_start_pose.json \
```

Document that this automatically moves to episode 242, verifies feedback, and only then connects to FastWAM. Keep the operator-at-estop, clear-workspace, mock-first, and hold-on-exit requirements.

- [ ] **Step 2: Run syntax, diff, and complete regression checks**

Run:

```bash
python -m py_compile scripts/piper_start_pose.py scripts/websocket_policy_client.py
git diff --check
python -m unittest \
  tests.test_piper_start_pose \
  tests.test_ros_joint_executor \
  tests.test_websocket_policy_client_start_pose \
  tests.test_fastwam_piper_runtime \
  tests.test_piper_fastwam_policy_server \
  tests.test_ws_policy_protocol -v
```

Expected: syntax succeeds, no whitespace errors, and all available tests pass.

- [ ] **Step 3: Commit documentation**

```bash
git add README.md
git commit -m "docs: use the Puzzle training pose for FastWAM startup"
```

- [ ] **Step 4: Deploy files without launching the robot client**

Copy the project files to their exact destinations:

```text
/bh/zbh_self/projects/piper-openpi-real-robot/scripts/websocket_policy_client.py
/bh/zbh_self/projects/piper-openpi-real-robot/scripts/piper_start_pose.py
/bh/zbh_self/projects/piper-openpi-real-robot/configs/piper_fastwam_puzzle_start_pose.json
/home/agilex/piper-openpi-real-robot/scripts/websocket_policy_client.py
/home/agilex/piper-openpi-real-robot/scripts/piper_start_pose.py
/home/agilex/piper-openpi-real-robot/configs/piper_fastwam_puzzle_start_pose.json
```

Do not run `websocket_policy_client.py`, publish `/enable_flag`, or publish either `/master/joint_*` topic during deployment.

- [ ] **Step 5: Verify deployed hashes and config loading only**

Run a non-ROS Python config-load command on the robot and compare SHA-256 hashes of all three deployed files with the local versions. Expected output includes:

```text
START_POSE_CONFIG_OK episode=242
```

No physical command is published.

- [ ] **Step 6: Hand off the supervised initialization-only and full inference commands**

The first command must set `--max-steps 1` and omit `--execute-actions` when testing model output. The later physical command retains `--execute-actions`, `--enable-on-start true`, the FastWAM transport profile, 30 Hz control, 25 rolling chunks, three required cameras, and the new start-pose config.
