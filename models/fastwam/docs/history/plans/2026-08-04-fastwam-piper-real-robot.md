# FastWAM Piper Real-Robot Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe FastWAM-backed WebSocket policy server that runs the completed Puzzle/`crimp` checkpoint on the Piper robot while preserving the existing robot client and OpenPI backend.

**Architecture:** Keep the existing JSON protocol, SSH tunnel, ROS client, camera topics, and joint publishers. Add a FastWAM runtime adapter that converts the Piper 7+7 wire layout into the model's 14D layout, composes the three camera views into the trained 384x320 input, normalizes state, performs FastWAM inference, validates the action, and converts it back to the unchanged Piper response schema. The real server is a small WebSocket wrapper around this adapter; all conversions and safety checks are independently unit-tested without CUDA or ROS.

**Tech Stack:** Python 3.12, NumPy, Pillow, PyTorch, Hydra/OmegaConf, FastWAM, `websockets`, ROS 1 client, `unittest`.

## Global Constraints

- Deploy on `ssh -p 3763 root@10.40.1.215`; use `/bh/zbh_self/projects/fastwam` and `/bh/zbh_self/envs/fastwam`.
- Never modify `/bh/media/unify` or `/bh/media/unify-fileset`.
- Keep `scripts/piper_openpi_policy_server.py` unchanged; FastWAM is a sibling server.
- Do not change the existing WebSocket request/response envelope or robot-side camera names (`head`, `left_wrist`, `right_wrist`).
- Use only the final Puzzle checkpoint `.../step_001695.pt` and matching `normalization/dataset_stats.json`.
- FastWAM action/state layout is `[L1..L6,R1..R6,Lg,Rg]`; Piper wire layout is `[L1..L6,Lg,R1..R6,Rg]`.
- Model inference is fixed at `action_horizon=32`, `num_video_frames=9`, and prompt task label `crimp`.
- Hardware motion remains disabled until the code tests, actual-GPU replay, image/state validation, and live safety checks pass.

---

## File Structure

- Create `scripts/fastwam_piper_runtime.py`: pure image, layout, normalization, safety, and FastWAM model-runtime functions; no ROS or WebSocket server loop.
- Create `scripts/piper_fastwam_policy_server.py`: CLI and JSON WebSocket service that uses `FastWAMPiperRuntime`.
- Create `configs/piper_fastwam_puzzle.yaml`: non-secret deployment defaults and explicit safety configuration.
- Create `tests/test_fastwam_piper_runtime.py`: CPU-only unit tests for all pure transformations and limits.
- Create `tests/test_piper_fastwam_policy_server.py`: mocked async request/response tests.
- Modify `README.md`: FastWAM deployment, compatibility rules, and guarded real-robot launch sequence.

### Task 1: Build and test the protocol-independent FastWAM/Piper adapter

**Files:**
- Create: `scripts/fastwam_piper_runtime.py`
- Test: `tests/test_fastwam_piper_runtime.py`

**Interfaces:**
- Consumes: decoded RGB `np.ndarray` camera images and the current Piper state mapping `{left_arm: list[float], right_arm: list[float]}`.
- Produces: `piper_wire_to_fastwam`, `fastwam_to_piper_wire`, `compose_fastwam_image`, `validate_and_limit_fastwam_actions`, and `FastWAMPiperRuntime.infer`.
- Later consumers: `piper_fastwam_policy_server.py` calls `FastWAMPiperRuntime.infer(request)` and receives an action chunk in Piper wire order.

- [ ] **Step 1: Write the failing CPU-only transformation tests**

Create `tests/test_fastwam_piper_runtime.py` with this coverage:

```python
def test_piper_wire_to_fastwam_reorders_grippers():
    state = {"left_arm": [1, 2, 3, 4, 5, 6, 7], "right_arm": [8, 9, 10, 11, 12, 13, 14]}
    assert runtime.piper_wire_to_fastwam(state).tolist() == [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 7, 14]

def test_fastwam_to_piper_wire_is_exact_inverse():
    action = np.asarray([[1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 7, 14]], dtype=np.float32)
    result = runtime.fastwam_to_piper_wire(action, dt=1 / 30)
    assert result["left_arm"] == [[1, 2, 3, 4, 5, 6, 7]]
    assert result["right_arm"] == [[8, 9, 10, 11, 12, 13, 14]]

def test_compose_fastwam_image_has_exact_robotwin_geometry():
    image = runtime.compose_fastwam_image(head, left, right)
    assert image.shape == (3, 384, 320)
    assert image.dtype == np.float32
    assert -1.0 <= image.min() <= image.max() <= 1.0

def test_nonfinite_or_wrong_dimension_state_is_rejected():
    with pytest.raises(ValueError, match="seven finite"):
        runtime.piper_wire_to_fastwam({"left_arm": [0] * 6, "right_arm": [0] * 7})

def test_joint_limits_and_per_step_delta_fail_closed():
    limited = runtime.validate_and_limit_fastwam_actions(prediction, current_state, limits)
    assert np.all(limited[:, :12] <= current_state[:12] + limits.max_joint_delta[:12])
    with pytest.raises(ValueError, match="absolute joint limit"):
        runtime.validate_and_limit_fastwam_actions(outside_limits, current_state, limits)
```

Use `unittest` assertions if `pytest` is unavailable; do not make this test suite import FastWAM, CUDA, ROS, or WebSocket libraries.

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```bash
python -m unittest tests.test_fastwam_piper_runtime -v
```

Expected: FAIL because `fastwam_piper_runtime` does not exist.

- [ ] **Step 3: Implement the pure adapter**

Implement these exact contracts in `scripts/fastwam_piper_runtime.py`:

```python
@dataclass(frozen=True)
class PiperSafetyLimits:
    lower: np.ndarray          # [14] in Piper wire order
    upper: np.ndarray          # [14] in Piper wire order
    max_delta: np.ndarray      # [14], all strictly positive

def piper_wire_to_fastwam(state: Mapping[str, Any]) -> np.ndarray:
    """Validate two 7D Piper vectors and return FastWAM-order float32 [14]."""

def fastwam_to_piper_wire(actions: np.ndarray, *, dt: float) -> dict[str, Any]:
    """Map [T,14] FastWAM outputs to the current JSON left_arm/right_arm schema."""

def compose_fastwam_image(head: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return float32 CHW [3,384,320] in [-1,1] using the RobotWin tile layout."""

def validate_and_limit_fastwam_actions(
    actions_fastwam: np.ndarray,
    current_fastwam: np.ndarray,
    limits_piper: PiperSafetyLimits,
) -> np.ndarray:
    """Convert to Piper order, reject absolute-limit violations, clamp delta, then return FastWAM order."""
```

Implement image resizing with `PIL.Image.Resampling.BILINEAR`; explicitly require HWC RGB/BGR images with three channels and make BGR-to-RGB conversion a caller-selected operation. Validate every scalar with `np.isfinite`. For action limits, convert both prediction and current state into Piper order, reject any predicted position outside `[lower, upper]`, clamp each timestep relative to the previously accepted Piper waypoint using `max_delta`, and convert the accepted trajectory back into FastWAM order.

- [ ] **Step 4: Run the adapter tests**

Run:

```bash
python -m unittest tests.test_fastwam_piper_runtime -v
```

Expected: PASS; output includes layout inversion, image shape, finite-value, and safety-limit tests.

- [ ] **Step 5: Commit the tested adapter**

```bash
git add scripts/fastwam_piper_runtime.py tests/test_fastwam_piper_runtime.py
git commit -m "feat: add FastWAM Piper runtime adapter"
```

### Task 2: Add FastWAM model loading and inference normalization

**Files:**
- Modify: `scripts/fastwam_piper_runtime.py`
- Modify: `tests/test_fastwam_piper_runtime.py`

**Interfaces:**
- Consumes: FastWAM project root, `sim_robotwin.yaml`, the trained task override, `.pt` checkpoint, and `dataset_stats.json`.
- Produces: `FastWAMPiperRuntime.from_config(config)` and `FastWAMPiperRuntime.infer(head, left, right, state)` returning an unnormalized `[32,14]` FastWAM action trajectory.
- Later consumers: WebSocket server creates exactly one runtime before listening.

- [ ] **Step 1: Add failing dependency-injected model tests**

Add tests using a fake model and fake processor rather than CUDA:

```python
def test_runtime_uses_training_prompt_and_fixed_temporal_shapes():
    runtime = FastWAMPiperRuntime(model=FakeModel(), processor=FakeProcessor(), config=fake_config)
    actions = runtime.infer(head=head, left=left, right=right, state=state)
    assert runtime.model.kwargs["prompt"] == "A video recorded from a robot's point of view executing the following instruction: crimp"
    assert runtime.model.kwargs["action_horizon"] == 32
    assert runtime.model.kwargs["num_video_frames"] == 9
    assert actions.shape == (32, 14)

def test_runtime_denormalizes_before_returning_actions():
    runtime = FastWAMPiperRuntime(model=FakeModel(), processor=ScalingProcessor(), config=fake_config)
    assert np.allclose(runtime.infer(head, left, right, state)[0], np.arange(14) + 10)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m unittest tests.test_fastwam_piper_runtime -v
```

Expected: FAIL because `FastWAMPiperRuntime` has no dependency-injected inference implementation.

- [ ] **Step 3: Implement runtime loading and model invocation**

Implement `FastWAMPiperRuntime` with two constructors:

```python
class FastWAMPiperRuntime:
    def __init__(self, *, model: Any, processor: Any, config: Mapping[str, Any]) -> None: ...

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "FastWAMPiperRuntime": ...

    def infer(
        self, *, head: np.ndarray, left: np.ndarray, right: np.ndarray, state: Mapping[str, Any]
    ) -> np.ndarray: ...
```

`from_config` must prepend `config["fastwam_root"]` and `config["fastwam_root"] + "/src"` to `sys.path`, compose `config["fastwam_root"] + "/configs/sim_robotwin.yaml"` with the `agilex_cobotmagic2_puzzle_joint_3cam_384_1e-4_baidu32` task override, set `model.load_text_encoder=True`, and set `skip_dit_load_from_pretrain=True` before instantiation. It must load `checkpoint_path`, instantiate `FastWAMProcessor`, load `dataset_stats_path` with `load_dataset_stats_from_json`, call `processor.set_normalizer_from_stats`, and put the model in evaluation mode.

`infer` must normalize state with the processor's state normalizer, call `model.infer_action`, then denormalize the returned action with the processor action normalizer. It must pass the exact prompt, the composed image tensor, `action_horizon=32`, `num_video_frames=9`, configured diffusion steps, and configured seed/CFG parameters. Reject any result not shaped `[32,14]` and any non-finite output.

- [ ] **Step 4: Run CPU unit tests and a real model load preflight on `3763`**

Run unit tests locally:

```bash
python -m unittest tests.test_fastwam_piper_runtime -v
```

Then, after deployment to `3763`, run the server's `--preflight-only` mode:

```bash
source /bh/zbh_self/miniforge3/etc/profile.d/conda.sh
conda activate /bh/zbh_self/envs/fastwam
CUDA_VISIBLE_DEVICES=0 python -u scripts/piper_fastwam_policy_server.py \
  --config configs/piper_fastwam_puzzle.yaml --preflight-only
```

Expected: model, checkpoint, and dataset statistics load successfully; output reports one A800 device, 32 action steps, 9 video frames, and no server port is opened.

- [ ] **Step 5: Commit model runtime support**

```bash
git add scripts/fastwam_piper_runtime.py tests/test_fastwam_piper_runtime.py
git commit -m "feat: load Puzzle FastWAM policy for Piper inference"
```

### Task 3: Implement the FastWAM WebSocket server and deployment configuration

**Files:**
- Create: `scripts/piper_fastwam_policy_server.py`
- Create: `configs/piper_fastwam_puzzle.yaml`
- Create: `tests/test_piper_fastwam_policy_server.py`

**Interfaces:**
- Consumes: existing JSON request envelope, `FastWAMPiperRuntime.infer`, and `PiperSafetyLimits`.
- Produces: existing response envelope `{"ok", "request_id", "step", "action", "error"}` and a CLI that binds to `127.0.0.1:7081`.
- Later consumers: the unchanged robot-side `websocket_policy_client.py` calls this server through the existing SSH tunnel.

- [ ] **Step 1: Write failing server tests with a fake runtime**

Create `tests/test_piper_fastwam_policy_server.py` with an async fake WebSocket request test:

```python
async def test_valid_request_returns_existing_piper_action_schema():
    server = PiperFastWAMServer(runtime=FakeRuntime(np.zeros((32, 14), dtype=np.float32)), config=fake_config)
    response = await server.handle_message(json.dumps(valid_request))
    assert response["ok"] is True
    assert response["request_id"] == valid_request["request_id"]
    assert len(response["action"]["left_arm"]) == fake_config["execution_horizon"]
    assert len(response["action"]["left_arm"][0]) == 7

async def test_model_or_safety_error_returns_no_action():
    server = PiperFastWAMServer(runtime=FailingRuntime(), config=fake_config)
    response = await server.handle_message(json.dumps(valid_request))
    assert response == {"ok": False, "request_id": valid_request["request_id"], "step": 3, "action": {}, "error": unittest.mock.ANY}
```

Test JPEG base64 decoding, the three required image names, configured RGB/BGR handling, the independent `execution_horizon <= 32` constraint, response action dimensions, and latency rejection.

- [ ] **Step 2: Run server tests to verify they fail**

Run:

```bash
python -m unittest tests.test_piper_fastwam_policy_server -v
```

Expected: FAIL because `piper_fastwam_policy_server` does not exist.

- [ ] **Step 3: Implement the server and checked-in configuration**

Implement `PiperFastWAMServer.handle_message` using the current OpenPI server's defensive JSON/base64 decoding pattern, but call the FastWAM runtime. Its execution order must be:

```python
request = json.loads(message)
head, left, right = decode_required_rgb_images(request["images"])
state_fastwam = piper_wire_to_fastwam(request["state"])
prediction = runtime.infer(head=head, left=left, right=right, state=request["state"])
safe_prediction = validate_and_limit_fastwam_actions(prediction, state_fastwam, safety_limits)
response_action = fastwam_to_piper_wire(safe_prediction[:execution_horizon], dt=action_dt)
return build_policy_response(request_id, step, response_action)
```

Catch all per-request errors, log a traceback, and return `ok: false` with `action: {}`. Measure monotonic request latency; reject after `max_inference_latency_s` before producing an action. Use `websockets.serve` with the configured max message size, ping interval, and ping timeout.

Create `configs/piper_fastwam_puzzle.yaml` with these fixed model fields:

```yaml
fastwam_root: /bh/zbh_self/projects/fastwam
sim_config_name: sim_robotwin.yaml
task_name: agilex_cobotmagic2_puzzle_joint_3cam_384_1e-4_baidu32
checkpoint_path: /bh/zbh_ckp/runs/fastwam/agilex_cobotmagic2_puzzle_joint_3cam_384_1e-4_baidu32/CogWAM_fastWAM-copy5_retry4/checkpoints/weights/step_001695.pt
dataset_stats_path: /bh/zbh_ckp/datasets/fastwam/agilex_cobotmagic2_puzzle_20260801_v21_puppet_joint/normalization/dataset_stats.json
task_label: crimp
model_action_horizon: 32
num_video_frames: 9
num_inference_steps: 10
device: cuda:0
host: 127.0.0.1
port: 7081
execution_horizon: 1
action_dt: 0.0333333333
```

Keep `joint_lower`, `joint_upper`, and `max_delta` as required 14-value fields in the same file, but do not invent values. The CLI must refuse to start for physical inference until all three vectors are populated with finite, ordered values. Mock and `--preflight-only` modes may use explicitly supplied test limits.

- [ ] **Step 4: Run mocked end-to-end server tests**

Run:

```bash
python -m unittest tests.test_piper_fastwam_policy_server -v
python -m unittest discover -s tests -v
```

Expected: PASS; the existing camera-recorder tests continue to pass and both successful and failed server responses preserve the established protocol.

- [ ] **Step 5: Commit server and configuration**

```bash
git add scripts/piper_fastwam_policy_server.py configs/piper_fastwam_puzzle.yaml tests/test_piper_fastwam_policy_server.py
git commit -m "feat: add FastWAM WebSocket policy server for Piper"
```

### Task 4: Document, deploy, and validate the server without robot motion

**Files:**
- Modify: `README.md`
- Modify: `docs/agilex_robot_ros_startup.md`
- Create: `scripts/run_piper_fastwam_server.sh`

**Interfaces:**
- Consumes: the checked-in deployment config and completed FastWAM server.
- Produces: repeatable A800 launch, tunnel, client dry-run, and image/action validation commands.
- Later consumers: an operator uses the documented staged commands for the full task attempt.

- [ ] **Step 1: Add documentation tests/command checks**

Add a shell syntax check for the launch script and a documentation assertion that the FastWAM launch references `piper_fastwam_policy_server.py`, `step_001695.pt`, `dataset_stats.json`, `--execute-actions`, and the explicit stop procedure.

```bash
bash -n scripts/run_piper_fastwam_server.sh
grep -q 'piper_fastwam_policy_server.py' README.md
grep -q 'step_001695.pt' README.md
```

- [ ] **Step 2: Run checks to verify they fail**

Run:

```bash
bash -n scripts/run_piper_fastwam_server.sh
```

Expected: FAIL because the launch script does not exist.

- [ ] **Step 3: Add launch script and staged operating procedure**

`scripts/run_piper_fastwam_server.sh` must:

```bash
#!/usr/bin/env bash
set -euo pipefail
source /bh/zbh_self/miniforge3/etc/profile.d/conda.sh
conda activate /bh/zbh_self/envs/fastwam
cd "$(dirname "$0")/.."
exec python -u scripts/piper_fastwam_policy_server.py \
  --config configs/piper_fastwam_puzzle.yaml "$@"
```

Update `README.md` with three separate commands: server preflight, tunnel, and robot dry-run without `--execute-actions`. Document that only after the operator verifies composed images, seven-value state order, valid `crimp` prompt, action range, and latency may they add `--execute-actions`. Update the ROS startup document with the required actual joint-limit source: extract lower/upper values from the specific Piper robot's active URDF/driver configuration and enter them in `configs/piper_fastwam_puzzle.yaml`; do not use generic internet values.

- [ ] **Step 4: Deploy to `3763` and run server-only preflight**

Copy the completed repository to `/bh/zbh_self/projects/piper-openpi-real-robot`; do not alter `/bh/zbh_self/projects/fastwam` or its existing dirty files. From the development Mac, use:

```bash
scp -P 3763 -r \
  /Users/hua/Documents/Codex/2026-08-03/ssh-p-3763-root-10-40/work/piper-openpi-real-robot \
  root@10.40.1.215:/bh/zbh_self/projects/
```

Then on `3763`, run:

```bash
ssh -p 3763 root@10.40.1.215
source /bh/zbh_self/miniforge3/etc/profile.d/conda.sh
conda activate /bh/zbh_self/envs/fastwam
cd /bh/zbh_self/projects/piper-openpi-real-robot
CUDA_VISIBLE_DEVICES=0 bash scripts/run_piper_fastwam_server.sh --preflight-only
```

Expected: model/checkpoint/stats load; logs report A800, 32 model actions, 9 video frames, and no open listening socket.

- [ ] **Step 5: Commit documentation and launch support**

```bash
git add README.md docs/agilex_robot_ros_startup.md scripts/run_piper_fastwam_server.sh
git commit -m "docs: add guarded FastWAM Piper deployment workflow"
```

### Task 5: Perform staged live validation and the complete `crimp` attempt

**Files:**
- Modify: `configs/piper_fastwam_puzzle.yaml` only after obtaining the active robot-specific limits.
- Output: a timestamped directory created under `/bh/zbh_ckp/runs/fastwam/piper_crimp_real_robot/` for server diagnostics; robot video recorder output remains in its documented location.

**Interfaces:**
- Consumes: live robot ROS topics, the running FastWAM server, tunnel, and an emergency-stop-ready operator.
- Produces: captured request images/state/action diagnostics and a bounded real execution result.

- [ ] **Step 1: Record live state layout and active joint limits without action publishing**

On the robot, read one `/puppet/joint_left` and `/puppet/joint_right` message and the active driver/URDF limit configuration. Confirm each message has seven entries ordered as six arm joints plus gripper. Enter only those verified 14 lower, upper, and maximum-delta values in the deployment config; retain the before/after config diff in the session log.

- [ ] **Step 2: Run the real-server dry run without `--execute-actions`**

Start the server with `--save-received-images` and start the unchanged client without `--execute-actions`. Verify server logs show three decoded RGB cameras, image tensor `[3,384,320]`, FastWAM state `[14]`, 32 predicted actions, response segment `[1,14]`, finite values, and action limits accepted.

- [ ] **Step 3: Run one constrained waypoint**

With the puzzle scene prepared and an operator at the physical emergency stop, set `execution_horizon: 1`, enable `--execute-actions`, and make exactly one request. Verify actual joint positions follow the expected direction, no status fault occurs, and the server/client logs have no rejected action or latency error. Stop and investigate if any expected condition is false.

- [ ] **Step 4: Run a short action chunk**

Increase `execution_horizon` only to the value that the latency measurement supports, beginning at 2. Execute one request and verify the robot remains within workspace and velocity expectations. Preserve the diagnostic images and three-camera recording.

- [ ] **Step 5: Execute and record the full `crimp` task**

After successful short-chunk validation, run the complete task with the task scene reset. Keep the operator at emergency stop, record all three cameras, and preserve server diagnostics in the timestamped run directory under `/bh/zbh_ckp/runs/fastwam/piper_crimp_real_robot/`. Stop immediately on a server safety rejection, ROS fault, state-layout mismatch, limit violation, inference timeout, or unexpected physical trajectory.
