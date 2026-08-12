# FastWAM Supervised Raw-Action Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in FastWAM WebSocket server that forwards finite model waypoints in their original order and magnitude, without the standard server's software action clamping or waypoint delta limiting.

**Architecture:** Keep the existing protected `piper_fastwam_policy_server.py` untouched. A new `piper_fastwam_policy_server_supervised.py` reuses its request/image/protocol helpers and `FastWAMPiperRuntime`, but truncates only to `execution_horizon` and immediately maps those finite actions to Piper wire order. The existing robot client remains the sole ROS command publisher and is unchanged.

**Tech Stack:** Python 3.10, NumPy, Pillow, PyYAML, WebSockets, FastWAM runtime, Python `unittest`, ROS Noetic client.

## Global Constraints

- The protected server, its config, and the deployed robot client must remain behaviorally unchanged.
- Raw mode accepts only model output with exact shape `(32, 14)` and finite values; malformed output produces the existing JSON error envelope with no action.
- Raw mode must not require, read, project to, or sequentially clamp by `joint_lower`, `joint_upper`, or `max_delta`.
- Raw mode preserves the existing JSON WebSocket protocol, 24-waypoint rolling execution horizon, action `dt`, image decoding, state mapping, checkpoint, recording, `Ctrl-C` behavior, ROS/CAN driver protections, and firmware protections.
- Terminal logging must visibly state `SUPERVISED RAW ACTION MODE` at server startup and when returning a raw action.
- Deploy only to `/bh/zbh_self/projects/piper-openpi-real-robot` on `ssh -p 7156 root@10.40.1.215`; do not modify `/bh/media`.

---

### Task 1: Add regression tests for raw forwarding

**Files:**
- Create: `tests/test_piper_fastwam_policy_server_supervised.py`

**Interfaces:**
- Consumes: `SupervisedRawPiperFastWAMServer(runtime: Any, config: Mapping[str, Any])` from `scripts/piper_fastwam_policy_server_supervised.py`.
- Produces: CPU-only behavioral tests for raw action mapping and rejection of malformed model outputs.

- [ ] **Step 1: Write the failing tests**

```python
import json
import unittest
from pathlib import Path

import numpy as np

from test_piper_fastwam_policy_server import FakeRuntime, make_request
import piper_fastwam_policy_server_supervised as raw_module


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_raw_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "input_color": "bgr",
        "execution_horizon": 2,
        "action_dt": 1 / 30,
    }
    config.update(overrides)
    return config


class SupervisedRawPiperFastWAMServerTest(unittest.IsolatedAsyncioTestCase):
    async def test_finite_out_of_range_action_is_forwarded_without_clamp_or_delta_limit(self) -> None:
        prediction = np.zeros((32, 14), dtype=np.float32)
        prediction[0, 1] = -0.127371117
        prediction[0, 2] = 0.089842878
        prediction[1, 1] = 2.5
        server = raw_module.SupervisedRawPiperFastWAMServer(
            runtime=FakeRuntime(prediction), config=make_raw_config()
        )

        response = await server.handle_message(json.dumps(make_request()))

        self.assertTrue(response["ok"])
        self.assertAlmostEqual(response["action"]["left_arm"][0][1], -0.127371117)
        self.assertAlmostEqual(response["action"]["left_arm"][0][2], 0.089842878)
        self.assertAlmostEqual(response["action"]["left_arm"][1][1], 2.5)

    async def test_non_finite_prediction_returns_existing_error_envelope(self) -> None:
        prediction = np.zeros((32, 14), dtype=np.float32)
        prediction[0, 0] = np.nan
        server = raw_module.SupervisedRawPiperFastWAMServer(
            runtime=FakeRuntime(prediction), config=make_raw_config()
        )

        response = await server.handle_message(json.dumps(make_request()))

        self.assertFalse(response["ok"])
        self.assertEqual(response["action"], {})
        self.assertIn("non-finite", response["error"])

    def test_config_does_not_require_standard_safety_vectors(self) -> None:
        raw_module.SupervisedRawPiperFastWAMServer(
            runtime=FakeRuntime(np.zeros((32, 14), dtype=np.float32)),
            config=make_raw_config(),
        )

```

- [ ] **Step 2: Run the new test module and verify it fails because the raw server module does not exist**

Run:

```bash
python -m unittest tests.test_piper_fastwam_policy_server_supervised -v
```

Expected: import failure for `piper_fastwam_policy_server_supervised`.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_piper_fastwam_policy_server_supervised.py
git commit -m "test: define supervised raw FastWAM behavior"
```

### Task 2: Implement the isolated raw-action WebSocket server

**Files:**
- Create: `scripts/piper_fastwam_policy_server_supervised.py`
- Test: `tests/test_piper_fastwam_policy_server_supervised.py`

**Interfaces:**
- Consumes: `decode_required_rgb_images`, `build_policy_response`, `load_config`, `FastWAMPiperRuntime`, `fastwam_to_piper_wire`, and `piper_wire_to_fastwam` from the existing protected server/runtime modules.
- Produces: `SupervisedRawPiperFastWAMServer.handle_message(message: str) -> dict[str, Any]` and CLI `python -u scripts/piper_fastwam_policy_server_supervised.py --config <yaml>`.

- [ ] **Step 1: Implement the server class so it uses only finite-output validation and wire-order mapping**

Create `scripts/piper_fastwam_policy_server_supervised.py` with this action-processing core. Do not import `PiperSafetyLimits` or `project_and_limit_fastwam_actions`.

```python
class SupervisedRawPiperFastWAMServer:
    def __init__(self, *, runtime: Any, config: Mapping[str, Any]) -> None:
        self.runtime = runtime
        self.config = dict(config)
        self.input_color = str(self.config.get("input_color", "bgr")).lower()
        if self.input_color not in {"rgb", "bgr"}:
            raise ValueError("input_color must be 'rgb' or 'bgr'")
        self.execution_horizon = _positive_int(self.config, "execution_horizon")
        if self.execution_horizon > FASTWAM_ACTION_HORIZON:
            raise ValueError("execution_horizon must be <= 32")
        self.action_dt = float(self.config.get("action_dt"))
        if not np.isfinite(self.action_dt) or self.action_dt <= 0.0:
            raise ValueError("action_dt must be a finite positive number")

    async def handle_message(self, message: str) -> dict[str, Any]:
        request_id, step, started = "unknown", -1, time.monotonic()
        try:
            request = json.loads(message)
            if not isinstance(request, Mapping):
                raise ValueError("websocket message must be a JSON object")
            request_id = str(request.get("request_id", request_id))
            step = int(request.get("step", step))
            head, left, right = decode_required_rgb_images(
                request.get("images"), input_color=self.input_color
            )
            state = request.get("state")
            if not isinstance(state, Mapping):
                raise ValueError("request.state must be a JSON object")
            piper_wire_to_fastwam(state)
            prediction = np.asarray(
                self.runtime.infer(head=head, left=left, right=right, state=state),
                dtype=np.float32,
            )
            if prediction.shape != (FASTWAM_ACTION_HORIZON, 14):
                raise ValueError("FastWAM runtime must return finite actions with shape (32, 14), got {}".format(prediction.shape))
            if not np.all(np.isfinite(prediction)):
                raise ValueError("FastWAM runtime returned non-finite actions")
            raw_prediction = prediction[: self.execution_horizon]
            action = fastwam_to_piper_wire(raw_prediction, dt=self.action_dt)
            LOG.warning(
                "SUPERVISED RAW ACTION MODE: step=%s request_id=%s latency_ms=%.1f action_horizon=%s; no software clamp or delta limit applied",
                step, request_id, (time.monotonic() - started) * 1000, self.execution_horizon,
            )
            return build_policy_response(request_id=request_id, step=step, action=action)
        except Exception as exc:
            LOG.exception("failed to process request_id=%s step=%s", request_id, step)
            return build_policy_response(request_id=request_id, step=step, action={}, error=str(exc))
```

Implement the same `handler`, `serve`, `build_arg_parser`, and `main` protocol lifecycle used by the protected server, with the CLI description `Supervised raw-action JSON WebSocket server for FastWAM Piper Puzzle policy`. After constructing the runtime and before opening the listener, emit:

```python
LOG.warning("SUPERVISED RAW ACTION MODE enabled; finite model waypoints are sent without software clamp or delta limit")
```

- [ ] **Step 2: Run the raw-server tests and verify they pass**

Run:

```bash
python -m unittest tests.test_piper_fastwam_policy_server_supervised -v
```

Expected: all three tests pass.

- [ ] **Step 3: Run the standard-server regression test and verify protected mode remains unchanged**

Run:

```bash
python -m unittest tests.test_piper_fastwam_policy_server -v
```

Expected: all standard-server tests pass, including clamp/rate-limit tests.

- [ ] **Step 4: Commit the implementation**

```bash
git add scripts/piper_fastwam_policy_server_supervised.py tests/test_piper_fastwam_policy_server_supervised.py
git commit -m "feat: add supervised raw FastWAM policy server"
```

### Task 3: Add explicit raw-mode configuration and launch entrypoint

**Files:**
- Create: `configs/piper_fastwam_puzzle_supervised_raw.yaml`
- Create: `scripts/run_piper_fastwam_supervised_raw_server.sh`
- Test: `tests/test_piper_fastwam_policy_server_supervised.py`

**Interfaces:**
- Consumes: `FastWAMPiperRuntime.from_config(config)` and the new raw server CLI.
- Produces: a config which selects the current `step_005000.pt` checkpoint and a launcher that never starts the protected server by accident.

- [ ] **Step 1: Add the raw deployment-config assertion to the test class**

Append this method to `SupervisedRawPiperFastWAMServerTest`:

```python
def test_raw_deployment_config_has_no_software_motion_limit_keys(self) -> None:
    config = raw_module.load_config(
        REPO_ROOT / "configs/piper_fastwam_puzzle_supervised_raw.yaml"
    )
    self.assertEqual(config["task_label"], "crimp")
    self.assertEqual(config["execution_horizon"], 24)
    self.assertNotIn("joint_lower", config)
    self.assertNotIn("joint_upper", config)
    self.assertNotIn("max_delta", config)
```

- [ ] **Step 2: Create the raw config with no software motion-limit fields**

Create `configs/piper_fastwam_puzzle_supervised_raw.yaml` with exactly:

```yaml
# Human-supervised FastWAM mode: finite model waypoints are forwarded without
# server-side joint projection or per-waypoint delta limiting. Do not use this
# config as the default deployment; start it only with the supervised launcher.
fastwam_root: /bh/zbh_self/projects/fastwam
model_base_path: /bh/zbh_ckp/models/fastwam
sim_config_name: sim_robotwin.yaml
task_name: agilex_cobotmagic2_puzzle_joint_3cam_384_1e-4_baidu32
checkpoint_path: /bh/zbh_ckp/runs/fastwam/agilex_cobotmagic2_puzzle_joint_3cam_384_1e-4_baidu32_statecontinue30000_wandb/CogWAM_fastWAM-puzzle-statecontinue32-30000-wandb-retry2/checkpoints/weights/step_005000.pt
dataset_stats_path: /bh/zbh_ckp/datasets/fastwam/agilex_cobotmagic2_puzzle_20260801_v21_puppet_joint/normalization/dataset_stats.json
task_label: crimp
model_action_horizon: 32
num_video_frames: 9
num_inference_steps: 10
seed: 42
device: cuda:0
host: 127.0.0.1
port: 7081
execution_horizon: 24
action_dt: 0.0333333333
input_color: bgr
max_message_size_mb: 64
ping_interval_s: 20
ping_timeout_s: 60
```

- [ ] **Step 3: Create the isolated launcher**

```bash
#!/usr/bin/env bash
set -euo pipefail

source /bh/zbh_self/miniforge3/etc/profile.d/conda.sh
conda activate /bh/zbh_self/envs/fastwam
cd "$(dirname "$0")/.."
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/bh/zbh_ckp/models/fastwam}"

exec python -u scripts/piper_fastwam_policy_server_supervised.py \
  --config configs/piper_fastwam_puzzle_supervised_raw.yaml "$@"
```

Then make it executable:

```bash
chmod 755 scripts/run_piper_fastwam_supervised_raw_server.sh
```

- [ ] **Step 4: Run config and source checks**

Run:

```bash
python -m unittest tests.test_piper_fastwam_policy_server_supervised -v
python -m py_compile scripts/piper_fastwam_policy_server_supervised.py
bash -n scripts/run_piper_fastwam_supervised_raw_server.sh
```

Expected: tests pass and both syntax checks are silent.

- [ ] **Step 5: Commit the config and launcher**

```bash
git add configs/piper_fastwam_puzzle_supervised_raw.yaml scripts/run_piper_fastwam_supervised_raw_server.sh tests/test_piper_fastwam_policy_server_supervised.py
git commit -m "config: add supervised raw FastWAM launcher"
```

### Task 4: Verify the full repository and deploy only the server-side raw mode

**Files:**
- Modify: `README.md`
- Deploy: `/bh/zbh_self/projects/piper-openpi-real-robot/scripts/piper_fastwam_policy_server_supervised.py`
- Deploy: `/bh/zbh_self/projects/piper-openpi-real-robot/configs/piper_fastwam_puzzle_supervised_raw.yaml`
- Deploy: `/bh/zbh_self/projects/piper-openpi-real-robot/scripts/run_piper_fastwam_supervised_raw_server.sh`

**Interfaces:**
- Consumes: checked-in source and current remote FastWAM environment.
- Produces: a verified Terminal 1 command for raw mode; Terminal 2 and Terminal 3 remain the current tunnel/client commands.

- [ ] **Step 1: Document the mode boundary and server command**

Add a short `FastWAM supervised raw mode` section to `README.md` that states the raw server must be selected explicitly, the protected server remains default, and gives this Terminal 1 command:

```bash
ssh -F /dev/null -tt -p 7156 root@10.40.1.215 \
  'cd /bh/zbh_self/projects/piper-openpi-real-robot && \
   exec bash scripts/run_piper_fastwam_supervised_raw_server.sh'
```

- [ ] **Step 2: Run the complete unit suite and static checks locally**

Run:

```bash
python -m unittest discover -s tests -v
python -m py_compile scripts/piper_fastwam_policy_server.py scripts/piper_fastwam_policy_server_supervised.py scripts/websocket_policy_client.py
git diff --check
```

Expected: all available tests pass; the two known OpenCV-dependent skips are acceptable; compilation and diff check are silent.


- [ ] **Step 3: Deploy only the new raw-mode files to the p7156 development machine**

Run from repository root:

```bash
scp -P 7156 scripts/piper_fastwam_policy_server_supervised.py \
  root@10.40.1.215:/bh/zbh_self/projects/piper-openpi-real-robot/scripts/
scp -P 7156 scripts/run_piper_fastwam_supervised_raw_server.sh \
  root@10.40.1.215:/bh/zbh_self/projects/piper-openpi-real-robot/scripts/
scp -P 7156 configs/piper_fastwam_puzzle_supervised_raw.yaml \
  root@10.40.1.215:/bh/zbh_self/projects/piper-openpi-real-robot/configs/
```

- [ ] **Step 4: Perform remote syntax and config preflight without opening a listener**

Run:

```bash
ssh -F /dev/null -tt -p 7156 root@10.40.1.215 \
  'cd /bh/zbh_self/projects/piper-openpi-real-robot && \
   source /bh/zbh_self/miniforge3/etc/profile.d/conda.sh && \
   conda activate /bh/zbh_self/envs/fastwam && \
   python -m py_compile scripts/piper_fastwam_policy_server_supervised.py && \
   python -u scripts/piper_fastwam_policy_server_supervised.py \
     --config configs/piper_fastwam_puzzle_supervised_raw.yaml --preflight-only'
```

Expected: FastWAM model load succeeds and the process exits without binding port `7081`.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md
git commit -m "docs: describe supervised raw FastWAM mode"
```
