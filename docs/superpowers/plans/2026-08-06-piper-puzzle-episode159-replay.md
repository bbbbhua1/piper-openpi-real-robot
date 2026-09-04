# Piper Puzzle Episode 159 Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replay Puzzle dataset episode 159 through a new independent WebSocket server, keeping all recorded targets and the 30 Hz relative cadence unchanged.

**Architecture:** A replay-only server loads the fixed LeRobot parquet, validates it against absolute Piper bounds, maps FastWAM layout to Piper layout, and serves all 610 actions once. The FastWAM inference server, model configuration, checkpoints, and existing robot client are not modified.

**Tech Stack:** Python 3.10, PyArrow, NumPy, websockets, LeRobot v2.1 parquet, ROS client already on the robot.

## Global Constraints

- Create replay-specific files only; do not edit `piper_fastwam_policy_server.py`, `websocket_policy_client.py`, `piper_fastwam_puzzle.yaml`, or model checkpoints.
- Dataset root is `/bh/zbh_ckp/datasets/fastwam/agilex_cobotmagic2_puzzle_20260801_v21_puppet_joint`; only episode `159` may be served.
- Preserve every recorded action target. No interpolation, smoothing, projection, per-waypoint rate limiting, or time rescaling is permitted.
- The existing protocol requires positive arrival times. The replay therefore schedules source timestamp zero at `+1/30 s`; it preserves every inter-frame timestamp difference. Action 0 equals the verified start pose already commanded by the client.
- Reject non-finite source values, wrong `[T,14]` layout, non-30-Hz source cadence, or absolute-limit violations before opening a listener.
- Serve exactly one response per server process; a second request returns an error.
- Existing arm-status fault behavior and Ctrl-C hold behavior are preserved by using the existing client unchanged.
- Use server port `7082` and Mac tunnel port `18002`.

---

## File Structure

- Create: `/bh/zbh_self/projects/piper-openpi-real-robot/scripts/piper_dataset_replay_server.py` — immutable data loader and one-shot WebSocket listener.
- Create: `/bh/zbh_self/projects/piper-openpi-real-robot/tests/test_piper_dataset_replay_server.py` — loader and protocol tests.
- Create: `/bh/zbh_self/projects/piper-openpi-real-robot/docs/superpowers/plans/2026-08-06-piper-puzzle-episode159-replay.md` — this plan.
- Copy: reference video to `/Users/hua/Downloads/fastwam-replay-reference/puzzle-episode-000159-head.mp4`.

### Task 1: Add a testable, immutable parquet loader

**Files:**

- Create: `tests/test_piper_dataset_replay_server.py`
- Create: `scripts/piper_dataset_replay_server.py`

**Interfaces:**

- Consumes: `load_start_pose_config(path) -> StartPoseConfig` from `piper_start_pose.py`.
- Produces: `load_episode(dataset_root: Path, episode_index: int, config: StartPoseConfig) -> ReplayEpisode`.
- Produces: `ReplayEpisode.as_piper_action() -> dict[str, object]`.

- [ ] **Step 1: Write failing loader tests**

```python
def test_loader_maps_source_fastwam_order_to_piper_order(tmp_path):
    write_episode(tmp_path, [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, .01, .02]] * 3)
    action = load_episode(tmp_path, 159, make_config()).as_piper_action()
    assert action["left_arm"][0] == [1., 2., 3., 4., 5., 6., .01]
    assert action["right_arm"][0] == [7., 8., 9., 10., 11., 12., .02]
    assert action["time_list"] == [1 / 30, 2 / 30, 3 / 30]

def test_loader_rejects_absolute_bound_violation(tmp_path):
    write_episode(tmp_path, [[0.] * 14, [9.] + [0.] * 13])
    with self.assertRaisesRegex(ValueError, "outside verified absolute bounds"):
        load_episode(tmp_path, 159, make_config())
```

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
cd /bh/zbh_self/projects/piper-openpi-real-robot
/bh/zbh_self/envs/fastwam/bin/python -m unittest tests.test_piper_dataset_replay_server -v
```

Expected: `ModuleNotFoundError: No module named 'piper_dataset_replay_server'`.

- [ ] **Step 3: Implement exact conversion and validation**

```python
PIPER_ORDER_FROM_FASTWAM = np.asarray([0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13])

@dataclass(frozen=True)
class ReplayEpisode:
    episode_index: int
    source_path: Path
    piper_actions: np.ndarray
    source_timestamps: np.ndarray

    def as_piper_action(self) -> dict[str, object]:
        relative = self.source_timestamps - self.source_timestamps[0]
        return {
            "left_arm": self.piper_actions[:, :7].tolist(),
            "right_arm": self.piper_actions[:, 7:].tolist(),
            "time_list": (relative + 1.0 / 30.0).tolist(),
            "dt": 1.0 / 30.0,
        }

def load_episode(dataset_root: Path, episode_index: int, config: StartPoseConfig) -> ReplayEpisode:
    source_path = dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    table = pq.read_table(source_path, columns=["timestamp", "action"])
    timestamps = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64)
    fastwam_actions = np.asarray(table.column("action").to_pylist(), dtype=np.float64)
    _validate(fastwam_actions, timestamps, config)
    return ReplayEpisode(episode_index, source_path, fastwam_actions[:, PIPER_ORDER_FROM_FASTWAM], timestamps)
```

`_validate` checks nonempty `[T,14]`, finite values, strictly increasing timestamps, each interval within `5e-5 s` of `1/30 s`, and inclusive Piper absolute bounds after the documented reorder. It never alters target values.

- [ ] **Step 4: Run loader tests**

Run:

```bash
cd /bh/zbh_self/projects/piper-openpi-real-robot
/bh/zbh_self/envs/fastwam/bin/python -m unittest tests.test_piper_dataset_replay_server -v
```

Expected: all loader tests pass.

- [ ] **Step 5: Commit if available**

Run:

```bash
git add scripts/piper_dataset_replay_server.py tests/test_piper_dataset_replay_server.py
git commit -m "feat: add immutable Piper dataset replay loader"
```

Expected: commit succeeds if `.git` exists. If it does not, record `git metadata unavailable` in the deployment log and do not fabricate a commit.

### Task 2: Add a one-shot replay WebSocket server

**Files:**

- Modify: `scripts/piper_dataset_replay_server.py`
- Modify: `tests/test_piper_dataset_replay_server.py`

**Interfaces:**

- Consumes: `ReplayEpisode.as_piper_action() -> dict[str, object]`.
- Produces: `DatasetReplayServer.handle_message(message: str) -> Awaitable[dict[str, object]]`.
- Uses: `build_policy_response()` from `ws_policy_protocol.py`.

- [ ] **Step 1: Write failing protocol test**

```python
async def test_server_serves_one_exact_action_chunk_then_refuses_reuse(self):
    server = DatasetReplayServer(replay_episode())
    first = await server.handle_message(json.dumps({"request_id": "one", "step": 0, "state": valid_state(), "images": {}}))
    self.assertTrue(first["ok"])
    self.assertEqual(first["action"], replay_episode().as_piper_action())
    second = await server.handle_message(json.dumps({"request_id": "two", "step": 1, "state": valid_state(), "images": {}}))
    self.assertFalse(second["ok"])
    self.assertIn("already served", second["error"])
```

- [ ] **Step 2: Verify the protocol test fails**

Run:

```bash
cd /bh/zbh_self/projects/piper-openpi-real-robot
/bh/zbh_self/envs/fastwam/bin/python -m unittest tests.test_piper_dataset_replay_server.DatasetReplayServerTest -v
```

Expected: failure because `DatasetReplayServer` is not defined.

- [ ] **Step 3: Implement the server and CLI**

```python
class DatasetReplayServer:
    def __init__(self, episode: ReplayEpisode) -> None:
        self.episode = episode
        self._served = False

    async def handle_message(self, message: str) -> dict[str, object]:
        request_id, step = "unknown", -1
        try:
            request = json.loads(message)
            if not isinstance(request, Mapping):
                raise ValueError("websocket message must be a JSON object")
            request_id = str(request.get("request_id", request_id))
            step = int(request.get("step", step))
            if self._served:
                raise RuntimeError("episode replay already served; restart the server for another run")
            self._served = True
            return build_policy_response(request_id, step, self.episode.as_piper_action())
        except Exception as exc:
            return build_policy_response(request_id, step, {}, error=str(exc))
```

The CLI accepts `--dataset-root`, `--episode-index`, `--start-pose-config`, `--host`, `--port`, and `--preflight-only`. It validates all source values before opening the listener and logs path, action count, duration, and cadence. It imports no model code.

- [ ] **Step 4: Run unit suites**

Run:

```bash
cd /bh/zbh_self/projects/piper-openpi-real-robot
/bh/zbh_self/envs/fastwam/bin/python -m unittest tests.test_piper_dataset_replay_server -v
/bh/zbh_self/envs/fastwam/bin/python -m unittest discover -s tests -v
```

Expected: replay tests pass and the existing suite remains green except documented optional-dependency skips.

- [ ] **Step 5: Commit if available**

Run:

```bash
git add scripts/piper_dataset_replay_server.py tests/test_piper_dataset_replay_server.py
git commit -m "feat: serve exact Puzzle episode replay over websocket"
```

Expected: commit succeeds if `.git` exists; otherwise retain test logs as the change record.

### Task 3: Preflight, deploy, and copy head reference

**Files:**

- Write: `/bh/zbh_self/logs/fastwam/piper_dataset_replay_episode159_preflight_20260806.log`.
- Copy: `/Users/hua/Downloads/fastwam-replay-reference/puzzle-episode-000159-head.mp4`.

**Interfaces:**

- Consumes: the Task 2 CLI and the unmodified robot client.
- Produces: server `127.0.0.1:7082`, tunnel port `18002`, one complete live replay recording.

- [ ] **Step 1: Compile and preflight**

Run:

```bash
cd /bh/zbh_self/projects/piper-openpi-real-robot
/bh/zbh_self/envs/fastwam/bin/python -m py_compile scripts/piper_dataset_replay_server.py
/bh/zbh_self/envs/fastwam/bin/python scripts/piper_dataset_replay_server.py \
  --dataset-root /bh/zbh_ckp/datasets/fastwam/agilex_cobotmagic2_puzzle_20260801_v21_puppet_joint \
  --episode-index 159 \
  --start-pose-config configs/piper_fastwam_puzzle_start_pose.json \
  --preflight-only
```

Expected: exactly 610 actions, 20.30 seconds, all absolute limits verified, and no model load.

- [ ] **Step 2: Download source head video**

Run on the operator Mac:

```bash
mkdir -p /Users/hua/Downloads/fastwam-replay-reference
scp -P 7156 root@10.40.1.215:/bh/zbh_ckp/datasets/fastwam/agilex_cobotmagic2_puzzle_20260801_v21_puppet_joint/videos/chunk-000/observation.images.cam_high/episode_000159.mp4 \
  /Users/hua/Downloads/fastwam-replay-reference/puzzle-episode-000159-head.mp4
```

Expected: local immutable source reference video; no source dataset changes.

- [ ] **Step 3: Use three terminals**

Terminal 1 runs only the new replay server with a timestamped log. Terminal 2 forwards `0.0.0.0:18002` to `127.0.0.1:7082` using SSH port 7156. Terminal 3 invokes the unmodified robot client with `--max-steps 1`, episode-159 start pose, ROS executor, all three camera topics, and `--uri ws://10.13.0.96:18002`.

- [ ] **Step 4: Verify live sequence**

Expected:

```text
[fastwam_puzzle_episode_159_start] moving to target ...
[start-pose] reached source_episode=159 ...
connected to policy server: ws://10.13.0.96:18002
[execute] waypoint 610/610 published ...
```

The first Ctrl-C starts the existing hold loop; the second exits. A ROS arm-status fault disables the robot and skips hold; do not retry until the fault is resolved.

