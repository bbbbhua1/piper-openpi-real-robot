# Verified OpenPI Workflow Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the real-robot-tested Piper/OpenPI client, three-camera recorder, prompt-aware policy server, tests, and exact operating commands directly to the repository's `main` branch.

**Architecture:** Preserve GitHub `main` as the history base, then port the already-verified recorder/client implementation and the one-line server prompt fix into that clone. Document the tested three-terminal dry run and full experiment separately, validate both code and command examples, and fast-forward `main` without force pushing.

**Tech Stack:** Python 3.9/3.11, ROS Noetic, OpenCV compressed ROS images, FFmpeg/FFprobe, asyncio WebSockets, OpenPI/JAX, SSH/SCP, GitHub CLI, Git.

## Global Constraints

- Start from GitHub `main` commit `cecdd4e80650be1e9507c9971d8cccbd1e08191a` and preserve all newer robot initialization documentation.
- Publish the finished result directly to `main`; do not create a PR, force push, rewrite history, tag, or release.
- Preserve the existing safety, home, hold, enable, dry-run, and action execution behavior.
- Dry-run communication testing uses one real ROS observation, `--executor mock`, `--max-steps 1`, and no `--execute-actions`.
- Record all three compressed camera callbacks continuously on the robot and upload only after MP4 finalization.
- Keep the robot-local recording if upload fails.
- Use development server `ssh -p 3763 root@10.40.1.215`.
- Use config `pi05_cmap_all_tasks_v21`.
- Use checkpoint `/bh/media/section/iclr_proj/zbh/openpi/training_runs/pi05_all_tasks_openpi_v21_8gpu_b512_80k/checkpoints/pi05_cmap_all_tasks_v21/pi05_all_tasks_openpi_v21_pi05_8gpu_b512_80k_correct_20260718/60000`.
- Use instruction `Arrange the dominoes on the table in a horizontal row and then knock them over.`
- Use robot-local recording root `/home/agilex/piper-openpi-real-robot/recordings`.
- Use server recording root `/bh/media/section/iclr_proj/zbh/piper_openpi_real_robot_videos`.
- Stop publication if remote `main` advances; integrate concurrent work before pushing.

---

### Task 1: Port The Verified Three-Camera Recorder

**Files:**
- Create: `scripts/camera_video_recorder.py`
- Create: `tests/__init__.py`
- Create: `tests/test_camera_video_recorder.py`

**Interfaces:**
- Consumes: compressed JPEG payloads from ROS `sensor_msgs/CompressedImage.data`.
- Produces: `CameraVideoRecorder.submit(camera_name: str, jpeg_payload: bytes) -> None`, `close() -> None`, `write_manifest() -> pathlib.Path`, and `upload() -> str`.
- Produces CLI defaults `DEFAULT_RECORD_DIR`, `DEFAULT_UPLOAD_HOST`, `DEFAULT_UPLOAD_PORT`, and `DEFAULT_UPLOAD_DIR` for the robot client.

- [ ] **Step 1: Add the recorder tests from the verified working copy**

Use the complete test suite from:

```text
/Users/hua/Documents/Codex/2026-07-28/bbbbhua1-piper-openpi-real-robot-https-3/work/piper-openpi-real-robot/tests/test_camera_video_recorder.py
```

The suite must retain these seven concrete tests:

```python
def test_builds_one_ffmpeg_command_per_camera()
def test_submit_writes_frames_and_manifest()
def test_full_queue_drops_oldest_frame_without_blocking_submitter()
def test_upload_creates_remote_root_then_copies_session()
def test_upload_failure_preserves_local_session()
def test_recording_cli_defaults()
def test_compressed_callback_submits_exact_jpeg_payload()
```

Also add an empty `tests/__init__.py` so the suite can be run as a module.

- [ ] **Step 2: Run the recorder tests and verify they fail before implementation**

Run:

```bash
python3 -m unittest -v tests.test_camera_video_recorder
```

Expected: failure because `scripts/camera_video_recorder.py` does not exist and
the existing client has no recording CLI.

- [ ] **Step 3: Add the verified recorder implementation**

Port the complete implementation byte-for-byte from:

```text
/Users/hua/Documents/Codex/2026-07-28/bbbbhua1-piper-openpi-real-robot-https-3/work/piper-openpi-real-robot/scripts/camera_video_recorder.py
```

The FFmpeg command must remain:

```python
[
    ffmpeg_bin,
    "-hide_banner",
    "-loglevel",
    "warning",
    "-nostdin",
    "-y",
    "-f",
    "image2pipe",
    "-framerate",
    fps_text,
    "-vcodec",
    "mjpeg",
    "-i",
    "pipe:0",
    "-map",
    "0:v:0",
    "-an",
    "-c:v",
    "copy",
    "-movflags",
    "+faststart",
    str(output_path),
]
```

The upload flow must remain:

```python
self._run_factory(
    [
        "ssh",
        "-p",
        str(self.upload_port),
        *ssh_options,
        self.upload_host,
        mkdir_command,
    ],
    check=True,
)
self._run_factory(
    [
        "scp",
        "-P",
        str(self.upload_port),
        *ssh_options,
        "-r",
        str(self.session_dir),
        "{}:{}/".format(self.upload_host, self.upload_dir),
    ],
    check=True,
)
```

After porting the verified behavior, update the current development-server
defaults to:

```python
DEFAULT_UPLOAD_HOST = "root@10.40.1.215"
DEFAULT_UPLOAD_PORT = 3763
```

Update the CLI-default test to assert these exact values. The upload directory
remains `/bh/media/section/iclr_proj/zbh/piper_openpi_real_robot_videos`.

- [ ] **Step 4: Run the focused recorder tests**

Run:

```bash
python3 -m unittest -v tests.test_camera_video_recorder
```

Expected at this point: five `CameraVideoRecorderTest` tests pass and two
`PolicyClientRecordingIntegrationTest` tests fail because the client recording
CLI/callback integration is added in Task 2.

---

### Task 2: Integrate Recording Into The Robot Client And Preserve The Prompt

**Files:**
- Modify: `scripts/websocket_policy_client.py`
- Modify: `scripts/piper_openpi_policy_server.py`
- Test: `tests/test_camera_video_recorder.py`

**Interfaces:**
- Consumes: `CameraVideoRecorder` and its four public lifecycle methods from Task 1.
- Produces: robot-client CLI flags `--record-videos`, `--record-dir`,
  `--record-fps`, `--record-queue-size`, `--record-upload-host`,
  `--record-upload-port`, `--record-upload-dir`, and
  `--record-session-name`.
- Produces: `RosObservationSource.finalize_recording() -> None`.
- Produces: an OpenPI inference repack that maps request `prompt` to policy
  `prompt`.

- [ ] **Step 1: Port the verified robot client**

Apply the verified production diff from:

```text
/Users/hua/Documents/Codex/2026-07-28/bbbbhua1-piper-openpi-real-robot-https-3/work/piper-openpi-real-robot/scripts/websocket_policy_client.py
```

Do not replace unrelated code blindly. Confirm the GitHub-base file matches the
pre-feature client, then apply only the recording integration. The callback
must pass the exact compressed bytes to both policy inference state and the
recorder:

```python
payload = bytes(message.data)
self.images[camera_name] = {
    "encoding": "jpeg",
    "data": payload,
}
if self.video_recorder is not None:
    self.video_recorder.submit(camera_name, payload)
```

The final cleanup must remain nested so recording finalization always runs
after robot hold/disable handling:

```python
finally:
    finalize_recording = getattr(source, "finalize_recording", None)
    if finalize_recording is not None:
        finalize_recording()
```

Recording validation must remain:

```python
if args.record_videos:
    if not args.compressed_images:
        raise ValueError("--record-videos requires --compressed-images")
    if not camera_topics:
        raise ValueError("--record-videos requires at least one --camera-topic")
```

- [ ] **Step 2: Apply the verified server prompt mapping**

First confirm the new development-server copy differs only at the prompt
mapping:

```bash
ssh -F /dev/null -p 3763 root@10.40.1.215 \
  'cat /bh/media/section/iclr_proj/zbh/openpi/scripts/piper_openpi_policy_server.py' \
  | diff -u scripts/piper_openpi_policy_server.py -
```

Then add this exact entry beside `state` in `inference_repack`:

```python
"prompt": "prompt",
```

Do not change the repository defaults in this task; the exact current machine
values belong in the README command examples.

- [ ] **Step 3: Verify protocol code is unchanged**

Run:

```bash
cmp \
  scripts/ws_policy_protocol.py \
  /Users/hua/Documents/Codex/2026-07-28/bbbbhua1-piper-openpi-real-robot-https-3/work/piper-openpi-real-robot/scripts/ws_policy_protocol.py
```

Expected: exit status 0 and no output. Do not modify the protocol file when it
is byte-identical.

- [ ] **Step 4: Run all tests and syntax checks**

Run:

```bash
python3 -m unittest -v tests.test_camera_video_recorder
python3 -m py_compile \
  scripts/camera_video_recorder.py \
  scripts/websocket_policy_client.py \
  scripts/piper_openpi_policy_server.py \
  scripts/ws_policy_protocol.py
```

Expected: 7 tests pass and `py_compile` produces no output.

- [ ] **Step 5: Review and commit production code**

Run:

```bash
git diff --check
git diff --stat
git status --short
```

Expected changed paths:

```text
scripts/camera_video_recorder.py
scripts/piper_openpi_policy_server.py
scripts/websocket_policy_client.py
tests/__init__.py
tests/test_camera_video_recorder.py
```

Commit:

```bash
git add \
  scripts/camera_video_recorder.py \
  scripts/piper_openpi_policy_server.py \
  scripts/websocket_policy_client.py \
  tests/__init__.py \
  tests/test_camera_video_recorder.py
git commit -m "feat: save verified robot camera recordings"
```

---

### Task 3: Replace Stale Examples With The Verified Workflow

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the CLI flags added in Task 2.
- Produces: exact copy-paste commands for policy service, tunnel, dry run, and
  full robot execution.

- [ ] **Step 1: Update repository layout and current-status sections**

Add the recorder and tests to the tree:

```text
scripts/
├── camera_video_recorder.py
├── piper_openpi_policy_server.py
├── websocket_policy_client.py
└── ws_policy_protocol.py
tests/
└── test_camera_video_recorder.py
```

State that `mode:=1` is required for the current joint-control workflow and
keep the link to `docs/agilex_robot_ros_startup.md`.

- [ ] **Step 2: Document the exact server command**

Use this command:

```bash
ssh -F /dev/null -p 3763 -tt root@10.40.1.215 "cd /bh/media/section/iclr_proj/zbh/openpi && \
export CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
PYTHONPATH=/bh/media/section/iclr_proj/zbh/openpi/src:/bh/media/section/iclr_proj/zbh/openpi/packages/openpi-client/src && \
exec /bh/media/section/iclr_proj/third_party/openpi/.venv/bin/python -u \
scripts/piper_openpi_policy_server.py \
--host 127.0.0.1 \
--port 7081 \
--openpi-root /bh/media/section/iclr_proj/zbh/openpi \
--config pi05_cmap_all_tasks_v21 \
--checkpoint /bh/media/section/iclr_proj/zbh/openpi/training_runs/pi05_all_tasks_openpi_v21_8gpu_b512_80k/checkpoints/pi05_cmap_all_tasks_v21/pi05_all_tasks_openpi_v21_pi05_8gpu_b512_80k_correct_20260718/60000 \
--horizon 10 \
--action-dt 0.1 \
--max-joint-step 0.05"
```

Document the success line:

```text
piper OpenPI policy server listening on ws://127.0.0.1:7081
```

- [ ] **Step 3: Document the tunnel command**

Use:

```bash
ssh -F /dev/null -p 3763 -N -g \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -L 0.0.0.0:18001:127.0.0.1:7081 \
  root@10.40.1.215
```

Explain that `18001` is the local listener and that the robot URI must use the
local computer's current robot-network IP. Use `10.13.10.63` only as the
verified example, not as a universal constant.

- [ ] **Step 4: Document the one-step no-action communication test**

Use the exact instruction and these behavior flags:

```text
--source ros
--executor mock
--max-steps 1
--compressed-images
--require-images
--record-videos
```

Include the three camera mappings:

```text
head=/camera_f/color/image_raw/compressed
left_wrist=/camera_l/color/image_raw/compressed
right_wrist=/camera_r/color/image_raw/compressed
```

Include explicit upload values:

```text
--record-upload-host root@10.40.1.215
--record-upload-port 3763
--record-upload-dir /bh/media/section/iclr_proj/zbh/piper_openpi_real_robot_videos
```

The SSH command must use `ssh -A -tt` so forwarded agent credentials are
available to the client's `BatchMode=yes` upload subprocess.

- [ ] **Step 5: Document the full experiment command**

Base it on the communication-test command, then make exactly these execution
changes:

```text
--executor ros
--execute-actions
--enable-on-start
```

Remove `--max-steps 1` so inference continues until interrupted. Keep all
recording flags and camera mappings. Do not add `--disable-on-exit`; the robot
must remain enabled while the hold loop maintains position.

Explain shutdown:

1. First `Ctrl-C` leaves inference and enters the hold loop.
2. Second `Ctrl-C` exits the hold loop.
3. MP4 files finalize and upload begins.
4. The robot-local session remains available whether upload succeeds or fails.

- [ ] **Step 6: Document verified evidence and file locations**

Record:

```text
Robot recordings: /home/agilex/piper-openpi-real-robot/recordings
Server recordings: /bh/media/section/iclr_proj/zbh/piper_openpi_real_robot_videos
```

State the completed evidence precisely:

```text
3 cameras × 150 frames, 1280×720, 30 FPS, zero dropped frames
single A800, successful policy response, 10 waypoints per arm
```

- [ ] **Step 7: Scan for stale machine values and commit documentation**

Run:

```bash
rg -n \
  '10\\.40\\.1\\.219|3164|/media/section|pi05_s40_3|49999|10\\.13\\.12\\.85|:18000' \
  README.md
```

Expected: no stale command examples. Historical prose may keep old values only
when explicitly labeled historical; the updated README should not need them.

Run:

```bash
git diff --check
git diff -- README.md
git add README.md
git commit -m "docs: update verified three-terminal experiment commands"
```

---

### Task 4: Final Verification And Direct Main Publication

**Files:**
- Verify: all files changed in Tasks 1–3

**Interfaces:**
- Consumes: verified code, tests, commands, and Git history.
- Produces: a fast-forwarded GitHub `main`.

- [ ] **Step 1: Run the complete local verification**

Run:

```bash
python3 -m unittest -v tests.test_camera_video_recorder
python3 -m py_compile \
  scripts/camera_video_recorder.py \
  scripts/websocket_policy_client.py \
  scripts/piper_openpi_policy_server.py \
  scripts/ws_policy_protocol.py
git diff --check origin/main...HEAD
git status --short --branch
```

Expected:

```text
Ran 7 tests
OK
## main...origin/main [ahead 3]
```

The exact ahead count may be greater if the design/plan documentation used
separate commits; the working tree must be clean.

- [ ] **Step 2: Confirm GitHub identity and concurrent-update safety**

Run:

```bash
gh auth status
git fetch origin main
git rev-parse origin/main
git merge-base --is-ancestor origin/main HEAD
```

Expected active GitHub account: `bbbbhua1`.

Expected `origin/main` before publication:

```text
cecdd4e80650be1e9507c9971d8cccbd1e08191a
```

If it differs, inspect and integrate the new commits before continuing. Never
use `--force`.

- [ ] **Step 3: Review the final patch**

Run:

```bash
git log --oneline --decorate origin/main..HEAD
git diff --stat origin/main...HEAD
git diff --name-status origin/main...HEAD
```

Expected production files, tests, README, design, and plan only. No checkpoints,
recordings, credentials, caches, or temporary files may appear.

- [ ] **Step 4: Push directly to main**

Run:

```bash
git push origin main
```

Expected: a normal fast-forward update with no force.

- [ ] **Step 5: Verify published main through GitHub CLI**

Run:

```bash
gh api repos/bbbbhua1/piper-openpi-real-robot/branches/main \
  --jq '{name: .name, sha: .commit.sha, protected: .protected}'
gh api repos/bbbbhua1/piper-openpi-real-robot/contents/scripts/camera_video_recorder.py \
  --jq '{path: .path, sha: .sha, size: .size}'
gh repo view bbbbhua1/piper-openpi-real-robot \
  --json nameWithOwner,defaultBranchRef,url
```

Expected:

- branch name is `main`;
- branch SHA equals local `HEAD`;
- `scripts/camera_video_recorder.py` exists;
- the repository default branch remains `main`.

- [ ] **Step 6: Report final state**

Report:

- final `main` commit SHA and GitHub URL;
- files added and modified;
- test and syntax-check results;
- server environment used for inspection:
  `/bh/media/section/iclr_proj/third_party/openpi/.venv`;
- no server files were modified;
- no large outputs were written on the new server;
- no cleanup remains except the user may delete local robot recordings after
  independently confirming server uploads.
