# FastWAM Model-Directory Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate FastWAM as an isolated `models/fastwam/` Piper workflow alongside `openpi_pi05` and `uva_dit`.

**Architecture:** Base the feature branch on the current model-oriented `origin/main`, then import only the completed FastWAM source from the old branch. Server code, configurations, launchers, diagnostics, and server tests live in `models/fastwam/server`; robot client code, protocol, start pose, recorder, and client tests live in `models/fastwam/client`. The root README is a model index and `models/fastwam/README.md` is the only FastWAM runbook.

**Tech Stack:** Git/GitHub, Python 3.10, FastWAM, NumPy, Pillow, PyYAML, WebSockets, ROS Noetic, Bash, SSH/SCP, `unittest`.

## Global Constraints

- Base on `origin/main` commit `7a2f914` or its current successor; do not retain the old root-layout README.
- Do not modify `models/openpi_pi05/`, `models/uva_dit/`, `reference/`, or generic `docs/`, other than FastWAM navigation in root `README.md`.
- Protected and supervised raw FastWAM are separate explicit server entrypoints.
- Do not commit weights, base models, data, statistics, recordings, credentials, secrets, or API keys.
- Deploy server files only to `/bh/zbh_self/projects/piper-openpi-real-robot/models/fastwam/server` on SSH port 7156. Do not modify `/bh/media`.
- Deploy client files only to `/home/agilex/piper-openpi-real-robot/models/fastwam/client` on robot `10.13.0.155`.

---

### Task 1: Rebase the working branch on current model-oriented main

**Files:**
- Reference: `backup/fastwam-root-layout`
- Modify: branch base for `agent/fastwam-piper-real-robot`

**Interfaces:**
- Consumes: completed FastWAM files on the backup branch.
- Produces: a clean branch containing current `models/openpi_pi05/` and `models/uva_dit/`.

- [ ] **Step 1: Preserve the completed root-layout implementation**

```bash
git branch backup/fastwam-root-layout agent/fastwam-piper-real-robot
git show --no-patch --oneline backup/fastwam-root-layout
```

- [ ] **Step 2: Reset the working branch before import**

```bash
git fetch origin --prune
git switch agent/fastwam-piper-real-robot
git reset --hard origin/main
git status -sb
```

Expected: a clean worktree based on the multi-model main branch.

- [ ] **Step 3: Verify existing models are unchanged**

```bash
git diff --exit-code origin/main -- models/openpi_pi05 models/uva_dit reference docs
```

Expected: no output and exit code zero.

### Task 2: Relocate FastWAM source into model-owned server and client directories

**Files:**
- Create: `models/fastwam/server/fastwam_piper_runtime.py`
- Create: `models/fastwam/server/piper_fastwam_policy_server.py`
- Create: `models/fastwam/server/piper_fastwam_policy_server_supervised.py`
- Create: `models/fastwam/server/run_piper_fastwam_server.sh`
- Create: `models/fastwam/server/run_piper_fastwam_supervised_raw_server.sh`
- Create: `models/fastwam/server/piper_fastwam_puzzle.yaml`
- Create: `models/fastwam/server/piper_fastwam_puzzle_supervised_raw.yaml`
- Create: `models/fastwam/server/diagnose_fastwam_validation_boundaries.py`
- Create: `models/fastwam/server/test_fastwam_piper_runtime.py`
- Create: `models/fastwam/server/test_piper_fastwam_policy_server.py`
- Create: `models/fastwam/server/test_piper_fastwam_policy_server_supervised.py`
- Create: `models/fastwam/client/websocket_policy_client.py`
- Create: `models/fastwam/client/ws_policy_protocol.py`
- Create: `models/fastwam/client/camera_video_recorder.py`
- Create: `models/fastwam/client/piper_start_pose.py`
- Create: `models/fastwam/client/piper_fastwam_puzzle_start_pose.json`
- Create: `models/fastwam/client/test_camera_video_recorder.py`
- Create: `models/fastwam/client/test_piper_start_pose.py`
- Create: `models/fastwam/client/test_ros_joint_executor.py`
- Create: `models/fastwam/client/test_websocket_policy_client_start_pose.py`
- Create: `models/fastwam/client/test_ws_policy_protocol.py`

**Interfaces:**
- Server imports only sibling modules `fastwam_piper_runtime` and `piper_fastwam_policy_server`.
- Client imports only sibling modules `camera_video_recorder`, `piper_start_pose`, and `ws_policy_protocol`.
- Produces: self-contained directories with no root `scripts/`, `configs/`, or `tests/` dependency.

- [ ] **Step 1: Restore only the FastWAM implementation files from backup**

```bash
git checkout backup/fastwam-root-layout -- configs scripts tests
```

Keep only names prefixed `piper_fastwam`, `fastwam_piper_runtime.py`, `diagnose_fastwam_validation_boundaries.py`, `run_piper_fastwam*.sh`, `websocket_policy_client.py`, `ws_policy_protocol.py`, `camera_video_recorder.py`, `piper_start_pose.py`, and their FastWAM tests. Remove any restored non-FastWAM root file before continuing.

- [ ] **Step 2: Move each owned file to its model directory**

```bash
mkdir -p models/fastwam/server models/fastwam/client
git mv scripts/fastwam_piper_runtime.py scripts/piper_fastwam_policy_server.py scripts/piper_fastwam_policy_server_supervised.py scripts/run_piper_fastwam_server.sh scripts/run_piper_fastwam_supervised_raw_server.sh scripts/diagnose_fastwam_validation_boundaries.py models/fastwam/server/
git mv configs/piper_fastwam_puzzle.yaml configs/piper_fastwam_puzzle_supervised_raw.yaml models/fastwam/server/
git mv scripts/websocket_policy_client.py scripts/ws_policy_protocol.py scripts/camera_video_recorder.py scripts/piper_start_pose.py models/fastwam/client/
git mv configs/piper_fastwam_puzzle_start_pose.json models/fastwam/client/
git mv tests/test_fastwam_piper_runtime.py tests/test_piper_fastwam_policy_server.py tests/test_piper_fastwam_policy_server_supervised.py models/fastwam/server/
git mv tests/test_camera_video_recorder.py tests/test_piper_start_pose.py tests/test_ros_joint_executor.py tests/test_websocket_policy_client_start_pose.py tests/test_ws_policy_protocol.py models/fastwam/client/
```

- [ ] **Step 3: Make both server launchers directory-local**

Protected launcher tail:

```bash
cd "$(dirname "$0")"
exec python -u piper_fastwam_policy_server.py --config piper_fastwam_puzzle.yaml "$@"
```

Raw-mode launcher tail:

```bash
cd "$(dirname "$0")"
exec python -u piper_fastwam_policy_server_supervised.py --config piper_fastwam_puzzle_supervised_raw.yaml "$@"
```

Keep the conda activation, `DIFFSYNTH_MODEL_BASE_PATH`, and executable mode `755` unchanged.

- [ ] **Step 4: Adjust every moved test's import root and data path**

Server tests must contain:

```python
SERVER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SERVER_DIR))
```

Client tests must contain:

```python
CLIENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CLIENT_DIR))
```

Use `SERVER_DIR / "piper_fastwam_puzzle_supervised_raw.yaml"` for supervised-server config tests and `CLIENT_DIR / "piper_fastwam_puzzle_start_pose.json"` for start-pose tests. The supervised-server test imports helpers with `from test_piper_fastwam_policy_server import FakeRuntime, make_request`.

- [ ] **Step 5: Verify source-local functionality**

```bash
python -m unittest discover -s models/fastwam/server -p 'test_*.py' -v
python -m unittest discover -s models/fastwam/client -p 'test_*.py' -v
python -m py_compile models/fastwam/server/*.py models/fastwam/client/*.py
bash -n models/fastwam/server/run_piper_fastwam_server.sh
bash -n models/fastwam/server/run_piper_fastwam_supervised_raw_server.sh
```

Expected: tests pass; only the known local OpenCV-dependent image transport tests may skip.

- [ ] **Step 6: Commit the isolated code**

```bash
git add models/fastwam
git commit -m "refactor: isolate FastWAM Piper model workflow"
```

### Task 3: Publish FastWAM-specific commands and root-model navigation

**Files:**
- Create: `models/fastwam/README.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: final Task 2 paths.
- Produces: copyable model-specific three-terminal commands and FastWAM as the third root index entry.

- [ ] **Step 1: Create the model runbook**

Include sections `目录`, `环境、权重与数据统计`, `机器人 ROS 初始化`, `部署`, `三终端 dry-run`, `完整真机执行`, `两种服务端模式`, `录像与停止`, and `checkpoint 切换`. It must use:

```text
/bh/zbh_self/projects/piper-openpi-real-robot/models/fastwam/server
/home/agilex/piper-openpi-real-robot/models/fastwam/client
```

Protected Terminal 1:

```bash
ssh -F /dev/null -tt -p 7156 root@10.40.1.215 'cd /bh/zbh_self/projects/piper-openpi-real-robot/models/fastwam/server && CUDA_VISIBLE_DEVICES=0 exec bash run_piper_fastwam_server.sh'
```

Raw-mode Terminal 1:

```bash
ssh -F /dev/null -tt -p 7156 root@10.40.1.215 'cd /bh/zbh_self/projects/piper-openpi-real-robot/models/fastwam/server && CUDA_VISIBLE_DEVICES=0 exec bash run_piper_fastwam_supervised_raw_server.sh'
```

Terminal 2:

```bash
ssh -F /dev/null -N -g -p 7156 -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -L 0.0.0.0:18001:127.0.0.1:7081 root@10.40.1.215
```

Terminal 3 must call `models/fastwam/client/websocket_policy_client.py`, set the FastWAM transport profile, three compressed camera topics, and the model-owned start-pose JSON. Dry run uses `--executor mock --max-steps 1` and no `--execute-actions`; full execution uses `--executor ros --execute-actions --max-steps 25` with recording. State that exactly one model server can bind port `7081`.

- [ ] **Step 2: Add FastWAM to the root model tree, table, and short launch index**

Add:

```markdown
| FastWAM Puzzle/crimp | `models/fastwam/server/` | `models/fastwam/client/` | [`models/fastwam/README.md`](models/fastwam/README.md) |
```

Add `models/fastwam/` to the tree with `README.md`, `server/`, and `client/`. Add a FastWAM three-terminal mock section after UVA-DiT that links to `models/fastwam/README.md` for protected/raw choice, full execution, start pose, video, and troubleshooting.

- [ ] **Step 3: Commit the runbook and root index**

```bash
git add README.md models/fastwam/README.md
git commit -m "docs: add FastWAM model deployment guide"
```

### Task 4: Move FastWAM history under its model and eliminate root duplicates

**Files:**
- Create: `models/fastwam/docs/README.md`
- Move: `docs/superpowers/plans/*fastwam*.md`
- Move: `docs/superpowers/specs/*fastwam*.md`
- Restore: `docs/agilex_robot_ros_startup.md` from `origin/main`

**Interfaces:**
- Produces: a single model-owned FastWAM source tree; generic ROS documentation remains generic.

- [ ] **Step 1: Move all FastWAM design records**

```bash
mkdir -p models/fastwam/docs/history/plans models/fastwam/docs/history/specs
git mv docs/superpowers/plans/*fastwam*.md models/fastwam/docs/history/plans/
git mv docs/superpowers/specs/*fastwam*.md models/fastwam/docs/history/specs/
```

- [ ] **Step 2: Add an explicit history warning**

Create `models/fastwam/docs/README.md`:

```markdown
# FastWAM design history

The files under `history/` retain original implementation decisions and paths.
They are not deployment runbooks. Use [`../README.md`](../README.md) for the
current model-owned commands.
```

- [ ] **Step 3: Restore generic ROS documentation and prove no root FastWAM source remains**

```bash
git checkout origin/main -- docs/agilex_robot_ros_startup.md
test ! -e scripts/piper_fastwam_policy_server.py
test ! -e scripts/piper_fastwam_policy_server_supervised.py
test ! -e scripts/fastwam_piper_runtime.py
test ! -e scripts/websocket_policy_client.py
test ! -e configs/piper_fastwam_puzzle.yaml
test ! -e configs/piper_fastwam_puzzle_supervised_raw.yaml
test ! -e configs/piper_fastwam_puzzle_start_pose.json
```

- [ ] **Step 4: Commit history migration and duplicate removal**

```bash
git add docs models/fastwam
git commit -m "refactor: move FastWAM records under model directory"
```

### Task 5: Validate, deploy, and update Draft PR #2

**Files:**
- Deploy: `models/fastwam/server/*`
- Deploy: `models/fastwam/client/*`
- Update: Draft PR #2 only

**Interfaces:**
- Produces: verified relocated server, checked client when robot is reachable, and an updated draft PR based on current main.

- [ ] **Step 1: Run final tests, syntax checks, and diff/secret checks**

```bash
python -m unittest discover -s models/fastwam/server -p 'test_*.py' -v
python -m unittest discover -s models/fastwam/client -p 'test_*.py' -v
python -m py_compile models/fastwam/server/*.py models/fastwam/client/*.py
bash -n models/fastwam/server/run_piper_fastwam_server.sh
bash -n models/fastwam/server/run_piper_fastwam_supervised_raw_server.sh
git diff --check origin/main...HEAD
git diff --binary origin/main...HEAD | rg -n -i '(WANDB_API_KEY|HF_TOKEN|HUGGINGFACE_HUB_TOKEN|ghp_[A-Za-z0-9]+|github_pat_|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|password\s*=|api[_-]?key\s*=)' || true
```

- [ ] **Step 2: Deploy only the relocated server directory and run a no-listener preflight**

```bash
ssh -F /dev/null -p 7156 root@10.40.1.215 'mkdir -p /bh/zbh_self/projects/piper-openpi-real-robot/models/fastwam/server'
scp -P 7156 models/fastwam/server/* root@10.40.1.215:/bh/zbh_self/projects/piper-openpi-real-robot/models/fastwam/server/
ssh -F /dev/null -tt -p 7156 root@10.40.1.215 'source /bh/zbh_self/miniforge3/etc/profile.d/conda.sh && conda activate /bh/zbh_self/envs/fastwam && cd /bh/zbh_self/projects/piper-openpi-real-robot/models/fastwam/server && CUDA_VISIBLE_DEVICES=0 python -u piper_fastwam_policy_server.py --config piper_fastwam_puzzle.yaml --preflight-only'
```

Expected: FastWAM loads without binding port `7081` or commanding the robot.

- [ ] **Step 3: Deploy client only when robot is reachable and syntax-check it**

```bash
ssh agilex@10.13.0.155 'mkdir -p /home/agilex/piper-openpi-real-robot/models/fastwam/client'
scp models/fastwam/client/* agilex@10.13.0.155:/home/agilex/piper-openpi-real-robot/models/fastwam/client/
ssh agilex@10.13.0.155 '/home/agilex/miniconda3/envs/xrocs-env/bin/python -m py_compile /home/agilex/piper-openpi-real-robot/models/fastwam/client/*.py'
```

- [ ] **Step 4: Force-update only the Draft PR feature branch**

```bash
git push --force-with-lease origin agent/fastwam-piper-real-robot
gh pr edit 2 --repo bbbbhua1/piper-openpi-real-robot --title 'Add FastWAM Piper real-robot workflow' --body 'Adds FastWAM as a third isolated model workflow under models/fastwam/. Protected and supervised raw modes, client, configs, tests, and runbook are model-owned. No weights, datasets, recordings, or credentials are included.'
```
