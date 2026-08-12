# FastWAM model-directory repository layout

## Purpose

Integrate the completed FastWAM Piper real-robot workflow into the repository
structure introduced on `main`: every model has an isolated directory with
its own server, client, tests, configuration, and deployment instructions.
FastWAM becomes a third peer model next to `openpi_pi05` and `uva_dit`.

## Target layout

```text
models/fastwam/
├── README.md
├── server/
│   ├── fastwam_piper_runtime.py
│   ├── piper_fastwam_policy_server.py
│   ├── piper_fastwam_policy_server_supervised.py
│   ├── run_piper_fastwam_server.sh
│   ├── run_piper_fastwam_supervised_raw_server.sh
│   ├── piper_fastwam_puzzle.yaml
│   ├── piper_fastwam_puzzle_supervised_raw.yaml
│   ├── diagnose_fastwam_validation_boundaries.py
│   └── test_*.py
├── client/
│   ├── websocket_policy_client.py
│   ├── ws_policy_protocol.py
│   ├── camera_video_recorder.py
│   ├── piper_start_pose.py
│   ├── piper_fastwam_puzzle_start_pose.json
│   └── test_*.py
└── docs/
    └── FastWAM-specific design and troubleshooting records
```

The generic repository documentation remains under the root `docs/` directory.
Only FastWAM-specific documentation moves into `models/fastwam/docs/`.

## Server boundary

`models/fastwam/server/` contains both explicit FastWAM policy modes:

- `piper_fastwam_policy_server.py` is the default protected deployment. It
  maps FastWAM outputs to Piper order, applies verified physical-boundary
  projection, and applies the configured per-waypoint delta limit.
- `piper_fastwam_policy_server_supervised.py` is an opt-in, continuously
  supervised diagnostic deployment. It accepts only finite, correctly-shaped
  model output but forwards finite waypoints without server-side projection or
  delta limiting.

Their launchers and YAML configurations stay alongside the server implementation
so a selected mode can be deployed as one self-contained server directory.

## Client boundary

`models/fastwam/client/` contains the complete robot-side copy needed by
FastWAM: WebSocket policy client, protocol adapter, camera recording,
task-specific start-pose movement and configuration, plus its client tests.
It intentionally does not share imports with another model's client; models
can be deployed and updated independently even when code is similar.

The client continues to provide FastWAM image resize before WebSocket,
three-camera recording, task-specific start-pose reset, rolling action-chunk
execution, hardware-fault handling, and `Ctrl-C` hold behavior.

## Documentation

The root `README.md` will mirror the current two-model index:

- add `models/fastwam/` to the directory tree and model table;
- add a FastWAM three-terminal dry-run section with direct links to the
  detailed model README; and
- preserve the existing OpenPI and UVA-DiT instructions unchanged.

`models/fastwam/README.md` will be the source of truth for FastWAM. It will
give separate, copyable Terminal 1 commands for protected and supervised raw
server modes; shared Terminal 2 tunnel and Terminal 3 robot dry-run commands;
then task start-pose, full execution, recording, stop, checkpoint, deployment,
and troubleshooting instructions. It will make the selected 7156 FastWAM
development machine and deployed robot paths explicit.

## Migration and compatibility

The implementation branch will first be rebased onto current `origin/main`.
FastWAM source files will be moved from the old repository-root `scripts/`,
`configs/`, and `tests/` locations into `models/fastwam/`. Imports, launchers,
tests, documentation links, and remote deployment commands will be updated to
match their new locations.

The old root FastWAM copies will be removed in the same change so no duplicate
source of truth remains. Files belonging to `models/openpi_pi05/`,
`models/uva_dit/`, root `reference/`, and general `docs/` are not modified
beyond adding FastWAM navigation to the root README.

No weights, base models, data, recordings, private addresses beyond the
already-public deployment examples, credentials, or API keys enter Git.

## Verification

Before updating the draft PR:

1. Run the full Python test discovery from the repository root.
2. Syntax-check both FastWAM servers and both server launchers.
3. Verify all FastWAM imports work from `models/fastwam/server` and client
   imports work from `models/fastwam/client`.
4. Inspect `git diff --check`, the change list relative to current `main`, and
   a secret-pattern scan.
5. On the 7156 development machine, deploy only the relocated server directory
   and run the FastWAM-environment syntax/config preflight without binding the
   policy port or commanding the robot.
