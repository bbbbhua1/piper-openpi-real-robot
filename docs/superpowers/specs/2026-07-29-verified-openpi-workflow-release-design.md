# Verified OpenPI Real-Robot Workflow Release Design

## Goal

Update `bbbbhua1/piper-openpi-real-robot` so its `main` branch contains the
complete OpenPI/Piper workflow that was verified on the real robot, including
three-camera recording and upload, the required OpenPI prompt repack, and
copy-pasteable commands for communication testing and full execution.

## Source Of Truth

The implementation starts from GitHub `main` at commit
`cecdd4e80650be1e9507c9971d8cccbd1e08191a`. This preserves the newer,
already-committed robot initialization documentation.

The verified client implementation comes from the local working copy at:

```text
work/piper-openpi-real-robot/scripts/
```

The verified server implementation comes from:

```text
ssh -p 3763 root@10.40.1.215
/bh/media/section/iclr_proj/zbh/openpi/scripts/piper_openpi_policy_server.py
```

The server copy differs from the current repository copy by the required
OpenPI inference repack entry:

```python
"prompt": "prompt",
```

## Production Code Changes

### Camera recorder

Add `scripts/camera_video_recorder.py`. It records each compressed ROS camera
stream with an independent bounded queue and FFmpeg writer thread, so camera
callbacks never wait for video encoding or network transfer.

Each session contains:

```text
head.mp4
left_wrist.mp4
right_wrist.mp4
head.ffmpeg.log
left_wrist.ffmpeg.log
right_wrist.ffmpeg.log
manifest.json
```

The recorder keeps the complete session on the robot. After client shutdown it
uses SSH and SCP to upload the session directory to the development server. An
upload failure is reported but never deletes the robot-local copy.

### Robot WebSocket client

Update `scripts/websocket_policy_client.py` to:

- attach the recorder to the same three compressed ROS camera callbacks used
  by policy inference;
- expose recording directory, FPS, queue size, session name, upload host,
  upload port, and upload directory as CLI arguments;
- require compressed camera topics when recording is enabled;
- finalize all MP4 files and write the manifest during client cleanup;
- upload only after recording finalization;
- preserve the existing safety, home, hold, enable, dry-run, and action
  execution behavior.

Dry-run communication testing uses `--executor mock`, omits
`--execute-actions`, and sends exactly one real observation using
`--max-steps 1`. It must not publish robot action commands.

### OpenPI policy server

Update `scripts/piper_openpi_policy_server.py` with the verified
`"prompt": "prompt"` repack mapping. The task instruction received from the
robot client then reaches the trained OpenPI policy as its prompt.

`scripts/ws_policy_protocol.py` remains unchanged unless a byte-for-byte
comparison reveals a real difference.

## Documentation Changes

Update `README.md` without removing the existing verified `mode:=1` robot
initialization guidance.

The README will contain:

1. The three-machine architecture.
2. The new development-server SSH target:
   `ssh -p 3763 root@10.40.1.215`.
3. The `/bh/media/section/iclr_proj/zbh/openpi` source path and the working
   OpenPI runtime/PYTHONPATH combination.
4. The verified config:
   `pi05_cmap_all_tasks_v21`.
5. The verified checkpoint:
   `/bh/media/section/iclr_proj/zbh/openpi/training_runs/pi05_all_tasks_openpi_v21_8gpu_b512_80k/checkpoints/pi05_cmap_all_tasks_v21/pi05_all_tasks_openpi_v21_pi05_8gpu_b512_80k_correct_20260718/60000`.
6. The exact instruction:
   `Arrange the dominoes on the table in a horizontal row and then knock them over.`
7. A single-A800 policy-server command.
8. An SSH tunnel command using local port `18001`.
9. A one-step dry-run communication-test command that records and uploads all
   three cameras without moving the arms.
10. A full real-robot command that enables and executes actions while recording.
11. The robot-local recording root:
    `/home/agilex/piper-openpi-real-robot/recordings`.
12. The server recording root:
    `/bh/media/section/iclr_proj/zbh/piper_openpi_real_robot_videos`.
13. Shutdown behavior, including the two `Ctrl-C` presses needed when the
    client enters its hold loop and the fact that upload starts only after the
    hold loop exits.

Machine-specific values such as the local computer IP are called out explicitly
so future users know what must be changed when the network changes.

## Tests And Verification

Add `tests/test_camera_video_recorder.py` and `tests/__init__.py`.

Verification before publishing includes:

- all recorder unit tests pass;
- both Python entrypoints and the recorder compile with `py_compile`;
- the documented server command matches the actual CLI and the verified
  runtime on the development server;
- the documented robot command matches the deployed client CLI;
- the local working tree is clean after commits;
- the update is a fast-forward from the current GitHub `main`;
- the final GitHub `main` contains the expected files and commit.

The previously completed real-hardware evidence is recorded in the README:
three 1280×720 streams at 30 FPS, 150 frames per stream, zero dropped frames,
successful upload, and a successful single-A800 policy response with ten
waypoints per arm.

## Version Control And Publication

Implementation is performed in the clone of GitHub `main`, not in the
unrelated local snapshot history. Changes are split into reviewable commits:

1. the approved design;
2. production code and tests;
3. updated commands and workflow documentation.

After all verification passes, GitHub CLI pushes the commits directly to
`main`. No pull request, force push, history rewrite, tag, or release is
created.

If remote `main` advances before publication, publishing stops. The new remote
commits must first be fetched and integrated; the workflow never force-pushes
over concurrent work.
