# LingBot-VA Piper deployment

This directory contains an independent LingBot-VA server and Piper robot
client. It does not change the existing OpenPI or FastWAM deployment paths.

## Server

The default checkpoint is:

```text
/bh/media/ZBH/lingbot-va-realrobot-ft-step17000-inference-20260831
```

Run a model-loading preflight:

```bash
bash models/lingbotva/server/run_piper_lingbotva_server.sh --preflight-only
```

Start the server on `127.0.0.1:7084`:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash models/lingbotva/server/run_piper_lingbotva_server.sh
```

The default configuration keeps the model, VAE and text encoder on the 80GB
GPU. This avoids the multi-minute first-request CPU/GPU transfer cost of
offload mode.

Create the same SSH tunnel pattern used by the existing deployments, mapping a
robot-reachable local port to server port `7084`.

## Robot dry-run

After the normal Piper ROS and camera startup, run one request without
publishing an action:

```bash
bash models/lingbotva/client/run_lingbotva_robot_client.sh \
  --uri ws://10.13.10.63:18004 \
  --max-steps 1 \
  --source ros \
  --executor ros \
  --compressed-images \
  --require-images \
  --camera-topic head=/camera_f/color/image_raw/compressed \
  --camera-topic left_wrist=/camera_l/color/image_raw/compressed \
  --camera-topic right_wrist=/camera_r/color/image_raw/compressed
```

The client sends `reset=true` on the first request. The first LingBot-VA result
contains a 16-waypoint conditioning frame followed by 16 executable
waypoints, so the client executes the latter half and samples one key frame.
Later 32-waypoint results are executed in two 16-waypoint groups, with one key
frame sampled after each group. The server uses those key frames and the
previous action grid to update the KV cache before the next inference.

## Physical execution

Only add `--execute-actions` after the dry-run response, start pose and joint
limits have been checked. Use `--execution-horizon N` to execute the first
`N` waypoints from each 32-point model chunk.

```bash
bash models/lingbotva/client/run_lingbotva_robot_client.sh \
  --uri ws://10.13.10.63:18004 \
  --execute-actions \
  --execution-horizon 32 \
  --compressed-images \
  --require-images \
  --camera-topic head=/camera_f/color/image_raw/compressed \
  --camera-topic left_wrist=/camera_l/color/image_raw/compressed \
  --camera-topic right_wrist=/camera_r/color/image_raw/compressed
```

The default instruction is `Complete the jigsaw puzzle on the table.` and can
be overridden with `--instruction`.
