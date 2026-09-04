#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || ! "$1" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: run_fastwam_domino_client.sh EXECUTION_HORIZON [CLIENT_ARGS...]" >&2
  echo "Example: bash scripts/run_fastwam_domino_client.sh 32" >&2
  exit 2
fi

execution_horizon="$1"
shift

source /opt/ros/noetic/setup.bash
source /home/agilex/agilex_ws/devel/setup.bash
source /home/agilex/cobot_magic/camera_ws/devel/setup.bash
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash

exec /home/agilex/miniconda3/envs/xrocs-env/bin/python \
  /home/agilex/piper-openpi-real-robot-20260831/scripts/websocket_policy_client.py \
  --uri ws://127.0.0.1:18007 \
  --instruction "Arrange the dominoes on the table in a horizontal row and then knock them over" \
  --source ros \
  --executor ros \
  --enable-on-start true \
  --execute-actions \
  --home-on-exit \
  --execution-horizon "${execution_horizon}" \
  --action-interp-factor 10 \
  --action-chunk-blend-steps 20 \
  --control-hz 10 \
  --compressed-images \
  --transport-image-profile fastwam \
  --require-images \
  --camera-topic head=/camera_f/color/image_raw/compressed \
  --camera-topic left_wrist=/camera_l/color/image_raw/compressed \
  --camera-topic right_wrist=/camera_r/color/image_raw/compressed \
  --record-videos \
  --record-dir /home/agilex/piper-openpi-real-robot-20260831/recordings \
  --record-session-name "fastwam-domino-$(date +%Y%m%d_%H%M%S)" \
  --record-upload-host root@10.40.1.215 \
  --record-upload-port 7156 \
  --record-upload-dir /bh/zbh_ckp/runs/fastwam/domino_real_robot \
  "$@"
