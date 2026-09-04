#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: run_pi05_domino_client.sh" >&2
  exit 2
fi

source /opt/ros/noetic/setup.bash
source /home/agilex/agilex_ws/devel/setup.bash
source /home/agilex/cobot_magic/camera_ws/devel/setup.bash
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash

exec /home/agilex/miniconda3/envs/xrocs-env/bin/python \
  /home/agilex/piper-openpi-real-robot-20260831/scripts/websocket_policy_client.py \
  --uri ws://127.0.0.1:18001 \
  --instruction "Arrange the dominoes on the table in a horizontal row and then knock them over" \
  --source ros \
  --executor ros \
  --enable-on-start true \
  --execute-actions \
  --control-hz 10 \
  --compressed-images \
  --require-images \
  --camera-topic head=/camera_f/color/image_raw/compressed \
  --camera-topic left_wrist=/camera_l/color/image_raw/compressed \
  --camera-topic right_wrist=/camera_r/color/image_raw/compressed \
  --record-videos \
  --record-dir /home/agilex/piper-openpi-real-robot-20260831/recordings \
  --record-session-name "pi05-domino-$(date +%Y%m%d_%H%M%S)" \
  --record-upload-host root@10.40.1.215 \
  --record-upload-port 7156 \
  --record-upload-dir /bh/zbh_ckp/runs/pi05/domino_real_robot
