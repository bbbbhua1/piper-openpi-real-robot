#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || ! "$1" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: run_pi05_puzzle_client.sh EXECUTION_HORIZON [smooth|interp] [CLIENT_ARGS...]" >&2
  echo "Example: bash scripts/run_pi05_puzzle_client.sh 50 smooth" >&2
  exit 2
fi

execution_horizon="$1"
shift
motion_mode="${1:-smooth}"
if [[ "$motion_mode" != "smooth" && "$motion_mode" != "interp" ]]; then
  echo "motion mode must be smooth or interp" >&2
  exit 2
fi
shift || true
motion_args=(--motion-mode "$motion_mode")
if [[ "$motion_mode" == "interp" ]]; then
  motion_args+=(--action-interp-factor 5 --action-chunk-blend-steps 20)
fi

source /opt/ros/noetic/setup.bash
source /home/agilex/agilex_ws/devel/setup.bash
source /home/agilex/cobot_magic/camera_ws/devel/setup.bash
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash

exec /home/agilex/miniconda3/envs/xrocs-env/bin/python \
  /home/agilex/piper-openpi-real-robot-20260831/scripts/websocket_policy_client.py \
  --uri ws://127.0.0.1:7081 \
  --instruction "Assemble the puzzle pieces on the table to complete the puzzle" \
  --source ros \
  --executor ros \
  --enable-on-start true \
  --execute-actions \
  --execution-horizon "${execution_horizon}" \
  "${motion_args[@]}" \
  --control-hz 10 \
  --compressed-images \
  --require-images \
  --camera-topic head=/camera_f/color/image_raw/compressed \
  --camera-topic left_wrist=/camera_l/color/image_raw/compressed \
  --camera-topic right_wrist=/camera_r/color/image_raw/compressed \
  --record-videos \
  --record-dir /home/agilex/piper-openpi-real-robot-20260831/recordings \
  --record-session-name "pi05-jigsaw-$(date +%Y%m%d_%H%M%S)" \
  --record-upload-host root@10.40.1.215 \
  --record-upload-port 7156 \
  --record-upload-dir /bh/zbh_ckp/runs/pi05/piper_jigsaw_real_robot \
  "$@"
