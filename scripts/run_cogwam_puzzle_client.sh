#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
execution_horizon="${1:-32}"
motion_mode="${2:-smooth}"
if [[ $# -gt 2 || ! "$execution_horizon" =~ ^[1-9][0-9]*$ || "$execution_horizon" -gt 32 || $((execution_horizon % 4)) -ne 0 || ("$motion_mode" != "smooth" && "$motion_mode" != "interp") ]]; then
  echo "Usage: run_cogwam_puzzle_client.sh [EXECUTION_HORIZON: 4|8|...|32] [smooth|interp]" >&2
  exit 2
fi
if [[ "$motion_mode" == "interp" ]]; then
  export COGWAM_CLIENT_EXTRA_ARGS="--motion-mode interp --action-interp-factor 5 --action-chunk-blend-steps 15"
else
  export COGWAM_CLIENT_EXTRA_ARGS="--motion-mode smooth"
fi
source /opt/ros/noetic/setup.bash
source /home/agilex/agilex_ws/devel/setup.bash
source /home/agilex/cobot_magic/camera_ws/devel/setup.bash
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash

exec /home/agilex/miniconda3/envs/xrocs-env/bin/python \
  /home/agilex/piper-openpi-real-robot-20260831/scripts/websocket_policy_client.py \
  --uri ws://127.0.0.1:7085 \
  --instruction "Assemble the puzzle pieces on the table to complete the puzzle" \
  --source ros --executor ros --enable-on-start true --execute-actions --reset-before-start \
  --execution-horizon "$execution_horizon" --control-hz 30 --compressed-images --require-images \
  ${COGWAM_CLIENT_EXTRA_ARGS:-} \
  --camera-topic head=/camera_f/color/image_raw/compressed \
  --camera-topic left_wrist=/camera_l/color/image_raw/compressed \
  --camera-topic right_wrist=/camera_r/color/image_raw/compressed \
  --record-videos --record-dir /home/agilex/piper-openpi-real-robot-20260831/recordings \
  --record-session-name "cogwam-jigsaw-$(date +%Y%m%d_%H%M%S)" \
  --record-upload-host root@10.40.1.215 --record-upload-port 7156 \
  --record-upload-dir /bh/zbh_ckp/runs/cogwam/piper_jigsaw_real_robot
