#!/usr/bin/env bash
# Launch Puzzle 0811-C (no image AR) with the Piper WebSocket policy server.
# Environment variables remain overridable for intentional experiments.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_DIR="${RUN_DIR:-/bh/media/unify/joyzhang/checkpoints_unidit_ur_usb/puzzle_0811_C_from_Abest_new0805_train406_eval20_p2p_noar_rightactionloss_rightgripsampler_lr1e6_12k_port23456}"
export CKPT_NAME="${CKPT_NAME:-checkpoint_step_00024000}"
export NORM_STATS="${NORM_STATS:-/bh/media/unify/joyzhang/checkpoints_unidit_ur_usb/pertask_norm_stats_puzzle_0805_train406_mask14_p2p_armsgrip_gripfull.pt}"
export PRETRAINED="${PRETRAINED:-/bh/media/unify/unidit/pretrained_models/unidit-base}"
export QWEN35="${QWEN35:-/bh/media/unify/unidit/pretrained_models/Qwen3.5-9B}"

# Generate 32 absolute joint-position waypoints, execute only the first 24,
# then use the next observation/state for a new prediction.  The paired client
# sends the latest 16 measured robot states as C's causal action-condition
# history; predicted waypoints are never reused as history.
export EXECUTE_STEPS="${EXECUTE_STEPS:-24}"
export REPLAN_EVERY_CALL="${REPLAN_EVERY_CALL:-0}"
export RUN_TAG="${RUN_TAG:-piper_puzzle_0811_c_noar_$(date +%Y%m%d_%H%M%S)}"
export RAW_ACTION_STEPS="${RAW_ACTION_STEPS:-8}"
export RAW_ACTION_HDF5="${RAW_ACTION_HDF5:-/bh/zbh_ckp/datasets/piper_uva_dit_raw_actions/raw_model_actions_${RUN_TAG}.hdf5}"

exec bash "${HERE}/run_piper_puzzle_policy_server.sh"
