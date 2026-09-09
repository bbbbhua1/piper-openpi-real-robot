#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /bh/zbh_self/miniforge3/etc/profile.d/conda.sh
conda activate "${COGWAM_ENV:-/bh/zbh_self/envs/cogwam}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="disabled"
export WANDB_ENTITY="${WANDB_ENTITY:-inference}"
export WANDB_PROJECT="${WANDB_PROJECT:-cogwam-inference}"
export PYTHONPATH="/bh/zbh_self/projects/CogWAM/qwen-vl-progress-estimation/qwen-vl-finetune:/bh/zbh_self/projects/CogWAM/src:/bh/zbh_self/projects/CogWAM:${PYTHONPATH:-}"
exec python -u scripts/cogwam_piper_policy_server.py --config configs/piper_cogwam_jigsaw.yaml "$@"
