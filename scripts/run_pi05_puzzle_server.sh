#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=/bh/zbh_self/projects/openpi-official/src:/bh/zbh_self/projects/openpi-official/packages/openpi-client/src:${PYTHONPATH:-}

exec /bh/zbh_self/envs/openpi-official/bin/python -u \
  scripts/piper_openpi_domino_policy_server.py \
  --config configs/piper_openpi_puzzle.yaml \
  "$@"
