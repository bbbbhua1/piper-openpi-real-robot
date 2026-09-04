#!/usr/bin/env bash
set -euo pipefail

source /bh/zbh_self/miniforge3/etc/profile.d/conda.sh
conda activate "${FASTWAM_ENV:-/bh/zbh_self/envs/fastwam}"
cd "$(dirname "$0")"

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/bh/zbh_ckp/models/fastwam}"

exec python -u piper_fastwam_policy_server_supervised.py \
  --config piper_fastwam_puzzle_supervised_raw.yaml "$@"
