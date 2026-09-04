#!/usr/bin/env bash
set -euo pipefail

source /bh/zbh_self/miniforge3/etc/profile.d/conda.sh
conda activate /bh/zbh_self/envs/fastwam
cd "$(dirname "$0")/.."

export DIFFSYNTH_MODEL_BASE_PATH=/bh/zbh_ckp/models/fastwam

exec python -u scripts/piper_fastwam_policy_server.py \
  --config configs/piper_fastwam_domino.yaml
