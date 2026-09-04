#!/usr/bin/env bash
set -euo pipefail

source /bh/zbh_self/miniforge3/etc/profile.d/conda.sh
conda activate "${LINGBOT_VA_ENV:-/bh/zbh_self/envs/lingbot-va-official}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LINGBOT_VA_ROOT="${LINGBOT_VA_ROOT:-/bh/zbh_self/projects/lingbot-va-official}"
export PYTHONPATH="${LINGBOT_VA_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec python -u "${SCRIPT_DIR}/lingbotva_piper_policy_server.py" \
  --config "${SCRIPT_DIR}/piper_lingbotva_realrobot.yaml" "$@"
