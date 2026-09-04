#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/scripts:${PYTHONPATH:-}"

CLIENT_PYTHON="${LINGBOTVA_CLIENT_PYTHON:-}"
if [[ -z "${CLIENT_PYTHON}" ]]; then
  if [[ -x /home/agilex/miniconda3/envs/xrocs-env/bin/python ]]; then
    CLIENT_PYTHON=/home/agilex/miniconda3/envs/xrocs-env/bin/python
  else
    CLIENT_PYTHON=python3
  fi
fi

exec "${CLIENT_PYTHON}" "${SCRIPT_DIR}/lingbotva_piper_robot_client.py" "$@"
