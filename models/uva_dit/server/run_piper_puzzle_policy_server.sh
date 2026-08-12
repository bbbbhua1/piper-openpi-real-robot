#!/usr/bin/env bash
# Launch a supported Puzzle UVA-DiT checkpoint behind the unchanged Piper WS API.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_ROOT="$(cd "${HERE}/../../.." && pwd)"
SERVER_SCRIPT="${HERE}/piper_puzzle_policy_server.py"
UVA_DIT_ROOT="${UVA_DIT_ROOT:-}"
cd "${WORKFLOW_ROOT}"

PYTHON="${PYTHON:-/bh/zbh_self/envs/uva-dit-piper/bin/python}"
GPU="${GPU:-0}"
PORT="${PORT:-7081}"
HOST="${HOST:-0.0.0.0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
MOCK_POLICY="${MOCK_POLICY:-0}"

# Model-time defaults.  The 32-step model chunk can be returned in smaller
# legacy responses, or the first 24 actions can be used before the next replan.
EXECUTE_STEPS="${EXECUTE_STEPS:-4}"
ACTION_DT="${ACTION_DT:-0.0333}"
REPLAN_EVERY_CALL="${REPLAN_EVERY_CALL:-0}"
VIDEO_STEPS="${VIDEO_STEPS:-25}"
ACTION_STEPS="${ACTION_STEPS:-25}"
INFER_SEED="${INFER_SEED:-42}"
TASK_PROMPT="${TASK_PROMPT:-拼图（圆形）}"
INACTIVE_ACTION_MODE="${INACTIVE_ACTION_MODE:-hold}"
MAX_JOINT_STEP="${MAX_JOINT_STEP:-0.0}"
STATE_OOB_MODE="${STATE_OOB_MODE:-clip}"
# The action latent is integrated in fp32 so sampled waypoints are not snapped
# onto the coarse bf16 grid; smoothing then removes the residual sampling noise
# from the whole 32-step chunk at once, which costs no phase lag.
ACTION_LATENT_FP32="${ACTION_LATENT_FP32:-1}"
ACTION_SMOOTH_WINDOW="${ACTION_SMOOTH_WINDOW:-11}"
ACTION_SMOOTH_POLY="${ACTION_SMOOTH_POLY:-2}"
ACTION_SMOOTH_GRIPPERS="${ACTION_SMOOTH_GRIPPERS:-0}"
ACTION_OUTPUT_CLIP="${ACTION_OUTPUT_CLIP:-0}"
ALLOW_MISSING_IMAGES="${ALLOW_MISSING_IMAGES:-0}"
RECORD_DIR="${RECORD_DIR:-/bh/zbh_ckp/runs/piper_uva_dit/${RUN_TAG}}"
RAW_ACTION_STEPS="${RAW_ACTION_STEPS:-8}"
RAW_ACTION_HDF5="${RAW_ACTION_HDF5:-}"

[[ -x "${PYTHON}" ]] || {
    echo "ERROR: Python runtime not found: ${PYTHON}" >&2
    echo "       Create or finish /bh/zbh_self/envs/uva-dit-piper first." >&2
    exit 2
}

COMMON_ARGS=(
    --host "${HOST}"
    --port "${PORT}"
    --execute-steps "${EXECUTE_STEPS}"
    --action-dt "${ACTION_DT}"
    --video-steps "${VIDEO_STEPS}"
    --action-steps "${ACTION_STEPS}"
    --seed "${INFER_SEED}"
    --expected-instruction "${TASK_PROMPT}"
    --inactive-action-mode "${INACTIVE_ACTION_MODE}"
    --max-joint-step "${MAX_JOINT_STEP}"
    --state-oob-mode "${STATE_OOB_MODE}"
    --raw-action-steps "${RAW_ACTION_STEPS}"
    --action-smooth-window "${ACTION_SMOOTH_WINDOW}"
    --action-smooth-poly "${ACTION_SMOOTH_POLY}"
)

if [[ "${ACTION_SMOOTH_GRIPPERS}" == "1" ]]; then
    COMMON_ARGS+=(--action-smooth-grippers)
fi

if [[ -n "${RAW_ACTION_HDF5}" ]]; then
    COMMON_ARGS+=(--raw-action-hdf5 "${RAW_ACTION_HDF5}")
fi

if [[ "${REPLAN_EVERY_CALL}" == "1" ]]; then
    COMMON_ARGS+=(--replan-every-call)
fi
if [[ "${ACTION_OUTPUT_CLIP}" == "1" ]]; then
    COMMON_ARGS+=(--action-output-clip)
fi
if [[ "${ALLOW_MISSING_IMAGES}" == "1" ]]; then
    COMMON_ARGS+=(--allow-missing-images)
fi

if [[ "${MOCK_POLICY}" == "1" ]]; then
    echo "========================================================"
    echo " Piper UVA-DiT protocol mock"
    echo " Host/port: ${HOST}:${PORT}"
    echo " Execute/dt: ${EXECUTE_STEPS} / ${ACTION_DT}"
    echo "========================================================"
    exec "${PYTHON}" "${SERVER_SCRIPT}" "${COMMON_ARGS[@]}" --mock-policy
fi

: "${UVA_DIT_ROOT:?Set UVA_DIT_ROOT to the full UVA_dit repository}"
[[ -f "${UVA_DIT_ROOT}/realmachine_deploy_v2/tianyi/infer_server_pi07_ar.py" ]] || {
    echo "ERROR: UVA-DiT source checkout not found: ${UVA_DIT_ROOT}" >&2
    echo "       Expected realmachine_deploy_v2/tianyi/infer_server_pi07_ar.py." >&2
    exit 2
}
cd "${UVA_DIT_ROOT}"

: "${RUN_DIR:?Set RUN_DIR to the selected Puzzle training run directory}"
CKPT_NAME="${CKPT_NAME:-checkpoint_latest}"
CHECKPOINT="${RUN_DIR%/}/${CKPT_NAME}"
: "${NORM_STATS:?Set NORM_STATS to the paired Puzzle train406 stats .pt file}"
: "${PRETRAINED:?Set PRETRAINED to the UniDiT base model directory}"
: "${QWEN35:?Set QWEN35 to the Qwen3.5-9B directory}"
QWEN35_ADAPTER="${QWEN35_ADAPTER:-${CHECKPOINT}/qwen3vl_proj.pt}"

MODEL_ARGS=(
    --checkpoint "${CHECKPOINT}"
    --pretrained "${PRETRAINED}"
    --qwen35 "${QWEN35}"
    --qwen35-adapter "${QWEN35_ADAPTER}"
    --norm-stats "${NORM_STATS}"
)

echo "========================================================"
echo " Piper UVA-DiT Puzzle policy server"
echo " Python:      ${PYTHON}"
echo " GPU/port:    ${GPU}/${PORT}"
echo " Checkpoint:  ${CHECKPOINT}"
echo " Norm stats:  ${NORM_STATS}"
echo " Base/Qwen:   ${PRETRAINED} / ${QWEN35}"
echo " Adapter:     ${QWEN35_ADAPTER}"
echo " Layout:      Piper 7+7 <-> physical14 <-> model32 padded"
echo " AR/history:  checkpoint-selected visual history; measured P→P state history16"
echo " Execute:     ${EXECUTE_STEPS}/32 at dt=${ACTION_DT}; replan=${REPLAN_EVERY_CALL}"
echo " Left side:   ${INACTIVE_ACTION_MODE}; right side active"
echo " Action num:  latent fp32=${ACTION_LATENT_FP32}; SavGol w=${ACTION_SMOOTH_WINDOW} p=${ACTION_SMOOTH_POLY} grippers=${ACTION_SMOOTH_GRIPPERS}"
echo " Record dir:  ${RECORD_DIR}"
echo " Raw actions: ${RAW_ACTION_HDF5:-disabled} (prefix=${RAW_ACTION_STEPS})"
echo "========================================================"

# This check reads only files and config; it initializes neither CUDA nor the model.
UVA_DIT_ROOT="${UVA_DIT_ROOT}" PYTHONPATH="${UVA_DIT_ROOT}:${PYTHONPATH:-}" \
    "${PYTHON}" "${SERVER_SCRIPT}" "${COMMON_ARGS[@]}" "${MODEL_ARGS[@]}" --config-only

if [[ "${CONFIG_ONLY:-0}" == "1" ]]; then
    exit 0
fi

mkdir -p "${RECORD_DIR}"
export UVA_DIT_ROOT
export PYTHONPATH="${UVA_DIT_ROOT}:${PYTHONPATH:-}"
export ACTION_LATENT_FP32
export HF_HOME="${HF_HOME:-/bh/zbh_ckp/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/bh/zbh_ckp/models/torch}"
# The base Pi07 engine otherwise writes this generated prompt trace into the
# repository CWD.  Keep per-run diagnostics with the action JSONL instead.
export INFER_PROMPT_LOG="${INFER_PROMPT_LOG:-${RECORD_DIR}/infer_prompt_exec.jsonl}"

exec env CUDA_VISIBLE_DEVICES="${GPU}" \
    "${PYTHON}" "${SERVER_SCRIPT}" "${COMMON_ARGS[@]}" "${MODEL_ARGS[@]}" \
    --record-dir "${RECORD_DIR}"
