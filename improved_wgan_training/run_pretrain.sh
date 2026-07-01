#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${PROJECT_ROOT}/scripts/env/gpu_env.sh"

DATASET="$1"
GPU_ID="$2"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/${DATASET}"

mkdir -p "${CHECKPOINT_DIR}/plots" "${CHECKPOINT_DIR}/samples"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
cd "${SCRIPT_DIR}"
exec "${PYTHON}" gan_language.py \
    --dataset "${DATASET}" \
    --mode pretrain \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --iters 3600 \
    --save-interval 200 \
    --max-n-examples 0 \
    2>&1 | tee "${CHECKPOINT_DIR}/train.log"
