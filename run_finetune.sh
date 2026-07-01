#!/bin/bash
set -euo pipefail

GPU_ID="${1:?Usage: run_finetune.sh GPU_ID CHECKPOINT_DIR RESTORE_CKPT LOG_FILE}"
CHECKPOINT_DIR="${2:?}"
RESTORE_CKPT="${3:?}"
LOG_FILE="${4:?}"

ITERS="${ITERS:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-100}"

CONDA_ENV="/root/miniconda3/envs/dnazymes-gpu"
SITE_PKGS="${CONDA_ENV}/lib/python3.10/site-packages"
NVIDIA_LIBS=$(find "${SITE_PKGS}/nvidia" -name 'lib' -type d 2>/dev/null | tr '\n' ':')

export PY="${CONDA_ENV}/bin/python"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib"
export PATH="/root/miniconda3/envs/hw-train-gpu/lib/python3.11/site-packages/triton/backends/nvidia/bin:${PATH}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CONDA_ENV}/lib"
export PYTHONUNBUFFERED=1
export TF_CUDNN_USE_AUTOTUNE=0
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

mkdir -p "${CHECKPOINT_DIR}/plots" "${CHECKPOINT_DIR}/samples"
mkdir -p "$(dirname "${LOG_FILE}")"

cd /root/dnazymes/improved_wgan_training

env LD_LIBRARY_PATH="${NVIDIA_LIBS}/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}" \
  "${PY}" gan_language.py \
  --dataset sequence_craft \
  --mode finetune \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --restore-checkpoint "${RESTORE_CKPT}" \
  --iters "${ITERS}" \
  --save-interval "${SAVE_INTERVAL}" \
  --sample-interval "${SAMPLE_INTERVAL}" \
  2>&1 | tee "${LOG_FILE}"
