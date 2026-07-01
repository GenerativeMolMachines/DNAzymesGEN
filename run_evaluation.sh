#!/bin/bash
set -euo pipefail

source /root/dnazymes/gpu_env.sh

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

CONDA_ENV="/root/miniconda3/envs/dnazymes-gpu"
SITE_PKGS="${CONDA_ENV}/lib/python3.10/site-packages"
NVIDIA_LIBS=$(find "${SITE_PKGS}/nvidia" -name 'lib' -type d 2>/dev/null | sort | tr '\n' ':')

OUTPUT="/root/dnazymes/checkpoints/evaluation_results.json"
rm -f "${OUTPUT}"

cd /root/dnazymes/improved_wgan_training

run_eval() {
  local label="$1"
  local ckpt="$2"
  env LD_LIBRARY_PATH="${NVIDIA_LIBS}${CONDA_ENV}/lib:/usr/lib/x86_64-linux-gnu" \
    "${PYTHON}" evaluate_checkpoints.py \
      --label "${label}" \
      --checkpoint "${ckpt}" \
      --sample-rounds 5 \
      --output "${OUTPUT}"
}

echo "=== EDS checkpoints ==="
run_eval "EDS pretrain (model-1400)" "/root/dnazymes/checkpoints/eds/model-1400"
run_eval "EDS finetune best/final (model-999)" "/root/dnazymes/checkpoints/eds_ft/model-999"

echo "=== MFE checkpoints ==="
run_eval "MFE pretrain (model-3000)" "/root/dnazymes/checkpoints/mfe/model-3000"
run_eval "MFE finetune best (model-800)" "/root/dnazymes/checkpoints/mfe_ft/model-800"
run_eval "MFE finetune final (model-999)" "/root/dnazymes/checkpoints/mfe_ft/model-999"

echo "=== Done. Results: ${OUTPUT} ==="
