#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/../env/gpu_env.sh"

SITE_PKGS="${CONDA_ENV}/lib/python3.10/site-packages"
NVIDIA_LIBS=$(find "${SITE_PKGS}/nvidia" -name 'lib' -type d 2>/dev/null | sort | tr '\n' ':')

NUM_SEQUENCES="${NUM_SEQUENCES:-250000}"
MIN_LENGTH="${MIN_LENGTH:-20}"
MAX_LENGTH="${MAX_LENGTH:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BATCHES_PER_ROUND="${BATCHES_PER_ROUND:-64}"
# Unfiltered runs only need GPU; skip NUPACK for speed.
NOFILTER_GPU="${NOFILTER_GPU:-0}"

run_one() {
  local gpu_id="$1"
  local label="$2"
  local checkpoint="$3"
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
  echo "=== ${label} (no MFE filter) on GPU ${gpu_id} ==="
  env LD_LIBRARY_PATH="${NVIDIA_LIBS}${CONDA_ENV}/lib:/usr/lib/x86_64-linux-gnu" \
    "${PYTHON}" generate_sequences.py \
      --label "${label}" \
      --checkpoint "${checkpoint}" \
      --num-sequences "${NUM_SEQUENCES}" \
      --min-length "${MIN_LENGTH}" \
      --max-length "${MAX_LENGTH}" \
      --batch-size "${BATCH_SIZE}" \
      --batches-per-round "${BATCHES_PER_ROUND}" \
      --no-mfe-filter \
      --skip-mfe-calc \
      2>&1 | tee "${PROJECT_ROOT}/generated/${label}/generate.log"
}

mkdir -p \
  "${PROJECT_ROOT}/generated/eds_pretrain_nofilter" \
  "${PROJECT_ROOT}/generated/mfe_pretrain_nofilter" \
  "${PROJECT_ROOT}/generated/eds_ft_nofilter" \
  "${PROJECT_ROOT}/generated/mfe_ft_nofilter" \
  "${PROJECT_ROOT}/generated/sequence_craft_nofilter"
cd "${PROJECT_ROOT}/improved_wgan_training"

run_one "${NOFILTER_GPU}" eds_pretrain_nofilter "${PROJECT_ROOT}/checkpoints/eds/model-1400"
run_one "${NOFILTER_GPU}" mfe_pretrain_nofilter "${PROJECT_ROOT}/checkpoints/mfe/model-3000"
run_one "${NOFILTER_GPU}" eds_ft_nofilter "${PROJECT_ROOT}/checkpoints/eds_ft/model-999"
run_one "${NOFILTER_GPU}" mfe_ft_nofilter "${PROJECT_ROOT}/checkpoints/mfe_ft/model-999"
run_one "${NOFILTER_GPU}" sequence_craft_nofilter "${PROJECT_ROOT}/checkpoints/sequence_craft/model-700"

echo "All unfiltered generation jobs completed."
