#!/bin/bash
set -euo pipefail

source /root/dnazymes/gpu_env.sh

CONDA_ENV="/root/miniconda3/envs/dnazymes-gpu"
SITE_PKGS="${CONDA_ENV}/lib/python3.10/site-packages"
NVIDIA_LIBS=$(find "${SITE_PKGS}/nvidia" -name 'lib' -type d 2>/dev/null | sort | tr '\n' ':')

NUM_SEQUENCES="${NUM_SEQUENCES:-250000}"
MAX_MFE="${MAX_MFE:--10.0}"
MIN_LENGTH="${MIN_LENGTH:-20}"
MFE_WORKERS="${MFE_WORKERS:-16}"
BATCH_SIZE="${BATCH_SIZE:-64}"
BATCHES_PER_ROUND="${BATCHES_PER_ROUND:-64}"

run_one() {
  local gpu_id="$1"
  local label="$2"
  local checkpoint="$3"
  export CUDA_VISIBLE_DEVICES="${gpu_id}"
  echo "=== ${label} on GPU ${gpu_id} ==="
  env LD_LIBRARY_PATH="${NVIDIA_LIBS}${CONDA_ENV}/lib:/usr/lib/x86_64-linux-gnu" \
    "${PYTHON}" generate_sequences.py \
      --label "${label}" \
      --checkpoint "${checkpoint}" \
      --num-sequences "${NUM_SEQUENCES}" \
      --max-mfe "${MAX_MFE}" \
      --min-length "${MIN_LENGTH}" \
      --batch-size "${BATCH_SIZE}" \
      --batches-per-round "${BATCHES_PER_ROUND}" \
      --mfe-workers "${MFE_WORKERS}" \
      2>&1 | tee "/root/dnazymes/generated/${label}/generate.log"
}

mkdir -p /root/dnazymes/generated/{eds_pretrain,eds_ft,mfe_pretrain,mfe_ft}
cd /root/dnazymes/improved_wgan_training

# Wave 1: one job per GPU
run_one 0 eds_pretrain /root/dnazymes/checkpoints/eds/model-1400 &
pid1=$!
run_one 1 mfe_pretrain /root/dnazymes/checkpoints/mfe/model-3000 &
pid2=$!
wait "$pid1" "$pid2"

# Wave 2
run_one 0 eds_ft /root/dnazymes/checkpoints/eds_ft/model-999 &
pid3=$!
run_one 1 mfe_ft /root/dnazymes/checkpoints/mfe_ft/model-999 &
pid4=$!
wait "$pid3" "$pid4"

echo "All generation jobs completed."
