#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/miniconda3/envs/dnazymes-gpu/bin/python
SCRIPT=/root/dnazymes/screening_pipeline/run_screening.py
LOG_DIR=/root/dnazymes/generated/screening/logs
mkdir -p "$LOG_DIR"

run_one() {
    local dataset="$1"
    local gpu="$2"
    local log="$LOG_DIR/${dataset}.log"
    echo "Starting $dataset on GPU $gpu -> $log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$SCRIPT" \
        --datasets "$dataset" \
        --batch-size 64 \
        --no-embeddings \
        2>&1 | tee "$log"
}

# Two GPUs: run pairs in parallel
run_one eds_pretrain_nofilter 0 &
run_one mfe_pretrain_nofilter 1 &
wait

run_one eds_ft_nofilter 0 &
run_one mfe_ft_nofilter 1 &
wait

echo "All screening jobs finished."
