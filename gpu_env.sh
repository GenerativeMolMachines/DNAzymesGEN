#!/bin/bash
CONDA_ENV="/root/miniconda3/envs/dnazymes-gpu"
SITE_PKGS="${CONDA_ENV}/lib/python3.10/site-packages"
NVCC_BIN="${SITE_PKGS}/nvidia/cuda_nvcc/bin"

# Stable lib order (cudnn/cublas before conda lib)
NVIDIA_LIBS=$(find "${SITE_PKGS}/nvidia" -name 'lib' -type d 2>/dev/null | sort | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${CONDA_ENV}/lib:/usr/lib/x86_64-linux-gnu"
export PATH="${NVCC_BIN}:${PATH}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CONDA_ENV}/lib"
export TF_CPP_MIN_LOG_LEVEL=2
export TF_CUDNN_USE_AUTOTUNE=0
export TF_ENABLE_ONEDNN_OPTS=0
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"
export PYTHONUNBUFFERED=1
export PYTHON="${CONDA_ENV}/bin/python"
