#!/bin/bash
# ==============================================================================
# IIT Delhi PADUM HPC - GPU Deep Learning Environment Setup (PyTorch + Transformers)
# Target: Skylake V100 (32 GB) / SCAI A100 (80 GB)
# ==============================================================================
set -euo pipefail

SCRATCH_BASE="/scratch/civil/btech/ce1240901"
PROJ_DIR="${SCRATCH_BASE}/amazon_ml"
MAMBA_BIN="${PROJ_DIR}/bin/micromamba"
GPU_ENV="${PROJ_DIR}/env_gpu"

export CONDA_PKGS_DIRS="${PROJ_DIR}/conda_pkgs"
export PIP_CACHE_DIR="${PROJ_DIR}/pipcache"
export XDG_CACHE_HOME="${PROJ_DIR}/cache"

if [ -f "$HOME/proxy.sh" ]; then
    nohup bash "$HOME/proxy.sh" > "${PROJ_DIR}/logs/proxy_gpu_setup.log" 2>&1 &
    sleep 5
fi
export http_proxy="http://proxy22.iitd.ac.in:3128"
export https_proxy="http://proxy22.iitd.ac.in:3128"

echo "=== Creating GPU Deep Learning Environment ==="
if [ ! -d "${GPU_ENV}/bin" ]; then
    "${MAMBA_BIN}" create -y -p "${GPU_ENV}" \
        -c pytorch -c nvidia -c conda-forge \
        python=3.10 \
        pytorch \
        torchvision \
        torchaudio \
        pytorch-cuda=11.8 \
        scikit-learn \
        pandas \
        numpy \
        scipy \
        lightgbm \
        rapidfuzz \
        joblib \
        pyarrow
fi

echo "=== Installing HuggingFace and Sentence-Transformers ==="
"${GPU_ENV}/bin/pip" install --no-cache-dir \
    transformers \
    sentence-transformers \
    accelerate \
    sparse_dot_topn \
    tqdm

echo "=== Testing PyTorch CUDA in GPU Environment ==="
"${GPU_ENV}/bin/python" -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Name:', torch.cuda.get_device_name(0))
    print('VRAM (GB):', torch.cuda.get_device_properties(0).total_memory / 1e9)
"
echo "=== GPU Environment Setup Complete! ==="
