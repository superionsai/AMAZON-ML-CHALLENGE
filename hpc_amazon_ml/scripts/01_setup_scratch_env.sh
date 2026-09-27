#!/bin/bash
# ==============================================================================
# IIT Delhi PADUM HPC - Amazon ML Challenge Environment Setup
# Scratch Directory: /scratch/civil/btech/ce1240901/amazon_ml
# Constraints: CentOS 7 (glibc 2.17), Zero Home writes, Authenticated Proxy
# ==============================================================================
set -euo pipefail

SCRATCH_BASE="/scratch/civil/btech/ce1240901"
PROJ_DIR="${SCRATCH_BASE}/amazon_ml"
BAKA_DIR="${SCRATCH_BASE}/Baka"

echo "=== [1/6] Initializing Directories in Scratch ==="
mkdir -p "${PROJ_DIR}"/{bin,env,env_gpu,dataset,models,output,logs,conda_pkgs,pipcache,cache}

# Redirect all caches away from full /home
export CONDA_PKGS_DIRS="${PROJ_DIR}/conda_pkgs"
export PIP_CACHE_DIR="${PROJ_DIR}/pipcache"
export XDG_CACHE_HOME="${PROJ_DIR}/cache"
export TORCH_HOME="${PROJ_DIR}/cache/torch"
export HF_HOME="${PROJ_DIR}/cache/huggingface"

echo "=== [2/6] Activating IITD Campus Internet Proxy ==="
if [ -f "$HOME/proxy.sh" ]; then
    nohup bash "$HOME/proxy.sh" > "${PROJ_DIR}/logs/proxy_setup.log" 2>&1 &
    sleep 6
fi
export http_proxy="http://proxy22.iitd.ac.in:3128"
export https_proxy="http://proxy22.iitd.ac.in:3128"
echo "Proxy set to: ${http_proxy}"
curl -sI https://example.com | head -n 1 || echo "Warning: Outbound proxy check returned non-200"

echo "=== [3/6] Setting up micromamba solver (glibc 2.17 compatible) ==="
MAMBA_BIN="${PROJ_DIR}/bin/micromamba"
if [ -x "${BAKA_DIR}/bin/micromamba" ]; then
    cp "${BAKA_DIR}/bin/micromamba" "${MAMBA_BIN}"
elif [ ! -x "${MAMBA_BIN}" ]; then
    echo "Downloading static micromamba binary..."
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "${PROJ_DIR}" bin/micromamba
fi
chmod +x "${MAMBA_BIN}"
"${MAMBA_BIN}" --version

echo "=== [4/6] Creating Python 3.10 CPU Environment in Scratch ==="
CPU_ENV="${PROJ_DIR}/env"
if [ ! -d "${CPU_ENV}/bin" ]; then
    "${MAMBA_BIN}" create -y -p "${CPU_ENV}" -c conda-forge \
        python=3.10 \
        numpy \
        scipy \
        pandas \
        scikit-learn \
        lightgbm \
        rapidfuzz \
        joblib \
        anyascii \
        pyarrow
fi

echo "=== [5/6] Installing additional Python packages via pip ==="
"${CPU_ENV}/bin/pip" install --no-cache-dir sparse_dot_topn tqdm

echo "=== [6/6] Verifying CPU Environment ==="
"${CPU_ENV}/bin/python" -c "
import lightgbm as lgb
import sklearn
import rapidfuzz
import sparse_dot_topn
import anyascii
import pandas as pd
import numpy as np
print('All core ML packages successfully loaded in HPC scratch environment!')
print('LightGBM version:', lgb.__version__)
print('RapidFuzz version:', rapidfuzz.__version__)
"

echo "=== Environment Setup Complete! ==="
echo "Scratch Project Path: ${PROJ_DIR}"
