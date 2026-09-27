# IIT Delhi HPC (PADUM) - Amazon ML Challenge 2026 Runbook
## High-Scale Multi-Core & GPU Training Architecture

---

### 1. Executive Summary & Hardware Leverage
With access to the IIT Delhi PADUM HPC cluster, we overcome local 16 GB RAM limits and can now train on **100% of the dataset** (`train_s1_frac = 1.0`, all 139,000 $S_1$ entities and all 1.73M queries):
- **CPU Cluster Nodes**: 24-core Haswell (63 GB RAM) and 40-core Skylake (120–192 GB RAM).
- **GPU Acceleration**: NVIDIA Tesla V100 (32 GB VRAM) and A100 (80 GB VRAM) for deep multilingual neural semantic bi-encoders.
- **Dedicated Scratch Workspace**: `/scratch/civil/btech/ce1240901/amazon_ml` (25 TB quota, Lustre parallel file system).

---

### 2. Mandatory Cluster Rules
> [!CAUTION]
> **HOME DISK QUOTA VIOLATION WARNING**:
> `/home/civil/btech/ce1240901` is currently at **506 GB used vs 100 GB quota**. Any file written to `~` will fail silently with `Disk quota exceeded`.
> **ALL repositories, virtual environments, conda packages, HuggingFace caches, and dataset files MUST reside exclusively in `/scratch/civil/btech/ce1240901/amazon_ml/`**.

> [!IMPORTANT]
> **OS & glibc Constraint**:
> The cluster runs **CentOS 7 with glibc 2.17**. Standard new Python wheels fail. Our setup uses `micromamba` from conda-forge, which is pre-compiled for glibc 2.17.

> [!NOTE]
> **Outbound Proxy Authentication**:
> Compute and login nodes require proxy authentication:
> `http_proxy=http://proxy22.iitd.ac.in:3128` and `https_proxy=http://proxy22.iitd.ac.in:3128`.
> Automated via `~/proxy.sh`.

---

### 3. Folder & File Structure
```
hpc_amazon_ml/
├── README.md                      # This operating guide
├── add_authorized_key.sh          # Authorize Windows SSH key
├── sync_dataset.sh                # Transfer dataset to scratch
├── scripts/
│   ├── 01_setup_scratch_env.sh    # Fast CPU environment creation via micromamba
│   ├── 01_setup_gpu_env.sh        # PyTorch + CUDA + Transformers GPU environment
│   ├── 02_train_hpc.py            # 100% dataset training with multi-seed LightGBM
│   ├── 03_train_semantic_biencoder.py # GPU deep multilingual bi-encoder
│   └── 04_predict_submission.py   # Test inference, F0.5 optimization & zip packaging
└── pbs_jobs/
    ├── 01_env_setup.pbs           # PBS job to build scratch environment
    ├── 02_train_cpu_full.pbs      # 24 Skylake cores, 120 GB RAM training job
    ├── 03_train_gpu_biencoder.pbs # 1 V100 GPU deep learning job
    ├── 04_predict_full.pbs        # Test set inference and validation job
    └── 05_full_pipeline.pbs       # All-in-one end-to-end training and submission job
```

---

### 4. Step-by-Step Execution Guide

#### Step 0: Authorize Windows SSH Key (One-Time)
Run this command from your Mac (which already has passwordless `ssh hpc`):
```bash
bash hpc_amazon_ml/add_authorized_key.sh
```
Or manually run on any authorized terminal:
```bash
ssh hpc "mkdir -p ~/.ssh && echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDzIcYAVGJhWcMheh1CQ4bHxi72Nq9nOMSmfodlIAZlY ce1240901_windows' >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"
```
After this, `ssh hpc` works passwordlessly from Windows directly.

#### Step 1: Clone Repository and Build Environment on Scratch
Connect to the cluster and clone into scratch:
```bash
ssh hpc
cd /scratch/civil/btech/ce1240901
mkdir -p amazon_ml && cd amazon_ml

# Activate proxy on login node
nohup bash ~/proxy.sh > ./logs/proxy_login.log 2>&1 &
sleep 5
export http_proxy=http://proxy22.iitd.ac.in:3128 https_proxy=http://proxy22.iitd.ac.in:3128

# Clone code
git clone https://github.com/superionsai/AMAZON-ML-CHALLENGE repo
cd repo

# Submit environment setup job to PBS
qsub hpc_amazon_ml/pbs_jobs/01_env_setup.pbs
```
Monitor status:
```bash
qstat -u ce1240901
tail -f /scratch/civil/btech/ce1240901/amazon_ml/logs/env_setup.out
```

#### Step 2: Transfer Dataset to Scratch
From your local terminal, transfer the data:
```bash
bash hpc_amazon_ml/sync_dataset.sh
```
Or on the cluster, if you have the files locally or on Google Drive / curl:
```bash
mkdir -p /scratch/civil/btech/ce1240901/amazon_ml/dataset
# Place train.tsv, train_labels.tsv, test.tsv here
```

#### Step 3: Run Full-Scale Training (100% Data)
Submit the 24-core, 120 GB RAM batch job:
```bash
qsub /scratch/civil/btech/ce1240901/amazon_ml/repo/hpc_amazon_ml/pbs_jobs/02_train_cpu_full.pbs
```
Monitor progress:
```bash
qstat -u ce1240901
tail -f /scratch/civil/btech/ce1240901/amazon_ml/logs/train_full.out
```

#### Step 4: Run Test Prediction & Generate Final Submission
Submit the prediction and packaging job:
```bash
qsub /scratch/civil/btech/ce1240901/amazon_ml/repo/hpc_amazon_ml/pbs_jobs/04_predict_full.pbs
```
When complete, the verified submission zip will be located at:
```
/scratch/civil/btech/ce1240901/amazon_ml/submissions/TCC_submission_hpc_v6.zip
```

#### (Alternative) Single-Command All-in-One Execution
To run training, prediction, validation, and packaging in one continuous 10-hour pipeline:
```bash
qsub /scratch/civil/btech/ce1240901/amazon_ml/repo/hpc_amazon_ml/pbs_jobs/05_full_pipeline.pbs
```

---

### 5. PBS Scheduler Monitoring Cheat Sheet
| Command | Purpose |
|---------|---------|
| `qstat -u ce1240901` | List all your active, queued, or running jobs |
| `qstat -f <jobid>` | Detailed resource utilization and execution node |
| `qdel <jobid>` | Terminate a running or queued job |
| `pbsnodes -a` | Check node status across the cluster |
| `tail -n 50 -f <logfile>` | Live monitoring of job stdout/stderr |
