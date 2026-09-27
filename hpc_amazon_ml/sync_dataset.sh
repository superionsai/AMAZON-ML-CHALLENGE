#!/bin/bash
# ==============================================================================
# IIT Delhi HPC - Dataset Sync Script
# Transfers local Amazon ML Challenge dataset files to cluster scratch storage
# ==============================================================================
set -euo pipefail

DEST_HOST="hpc"
DEST_PATH="/scratch/civil/btech/ce1240901/amazon_ml/dataset"

echo "Ensuring remote destination directory exists: ${DEST_PATH}..."
ssh "${DEST_HOST}" "mkdir -p ${DEST_PATH}"

echo "Transferring training and test TSV files to HPC scratch..."
# Checks for dataset files in standard student_resource/dataset directory
if [ -d "student_resource/dataset" ]; then
    rsync -avz --progress student_resource/dataset/*.tsv "${DEST_HOST}:${DEST_PATH}/"
elif [ -d "dataset" ]; then
    rsync -avz --progress dataset/*.tsv "${DEST_HOST}:${DEST_PATH}/"
else
    echo "Warning: Could not find local dataset/ folder. Please specify files manually."
fi

echo "Verifying remote dataset on HPC scratch..."
ssh "${DEST_HOST}" "ls -lh ${DEST_PATH}"
echo "Dataset transfer complete!"
