#!/bin/bash
# =============================================================================
# SLURM job script — Habrok (University of Groningen)
#
# Submit:  sbatch job.sh
# Monitor: squeue -u $USER
# Logs:    logs/gut_cnn_<jobid>.out  /  logs/gut_cnn_<jobid>.err
# =============================================================================

#SBATCH --job-name=gut-cnn
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=logs/gut_cnn_%j.out
#SBATCH --error=logs/gut_cnn_%j.err

# -----------------------------------------------------------------------------
# Python module
# -----------------------------------------------------------------------------
module purge
module load Python/3.11.3-GCCcore-12.3.0

# -----------------------------------------------------------------------------
# Project & venv
# -----------------------------------------------------------------------------
PROJECT=$HOME/thesis/gut-microbiome-cnn-disease-classification
cd "$PROJECT" || { echo "ERROR: project dir not found: $PROJECT"; exit 1; }

mkdir -p logs

VENV=$PROJECT/venv
if [ ! -d "$VENV" ]; then
    echo "[setup] Creating virtual environment..."
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install --upgrade pip --quiet
    # Unified requirements.txt — environment markers select tensorflow[and-cuda]
    # on Linux automatically (and skip tensorflow-metal which is macOS-only)
    pip install -r requirements.txt --quiet
    echo "[setup] venv ready."
else
    source "$VENV/bin/activate"
fi

# -----------------------------------------------------------------------------
# Confirm GPU is visible to TF before committing to a 12-hour run
# -----------------------------------------------------------------------------
python3 -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'TF {tf.__version__}  |  GPUs found: {gpus}')
if not gpus:
    raise RuntimeError('No GPU detected — check partition / gres request')
"

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
echo "============================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "GPUs     : $CUDA_VISIBLE_DEVICES"
echo "Started  : $(date)"
echo "============================================"

python3 run_limited_metadata_pipeline.py

echo "============================================"
echo "Finished : $(date)"
echo "============================================"
