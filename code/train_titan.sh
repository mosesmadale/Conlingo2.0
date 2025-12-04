#!/bin/bash
#SBATCH --job-name=conlingo_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

echo "========================================"
echo "ConLingo 2.0 - Training"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/venv"

module purge
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.3.0

source "$VENV_DIR/bin/activate"

export HF_HOME=~/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=~/.cache/huggingface

cd "$SCRIPT_DIR"

python scripts/train_model.py

deactivate

echo ""
echo "========================================"
echo "Training completed: $(date)"
echo "========================================"
