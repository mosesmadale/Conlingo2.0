#!/bin/bash
#SBATCH --job-name=conlingo_test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

echo "========================================"
echo "ConLingo 2.0 - Testing"
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

python scripts/test_model.py

deactivate

echo ""
echo "========================================"
echo "Testing completed: $(date)"
echo "========================================"
