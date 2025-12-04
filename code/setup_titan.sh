#!/bin/bash
#SBATCH --job-name=conlingo_setup
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

echo "========================================"
echo "ConLingo 2.0 - Titan Environment Setup"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$SLURM_SUBMIT_DIR"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/venv"

echo "Project root: ${PROJECT_ROOT}"
echo "Virtual environment: ${VENV_DIR}"
echo ""

# Load required modules
module purge
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.3.0

echo "Loaded modules:"
module list
echo ""

# Create logs directory
mkdir -p "${SCRIPT_DIR}/logs"

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi
echo ""

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo ""

# Install PyTorch with CUDA 12.1
echo "Installing PyTorch with CUDA 12.1..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
echo ""

# Install other requirements
echo "Installing other dependencies..."
pip install -r "${SCRIPT_DIR}/requirements.txt"
echo ""

# Verify installation
echo "========================================"
echo "Verifying Installation"
echo "========================================"
nvidia-smi
echo ""

python3 << 'PYTHON_CHECK'
import torch
import sys

print("=" * 60)
print("Installation Verification:")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("CUDA not available!")
    sys.exit(1)

print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

from transformers import AutoTokenizer
print("Transformers installed")

from peft import LoraConfig
print("PEFT installed")

import bitsandbytes
print("bitsandbytes installed")

print("")
print("=" * 60)
print("Environment setup complete!")
print("=" * 60)
PYTHON_CHECK

# Auto-login to HuggingFace with environment variable
echo ""
if [ ! -z "$HF_TOKEN" ]; then
    echo "Logging into HuggingFace..."
    echo "$HF_TOKEN" | python3 -c "from huggingface_hub import login; import sys; login(token=sys.stdin.read().strip())"
    echo "HuggingFace login complete"
else
    echo "Warning: HF_TOKEN not set. You'll need to login manually with: huggingface-cli login"
fi

deactivate

echo ""
echo "========================================"
echo "Setup completed: $(date)"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Run training: sbatch train_titan.sh"
echo "2. After training: sbatch test_titan.sh"
echo ""
