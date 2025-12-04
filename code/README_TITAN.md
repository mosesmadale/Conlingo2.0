# ConLingo 2.0 - Titan Deployment Guide

Fine-tuned LLaMA-3 8B model for Indian cultural awareness in Christian contexts.

## Quick Start (4 Commands)
```bash
# 0. Load git (if not already loaded)
module load git

# 1. Clone repository
git clone https://github.com/mosesmadale/Conlingo2.0.git
cd Conlingo2.0/code

# 2. Setup environment (10-15 minutes)
sbatch setup_titan.sh

# 3. After setup completes, run training (1-2 hours)
sbatch train_titan.sh
```

## Prerequisites

- Titan cluster access with GPU partition
- HuggingFace account with LLaMA-3 access
- HuggingFace token

## Detailed Steps

### 1. HuggingFace Setup

Before running setup, get HuggingFace access:

a. Create account at https://huggingface.co
b. Accept LLaMA-3 license: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
c. Get token from: https://huggingface.co/settings/tokens

### 2. Environment Setup
```bash
sbatch setup_titan.sh
```

Wait for job to complete (check logs/setup_*.out).

Then login to HuggingFace:
```bash
module load Python/3.12.3-GCCcore-13.3.0
source ../venv/bin/activate
huggingface-cli login
# Paste your token when prompted
deactivate
```

### 3. Training
```bash
sbatch train_titan.sh
```

Training time: 1-2 hours on 24GB GPU
Output: trained_model/final_model/

Monitor progress:
```bash
tail -f logs/train_*.out
```

### 4. Testing
```bash
sbatch test_titan.sh
```

Results saved to: results/test_output.txt

View results:
```bash
cat results/test_output.txt
```

## Directory Structure
```
code/
├── setup_titan.sh          # Environment setup (SLURM job)
├── train_titan.sh          # Training script (SLURM job)
├── test_titan.sh           # Testing script (SLURM job)
├── requirements.txt        # Python dependencies
├── data/                   # Training data (5 JSONL files)
├── scripts/
│   ├── train_model.py      # Training implementation
│   └── test_model.py       # Testing implementation
├── logs/                   # Job outputs (created automatically)
├── trained_model/          # Model checkpoints (created by training)
└── results/                # Test outputs (created by testing)
```

## Troubleshooting

### Setup fails
- Check logs/setup_*.err
- Verify Python/CUDA modules loaded
- Ensure sufficient disk space

### Training fails
- Verify HuggingFace login: `huggingface-cli whoami`
- Check GPU availability: `squeue -u $USER`
- Review logs/train_*.err

### Out of memory
- Training uses 24GB GPU
- If using smaller GPU, reduce batch_size in train_model.py

## Configuration

Edit `scripts/train_model.py` to modify:
- num_train_epochs (default: 3)
- per_device_train_batch_size (default: 2)
- learning_rate (default: 2e-4)

## Test Questions

The model answers 3 culturally-sensitive questions:
1. Hindu deities in Christmas homilies sensitivities
2. Caste-neutral church seating strategies  
3. Christians using caste surnames discussion

## Support

For issues, contact: mmadale4@oru.edu
