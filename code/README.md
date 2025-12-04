# ConLingo 2.0 - Titan Deployment Guide

Fine-tuned LLaMA-3 8B model for Indian cultural awareness in Christian contexts.

## Prerequisites

- Titan cluster access with GPU partition
- HuggingFace account with LLaMA-3 access approval

## Setup Instructions

### Step 1: Get HuggingFace Access

1. Create account at https://huggingface.co
2. Request access to LLaMA-3: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
3. Wait for approval email (usually within hours)
4. Get your access token: https://huggingface.co/settings/tokens

### Step 2: Clone Repository
```bash
module load git
git clone https://github.com/mosesmadale/Conlingo2.0.git
cd Conlingo2.0/code
```

### Step 3: Setup Environment

Set your HuggingFace token (replace with your actual token):
```bash
export HF_TOKEN="<paste your token here>"
```

Run setup (takes 10-15 minutes):
```bash
sbatch --export=HF_TOKEN setup_titan.sh
```

Monitor setup progress:
```bash
tail -f logs/setup_*.out
```

Wait until you see "Setup completed" message.

### Step 4: Run Training

Start training (takes 1-2 hours on 24GB GPU):
```bash
sbatch train_titan.sh
```

Monitor training progress:
```bash
tail -f logs/train_*.out
```

Training output will be saved to: `trained_model/final_model/`

### Step 5: Run Testing

After training completes:
```bash
sbatch test_titan.sh
```

View results:
```bash
cat results/test_output.txt
```

## Directory Structure
```
code/
├── setup_titan.sh          # Environment setup script
├── train_titan.sh          # Training script
├── test_titan.sh           # Testing script
├── requirements.txt        # Python dependencies
├── data/                   # Training data (5 JSONL files, 3MB total)
├── scripts/
│   ├── train_model.py      # Training implementation
│   └── test_model.py       # Testing implementation
├── logs/                   # Job outputs (auto-created)
├── trained_model/          # Model checkpoints (auto-created)
└── results/                # Test outputs (auto-created)
```

## Test Questions

The model answers 3 culturally-sensitive questions:

1. What sensitivities should pastors consider when mentioning Hindu deities in Christmas homilies?
2. How can churches ensure caste-neutral seating and participation during worship?
3. Why might some Christians still use caste surnames, and how should this be discussed?

## Troubleshooting

### Setup fails

Check error log:
```bash
cat logs/setup_*.err
```

Common issues:
- HF_TOKEN not set: Re-run with `export HF_TOKEN="your_token"` and `sbatch --export=HF_TOKEN setup_titan.sh`
- No disk space: Check with `df -h ~`

### Training fails

Check if logged into HuggingFace:
```bash
module load Python/3.12.3-GCCcore-13.3.0
source ../venv/bin/activate
python -c "from huggingface_hub import whoami; print(whoami())"
deactivate
```

Check error log:
```bash
cat logs/train_*.err
```

### Out of memory

Training requires 24GB GPU. If using smaller GPU, edit `scripts/train_model.py`:
- Reduce `per_device_train_batch_size` from 2 to 1
- Reduce `max_length` from 512 to 384

## Configuration

Edit `scripts/train_model.py` to modify:
- `num_train_epochs` (default: 3)
- `per_device_train_batch_size` (default: 2)
- `learning_rate` (default: 2e-4)

## Dataset Information

Training uses 5 datasets (3,031 total examples):
- YouTube transcripts: 512 examples (16.9%)
- TED Talks: 596 examples (19.7%)
- Wikipedia articles: 500 examples (16.5%)
- Indian Constitution: 500 examples (16.5%)
- Cultural superstitions: 923 examples (30.5%)

## Support

For issues, contact: mmadale4@oru.edu

## References

Claude Sonnet 4.5
- Used Claude Sonnet 4.5 LLM to assist with troubleshooting code errors.
- Used Claude Sonnet 4.5 LLM to assist with optimizing the model training time.