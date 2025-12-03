# ConLingo 2.0 - Cultural AI Model Training

Fine-tuned LLaMA-3 8B model for Indian cultural context and Christian-Hindu dialogue.

## System Requirements

### Hardware
- NVIDIA GPU with 24GB+ VRAM (RTX 3090, RTX 4090, or A5000+)
- 32GB+ RAM recommended
- 50GB free disk space

### Software
- Windows 10/11
- Python 3.10 or 3.11
- Git

## Pre-Setup: Get HuggingFace Token

1. Create account at [https://huggingface.co/join](https://huggingface.co/join)
2. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Click "New token"
4. Name: "conlingo-demo", Type: "Read"
5. Click "Generate" and **COPY THE TOKEN**
6. Accept LLaMA 3 license at: [https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

### 1. Install Python 3.11

**IMPORTANT**: Python 3.13+ is not yet supported by PyTorch.

1. Download Python 3.11.9: [https://www.python.org/downloads/release/python-3119/](https://www.python.org/downloads/release/python-3119/)
2. Choose "Windows installer (64-bit)"
3. Run installer
4. **Check "Add Python to PATH"** during installation
5. Restart Command Prompt
6. Verify: `python --version` should show 3.11.x

## 2. Git Setup (if needed)

Before starting, verify Git is installed:
```cmd
git --version
```

If Git is not found:
1. Download from [https://git-scm.com/download/win](https://git-scm.com/download/win)
2. Run installer (requires admin access)
3. Check "Add Git to PATH" during installation
4. Restart Command Prompt
5. Verify: `git --version`

## Quick Start

### Step 1: Clone Repository

Open Command Prompt:
```bash
cd Desktop
git clone https://github.com/mosesmadale/Conlingo2.0.git
cd Conlingo2.0\code
```

### Step 2: Run Setup
```bash
setup_windows.bat
```

What this does:
- Checks Python version (requires 3.10+)
- Creates virtual environment
- Installs dependencies (10-15 minutes)
- Prompts for HuggingFace token
- Authenticates with HuggingFace

When prompted, paste your HuggingFace token.

### Step 3: Train Model
```bash
train.bat
```

What this does:
- Loads 2,500 training examples from 5 datasets
- Downloads LLaMA-3 8B base model (16GB)
- Trains with LoRA fine-tuning for 3 epochs

**Expected time**: 1-2 hours on GPU

You'll see:
- Training loss decreasing
- Progress bars
- Evaluation metrics every 100 steps

### Step 4: Test Model
```bash
test.bat
```

What this does:
- Loads trained model
- Tests with 3 questions about Indian Christianity
- Generates cultural responses
- Saves to `test_results.txt`
- Opens results in Notepad

**Expected time**: 2-3 minutes

## Project Structure
```
Conlingo2.0/
├── code/
│   ├── data/                    # Training datasets (5 JSONL files)
│   │   ├── youtube.jsonl
│   │   ├── ted_talks.jsonl
│   │   ├── wikipedia.jsonl
│   │   ├── constitution.jsonl
│   │   └── superstitions.jsonl
│   ├── scripts/
│   │   ├── train_model.py       # Training implementation
│   │   └── test_model.py        # Testing implementation
│   ├── setup_windows.bat        # Setup script
│   ├── train.bat                # Training launcher
│   ├── test.bat                 # Testing launcher
│   └── requirements.txt         # Python dependencies
└── README.md                    # This file
```

## Training Data Sources

1. **YouTube Transcripts** (512 examples) - Indian cultural content
2. **TED Talks** (596 examples) - Indian speakers and topics
3. **Wikipedia** (500 examples) - Indian culture and history
4. **Constitution** (500 examples) - Indian legal and constitutional knowledge
5. **Superstitions** (923 examples) - Regional beliefs and practices

**Total**: 2,531 training examples covering diverse Indian cultural contexts

## Test Questions

The model is tested with three questions about Christianity in Indian context:

1. What sensitivities should pastors consider when mentioning Hindu deities in Christmas homilies?
2. How can churches ensure caste-neutral seating and participation during worship?
3. Why might some Christians still use caste surnames, and how should this be discussed?

Expected: ~100 word culturally-aware responses for each question.

## Troubleshooting

### Python not found
- Install from [python.org](https://python.org)
- Check "Add Python to PATH" during installation
- Restart Command Prompt after installation

### GPU not detected
- Verify with: `nvidia-smi` in Command Prompt
- Install drivers from [nvidia.com](https://nvidia.com)
- Training will fall back to CPU (24+ hours, not recommended)

### HuggingFace authentication failed
- Verify token is correct
- Confirm you accepted LLaMA 3 license
- Token must have READ permission
- Try creating new token

### Out of memory during training
- Close other applications
- Edit `code/scripts/train_model.py` line 143
- Change `per_device_train_batch_size=2` to `=1`

### Slow model download
- LLaMA-3 is 16GB
- First download: 20-30 minutes
- Subsequent runs use cached version

## Technical Details

### Model Architecture
- Base: LLaMA-3 8B Instruct
- Fine-tuning: LoRA (Low-Rank Adaptation)
- Trainable parameters: 0.17% of total
- Training epochs: 3
- Learning rate: 2e-4

### Training Configuration
- Batch size: 2 per device
- Gradient accumulation: 16 steps
- Effective batch size: 32
- Max sequence length: 512 tokens
- Optimizer: AdamW with cosine scheduling

### Hardware Usage
- GPU VRAM: ~20GB during training
- Disk space: ~35GB (model + cache)
- Training time: 1-2 hours (GPU), 24+ hours (CPU)

## Output Files

After successful run:
- `trained_model/final_model/` - Fine-tuned model weights
- `test_results.txt` - Test outputs with 3 questions answered

## Demo Checklist

For running this demo:

- [ ] Python 3.10+ installed
- [ ] Git installed
- [ ] HuggingFace account created
- [ ] HuggingFace token obtained
- [ ] LLaMA 3 license accepted
- [ ] GPU drivers installed
- [ ] 50GB free disk space
- [ ] Estimated total time: 2-3 hours

## Authors

Moses Madale - Oral Roberts University  
Email: mmadale4@oru.edu

## License

This project uses LLaMA-3 which requires accepting Meta's license agreement.
Training data compiled from public sources for educational purposes.