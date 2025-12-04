#!/usr/bin/env python3
import torch
import json
import os
import sys
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print("="*80)
print("ConLingo 2.0 - Training Pipeline")
print("="*80)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "trained_model")
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\nBase directory: {BASE_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

DATA_FILES = {
    "youtube": os.path.join(DATA_DIR, "youtube.jsonl"),
    "ted_talks": os.path.join(DATA_DIR, "ted_talks.jsonl"),
    "wikipedia": os.path.join(DATA_DIR, "wikipedia.jsonl"),
    "constitution": os.path.join(DATA_DIR, "constitution.jsonl"),
    "superstitions": os.path.join(DATA_DIR, "superstitions.jsonl")
}

print("\n" + "="*80)
print("Step 1: Loading Training Data")
print("="*80)

all_examples = []
dataset_stats = {}

for dataset_name, data_path in DATA_FILES.items():
    print(f"\nLoading {dataset_name}...")
    
    if not os.path.exists(data_path):
        print(f"  ERROR: File not found: {data_path}")
        sys.exit(1)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    count = 0
    for item in data:
        if "instruction" in item and "response" in item:
            question = item["instruction"]
            answer = item["response"]
        elif "question" in item and "answer" in item:
            question = item["question"]
            answer = item["answer"]
        else:
            continue
        
        all_examples.append({
            "question": question,
            "answer": answer,
            "source": dataset_name
        })
        count += 1
    
    dataset_stats[dataset_name] = count
    print(f"  Loaded {count} examples")

print(f"\nTotal examples: {len(all_examples)}")

train_data, val_data = train_test_split(all_examples, test_size=0.1, random_state=42)
print(f"Training examples: {len(train_data)}")
print(f"Validation examples: {len(val_data)}")

print("\n" + "="*80)
print("Step 2: Loading Tokenizer")
print("="*80)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Tokenizer loaded successfully")

print("\n" + "="*80)
print("Step 3: Detecting GPU")
print("="*80)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Total GPU Memory: {gpu_memory:.1f} GB")
    
    if gpu_memory < 20:
        print("\nWARNING: GPU has less than 20GB memory")
        print("Training will use reduced batch size and sequence length")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != "yes":
            print("Training cancelled.")
            sys.exit(0)

batch_size = 1
grad_accum = 32
max_length = 384

print(f"\nTraining Configuration:")
print(f"  Batch size: {batch_size}")
print(f"  Gradient accumulation: {grad_accum}")
print(f"  Max sequence length: {max_length}")
print(f"  Effective batch size: {batch_size * grad_accum}")

print("\n" + "="*80)
print("Step 4: Preparing Datasets")
print("="*80)

def format_instruction(example):
    text = f"### Question:\n{example['question']}\n\n### Answer:\n{example['answer']}"
    return text

def tokenize_function(example):
    text = format_instruction(example)
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

print("Tokenizing training data...")
train_dataset = train_dataset.map(
    tokenize_function,
    remove_columns=train_dataset.column_names
)

print("Tokenizing validation data...")
val_dataset = val_dataset.map(
    tokenize_function,
    remove_columns=val_dataset.column_names
)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

print("\n" + "="*80)
print("Step 5: Loading Base Model")
print("="*80)
print("This may take several minutes...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
    load_in_8bit=False,
    load_in_4bit=False
)

model.gradient_checkpointing_enable()
print("Base model loaded successfully")

print("\n" + "="*80)
print("Step 6: Configuring LoRA")
print("="*80)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.4f}%)")
print(f"Total parameters: {total_params:,}")

print("\n" + "="*80)
print("Step 7: Setting Up Training")
print("="*80)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True if device == "cuda" else False,
    report_to="none",
    save_total_limit=2,
    remove_unused_columns=False,
    gradient_checkpointing=True
)

print("Training configuration:")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  Learning rate: {training_args.learning_rate}")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

print("\n" + "="*80)
print("Step 8: Starting Training")
print("="*80)
if device == "cuda":
    if gpu_memory < 16:
        print("Training will take approximately 3-4 hours on this GPU")
    else:
        print("Training will take approximately 1-2 hours on this GPU")
else:
    print("Training will take 24-48 hours on CPU")
print("Progress will be displayed below...")
print("="*80 + "\n")

trainer.train()

print("\n" + "="*80)
print("Training Complete!")
print("="*80)

print("\n" + "="*80)
print("Step 9: Saving Model")
print("="*80)

final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

print(f"Model saved to: {final_model_dir}")

print("\n" + "="*80)
print("Step 10: Final Evaluation")
print("="*80)

eval_results = trainer.evaluate()
for key, value in eval_results.items():
    print(f"  {key}: {value:.4f}")

print("\n" + "="*80)
print("Training Pipeline Complete!")
print("="*80)
print(f"\nTrained model location: {final_model_dir}")
print("You can now run the testing script.")