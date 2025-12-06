#!/usr/bin/env python3
"""
LoRA Fine-Tuning with ALL Indian Cultural Data
Combines: YouTube, TED Talks, Wikipedia, Constitution, Superstitions
"""

import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import os
from sklearn.model_selection import train_test_split

# Used Claude Sonnet 4.5 to add detailed logging and prettified separators throughout the script
# to improve readability of the output of the code.
print("="*60)
print("Combined All-Data LoRA Fine-Tuning Pipeline")
print("="*60)

# Paths
DATA_PATHS = {
    "youtube": "data/youtube.jsonl",
    "ted_talks": "data/ted_talks.jsonl",
    "wikipedia": "data/wikipedia.jsonl",
    "constitution": "data/constitution.jsonl",
    "superstitions": "data/superstitions.jsonl"
}

OUTPUT_DIR = "trained_model"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n1. Loading and combining all datasets...")

all_examples = []
dataset_stats = {}

for dataset_name, data_path in DATA_PATHS.items():
    print(f"\n  Loading {dataset_name}...")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    count = 0
    for item in data:
        # Normalize to question/answer format
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
    print(f"    Loaded {count} examples from {dataset_name}")

print(f"\n  Dataset Statistics:")
for dataset_name, count in dataset_stats.items():
    percentage = (count / len(all_examples)) * 100
    print(f"    {dataset_name}: {count} examples ({percentage:.1f}%)")

print(f"\n  Total combined examples: {len(all_examples)}")

# Split into train/validation (90/10)
train_data, val_data = train_test_split(all_examples, test_size=0.1, random_state=42)

print(f"\n  Training examples: {len(train_data)}")
print(f"  Validation examples: {len(val_data)}")

print(f"\n2. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"  Tokenizer loaded: {tokenizer.__class__.__name__}")

print(f"\n3. Preparing datasets...")

def format_instruction(example):
    """Format question-answer pair for training"""
    text = f"### Question:\n{example['question']}\n\n### Answer:\n{example['answer']}"
    return text

def tokenize_function(example):
    """Tokenize examples with padding and truncation"""
    text = format_instruction(example)
    
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors=None
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Convert to HuggingFace Dataset format
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# Tokenize
print("  Tokenizing training data...")
train_dataset = train_dataset.map(
    tokenize_function,
    remove_columns=train_dataset.column_names
)

print("  Tokenizing validation data...")
val_dataset = val_dataset.map(
    tokenize_function,
    remove_columns=val_dataset.column_names
)

print(f"  Training dataset size: {len(train_dataset)}")
print(f"  Validation dataset size: {len(val_dataset)}")

print(f"\n4. Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

model.gradient_checkpointing_enable()

print(f"  Model loaded: {model.__class__.__name__}")
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

print(f"\n5. Configuring LoRA...")
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

print(f"  LoRA configured successfully")
print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.4f}%)")
print(f"  Total parameters: {total_params:,}")

print(f"\n6. Setting up training arguments...")

# Used Claude Sonnet 4.5 to help me choose optimal training parameters for LoRA fine-tuning on this dataset.
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    report_to="none",
    save_total_limit=2,
    remove_unused_columns=False,
    gradient_checkpointing=True
)

print("  Training configuration:")
print(f"    Epochs: {training_args.num_train_epochs}")
print(f"    Batch size: {training_args.per_device_train_batch_size}")
print(f"    Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"    Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"    Learning rate: {training_args.learning_rate}")
print(f"    Total training steps: ~{len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")

print(f"\n7. Initializing trainer...")

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

print("  Trainer initialized successfully")

print(f"\n8. Starting training...")
print("="*60)

trainer.train()

print("\n" + "="*60)
print("Training complete")
print("="*60)

print(f"\n9. Saving final model...")
model.save_pretrained(f"{OUTPUT_DIR}/final_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")

print(f"  Model saved to: {OUTPUT_DIR}/final_model")

print(f"\n10. Final evaluation metrics:")
eval_results = trainer.evaluate()
for key, value in eval_results.items():
    print(f"    {key}: {value:.4f}")

# Used Claude Sonnet 4.5 to help me structure a clear and informative training summary file.
# Save training summary
summary_file = f"{OUTPUT_DIR}/training_summary.txt"
with open(summary_file, 'w') as f:
    f.write("="*60 + "\n")
    f.write("Combined All-Data Model Training Summary\n")
    f.write("="*60 + "\n\n")
    f.write("Datasets Used:\n")
    for dataset_name, count in dataset_stats.items():
        percentage = (count / len(all_examples)) * 100
        f.write(f"  - {dataset_name}: {count} examples ({percentage:.1f}%)\n")
    f.write(f"\nTotal Examples: {len(all_examples)}\n")
    f.write(f"Training Examples: {len(train_data)}\n")
    f.write(f"Validation Examples: {len(val_data)}\n\n")
    f.write("Training Configuration:\n")
    f.write(f"  Epochs: {training_args.num_train_epochs}\n")
    f.write(f"  Learning Rate: {training_args.learning_rate}\n")
    f.write(f"  Batch Size: {training_args.per_device_train_batch_size}\n\n")
    f.write("Final Metrics:\n")
    for key, value in eval_results.items():
        f.write(f"  {key}: {value:.4f}\n")

print(f"\n  Training summary saved to: {summary_file}")


print("\n" + "="*60)
print("Fine-tuning pipeline complete")
print("="*60)
