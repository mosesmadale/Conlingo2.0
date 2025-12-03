#!/usr/bin/env python3
import torch
import os
import sys
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datetime import datetime

print("="*80)
print("ConLingo 2.0 - Model Testing")
print("="*80)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "trained_model", "final_model")
OUTPUT_FILE = os.path.join(BASE_DIR, "test_results.txt")
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

if not os.path.exists(MODEL_DIR):
    print(f"\nERROR: Trained model not found at {MODEL_DIR}")
    print("Please run train.bat first to train the model.")
    sys.exit(1)

CULTURAL_SYSTEM_PROMPT = """You are a cultural anthropologist and contextual researcher with deep experience studying values, beliefs, customs, and worldview formation across diverse Indian communities.

Your expertise includes:
- Core cultural values and virtues across Indian regions
- Family and social structures in Indian society
- Spiritual and religious norms (Hinduism, Christianity, Islam, Sikhism)
- Cultural symbols, celebrations, and identity markers
- Regional variations (North/South/East/West India)
- Traditional vs modern worldview tensions
- Caste dynamics and social hierarchies
- Hindu-Christian dialogue and interfaith relations

When answering questions about Indian culture and Christianity in India:
1. Draw from your deep knowledge of Indian regional diversity, historical contexts, and contemporary practices
2. Include specific examples from everyday life showing how values and beliefs manifest
3. Acknowledge regional, religious, and generational variations
4. Demonstrate sensitivity to both Hindu and Christian perspectives
5. Focus on worldview - why people believe what they do, not just what they do
6. Provide practical, actionable insights that show cultural logic

Response Guidelines:
- Length: Approximately 100 words
- Tone: Expert yet conversational, like explaining to someone unfamiliar with the region
- Structure: Clear and well-organized with natural flow
- Content: Highly specific to Indian cultural context with real examples
- Avoid: Generic statements, Western-centric views, oversimplifications
- Include: Regional nuances, historical context, modern tensions, specific practices
"""

QUESTIONS = [
    "What sensitivities should pastors consider when mentioning Hindu deities in Christmas homilies?",
    "How can churches ensure caste-neutral seating and participation during worship?",
    "Why might some Christians still use caste surnames, and how should this be discussed?"
]

print(f"\nModel directory: {MODEL_DIR}")
print(f"Output file: {OUTPUT_FILE}")

print("\n" + "="*80)
print("Step 1: Loading Tokenizer")
print("="*80)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
print("Tokenizer loaded successfully")

print("\n" + "="*80)
print("Step 2: Loading Base Model")
print("="*80)
print("This may take several minutes...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True
)
print("Base model loaded successfully")

print("\n" + "="*80)
print("Step 3: Loading Fine-tuned Adapters")
print("="*80)

model = PeftModel.from_pretrained(base_model, MODEL_DIR)
print("Fine-tuned model loaded successfully")

print("\n" + "="*80)
print("Step 4: Running Tests")
print("="*80)

def clean_response(text):
    text = re.sub(r'\[Insert.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[Acknowledge.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[Practical.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("ConLingo 2.0 - Test Results\n")
    f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*80 + "\n\n")
    
    for i, question in enumerate(QUESTIONS, 1):
        print(f"\nTesting Question {i}/3...")
        print(f"Q: {question[:60]}...")
        
        prompt = f"""{CULTURAL_SYSTEM_PROMPT}

Question: {question}

CRITICAL OUTPUT REQUIREMENTS:
- Write ONLY in complete, natural sentences
- Use flowing prose in paragraph form
- NO brackets, placeholders, or template markers
- NO meta-instructions or instructional notes
- NO hashtags, links, or web formatting
- Write as if naturally speaking to an audience

Provide a comprehensive response of approximately 100 words that demonstrates deep cultural understanding of the Indian context.

Response:"""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                min_new_tokens=90,
                temperature=0.4,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Response:" in full_response:
            answer = full_response.split("Response:")[-1].strip()
        elif question in full_response:
            parts = full_response.split(question)
            if len(parts) > 1:
                answer = parts[-1].strip()
            else:
                answer = full_response.strip()
        else:
            answer = full_response.strip()
        
        answer = clean_response(answer)
        
        word_count = len(answer.split())
        
        f.write(f"Question {i}:\n")
        f.write(f"{question}\n\n")
        f.write(f"Answer:\n")
        f.write(f"{answer}\n")
        f.write(f"\n[Word count: {word_count}]\n")
        f.write("\n" + "-"*80 + "\n\n")
        
        print(f"  Answer generated ({word_count} words)")
    
    f.write("="*80 + "\n")
    f.write(f"Testing Complete - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*80 + "\n")

print("\n" + "="*80)
print("Testing Complete!")
print("="*80)
print(f"\nResults saved to: {OUTPUT_FILE}")
print("\nYou can view the results by opening test_results.txt")
