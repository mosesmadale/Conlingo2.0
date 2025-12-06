#!/usr/bin/env python3
# Used Claude Sonnet 4.5 to generate descriptive docstrings to allow good readability for code readers.
"""
Enhanced Test: Combined Model with RAG-Style Prompting
3 Critical Questions - 100 word responses with cultural depth
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datetime import datetime
import os
import re

# Used Claude Sonnet 4.5 to add detailed logging and prettified separators throughout the script
# to improve readability of the output of the code.
print("="*80)
print("Combined Model - Enhanced Prompting Test (3 Questions)")
print("="*80)

BASE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
FINETUNED_MODEL_PATH = "trained_model/final_model"
OUTPUT_FILE = "results/test_output.txt"

# Claude Sonnet 4.5 was used to help me structure the following prompt to the AI Llama model which I originally developed myself based on the prompt that was given to the Conlingo RAG model so that it is more LLM friendly.
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

print(f"\n1. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
print(f"  Tokenizer loaded")

print(f"\n2. Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
print(f"  Base model loaded")

print(f"\n3. Loading fine-tuned LoRA adapters...")
model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_PATH)
print(f"  Fine-tuned combined model loaded")

print(f"\n4. Running enhanced 3-question test...")
print(f"  Target: ~100 words per response")
print(f"  Temperature: 0.4 (analytical precision)")
print(f"  Output: {OUTPUT_FILE}")
print("="*80)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Used Claude Sonnet 4.5 to help me develop a comprehensive set of regular expressions to clean the model's output
# removing any weird characters that might show up because of temperature settings.
def clean_response(text):
    """Remove all artifacts and ensure clean prose"""
    text = re.sub(r'\[Insert.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[Acknowledge.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[Practical.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[.*?example.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[.*?variation.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[.*?insight.*?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[approx\.?\s*\d*\s*words?\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip('"').strip("'").strip()
    return text

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("Combined Model (All 5 Datasets) - Enhanced Prompting\n")
    f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Configuration: RAG-style cultural anthropologist prompting\n")
    f.write(f"Target: 100-word comprehensive responses\n")
    f.write("="*80 + "\n\n")
    
    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n[Question {i}/3]")
        print(f"Q: {question}")
        
        prompt = f"""{CULTURAL_SYSTEM_PROMPT}

Question: {question}

CRITICAL OUTPUT REQUIREMENTS:
- Write ONLY in complete, natural sentences
- Use flowing prose in paragraph form
- NO brackets, placeholders, or template markers like [Insert example]
- NO meta-instructions or instructional notes
- NO hashtags, links, or web formatting
- Write as if naturally speaking to an audience

Provide a comprehensive response of approximately 100 words that demonstrates deep cultural understanding of the Indian context. Include specific examples naturally within your prose and acknowledge regional or religious variations where relevant.

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
                if answer.startswith("CRITICAL OUTPUT") or answer.startswith("Provide a comprehensive"):
                    lines = answer.split("\n")
                    answer = "\n".join([line for line in lines if not any(x in line for x in ["CRITICAL OUTPUT", "Provide a comprehensive", "approximately 100 words"])])
                    answer = answer.strip()
            else:
                answer = full_response.strip()
        else:
            answer = full_response.strip()
        
        answer = clean_response(answer)
        
        answer_lines = answer.split("\n")
        cleaned_lines = []
        for line in answer_lines:
            if not any(x in line.lower() for x in ["provide a comprehensive", "approximately 100 words", "cultural understanding", "critical output", "write only"]):
                cleaned_lines.append(line)
        answer = "\n".join(cleaned_lines).strip()
        
        if len(answer) > 1000:
            sentences = answer.split('. ')
            answer = '. '.join(sentences[:8])
            if not answer.endswith('.'):
                answer += '.'
        
        word_count = len(answer.split())
        
        f.write(f"Question {i}:\n")
        f.write(f"{question}\n\n")
        f.write(f"Answer:\n")
        f.write(f"{answer}\n")
        f.write(f"\n[Word count: {word_count}]\n")
        f.write("\n" + "-"*80 + "\n\n")
        
        print(f"  Response generated: {word_count} words")
        print(f"  Preview: {answer[:120]}...")
        print("-"*80)
    
    f.write("="*80 + "\n")
    f.write(f"Test Complete - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*80 + "\n")

print("\n" + "="*80)
print("Test Complete")
print("="*80)
print(f"\nResults saved to: {OUTPUT_FILE}")
print("\nYou can view the results with: cat {OUTPUT_FILE}")
print("\n" + "="*80)