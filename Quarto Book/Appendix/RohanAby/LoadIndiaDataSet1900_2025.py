from datasets import load_dataset
import re

# Load English Wikipedia snapshot (latest available)
dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

# Define regex for years between 1900 and 2025
year_pattern = re.compile(r"\b(19[0-9]{2}|20[0-2][0-9]|2025)\b")

def mentions_year_range(example):
    text = example["text"]
    return bool(year_pattern.search(text))

# Filter dataset for pages mentioning years 1900–2025
dataset_filtered = dataset.filter(mentions_year_range)

print(f"Filtered dataset size: {len(dataset_filtered)}")
dataset_filtered.save_to_disk("wikipedia_1900_2025")
