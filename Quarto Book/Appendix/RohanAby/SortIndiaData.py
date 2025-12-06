import os
import json
from datasets import load_from_disk
import re

# ------------------------------
# Load dataset
# ------------------------------
dataset_dir = "wikipedia_1900_2025"
dataset = load_from_disk(dataset_dir)
print(f"Total articles loaded: {len(dataset)}")

# ------------------------------
# Define categories & keywords
# ------------------------------
categories = {
    "Non-Material Culture": {
        "Values & Beliefs": ["values", "beliefs", "morality", "ethics"],
        "Regional Indian Superstitions & Beliefs": ["superstition", "omen", "ritual", "folk belief"],
        "Norms & Customs": ["norms", "customs", "traditions", "practices"],
        "India News Headlines": ["news", "headlines", "current events", "india"],
        "Language": ["language", "dialect", "linguistic", "vernacular"],
        "Religion & Spirituality": ["hinduism", "buddhism", "sikhism", "islam", "christianity", "spirituality", "temple", "yoga"],
        "Arts and Literature": ["art", "literature", "painting", "music", "dance", "poetry"],
        "Social Organization": ["caste", "community", "society", "family", "clan"],
    },
    "Material Culture": {
        "Artifacts & Technology": ["artifact", "tool", "technology", "weapon", "craft"],
        "Government & Economic Systems": ["government", "politics", "economy", "trade", "industry"],
    }
}

# ------------------------------
# Create output directory
# ------------------------------
output_dir = "india_categorized"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# Helper function: check India context
# ------------------------------
def is_india_paragraph(paragraph):
    paragraph_lower = paragraph.lower()
    # Keep paragraph only if it mentions India/Indian/Bharat
    return any(x in paragraph_lower for x in ["india", "indian", "bharat"])

# ------------------------------
# Filter and categorize
# ------------------------------
for super_category, subcats in categories.items():
    for subcat_name, keywords in subcats.items():
        filtered_articles = []
        for example in dataset:
            title = example.get("title", "").strip()
            text = example.get("text", "")

            if not text:
                continue

            # Split into paragraphs to avoid unrelated global mentions
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
            india_paragraphs = [p for p in paragraphs if is_india_paragraph(p)]

            if not india_paragraphs:
                continue

            # Check category keywords in India paragraphs
            matched_paragraphs = []
            for p in india_paragraphs:
                if any(k.lower() in p.lower() for k in keywords):
                    matched_paragraphs.append(p)

            if matched_paragraphs:
                filtered_articles.append({"title": title, "text": "\n".join(matched_paragraphs)})

        # Save subcategory as JSONL
        subcat_file = os.path.join(output_dir, f"{subcat_name.replace(' ', '_')}.jsonl")
        with open(subcat_file, "w", encoding="utf-8") as f:
            for article in filtered_articles:
                json.dump(article, f)
                f.write("\n")

        print(f"{subcat_name}: {len(filtered_articles)} articles saved.")
