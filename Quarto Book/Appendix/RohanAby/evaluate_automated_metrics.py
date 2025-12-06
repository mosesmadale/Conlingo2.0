import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
import math

def compute_perplexity(probabilities):
    return math.exp(-sum(math.log(p) for p in probabilities) / len(probabilities))

def evaluate_automated_metrics(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    rouge = Rouge()
    results = []

    for _, row in df.iterrows():
        ref = row['reference']
        gen = row['generated']

        # BLEU
        bleu = sentence_bleu([ref.split()], gen.split())

        # ROUGE
        rouge_scores = rouge.get_scores(gen, ref)[0]['rouge-l']['f']

        # Semantic Similarity
        emb1 = model.encode(ref, convert_to_tensor=True)
        emb2 = model.encode(gen, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(emb1, emb2).item()

        results.append({
            "BLEU": bleu,
            "ROUGE-L": rouge_scores,
            "Semantic Similarity": sim
        })

    return pd.DataFrame(results)
