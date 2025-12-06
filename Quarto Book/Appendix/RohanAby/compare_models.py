import pandas as pd

def compare_models(rag_df, finetuned_df, baseline_df):
    comparison = pd.DataFrame({
        "Metric": ["BLEU", "ROUGE-L", "Semantic Similarity", "CSI_Score"],
        "RAG": [rag_df["BLEU"].mean(), rag_df["ROUGE-L"].mean(), rag_df["Semantic Similarity"].mean(), rag_df["CSI_Score"].mean()],
        "Fine-Tuned": [finetuned_df["BLEU"].mean(), finetuned_df["ROUGE-L"].mean(), finetuned_df["Semantic Similarity"].mean(), finetuned_df["CSI_Score"].mean()],
        "Baseline": [baseline_df["BLEU"].mean(), baseline_df["ROUGE-L"].mean(), baseline_df["Semantic Similarity"].mean(), baseline_df["CSI_Score"].mean()]
    })
    return comparison
