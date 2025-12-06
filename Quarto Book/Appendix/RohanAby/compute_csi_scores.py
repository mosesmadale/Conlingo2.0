import pandas as pd

def compute_csi_score(row):
    weights = {"Accuracy": 0.3, "Tone": 0.3, "Context": 0.2, "Empathy": 0.2}
    return sum(row[k] * w for k, w in weights.items())

def apply_csi_rubric(df):
    df["CSI_Score"] = df.apply(compute_csi_score, axis=1)
    return df[["Question_ID", "Accuracy", "Tone", "Context", "Empathy", "CSI_Score"]]
