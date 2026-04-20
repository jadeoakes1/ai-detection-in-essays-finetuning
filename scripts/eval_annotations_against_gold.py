import os

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.makedirs("analysis_outputs/annotation", exist_ok=True)

# =========================
# FILE PATHS
# =========================
annotator1_file = "data/annotation/annotator1_200_samples.csv"
annotator2_file = "data/annotation/annotator2_200_samples.csv"
gold_file = "data/splits/balanced/test.csv"

report_output_file = "analysis_outputs/annotation/gold_comparison_metrics.txt"
merged_output_file = "analysis_outputs/annotation/merged_annotations_with_gold.csv"

# =========================
# LOAD DATA
# =========================
df1 = pd.read_csv(annotator1_file)
df2 = pd.read_csv(annotator2_file)
df_gold = pd.read_csv(gold_file)

# =========================
# KEEP NEEDED COLUMNS
# =========================
df1 = df1[["id", "text", "label_guess", "confidence"]].rename(
    columns={
        "id": "annot_id_1",
        "label_guess": "label_1",
        "confidence": "confidence_1",
    }
)

df2 = df2[["id", "text", "label_guess", "confidence"]].rename(
    columns={
        "id": "annot_id_2",
        "label_guess": "label_2",
        "confidence": "confidence_2",
    }
)

df_gold = df_gold[["text", "label", "prompt_name", "source"]].copy()

# =========================
# NORMALIZE TEXT FOR MATCHING
# =========================
def normalize_text(s):
    if pd.isna(s):
        return ""
    return " ".join(str(s).strip().split())

df1["text_norm"] = df1["text"].apply(normalize_text)
df2["text_norm"] = df2["text"].apply(normalize_text)
df_gold["text_norm"] = df_gold["text"].apply(normalize_text)

# Normalize annotator labels
df1["label_1"] = df1["label_1"].astype(str).str.strip().str.lower()
df2["label_2"] = df2["label_2"].astype(str).str.strip().str.lower()

# Gold labels: 0 = human, 1 = ai
label_map = {0: "human", 1: "ai"}
df_gold["gold_label"] = df_gold["label"].map(label_map)

# =========================
# CHECK FOR DUPLICATE TEXTS
# =========================
gold_dupes = df_gold["text_norm"].duplicated().sum()
ann1_dupes = df1["text_norm"].duplicated().sum()
ann2_dupes = df2["text_norm"].duplicated().sum()

# =========================
# MERGE ANNOTATOR FILES TOGETHER
# =========================
df_ann = pd.merge(
    df1,
    df2[["text_norm", "label_2", "confidence_2", "annot_id_2"]],
    on="text_norm",
    how="inner",
)

df_ann = df_ann.rename(columns={"text": "text_original"})

# =========================
# MERGE WITH GOLD BY NORMALIZED TEXT
# =========================
df = pd.merge(
    df_ann,
    df_gold[["text_norm", "gold_label", "prompt_name", "source"]],
    on="text_norm",
    how="left",
)

# =========================
# MATCH STATS
# =========================
n_ann = len(df_ann)
n_matched_gold = df["gold_label"].notna().sum()
n_unmatched_gold = df["gold_label"].isna().sum()

df_eval = df[df["gold_label"].notna()].copy()

# =========================
# METRICS FUNCTION
# =========================
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=["human", "ai"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=["human", "ai"])
    cm_df = pd.DataFrame(
        cm,
        index=["gold_human", "gold_ai"],
        columns=["pred_human", "pred_ai"],
    )
    return acc, report, cm_df

# =========================
# COMPUTE ANNOTATOR-vs-GOLD METRICS
# =========================
acc1 = report1 = cm1 = None
acc2 = report2 = cm2 = None

if len(df_eval) > 0:
    acc1, report1, cm1 = compute_metrics(df_eval["gold_label"], df_eval["label_1"])
    acc2, report2, cm2 = compute_metrics(df_eval["gold_label"], df_eval["label_2"])

# =========================
# CONSENSUS VS GOLD
# =========================
df_eval["consensus_label"] = df_eval.apply(
    lambda row: row["label_1"] if row["label_1"] == row["label_2"] else "disagree",
    axis=1,
)

df_consensus = df_eval[df_eval["consensus_label"] != "disagree"].copy()
n_consensus = len(df_consensus)
n_disagree = (df_eval["consensus_label"] == "disagree").sum()

acc_cons = report_cons = cm_cons = None
if n_consensus > 0:
    acc_cons, report_cons, cm_cons = compute_metrics(
        df_consensus["gold_label"], df_consensus["consensus_label"]
    )

# =========================
# SAVE MERGED DATA
# =========================
df.to_csv(merged_output_file, index=False)

# =========================
# SAVE TXT REPORT
# =========================
with open(report_output_file, "w", encoding="utf-8") as f:
    f.write("===== ANNOTATOR vs GOLD EVALUATION =====\n\n")

    f.write("Matching method: normalized text match\n\n")

    f.write(f"Gold duplicate normalized texts: {gold_dupes}\n")
    f.write(f"Annotator 1 duplicate normalized texts: {ann1_dupes}\n")
    f.write(f"Annotator 2 duplicate normalized texts: {ann2_dupes}\n\n")

    f.write(f"Annotated samples after merging annotator files: {n_ann}\n")
    f.write(f"Annotated samples matched to gold:            {n_matched_gold}\n")
    f.write(f"Annotated samples NOT matched to gold:        {n_unmatched_gold}\n\n")

    if len(df_eval) == 0:
        f.write("No annotated samples matched to gold. No evaluation metrics computed.\n")
    else:
        f.write("=== ANNOTATOR 1 vs GOLD ===\n")
        f.write(f"Accuracy: {acc1:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(pd.DataFrame(report1).transpose().to_string())
        f.write("\n\n")
        f.write("Confusion matrix (rows=gold, cols=pred):\n")
        f.write(cm1.to_string())
        f.write("\n\n")

        f.write("=== ANNOTATOR 2 vs GOLD ===\n")
        f.write(f"Accuracy: {acc2:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(pd.DataFrame(report2).transpose().to_string())
        f.write("\n\n")
        f.write("Confusion matrix (rows=gold, cols=pred):\n")
        f.write(cm2.to_string())
        f.write("\n\n")

        f.write("=== CONSENSUS vs GOLD ===\n")
        f.write(f"Consensus cases: {n_consensus}\n")
        f.write(f"Annotator disagreement cases: {n_disagree}\n\n")

        if n_consensus > 0:
            f.write(f"Accuracy: {acc_cons:.4f}\n\n")
            f.write("Classification report:\n")
            f.write(pd.DataFrame(report_cons).transpose().to_string())
            f.write("\n\n")
            f.write("Confusion matrix (rows=gold, cols=pred):\n")
            f.write(cm_cons.to_string())
            f.write("\n")
        else:
            f.write("No consensus cases available for evaluation.\n")