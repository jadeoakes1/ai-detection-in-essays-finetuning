import os
from collections import Counter

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

os.makedirs("analysis_outputs/annotation", exist_ok=True)

# =========================
# FILE PATHS
# =========================
file_annotator1 = "data/annotation/annotator1_200_samples.csv"
file_annotator2 = "data/annotation/annotator2_200_samples.csv"

merged_output_file = "analysis_outputs/annotation/merged_annotation_labels.csv"
report_output_file = "analysis_outputs/annotation/inter_annotator_agreement.txt"

# =========================
# LOAD DATA
# =========================
df1 = pd.read_csv(file_annotator1)
df2 = pd.read_csv(file_annotator2)

df1 = df1[["id", "label_guess", "confidence"]].rename(
    columns={"label_guess": "label_1", "confidence": "confidence_1"}
)

df2 = df2[["id", "label_guess", "confidence"]].rename(
    columns={"label_guess": "label_2", "confidence": "confidence_2"}
)

df = pd.merge(df1, df2, on="id", how="inner")

# =========================
# CLEAN / NORMALIZE
# =========================
for col in ["label_1", "label_2"]:
    df[col] = df[col].fillna("").astype(str).str.strip().str.lower()

for col in ["confidence_1", "confidence_2"]:
    df[col] = df[col].fillna("").astype(str).str.strip().str.lower()
    df[col] = df[col].replace(
        {
            "": "missing",
            "nan": "missing",
            "none": "missing",
            "n/a": "missing",
            "na": "missing",
            "med": "medium",
        }
    )

# =========================
# MATCH COLUMNS
# =========================
df["label_match"] = df["label_1"] == df["label_2"]
df["confidence_match"] = df["confidence_1"] == df["confidence_2"]

# =========================
# LABEL METRICS (ALL ROWS)
# =========================
n_total = len(df)

label_agreement = df["label_match"].mean()
label_kappa = cohen_kappa_score(df["label_1"], df["label_2"])

labels_sorted = sorted(set(df["label_1"]).union(set(df["label_2"])))
cm_labels = confusion_matrix(df["label_1"], df["label_2"], labels=labels_sorted)
cm_labels_df = pd.DataFrame(
    cm_labels,
    index=[f"A1_{x}" for x in labels_sorted],
    columns=[f"A2_{x}" for x in labels_sorted],
)

# =========================
# CONFIDENCE METRICS
# =========================
df_conf = df[
    (df["confidence_1"] != "missing") &
    (df["confidence_2"] != "missing")
].copy()

n_conf = len(df_conf)
n_conf_excluded = n_total - n_conf

conf_agreement = None
conf_kappa = None
weighted_conf_kappa = None
cm_conf_df = None

if n_conf > 0:
    conf_agreement = df_conf["confidence_match"].mean()
    conf_kappa = cohen_kappa_score(df_conf["confidence_1"], df_conf["confidence_2"])

    conf_sorted = sorted(set(df_conf["confidence_1"]).union(set(df_conf["confidence_2"])))
    cm_conf = confusion_matrix(df_conf["confidence_1"], df_conf["confidence_2"], labels=conf_sorted)
    cm_conf_df = pd.DataFrame(
        cm_conf,
        index=[f"A1_{x}" for x in conf_sorted],
        columns=[f"A2_{x}" for x in conf_sorted],
    )

    confidence_order = {"low": 0, "medium": 1, "high": 2}
    conf_values = set(df_conf["confidence_1"]).union(set(df_conf["confidence_2"]))

    if conf_values.issubset(confidence_order.keys()):
        conf1_num = df_conf["confidence_1"].map(confidence_order)
        conf2_num = df_conf["confidence_2"].map(confidence_order)
        weighted_conf_kappa = cohen_kappa_score(conf1_num, conf2_num, weights="quadratic")

# =========================
# DISAGREEMENTS
# =========================
disagreements = df[~df["label_match"]].copy()
pair_counts = Counter(zip(disagreements["label_1"], disagreements["label_2"]))

# =========================
# LABEL AGREEMENT BY CONFIDENCE PATTERN
# =========================
if n_conf > 0:
    agreement_by_conf = (
        df_conf.groupby(["confidence_1", "confidence_2"])["label_match"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "label_agreement_rate"})
    )
else:
    agreement_by_conf = pd.DataFrame(
        columns=["confidence_1", "confidence_2", "count", "label_agreement_rate"]
    )

# =========================
# SAVE MERGED CSV
# =========================
df.to_csv(merged_output_file, index=False)

# =========================
# SAVE RESULTS TO TXT
# =========================
with open(report_output_file, "w", encoding="utf-8") as f:
    f.write("===== ANNOTATION AGREEMENT RESULTS =====\n\n")

    f.write(f"Number of doubly annotated samples: {n_total}\n\n")

    f.write("=== LABEL AGREEMENT (ALL SAMPLES) ===\n")
    f.write(f"Percent agreement: {label_agreement:.4f} ({label_agreement * 100:.2f}%)\n")
    f.write(f"Cohen's kappa:     {label_kappa:.4f}\n\n")

    f.write("Label confusion matrix:\n")
    f.write(cm_labels_df.to_string())
    f.write("\n\n")

    f.write("=== CONFIDENCE AGREEMENT ===\n")
    f.write(f"Rows with confidence from both annotators: {n_conf}\n")
    f.write(f"Rows excluded from confidence analysis:   {n_conf_excluded}\n\n")

    if n_conf > 0:
        f.write(f"Percent agreement: {conf_agreement:.4f} ({conf_agreement * 100:.2f}%)\n")
        f.write(f"Cohen's kappa:     {conf_kappa:.4f}\n")
        if weighted_conf_kappa is not None:
            f.write(f"Weighted Cohen's kappa: {weighted_conf_kappa:.4f}\n")

        f.write("\nConfidence confusion matrix:\n")
        f.write(cm_conf_df.to_string())
        f.write("\n\n")
    else:
        f.write("No rows had confidence values from both annotators, so confidence agreement was not computed.\n\n")

    f.write("=== DISAGREEMENT SUMMARY ===\n")
    f.write(f"Number of label disagreements: {len(disagreements)}\n")

    if len(disagreements) > 0:
        f.write("\nDisagreement label pairs:\n")
        for pair, count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{pair}: {count}\n")
    f.write("\n")

    f.write("=== LABEL AGREEMENT BY CONFIDENCE ===\n")
    if not agreement_by_conf.empty:
        f.write(agreement_by_conf.to_string(index=False))
        f.write("\n")
    else:
        f.write("No confidence-based agreement table available.\n")