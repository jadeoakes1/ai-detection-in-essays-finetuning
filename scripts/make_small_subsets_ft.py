#!/usr/bin/env python
"""
Create random train/dev CSV subsets from a labeled pool.

- Train: 200 samples
- Dev: 100 samples
- No overlap between splits
- Preserves original columns
"""

import os
import pandas as pd

INPUT_FILE = "data/clean/train_clean_pool.csv"
OUTPUT_DIR = "data/splits/balanced"
TRAIN_SIZE = 200
DEV_SIZE = 100
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(msg)


df = pd.read_csv(INPUT_FILE)

required_cols = {"text", "label"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df = df.copy()
df["label"] = df["label"].astype(int)

log("===== INPUT DATA =====")
log(f"Rows: {len(df)}")
log(str(df["label"].value_counts().sort_index()))

total_needed = TRAIN_SIZE + DEV_SIZE
if len(df) < total_needed:
    raise ValueError(f"Need at least {total_needed} rows, found {len(df)}")

# Shuffle once so train/dev are random and non-overlapping
df_shuffled = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

train_df = df_shuffled.iloc[:TRAIN_SIZE].copy()
dev_df = df_shuffled.iloc[TRAIN_SIZE:TRAIN_SIZE + DEV_SIZE].copy()

train_path = os.path.join(OUTPUT_DIR, "train.csv")
dev_path = os.path.join(OUTPUT_DIR, "dev.csv")

train_df.to_csv(train_path, index=False)
dev_df.to_csv(dev_path, index=False)

log("\n===== TRAIN =====")
log(f"Saved: {train_path}")
log(f"Rows: {len(train_df)}")
log(str(train_df['label'].value_counts().sort_index()))

log("\n===== DEV =====")
log(f"Saved: {dev_path}")
log(f"Rows: {len(dev_df)}")
log(str(dev_df['label'].value_counts().sort_index()))

stats_path = os.path.join(OUTPUT_DIR, "stats.txt")
with open(stats_path, "w", encoding="utf-8") as f:
    for line in log_lines:
        f.write(line + "\n")

print(f"\nSaved stats to: {stats_path}")