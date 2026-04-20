import os
import pandas as pd

INPUT_FILE = "data/clean/train_clean_pool.csv"
OUTPUT_DIR = "data/splits/clean_subsets"
SIZES = [2000, 10000, 19310]
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

df["label"] = df["label"].astype(int)

log("===== CLEAN POOL =====")
log(f"Rows: {len(df)}")
log(str(df["label"].value_counts().sort_index()))

df_human = df[df["label"] == 0].copy()
df_ai = df[df["label"] == 1].copy()

max_per_class = min(len(df_human), len(df_ai))
max_total_balanced = max_per_class * 2

log(f"\nMax balanced total available: {max_total_balanced}")
log(f"Max per class: {max_per_class}")

# Shuffle once → nested subsets
human_order = df_human.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
ai_order = df_ai.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

for size in SIZES:
    if size % 2 != 0:
        raise ValueError(f"Size must be even: {size}")

    per_class = size // 2

    if per_class > max_per_class:
        log(f"\nSkipping {size}: not enough samples per class")
        continue

    log(f"\n===== BUILDING {size} =====")
    log(f"Per class: {per_class}")

    human_subset = human_order.iloc[:per_class].copy()
    ai_subset = ai_order.iloc[:per_class].copy()

    subset_df = pd.concat([human_subset, ai_subset], ignore_index=True)
    subset_df = subset_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    out_path = os.path.join(OUTPUT_DIR, f"train_clean_balanced_{size}.csv")
    subset_df.to_csv(out_path, index=False)

    counts = subset_df["label"].value_counts().sort_index()

    log(f"Saved: {out_path}")
    log(str(counts))
    log(f"Rows: {len(subset_df)}")

# Save stats file
stats_path = os.path.join(OUTPUT_DIR, "stats.txt")
with open(stats_path, "w") as f:
    for line in log_lines:
        f.write(line + "\n")

print(f"\nSaved stats to: {stats_path}")