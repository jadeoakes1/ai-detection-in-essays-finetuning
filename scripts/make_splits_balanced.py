# scripts/make_splits_balanced.py
import pandas as pd
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------
RANDOM_STATE = 42

AI_DEV = 1250
AI_TEST = 1250
HUMAN_DEV = 1250
HUMAN_TEST = 1250

# Human balancing config
HUMAN_TEST_SOURCE_ALWAYS_INCLUDE = "train_essays"     # human source to always include
HUMAN_TRAIN_SOURCE_SAMPLE_FROM = "persuade_corpus"    # human source to downsample

# Base directory = Capstone root
BASE_DIR = Path(__file__).resolve().parent.parent  # parent of scripts/

# Paths
RAW_FILE = BASE_DIR / "data" / "raw" / "daigt_v2" / "train_v2_drcat_02.csv"
OUT_DIR = BASE_DIR / "data" / "splits" / "balanced"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Helper: sample equal AI per source (round-robin)
# -----------------------------
def sample_ai_round_robin(ai_df: pd.DataFrame, total_n: int, random_state: int = 42) -> pd.DataFrame:
    sources = list(ai_df["source"].unique())

    # Shuffle indices per source
    per_source_idx = {
        s: ai_df[ai_df["source"] == s].sample(frac=1, random_state=random_state).index.tolist()
        for s in sources
    }

    picked_idx = []
    while len(picked_idx) < total_n:
        made_progress = False
        for s in sources:
            if per_source_idx[s]:
                picked_idx.append(per_source_idx[s].pop(0))
                made_progress = True
                if len(picked_idx) == total_n:
                    break
        if not made_progress:
            break

    return ai_df.loc[picked_idx].copy()

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(RAW_FILE)

# Remove mislabeled AI rows from train_essays
mislabeled_mask = (df["source"] == "train_essays") & (df["label"] == 1)
num_removed = mislabeled_mask.sum()

if num_removed > 0:
    print(f"Removing {num_removed} mislabeled AI rows from train_essays")
    df = df[~mislabeled_mask].copy()

# Split by label
ai_all = df[df["label"] == 1].copy()
human_all = df[df["label"] == 0].copy()

# -----------------------------
# Balance HUMAN pool to match AI count
# -----------------------------
# Always-include human source (e.g., train_essays)
human_fixed = human_all[human_all["source"] == HUMAN_TEST_SOURCE_ALWAYS_INCLUDE].copy()

# Sample-from human source (e.g., persuade_corpus)
human_sample_from = human_all[human_all["source"] == HUMAN_TRAIN_SOURCE_SAMPLE_FROM].copy()

# (Optional) If you want to allow other human sources too, you could add them here.
# For now, we only use train_essays + persuade_corpus as you requested.

target_human_total = len(ai_all)
need_from_persuade = target_human_total - len(human_fixed)

if need_from_persuade < 0:
    # train_essays alone exceeds AI count; downsample train_essays to match AI
    print(
        f"[WARN] {HUMAN_TEST_SOURCE_ALWAYS_INCLUDE} has {len(human_fixed)} rows, "
        f"which is > AI count ({target_human_total}). Downsampling {HUMAN_TEST_SOURCE_ALWAYS_INCLUDE}."
    )
    human_fixed = human_fixed.sample(n=target_human_total, random_state=RANDOM_STATE)
    human_balanced = human_fixed
else:
    if need_from_persuade > len(human_sample_from):
        raise ValueError(
            f"Not enough rows in {HUMAN_TRAIN_SOURCE_SAMPLE_FROM}: "
            f"need {need_from_persuade}, but only have {len(human_sample_from)}"
        )

    human_persuade_sample = human_sample_from.sample(n=need_from_persuade, random_state=RANDOM_STATE)
    human_balanced = pd.concat([human_fixed, human_persuade_sample], ignore_index=True)

# Shuffle balanced human pool
human_balanced = human_balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print("=== BALANCING SUMMARY ===")
print(f"AI total: {len(ai_all)}")
print(f"Human fixed ({HUMAN_TEST_SOURCE_ALWAYS_INCLUDE}): {len(human_fixed)}")
print(f"Human sampled ({HUMAN_TRAIN_SOURCE_SAMPLE_FROM}): {len(human_balanced) - len(human_fixed)}")
print(f"Human balanced total: {len(human_balanced)}")
print("=========================")

# -----------------------------
# Create splits
# -----------------------------

# --- AI splits ---
dev_ai = sample_ai_round_robin(ai_all, AI_DEV, RANDOM_STATE)
ai_remaining = ai_all.drop(dev_ai.index)

test_ai = sample_ai_round_robin(ai_remaining, AI_TEST, RANDOM_STATE)
ai_remaining = ai_remaining.drop(test_ai.index)

# --- Human splits (from balanced pool) ---
dev_human = human_balanced.sample(n=HUMAN_DEV, random_state=RANDOM_STATE)
human_remaining = human_balanced.drop(dev_human.index)

test_human = human_remaining.sample(n=HUMAN_TEST, random_state=RANDOM_STATE)
human_remaining = human_remaining.drop(test_human.index)

# -----------------------------
# Combine + save
# -----------------------------
train = pd.concat([human_remaining, ai_remaining]).sample(frac=1, random_state=RANDOM_STATE)
dev = pd.concat([dev_ai, dev_human]).sample(frac=1, random_state=RANDOM_STATE)
test = pd.concat([test_ai, test_human]).sample(frac=1, random_state=RANDOM_STATE)

train.to_csv(OUT_DIR / "train.csv", index=False)
dev.to_csv(OUT_DIR / "dev.csv", index=False)
test.to_csv(OUT_DIR / "test.csv", index=False)

print("\nSplits created successfully (balanced classes).")
print(f"Train size: {len(train)} (Human={sum(train.label==0)}, AI={sum(train.label==1)})")
print(f"Dev size:   {len(dev)}   (Human={sum(dev.label==0)}, AI={sum(dev.label==1)})")
print(f"Test size:  {len(test)}  (Human={sum(test.label==0)}, AI={sum(test.label==1)})")

print("\nTrain human sources:")
print(train[train["label"] == 0]["source"].value_counts())
print("\nTrain AI sources:")
print(train[train["label"] == 1]["source"].value_counts())

# SAVE STATS
stats_path = OUT_DIR / "stats.txt"
with open(stats_path, "w", encoding="utf-8") as f:
    f.write("=== BALANCING SUMMARY ===\n")
    f.write(f"AI total: {len(ai_all)}\n")
    f.write(f"Human fixed ({HUMAN_TEST_SOURCE_ALWAYS_INCLUDE}): {len(human_fixed)}\n")
    f.write(f"Human sampled ({HUMAN_TRAIN_SOURCE_SAMPLE_FROM}): {len(human_balanced) - len(human_fixed)}\n")
    f.write(f"Human balanced total: {len(human_balanced)}\n\n")

    f.write("=== SPLIT SIZES ===\n")
    f.write(f"Train size: {len(train)} (Human={sum(train.label==0)}, AI={sum(train.label==1)})\n")
    f.write(f"Dev size:   {len(dev)} (Human={sum(dev.label==0)}, AI={sum(dev.label==1)})\n")
    f.write(f"Test size:  {len(test)} (Human={sum(test.label==0)}, AI={sum(test.label==1)})\n\n")

    f.write("=== TRAIN HUMAN SOURCES (top 20) ===\n")
    f.write(train[train["label"] == 0]["source"].value_counts().to_string())
    f.write("\n\n=== TRAIN AI SOURCES (top 20) ===\n")
    f.write(train[train["label"] == 1]["source"].value_counts().to_string())
    f.write("\n")
print(f"\nWrote split stats to: {stats_path}")