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

# Base directory = Capstone root
BASE_DIR = Path(__file__).resolve().parent.parent  # parent of scripts/

# Paths
RAW_FILE = BASE_DIR / "data" / "raw" / "daigt_v2" / "train_v2_drcat_02.csv"
OUT_DIR = BASE_DIR / "data" / "splits" / "baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv(RAW_FILE)

# Get labels
ai_df = df[df["label"] == 1].copy()       # if 1 = AI
human_df = df[df["label"] == 0].copy()    # if 0 = human


# List of AI sources in deterministic order for round-robin sampling
AI_SOURCES = sorted(ai_df["source"].unique())

# -----------------------------
# Helper: sample equal AI per source
# -----------------------------

import pandas as pd

def sample_ai_round_robin(ai_df, total_n, random_state=42):
    """
    Sample AI data equally from all sources using round-robin.
    """
    rng = pd.Series(random_state).sample  # optional reproducibility
    sources = ai_df["source"].unique()
    
    # Shuffle each source individually
    shuffled = {s: ai_df[ai_df["source"] == s].sample(frac=1, random_state=random_state).copy()
                for s in sources}

    sampled_rows = []
    i = 0
    while len(sampled_rows) < total_n:
        for s in sources:
            if len(shuffled[s]) > 0:
                sampled_rows.append(shuffled[s].iloc[0])
                shuffled[s] = shuffled[s].iloc[1:]  # remove taken row
                if len(sampled_rows) == total_n:
                    break
        i += 1

    return pd.DataFrame(sampled_rows)

# -----------------------------
# Create splits
# -----------------------------

# --- AI splits ---
dev_ai = sample_ai_round_robin(ai_df, AI_DEV, RANDOM_STATE)
ai_df = ai_df.drop(dev_ai.index)

test_ai = sample_ai_round_robin(ai_df, AI_TEST, RANDOM_STATE)
ai_df = ai_df.drop(test_ai.index)

# --- Human splits ---
dev_human = human_df.sample(n=HUMAN_DEV, random_state=RANDOM_STATE)
human_df = human_df.drop(dev_human.index)

test_human = human_df.sample(n=HUMAN_TEST, random_state=RANDOM_STATE)

# -----------------------------
# Combine + save
# -----------------------------

train = pd.concat([human_df, ai_df]).sample(frac=1, random_state=RANDOM_STATE)
dev = pd.concat([dev_ai, dev_human]).sample(frac=1, random_state=RANDOM_STATE)
test = pd.concat([test_ai, test_human]).sample(frac=1, random_state=RANDOM_STATE)

train.to_csv(OUT_DIR / "train.csv", index=False)
dev.to_csv(OUT_DIR / "dev.csv", index=False)
test.to_csv(OUT_DIR / "test.csv", index=False)

print("Splits created successfully")
print(f"Dev set size: {len(dev)}")
print(f"Test set size: {len(test)}")
