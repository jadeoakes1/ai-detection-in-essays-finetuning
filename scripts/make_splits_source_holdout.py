# scripts/make_splits_source_holdout.py
import argparse
from pathlib import Path
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
RANDOM_STATE = 42

AI_DEV = 1250
AI_TEST = 1250
HUMAN_DEV = 1250
HUMAN_TEST = 1250

# Defaults (override via CLI)
DEFAULT_HUMAN_TRAIN_SOURCE = "persuade_corpus"
DEFAULT_HUMAN_TEST_SOURCE = "train_essays"

# AI sources to hold out for TEST ONLY (override via CLI)
DEFAULT_AI_TEST_SOURCES = "darragh_claude_v6,darragh_claude_v7"

# Optional: cap training set size per class (None = keep all)
DEFAULT_TRAIN_CAP_PER_CLASS = None  # e.g. 25000 if you want

# Base directory = Capstone root (parent of scripts/)
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW_FILE = BASE_DIR / "data" / "raw" / "daigt_v2" / "train_v2_drcat_02.csv"
DEFAULT_OUT_DIR = BASE_DIR / "data" / "splits" / "source_holdout"


# -----------------------------
# Helper: sample equal-ish AI per source (round-robin)
# -----------------------------
def sample_ai_round_robin(ai_df: pd.DataFrame, total_n: int, random_state: int = 42) -> pd.DataFrame:
    sources = sorted(ai_df["source"].unique())

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


def sample_df(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    """Sample n rows (or return all if n is None)."""
    if n is None:
        return df.copy()
    if n > len(df):
        return df.sample(frac=1, random_state=seed).copy()
    return df.sample(n=n, random_state=seed).copy()


def exact_text_overlap(a: pd.DataFrame, b: pd.DataFrame) -> int:
    a_texts = set(a["text"].astype(str).tolist())
    b_texts = set(b["text"].astype(str).tolist())
    return len(a_texts.intersection(b_texts))


def main():
    parser = argparse.ArgumentParser(description="Create DAIGT v2 splits with AI source holdout.")
    parser.add_argument("--raw_file", type=str, default=str(DEFAULT_RAW_FILE))
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))

    parser.add_argument("--ai_dev", type=int, default=AI_DEV)
    parser.add_argument("--ai_test", type=int, default=AI_TEST)
    parser.add_argument("--human_dev", type=int, default=HUMAN_DEV)
    parser.add_argument("--human_test", type=int, default=HUMAN_TEST)

    parser.add_argument("--human_train_source", type=str, default=DEFAULT_HUMAN_TRAIN_SOURCE)
    parser.add_argument("--human_test_source", type=str, default=DEFAULT_HUMAN_TEST_SOURCE)

    parser.add_argument(
        "--ai_test_sources",
        type=str,
        default=DEFAULT_AI_TEST_SOURCES,
        help="Comma-separated AI sources to hold out for test only."
    )

    parser.add_argument(
        "--train_cap_per_class",
        type=int,
        default=-1,
        help="Cap training examples per class. Use -1 for no cap."
    )

    args = parser.parse_args()

    RAW_FILE = Path(args.raw_file)
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_cap = None if args.train_cap_per_class == -1 else args.train_cap_per_class

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(RAW_FILE)

    # Remove mislabeled AI rows from train_essays (same as your balanced script)
    mislabeled_mask = (df["source"] == "train_essays") & (df["label"] == 1)
    num_removed = int(mislabeled_mask.sum())
    if num_removed > 0:
        print(f"Removing {num_removed} mislabeled AI rows from train_essays")
        df = df[~mislabeled_mask].copy()

    # Parse AI test sources
    AI_TEST_SOURCES = [s.strip() for s in args.ai_test_sources.split(",") if s.strip()]
    if not AI_TEST_SOURCES:
        raise ValueError("ai_test_sources is empty. Provide at least one AI test source.")

    # Split by label
    ai_all = df[df["label"] == 1].copy()
    human_all = df[df["label"] == 0].copy()

    # -----------------------------
    # Pools (enforce holdout)
    # -----------------------------
    # Humans: train/dev from one source, test from another source
    human_train_pool = human_all[human_all["source"] == args.human_train_source].copy()
    human_test_pool = human_all[human_all["source"] == args.human_test_source].copy()

    # AI: train/dev from non-heldout sources, test from heldout sources
    ai_test_pool = ai_all[ai_all["source"].isin(AI_TEST_SOURCES)].copy()
    ai_train_pool = ai_all[~ai_all["source"].isin(AI_TEST_SOURCES)].copy()

    print("\n=== SOURCE HOLDOUT SUMMARY (POOLS) ===")
    print(f"Raw file: {RAW_FILE}")
    print(f"Out dir:  {OUT_DIR}")
    print(f"Removed mislabeled AI rows from train_essays: {num_removed}")
    print()
    print(f"Human train pool source: {args.human_train_source} -> {len(human_train_pool)} rows")
    print(f"Human test  pool source: {args.human_test_source} -> {len(human_test_pool)} rows")
    print(f"AI train pool (non-heldout) -> {len(ai_train_pool)} rows")
    print(f"AI test  pool (heldout only) -> {len(ai_test_pool)} rows")
    print("Held-out AI sources (test-only):")
    for s in AI_TEST_SOURCES:
        print(f" - {s}")
    print("=====================================\n")

    print("=== SPLIT TARGETS ===")
    print(f"Dev per class:   Human={args.human_dev}, AI={args.ai_dev}")
    print("Test per class:  BALANCED from heldout pools (max possible)")
    print("=====================\n")

    # -----------------------------
    # Create splits
    # -----------------------------

    # --- Dev (from TRAIN pools only) ---
    # AI dev: round-robin across train sources
    if args.ai_dev > len(ai_train_pool):
        raise ValueError(f"Not enough AI in train pool for ai_dev={args.ai_dev}. Available={len(ai_train_pool)}")
    dev_ai = sample_ai_round_robin(ai_train_pool, args.ai_dev, RANDOM_STATE)
    ai_train_remaining = ai_train_pool.drop(dev_ai.index)

    # Human dev
    if args.human_dev > len(human_train_pool):
        raise ValueError(f"Not enough Human in train pool for human_dev={args.human_dev}. Available={len(human_train_pool)}")
    dev_human = sample_df(human_train_pool, args.human_dev, RANDOM_STATE)
    human_train_remaining = human_train_pool.drop(dev_human.index)

    # --- Test (from TEST pools only) ---
    # Recommended approach: use the maximum BALANCED test possible from heldout pools.
    # (Downsample the larger side; do NOT limit by args.ai_test/args.human_test.)
    test_n = min(len(human_test_pool), len(ai_test_pool))

    if test_n == 0:
        raise ValueError(
            f"Test set would be empty. "
            f"Available: AI_test_pool={len(ai_test_pool)}, Human_test_pool={len(human_test_pool)}"
        )

    print(f"Balanced heldout test_n = min(Human_test_pool, AI_test_pool) = {test_n}")
    print(f"[NOTE] Ignoring --ai_test/--human_test; heldout test uses max balanced test_n={test_n}.")
    
    print("\nAI test pool counts by source:")
    print(ai_test_pool["source"].value_counts())

    print("\nHuman test pool counts by source:")
    print(human_test_pool["source"].value_counts())

    # AI test: round-robin across heldout sources (important!)
    test_ai = sample_ai_round_robin(ai_test_pool, test_n, RANDOM_STATE)

    # Human test: simple sample from designated test source
    test_human = sample_df(human_test_pool, test_n, RANDOM_STATE)

    # --- Train (remaining from TRAIN pools only; optional cap) ---
    train_ai = sample_df(ai_train_remaining, train_cap, RANDOM_STATE)
    train_human = sample_df(human_train_remaining, train_cap, RANDOM_STATE)

    # Balance remaining samples for train dataset
    n = min(len(train_ai), len(train_human))
    train_ai = train_ai.sample(n=n, random_state=RANDOM_STATE)
    train_human = train_human.sample(n=n, random_state=RANDOM_STATE)

    # Combine + shuffle
    train = pd.concat([train_human, train_ai]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    dev = pd.concat([dev_human, dev_ai]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    test = pd.concat([test_human, test_ai]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # -----------------------------
    # Leakage checks (mirror your train script checks)
    # -----------------------------
    overlap_train_dev = exact_text_overlap(train, dev)
    overlap_train_test = exact_text_overlap(train, test)
    print(f"Exact text overlap train↔dev:  {overlap_train_dev}")
    print(f"Exact text overlap train↔test: {overlap_train_test}")

    # Save
    train.to_csv(OUT_DIR / "train.csv", index=False)
    dev.to_csv(OUT_DIR / "dev.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)

    print("\nSplits created successfully (source holdout).")
    print(f"Train size: {len(train)} (Human={sum(train.label==0)}, AI={sum(train.label==1)})")
    print(f"Dev size:   {len(dev)}   (Human={sum(dev.label==0)}, AI={sum(dev.label==1)})")
    print(f"Test size:  {len(test)}  (Human={sum(test.label==0)}, AI={sum(test.label==1)})")

    print("\nTrain human sources:")
    print(train[train["label"] == 0]["source"].value_counts())
    print("\nTrain AI sources:")
    print(train[train["label"] == 1]["source"].value_counts())

    print("\nTest human sources:")
    print(test[test["label"] == 0]["source"].value_counts())
    print("\nTest AI sources:")
    print(test[test["label"] == 1]["source"].value_counts())

    # SAVE STATS (like balanced)
    stats_path = OUT_DIR / "stats.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("=== SOURCE HOLDOUT SUMMARY (POOLS) ===\n")
        f.write(f"Raw file: {RAW_FILE}\n")
        f.write(f"Out dir:  {OUT_DIR}\n")
        f.write(f"Removed mislabeled AI rows from train_essays: {num_removed}\n\n")
        f.write(f"Human train pool source: {args.human_train_source} -> {len(human_train_pool)} rows\n")
        f.write(f"Human test  pool source: {args.human_test_source} -> {len(human_test_pool)} rows\n")
        f.write(f"AI train pool (non-heldout) -> {len(ai_train_pool)} rows\n")
        f.write(f"AI test  pool (heldout only) -> {len(ai_test_pool)} rows\n")
        f.write("Held-out AI sources (test-only):\n")
        for s in AI_TEST_SOURCES:
            f.write(f" - {s}\n")
        f.write("\n")

        f.write("\n=== TEST BALANCING ===\n")
        f.write(f"test_n = min(len(human_test_pool), len(ai_test_pool)) = {test_n}\n")
        f.write(f"NOTE: Ignoring --ai_test/--human_test; heldout test uses max balanced test_n={test_n}.\n\n")

        f.write("=== SPLIT SIZES ===\n")
        f.write(f"Train size: {len(train)} (Human={sum(train.label==0)}, AI={sum(train.label==1)})\n")
        f.write(f"Dev size:   {len(dev)} (Human={sum(dev.label==0)}, AI={sum(dev.label==1)})\n")
        f.write(f"Test size:  {len(test)} (Human={sum(test.label==0)}, AI={sum(test.label==1)})\n\n")

        f.write("=== LEAKAGE CHECKS ===\n")
        f.write(f"Exact text overlap train↔dev:  {overlap_train_dev}\n")
        f.write(f"Exact text overlap train↔test: {overlap_train_test}\n\n")

        f.write("=== TRAIN HUMAN SOURCES ===\n")
        f.write(train[train["label"] == 0]["source"].value_counts().to_string())
        f.write("\n\n=== TRAIN AI SOURCES ===\n")
        f.write(train[train["label"] == 1]["source"].value_counts().to_string())
        f.write("\n\n=== TEST HUMAN SOURCES ===\n")
        f.write(test[test["label"] == 0]["source"].value_counts().to_string())
        f.write("\n\n=== TEST AI SOURCES ===\n")
        f.write(test[test["label"] == 1]["source"].value_counts().to_string())
        f.write("\n")

    print(f"\nWrote split stats to: {stats_path}")


if __name__ == "__main__":
    main()