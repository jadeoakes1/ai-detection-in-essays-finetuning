import pandas as pd
from pathlib import Path
import sys


OUT_FILE = Path("data") / "daigtv2_stats_2.txt"


def inspect_full_dataset():
    print("---------------------------------------------")
    print("FULL DATASET STATS")
    print("---------------------------------------------")

    df = pd.read_csv("data/raw/daigt_v2/train_v2_drcat_02.csv")

    print(df.columns)
    print(df.head())
    print()

    print("Overall label counts:")
    print(df["label"].value_counts())
    print()

    print("Overall source counts:")
    print(df["source"].value_counts())
    print()

    counts_by_source_label = df.groupby(["source", "label"]).size().unstack(fill_value=0)
    counts_by_source_label = counts_by_source_label.rename(columns={0: "Human", 1: "AI"})

    print("Counts by source and label:")
    print(counts_by_source_label)
    print()

    total_human = df[df["label"] == 0].shape[0]
    total_ai = df[df["label"] == 1].shape[0]

    print(f"Total human rows: {total_human}")
    print(f"Total AI rows: {total_ai}")
    print(f"Total rows: {len(df)}")
    print()

    return df


def inspect_text_lengths(df):
    print("---------------------------------------------")
    print("TEXT LENGTH STATS (ALL ROWS)")
    print("---------------------------------------------")

    pd.options.display.float_format = "{:.2f}".format

    char_lens = df["text"].astype(str).str.len()
    word_lens = df["text"].astype(str).str.split().str.len()

    print("Character length summary:")
    print(char_lens.describe())
    print()

    print("Word length summary:")
    print(word_lens.describe())
    print()

    print("Empty-text rows:", int((char_lens == 0).sum()))
    print("NaN-text rows:", int(df["text"].isna().sum()))
    print()

    pd.reset_option("display.float_format")


def inspect_text_lengths_by_label(df):
    print("---------------------------------------------")
    print("TEXT LENGTH STATS BY LABEL")
    print("---------------------------------------------")

    tmp = df.copy()
    tmp["char_len"] = tmp["text"].astype(str).str.len()
    tmp["word_len"] = tmp["text"].astype(str).str.split().str.len()

    stats = tmp.groupby("label")[["char_len", "word_len"]].agg(["mean", "median"])
    stats.index = stats.index.map({0: "Human", 1: "AI"})

    pd.options.display.float_format = "{:.2f}".format

    print(stats)
    print()

    pd.reset_option("display.float_format")
    


def inspect_splits():
    BASE_DIR = Path("data/splits/balanced")
    split_files = ["dev.csv", "test.csv", "train.csv"]

    for split_file in split_files:
        path = BASE_DIR / split_file
        df_split = pd.read_csv(path)

        print(f"--- {split_file.upper()} STATS ---")
        print("Total rows:", len(df_split))

        print("\nLabel counts:")
        print(df_split["label"].value_counts())

        counts_by_source_label = df_split.groupby(["source", "label"]).size().unstack(fill_value=0)
        counts_by_source_label = counts_by_source_label.rename(columns={0: "Human", 1: "AI"})

        print("\nCounts by source and label:")
        print(counts_by_source_label)

        print("\n" + "-" * 50 + "\n")


def inspect_train_essays(df):
    df_dev = pd.read_csv("data/splits/balanced/dev.csv")
    train_essays_rows = df_dev[df_dev["source"] == "train_essays"]

    print("---------------------------------------------")
    print("INSPECT train_essays SOURCE")
    print("---------------------------------------------")
    print(train_essays_rows.head())
    print("Number of train_essays rows:", len(train_essays_rows))
    print()

    pd.set_option("display.max_colwidth", None)

    train_essays_ai = df[(df["source"] == "train_essays") & (df["label"] == 1)]

    print("---------------------------------------------")
    print("TRAIN_ESSAYS AI-LABELED ROWS")
    print("---------------------------------------------")
    print(train_essays_ai)
    print("\nNumber of AI-labeled train_essays rows:", len(train_essays_ai))

    pd.reset_option("display.max_colwidth")


def inspect_prompts(df):
    print("---------------------------------------------")
    print("PROMPT STATS")
    print("---------------------------------------------")

    if "prompt_name" not in df.columns:
        print("Column 'prompt_name' not found in dataset.")
        print()
        return

    print("Number of unique prompts:", df["prompt_name"].nunique())
    print()

    print("Prompt counts:")
    print(df["prompt_name"].value_counts())
    print()

    print("Prompt counts by label:")
    prompt_label_counts = (
        df.groupby(["prompt_name", "label"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "Human", 1: "AI"})
        .sort_values(by=["Human", "AI"], ascending=False)
    )
    print(prompt_label_counts)
    print()

    print("Prompts that appear in BOTH Human and AI:")
    both_labels = prompt_label_counts[
        (prompt_label_counts["Human"] > 0) & (prompt_label_counts["AI"] > 0)
    ]
    print(f"Count: {len(both_labels)}")
    print(both_labels)
    print()

    print("Prompts that appear ONLY in Human:")
    human_only = prompt_label_counts[
        (prompt_label_counts["Human"] > 0) & (prompt_label_counts["AI"] == 0)
    ]
    print(f"Count: {len(human_only)}")
    print(human_only)
    print()

    print("Prompts that appear ONLY in AI:")
    ai_only = prompt_label_counts[
        (prompt_label_counts["Human"] == 0) & (prompt_label_counts["AI"] > 0)
    ]
    print(f"Count: {len(ai_only)}")
    print(ai_only)
    print()


def inspect_balanced_split_word_counts():
    print("---------------------------------------------")
    print("BALANCED SPLIT WORD COUNT STATS")
    print("---------------------------------------------")

    base_dir = Path("data/splits/balanced")
    split_files = ["train.csv", "dev.csv", "test.csv"]

    dfs = []
    for split_file in split_files:
        path = base_dir / split_file
        df_split = pd.read_csv(path)
        df_split["split"] = split_file.replace(".csv", "")
        dfs.append(df_split)

    data = pd.concat(dfs, ignore_index=True)
    data["word_count"] = data["text"].astype(str).str.split().str.len()

    def summarize(df, name):
        print(f"===== {name} =====")
        print(f"N = {len(df)}")

        wc = df["word_count"]
        print(f"mean:   {wc.mean():.2f}")
        print(f"median: {wc.median():.2f}")
        print(f"min:    {wc.min()}")
        print(f"max:    {wc.max()}")
        print(f"std:    {wc.std():.2f}")
        print()

    # overall
    summarize(data, "OVERALL")

    # by split
    for split in ["train", "dev", "test"]:
        summarize(data[data["split"] == split], f"SPLIT: {split}")

    # by label
    for label in sorted(data["label"].unique()):
        label_name = "Human" if label == 0 else "AI"
        summarize(data[data["label"] == label], f"LABEL: {label_name}")

    # by source
    print("===== BY SOURCE =====")
    source_stats = (
        data.groupby("source")["word_count"]
        .agg(["mean", "median", "min", "max", "count"])
        .sort_values(by="mean", ascending=False)
    )
    print(source_stats)
    print()


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        sys.stdout = f

        df = inspect_full_dataset()
        inspect_text_lengths(df)
        inspect_text_lengths_by_label(df)
        inspect_balanced_split_word_counts()
        inspect_prompts(df)
        inspect_splits()
        inspect_train_essays(df)

        sys.stdout = sys.__stdout__

    print(f"All stats saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
