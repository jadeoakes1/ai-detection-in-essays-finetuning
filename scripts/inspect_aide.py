import pandas as pd
from pathlib import Path
import sys


OUT_FILE = Path("data") / "aide_stats.txt"


def inspect_full_dataset():
    print("---------------------------------------------")
    print("FULL DATASET STATS (AIDE)")
    print("---------------------------------------------")

    df = pd.read_csv("data/raw/aide/AIDE_train_essays.csv")

    print(df.columns)
    print(df.head())
    print()

    # Overall label counts
    print("Overall label counts (ground truth):")
    print(df["generated"].value_counts())
    print()

    # Overall prompt counts
    print("Overall prompt_id counts:")
    print(df["prompt_id"].value_counts())
    print()

    # Counts by prompt and label
    print("Counts by prompt_id and label (Human vs AI):")
    counts_by_prompt_label = df.groupby(["prompt_id", "generated"]).size().unstack(fill_value=0)
    counts_by_prompt_label = counts_by_prompt_label.rename(columns={0: "Human", 1: "AI"})
    print(counts_by_prompt_label)
    print()

    # Totals
    total_human = (df["generated"] == 0).sum()
    total_ai = (df["generated"] == 1).sum()
    print(f"Total human rows: {total_human}")
    print(f"Total AI rows: {total_ai}")
    print(f"Total rows: {len(df)}")
    print()

    return df


def inspect_text_lengths(df):
    print("---------------------------------------------")
    print("TEXT LENGTH STATS")
    print("---------------------------------------------")

    # Make printing prettier (no huge decimals)
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

    # Reset formatting in case you print later tables
    pd.reset_option("display.float_format")


def inspect_ai_examples(df, n=25):
    print("---------------------------------------------")
    print("AI-LABELED ROWS (generated == 1)")
    print("---------------------------------------------")

    ai_rows = df[df["generated"] == 1]

    print("Number of AI-labeled rows:", len(ai_rows))
    print()

    if len(ai_rows) == 0:
        print("No AI-labeled rows found.")
        print()
        return

    # Make sure the full text prints into the stats file
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    print(f"Showing up to first {n} AI rows (full text):")
    print(ai_rows[["id", "prompt_id", "generated", "text"]].head(n))
    print()

    print("AI rows by prompt_id:")
    print(ai_rows["prompt_id"].value_counts())
    print()

    # Reset so you don't mess up other prints
    pd.reset_option("display.max_colwidth")
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_columns")


def inspect_prompts_file():
    prompts_path = Path("data/raw/aide/train_prompts.csv")
    if not prompts_path.exists():
        print("---------------------------------------------")
        print("PROMPTS FILE")
        print("---------------------------------------------")
        print("train_prompts.csv not found at:", prompts_path)
        print()
        return

    print("---------------------------------------------")
    print("PROMPTS FILE (train_prompts.csv)")
    print("---------------------------------------------")

    prompts = pd.read_csv(prompts_path)
    print(prompts.columns)
    print(prompts.head())
    print()
    print("Number of prompts:", len(prompts))
    print()


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        sys.stdout = f

        df = inspect_full_dataset()
        inspect_text_lengths(df)
        inspect_ai_examples(df, n=25)
        inspect_prompts_file()

        sys.stdout = sys.__stdout__

    print(f"All stats saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
