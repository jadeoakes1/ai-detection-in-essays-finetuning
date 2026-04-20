#!/usr/bin/env python
# scripts/prepare_clean_data.py

import os
import re
import argparse
from typing import List, Tuple

import pandas as pd


SUSPICIOUS_PATTERNS = [
    # obvious AI / meta boilerplate
    r"\bas an ai language model\b",
    r"\bhere(?:'s| is) (?:the )?essay\b",
    r"\bplease grade this essay\b",
    r"\bprovide feedback\b",
    r"\bfictional essay\b",
    r"\bintentional spelling and grammar mistakes\b",
    r"\bmake it more realistic\b",

    # placeholder / template metadata
    r"\[your name\]",
    r"\bstudent_name\b",
    r"\bteacher_name\b",
    r"\bschool_name\b",
    r"\bother_name\b",
    r"\bgeneric_name\b",

    # structured prompt / scaffold leftovers
    r"\btitle:\b",
    r"\bintroduction:\b",
    r"\bconclusion:\b",
    r"\bword count:\b",
    r"\bprompt:\b",
]


def whitespace_token_count(text: str) -> int:
    return len(str(text).split())


def normalize_text_for_dedup(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def has_suspicious_pattern(text: str) -> Tuple[bool, List[str]]:
    matches = []
    lower = str(text).lower()
    for pat in SUSPICIOUS_PATTERNS:
        if re.search(pat, lower):
            matches.append(pat)
    return (len(matches) > 0, matches)


def has_placeholder_metadata(text: str) -> Tuple[bool, List[str]]:
    placeholder_patterns = [
        r"\[your name\]",
        r"\bstudent_name\b",
        r"\bteacher_name\b",
        r"\bschool_name\b",
        r"\bother_name\b",
        r"\bgeneric_name\b",
    ]

    matches = []
    lower = str(text).lower()
    for pat in placeholder_patterns:
        if re.search(pat, lower):
            matches.append(pat)
    return (len(matches) > 0, matches)


def looks_truncated(text: str) -> bool:
    """
    Heuristic only.
    Flags text that seems cut off mid-sentence / mid-token.
    """
    text = str(text).strip()
    if not text:
        return False

    # Ends with ellipsis or abrupt dash
    if re.search(r"(\.\.\.|—|-)\s*$", text):
        return True

    # Last character is not typical sentence-ending punctuation / quote / bracket
    if text[-1] not in {".", "!", "?", "\"", "'", "”", ")", "]"}:
        # If the last token is very short or looks incomplete, flag it
        last_token = text.split()[-1] if text.split() else ""
        if len(last_token) <= 3 or re.match(r".*[a-zA-Z]$", last_token):
            return True

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--review_csv", type=str, required=True)

    parser.add_argument("--train_jsonl", type=str, default=None)
    parser.add_argument("--valid_jsonl", type=str, default=None)

    parser.add_argument("--train_per_class", type=int, default=100)
    parser.add_argument("--valid_per_class", type=int, default=50)

    parser.add_argument("--min_words", type=int, default=150)
    parser.add_argument("--max_words", type=int, default=1200)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean_pool_csv", type=str, default=None)

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    required_cols = {"text", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    # Basic stats
    df["word_count"] = df["text"].apply(whitespace_token_count)
    df["normalized_text"] = df["text"].apply(normalize_text_for_dedup)

    # Duplicate detection
    dup_mask = df["normalized_text"].duplicated(keep=False)
    df["is_duplicate"] = dup_mask.astype(int)

    # Suspicious pattern detection
    suspicious_flags = df["text"].apply(has_suspicious_pattern)
    df["has_suspicious_pattern"] = suspicious_flags.apply(lambda x: int(x[0]))
    df["suspicious_matches"] = suspicious_flags.apply(lambda x: " | ".join(x[1]))

    # Placeholder metadata detection
    placeholder_flags = df["text"].apply(has_placeholder_metadata)
    df["has_placeholder_metadata"] = placeholder_flags.apply(lambda x: int(x[0]))
    df["placeholder_matches"] = placeholder_flags.apply(lambda x: " | ".join(x[1]))

    # Truncation heuristic
    df["looks_truncated"] = df["text"].apply(lambda x: int(looks_truncated(x)))

    # Empty / length filters
    df["is_empty"] = df["text"].str.strip().eq("").astype(int)
    df["too_short"] = (df["word_count"] < args.min_words).astype(int)
    df["too_long"] = (df["word_count"] > args.max_words).astype(int)

    # Keep source if available
    if "source" not in df.columns:
        df["source"] = ""

    # Overall bad flag
    df["auto_exclude"] = (
        (df["is_empty"] == 1) |
        (df["too_short"] == 1) |
        (df["too_long"] == 1) |
        (df["is_duplicate"] == 1) |
        (df["has_suspicious_pattern"] == 1) |
        (df["has_placeholder_metadata"] == 1) |
        (df["looks_truncated"] == 1)
    ).astype(int)

    # Candidate pool = rows not auto-excluded
    clean_df = df[df["auto_exclude"] == 0].copy()
    if args.clean_pool_csv:
        if os.path.dirname(args.clean_pool_csv):
            os.makedirs(os.path.dirname(args.clean_pool_csv), exist_ok=True)
        clean_df.to_csv(args.clean_pool_csv, index=False)
        print(f"Saved clean pool CSV to: {args.clean_pool_csv}")

    print("\n===== FULL DATASET =====")
    print(f"Rows: {len(df)}")
    print(df["label"].value_counts().sort_index())

    print("\n===== AUTO-EXCLUDE COUNTS =====")
    for col in [
        "is_empty", "too_short", "too_long", "is_duplicate",
        "has_suspicious_pattern", "has_placeholder_metadata",
        "looks_truncated", "auto_exclude"
    ]:
        print(f"{col}: {int(df[col].sum())}")

    print("\n===== CLEAN POOL =====")
    print(f"Rows: {len(clean_df)}")
    print(clean_df["label"].value_counts().sort_index())

    # Build review sample: slightly oversample so you can manually choose
    review_per_class = (args.train_per_class + args.valid_per_class) * 2

    human_pool = clean_df[clean_df["label"] == 0]
    ai_pool = clean_df[clean_df["label"] == 1]

    if len(human_pool) < review_per_class:
        print(f"\nWarning: only {len(human_pool)} clean Human rows available for review.")
    if len(ai_pool) < review_per_class:
        print(f"Warning: only {len(ai_pool)} clean AI rows available for review.")

    human_review = human_pool.sample(
        n=min(review_per_class, len(human_pool)),
        random_state=args.seed
    )
    ai_review = ai_pool.sample(
        n=min(review_per_class, len(ai_pool)),
        random_state=args.seed
    )

    review_df = pd.concat([human_review, ai_review], ignore_index=True)
    review_df = review_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Add manual review columns
    review_df["split"] = ""   # train or valid
    review_df["review_notes"] = ""

    # Assign splits automatically (balanced)
    train_n = args.train_per_class
    valid_n = args.valid_per_class

    train_rows = []
    valid_rows = []

    for label in [0, 1]:
        label_df = review_df[review_df["label"] == label].sample(
            n=train_n + valid_n,
            random_state=args.seed
        ).reset_index(drop=True)

        train_part = label_df.iloc[:train_n].copy()
        valid_part = label_df.iloc[train_n:].copy()

        train_part["split"] = "train"
        valid_part["split"] = "valid"

        train_rows.append(train_part)
        valid_rows.append(valid_part)

    final_df = pd.concat(train_rows + valid_rows).reset_index(drop=True)
    review_df = final_df

    # Helpful output columns
    out_cols = [
        "label", "source", "word_count",
        "is_duplicate",
        "has_suspicious_pattern", "suspicious_matches",
        "has_placeholder_metadata", "placeholder_matches",
        "looks_truncated",
        "text",
        "split", "review_notes"
    ]
    if os.path.dirname(args.review_csv):
        os.makedirs(os.path.dirname(args.review_csv), exist_ok=True)
    review_df[out_cols].to_csv(args.review_csv, index=False)
    print(f"\nSaved review CSV to: {args.review_csv}")


if __name__ == "__main__":
    main()