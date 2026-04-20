#!/usr/bin/env python
"""
Convert an existing CSV dataset with `text` and `label` columns into
OpenAI or Gemini supervised fine-tuning JSONL format.

Usage examples:

# OpenAI train subset
python scripts/convert_csv_to_ft_jsonl.py \
  --input_csv data/splits/clean_subsets/train_clean_balanced_10000.csv \
  --output_jsonl data/finetune/openai/train_clean_balanced_10000.jsonl \
  --format openai

# Gemini train subset
python scripts/convert_csv_to_ft_jsonl.py \
  --input_csv data/splits/clean_subsets/train_clean_balanced_10000.csv \
  --output_jsonl data/finetune/gemini/train_clean_balanced_10000.jsonl \
  --format gemini

# OpenAI validation from dev.csv
python scripts/convert_csv_to_ft_jsonl.py \
  --input_csv data/splits/balanced/dev.csv \
  --output_jsonl data/finetune/openai/dev_2500.jsonl \
  --format openai

# Gemini validation from dev.csv
python scripts/convert_csv_to_ft_jsonl.py \
  --input_csv data/splits/balanced/dev.csv \
  --output_jsonl data/finetune/gemini/dev_2500.jsonl \
  --format gemini
"""

import argparse
import json
import os
from typing import Any, Dict

import pandas as pd

from ft_formatters import build_openai_record, build_gemini_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to input CSV with text and label columns")
    parser.add_argument("--output_jsonl", required=True, help="Path to output JSONL")
    parser.add_argument("--format", required=True, choices=["openai", "gemini"], help="Target fine-tuning format")
    parser.add_argument("--text_col", default="text", help="Name of text column")
    parser.add_argument("--label_col", default="label", help="Name of label column")
    parser.add_argument(
        "--drop_blank_text",
        action="store_true",
        help="Drop rows where stripped text is empty before export",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    required_cols = {args.text_col, args.label_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df[args.text_col] = df[args.text_col].astype(str)
    df[args.label_col] = df[args.label_col].astype(int)

    if args.drop_blank_text:
        before = len(df)
        df = df[df[args.text_col].str.strip() != ""].copy()
        print(f"Dropped {before - len(df)} blank rows")

    builders = {
        "openai": build_openai_record,
        "gemini": build_gemini_record,
    }
    build_record = builders[args.format]

    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = build_record(row[args.text_col], row[args.label_col])
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts = df[args.label_col].value_counts().sort_index().to_dict()
    print(f"Saved {args.format} JSONL to: {args.output_jsonl}")
    print(f"Rows exported: {len(df)}")
    print(f"Label counts: {counts}")


if __name__ == "__main__":
    main()
