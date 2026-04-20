#!/usr/bin/env python
"""
Batch-convert your existing train subsets plus dev.csv into both OpenAI and Gemini JSONL.

Edit the paths below if needed, then run:
python scripts/batch_convert_ft_datasets.py
"""

from pathlib import Path
import subprocess
import sys


SCRIPT_PATH = Path("scripts/convert_csv_to_ft_jsonl.py")

TRAIN_INPUTS = [
    ("data/splits/clean_subsets/train_clean_balanced_2000.csv", "train_clean_balanced_2000"),
    ("data/splits/clean_subsets/train_clean_balanced_10000.csv", "train_clean_balanced_10000"),
    ("data/splits/clean_subsets/train_clean_balanced_19310.csv", "train_clean_balanced_19310"),
]

DEV_INPUT = ("data/splits/dev.csv", "dev_2500")

TARGETS = [
    ("openai", Path("data/finetune/openai")),
    ("gemini", Path("data/finetune/gemini")),
]


def run_conversion(input_csv: str, stem: str, target_format: str, output_dir: Path) -> None:
    output_jsonl = output_dir / f"{stem}.jsonl"
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--input_csv", input_csv,
        "--output_jsonl", str(output_jsonl),
        "--format", target_format,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    for fmt, out_dir in TARGETS:
        out_dir.mkdir(parents=True, exist_ok=True)

        for input_csv, stem in TRAIN_INPUTS:
            run_conversion(input_csv, stem, fmt, out_dir)

        run_conversion(DEV_INPUT[0], DEV_INPUT[1], fmt, out_dir)

    print("\nDone.")
    print("You now have train subset JSONL files for both OpenAI and Gemini, plus dev_2500.jsonl for validation.")


if __name__ == "__main__":
    main()
