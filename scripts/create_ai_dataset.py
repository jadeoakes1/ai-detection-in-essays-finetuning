#!/usr/bin/env python3

import csv
from pathlib import Path


def collect_essays(input_dir):
    root = Path(input_dir)

    rows = []
    essay_id = 0

    for provider_dir in sorted(root.iterdir()):
        if not provider_dir.is_dir():
            continue

        provider = provider_dir.name

        provider_to_source = {
            "openai": "gpt-5.4",
            "anthropic": "claude-sonnet-4-6",
            "gemini": "gemini-3-flash"
        }

        source = provider_to_source.get(provider, provider)

        for prompt_dir in sorted(provider_dir.iterdir()):
            if not prompt_dir.is_dir():
                continue

            prompt_raw = prompt_dir.name  # "prompt_1"
            prompt_num = prompt_raw.split("_")[1]  # "1"
            prompt_id = f"p{prompt_num}"  # "p1"

            for file in sorted(prompt_dir.glob("*.txt")):
                text = file.read_text(encoding="utf-8").strip()

                rows.append({
                "essay_id": essay_id,
                "text": text,
                "label": 1,
                "prompt_id": prompt_id,
                "source": source
            })

                essay_id += 1

    return rows


def write_csv(rows, output_file):
    fieldnames = ["essay_id", "text", "label", "prompt_id", "source"]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    rows = collect_essays(args.input_dir)
    write_csv(rows, args.output_file)

    print(f"Wrote {len(rows)} essays to {args.output_file}")


if __name__ == "__main__":
    main()