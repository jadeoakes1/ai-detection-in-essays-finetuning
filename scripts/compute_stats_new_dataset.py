#!/usr/bin/env python3

import re
from pathlib import Path
from collections import defaultdict


def count_words(text):
    return len(re.findall(r"\b[\w'-]+\b", text))


def compute_stats(root_dir):
    root = Path(root_dir)

    stats = {
        "total_essays": 0,
        "by_provider": defaultdict(int),
        "by_prompt": defaultdict(int),
        "word_counts": [],
        "per_essay": []
    }

    for provider_dir in root.iterdir():
        if not provider_dir.is_dir():
            continue

        provider = provider_dir.name

        for prompt_dir in provider_dir.iterdir():
            if not prompt_dir.is_dir():
                continue

            prompt_id = prompt_dir.name

            for file in prompt_dir.glob("*.txt"):
                text = file.read_text(encoding="utf-8")
                wc = count_words(text)

                stats["total_essays"] += 1
                stats["by_provider"][provider] += 1
                stats["by_prompt"][prompt_id] += 1
                stats["word_counts"].append(wc)

                stats["per_essay"].append({
                    "file": str(file),
                    "provider": provider,
                    "prompt": prompt_id,
                    "word_count": wc
                })

    return stats


def write_stats(stats, output_file):
    lines = []

    lines.append("=== DATASET STATISTICS ===\n")
    lines.append(f"Total essays: {stats['total_essays']}\n")

    # Provider counts
    lines.append("\n--- Essays by provider ---")
    for provider, count in stats["by_provider"].items():
        lines.append(f"{provider}: {count}")

    # Prompt counts
    lines.append("\n--- Essays by prompt ---")
    for prompt, count in stats["by_prompt"].items():
        lines.append(f"{prompt}: {count}")

    # Word count summary
    if stats["word_counts"]:
        wc = stats["word_counts"]

        lines.append("\n--- Word count summary ---")
        lines.append(f"Min: {min(wc)}")
        lines.append(f"Max: {max(wc)}")
        lines.append(f"Mean: {sum(wc)/len(wc):.2f}")

        outside = sum(1 for w in wc if w < 300 or w > 400)
        lines.append(f"Outside 300-400: {outside}")

    # Per-essay breakdown
    lines.append("\n--- Per-essay word counts ---")
    for entry in sorted(stats["per_essay"], key=lambda x: x["file"]):
        path_obj = Path(entry["file"])
        short_name = f"{path_obj.parent.name}/{path_obj.name}"

        lines.append(
            f"{entry['provider']:<10} | {entry['prompt']:<8} | {entry['word_count']:>3} words | {short_name}"
        )

    with open(output_file, "w") as f:
        f.write("\n".join(lines))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_file", default="data/new_dataset/ai_essays/stats.txt")
    args = parser.parse_args()

    stats = compute_stats(args.input_dir)
    write_stats(stats, args.output_file)

    print(f"Stats written to {args.output_file}")


if __name__ == "__main__":
    main()