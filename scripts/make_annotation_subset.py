# This is for the human annotation part of the project - takes samples from the test set

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True,
                        help="Path to the labeled test CSV")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save annotation files")
    parser.add_argument("--n_per_class", type=int, default=100,
                        help="Number of samples per class")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--text_col", type=str, default="text",
                        help="Name of text column")
    parser.add_argument("--label_col", type=str, default="label",
                        help="Name of gold label column")
    parser.add_argument("--source_col", type=str, default="source",
                        help="Name of source column")
    parser.add_argument("--model_pred_col", type=str, default=None,
                        help="Optional model prediction column")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.test_file)

    required_cols = [args.text_col, args.label_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {args.test_file}")

    # Balanced sample: n_per_class from each label
    human_df = df[df[args.label_col] == 0]
    ai_df = df[df[args.label_col] == 1]

    if len(human_df) < args.n_per_class:
        raise ValueError(f"Not enough human samples: requested {args.n_per_class}, found {len(human_df)}")
    if len(ai_df) < args.n_per_class:
        raise ValueError(f"Not enough AI samples: requested {args.n_per_class}, found {len(ai_df)}")

    human_sample = human_df.sample(n=args.n_per_class, random_state=args.seed)
    ai_sample = ai_df.sample(n=args.n_per_class, random_state=args.seed)

    subset = pd.concat([human_sample, ai_sample], axis=0)
    subset = subset.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Build master file
    master = pd.DataFrame({
        "id": range(len(subset)),
        "text": subset[args.text_col].values,
        "gold_label": subset[args.label_col].values
    })

    if args.source_col in subset.columns:
        master["source"] = subset[args.source_col].values

    if args.model_pred_col is not None:
        if args.model_pred_col not in subset.columns:
            raise ValueError(f"Model prediction column '{args.model_pred_col}' not found")
        master["model_prediction"] = subset[args.model_pred_col].values

    master_path = out_dir / "annotation_subset_master.csv"
    master.to_csv(master_path, index=False)

    # Create annotator templates
    template = master[["id", "text"]].copy()
    template["label_guess"] = ""
    template["confidence"] = ""

    ann1_path = out_dir / "annotator1_template.csv"
    ann2_path = out_dir / "annotator2_template.csv"

    template.to_csv(ann1_path, index=False)
    template.to_csv(ann2_path, index=False)

    print("Created annotation files:")
    print(f"  Master:      {master_path}")
    print(f"  Annotator 1: {ann1_path}")
    print(f"  Annotator 2: {ann2_path}")
    print()
    print("Subset summary:")
    print(master["gold_label"].value_counts().sort_index())
    print()
    print("Label mapping:")
    print("  0 = Human")
    print("  1 = AI")


if __name__ == "__main__":
    main()