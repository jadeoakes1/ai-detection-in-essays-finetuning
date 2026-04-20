# This is for the annotation scheme - provides examples of data taken from the training set

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to the labeled train CSV")
    parser.add_argument("--out_file", type=str, required=True,
                        help="Path to save the example subset CSV")
    parser.add_argument("--n_per_class", type=int, default=10,
                        help="Number of samples to pull per class")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--text_col", type=str, default="text",
                        help="Name of text column")
    parser.add_argument("--label_col", type=str, default="label",
                        help="Name of label column")
    args = parser.parse_args()

    df = pd.read_csv(args.train_file)

    required_cols = [args.text_col, args.label_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {args.train_file}")

    human_df = df[df[args.label_col] == 0]
    ai_df = df[df[args.label_col] == 1]

    if len(human_df) < args.n_per_class:
        raise ValueError(
            f"Not enough human samples: requested {args.n_per_class}, found {len(human_df)}"
        )
    if len(ai_df) < args.n_per_class:
        raise ValueError(
            f"Not enough AI samples: requested {args.n_per_class}, found {len(ai_df)}"
        )

    human_sample = human_df.sample(n=args.n_per_class, random_state=args.seed).copy()
    ai_sample = ai_df.sample(n=args.n_per_class, random_state=args.seed).copy()

    human_sample["label_name"] = "Human"
    ai_sample["label_name"] = "AI"

    subset = pd.concat([human_sample, ai_sample], axis=0).reset_index(drop=True)

    output_df = subset[[args.text_col, args.label_col, "label_name"]].copy()
    output_df = output_df.rename(columns={
        args.text_col: "text",
        args.label_col: "label"
    })

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(out_path, index=False)

    print(f"Saved example subset to: {out_path}")
    print()
    print("Label counts:")
    print(output_df["label_name"].value_counts())


if __name__ == "__main__":
    main()