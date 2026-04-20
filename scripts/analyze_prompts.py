import pandas as pd
from pathlib import Path
from datetime import datetime

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))

def main():
    input_path = "data/raw/daigt_v2/train_v2_drcat_02.csv"
    output_dir = Path("analysis_outputs")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"daigt_prompt_analysis_{timestamp}.txt"

    df = pd.read_csv(input_path)

    with open(output_file, "w", encoding="utf-8") as f:

        def write(text):
            print(text)
            f.write(text + "\n")

        write("=== DAIGT V2 PROMPT ANALYSIS ===")
        write(f"Input file: {input_path}")
        write(f"Timestamp: {timestamp}")
        write("")

        # Basic stats
        write("=== Basic Counts ===")
        write(f"Rows: {len(df)}")
        write(f"Label distribution:\n{df['label'].value_counts()}\n")
        write(f"Unique sources: {df['source'].nunique()}")
        write(f"Unique prompts: {df['prompt_name'].nunique()}")
        write("")

        # Prompt overlap across labels (global)
        human_prompts = set(df.loc[df["label"] == 0, "prompt_name"].dropna().unique())
        ai_prompts    = set(df.loc[df["label"] == 1, "prompt_name"].dropna().unique())

        write("=== Prompt Overlap Across Labels (GLOBAL) ===")
        write(f"Human unique prompts: {len(human_prompts)}")
        write(f"AI unique prompts: {len(ai_prompts)}")
        write(f"Overlap prompts: {len(human_prompts & ai_prompts)}")
        write(f"Jaccard(human, ai): {round(jaccard(human_prompts, ai_prompts), 4)}")
        write("")

        human_only = sorted(list(human_prompts - ai_prompts))
        ai_only = sorted(list(ai_prompts - human_prompts))

        write("Human-only prompts (first 30):")
        write(str(human_only[:30]))
        write("")
        write("AI-only prompts (first 30):")
        write(str(ai_only[:30]))
        write("")

        # Prompt exclusivity leakage (global)
        prompt_to_labels = df.groupby("prompt_name")["label"].nunique()
        exclusive_prompts = set(prompt_to_labels[prompt_to_labels == 1].index)
        leakage_rows = df["prompt_name"].isin(exclusive_prompts).mean()

        write("=== Prompt Exclusivity Leakage (GLOBAL) ===")
        write(f"Exclusive prompts: {len(exclusive_prompts)} / {df['prompt_name'].nunique()}")
        write(f"Fraction of rows from exclusive prompts: {round(leakage_rows, 4)}")
        write("")

        # Source x Label
        write("=== Source x Label Counts ===")
        src_ct = pd.crosstab(df["source"], df["label"])
        write(str(src_ct))
        write("")

        # NEW: source purity / label-locked sources
        write("=== Source Purity / Label-locked Sources ===")
        src_label_nunique = df.groupby("source")["label"].nunique()
        label_locked_sources = src_label_nunique[src_label_nunique == 1].index.tolist()

        write(f"Label-locked sources (label nunique == 1): {len(label_locked_sources)} / {df['source'].nunique()}")
        write("Label-locked source list:")
        write(str(label_locked_sources))
        write("")

        locked_rows_frac = df["source"].isin(label_locked_sources).mean()
        write(f"Fraction of rows from label-locked sources: {round(locked_rows_frac, 4)}")
        write("")

        # ------------------------------------------------------------
        # Source -> prompts breakdown
        # ------------------------------------------------------------
        write("=== Source -> Prompt Breakdown ===")

        # Avoid NaNs
        df_src = df.dropna(subset=["source", "prompt_name"]).copy()

        for source, sdf in df_src.groupby("source"):
            n_rows = len(sdf)
            n_prompts = sdf["prompt_name"].nunique()
            label_counts = sdf["label"].value_counts().to_dict()

            s_human_prompts = set(sdf.loc[sdf["label"] == 0, "prompt_name"].unique())
            s_ai_prompts    = set(sdf.loc[sdf["label"] == 1, "prompt_name"].unique())
            s_overlap = len(s_human_prompts & s_ai_prompts)
            s_jacc = jaccard(s_human_prompts, s_ai_prompts)

            top_prompts = sdf["prompt_name"].value_counts().head(10)

            write(f"--- Source: {source} ---")
            write(f"Rows: {n_rows}")
            write(f"Label counts: {label_counts}  (0=Human, 1=AI)")
            write(f"Unique prompts: {n_prompts}")
            write(f"Human prompts: {len(s_human_prompts)} | AI prompts: {len(s_ai_prompts)}")
            write(f"Overlap prompts within source: {s_overlap}")
            write(f"Jaccard(human, ai) within source: {round(s_jacc, 4)}")
            write("Top prompts (count):")
            for prompt_name, cnt in top_prompts.items():
                write(f"  {prompt_name}: {cnt}")
            write("")

    print(f"\nSaved analysis to: {output_file}")

if __name__ == "__main__":
    main()
