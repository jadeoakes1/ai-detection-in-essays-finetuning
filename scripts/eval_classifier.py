#!/usr/bin/env python
# scripts/eval_classifier.py

import argparse
import os
import json
from datetime import datetime
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


class EssayDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def evaluate(model, loader, device, target_names=None):
    model.eval()
    preds, true_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask=attention_mask).logits
            batch_preds = torch.argmax(logits, dim=-1)

            preds.extend(batch_preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(true_labels, preds)
    report = classification_report(
        true_labels,
        preds,
        labels=[0, 1],
        target_names=target_names,
        zero_division=0
    )
    return acc, report, preds, true_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--save_predictions", type=str, default=None)
    parser.add_argument("--save_metrics", type=str, default=None)
    args = parser.parse_args()

    print("Using device:", DEVICE)

    df = pd.read_csv(args.data_file)
    required_cols = {"text", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {args.data_file}: {sorted(missing)}")

    print(f"Eval file: {args.data_file}")
    print(f"Eval size: {len(df)}")

    print("\nLoading model checkpoint...")
    print(f"Model directory: {args.model_dir}")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    if hasattr(model.config, "id2label") and model.config.id2label:
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        target_names = [id2label.get(0, "Human"), id2label.get(1, "AI")]
    else:
        target_names = ["Human", "AI"]

    dataset = EssayDataset(df["text"].tolist(), df["label"].tolist(), tokenizer, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    print("\nEvaluating model...")
    acc, report, preds, true_labels = evaluate(
        model,
        loader,
        device=DEVICE,
        target_names=target_names,
    )

    true_counts = np.bincount(np.array(true_labels), minlength=2)
    pred_counts = np.bincount(np.array(preds), minlength=2)

    print("\nLabel distribution:")
    print(f"True counts: Human={true_counts[0]}, AI={true_counts[1]}")
    print(f"Pred counts: Human={pred_counts[0]}, AI={pred_counts[1]}")

    majority_acc = max(true_counts) / len(true_labels)
    print(f"\nMajority-class baseline accuracy: {majority_acc:.4f}")

    cm = confusion_matrix(true_labels, preds, labels=[0, 1])
    print("\nConfusion matrix (rows=true, cols=pred) [labels: 0=Human, 1=AI]:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    print("\nPer-class breakdown:")
    print(f"Human (0): support={tn+fp}, correct={tn}, incorrect={fp}, recall={tn/(tn+fp+1e-12):.4f}")
    print(f"AI    (1): support={fn+tp}, correct={tp}, incorrect={fn}, recall={tp/(fn+tp+1e-12):.4f}")

    print(f"\nTest Accuracy: {acc:.4f}")
    print(report)
    print(f"\nModel checkpoint used: {args.model_dir}")

    df2 = df.copy()
    df2["pred"] = preds
    df2["correct"] = (df2["pred"] == df2["label"]).astype(int)

    if "source" in df2.columns:
        print("\nAccuracy by source (test):")
        print(df2.groupby("source")["correct"].mean().sort_values(ascending=False))

        print("\nCounts by source (test):")
        print(df2["source"].value_counts())
    else:
        print("\nNo 'source' column found; skipping per-source analysis.")

    if args.save_predictions:
        df2.to_csv(args.save_predictions, index=False)
        print(f"\nSaved predictions to: {args.save_predictions}")

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "model_dir": args.model_dir,
        "data_file": args.data_file,
        "eval_size": len(df),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "accuracy": acc,
        "majority_class_baseline_accuracy": majority_acc,
        "true_counts": {"Human": int(true_counts[0]), "AI": int(true_counts[1])},
        "pred_counts": {"Human": int(pred_counts[0]), "AI": int(pred_counts[1])},
        "confusion_matrix": cm.tolist(),
        "classification_report_text": report,
    }

    if args.save_metrics:
        if os.path.dirname(args.save_metrics):
            os.makedirs(os.path.dirname(args.save_metrics), exist_ok=True)
        save_json(metrics, args.save_metrics)
        print(f"Saved metrics to: {args.save_metrics}")


if __name__ == "__main__":
    main()