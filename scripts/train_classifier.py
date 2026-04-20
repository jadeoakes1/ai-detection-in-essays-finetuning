#!/usr/bin/env python
# scripts/train_classifier.py

import os
import random
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EssayDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def evaluate(model, loader, device):
    model.eval()
    preds, true_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            preds.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    report = classification_report(
        true_labels, 
        preds, 
        labels=[0, 1],
        target_names=["Human", "AI"],
        zero_division=0,
    )

    return acc, report


def exact_text_overlap(a: pd.DataFrame, b: pd.DataFrame) -> int:
    a_texts = set(a["text"].astype(str).tolist())
    b_texts = set(b["text"].astype(str).tolist())
    return len(a_texts.intersection(b_texts))


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)

    # ======================
    # SANITY CHECK FLAGS
    # ======================
    # Use via CLI:
    #   --shuffle_train_labels     → expect ~50% accuracy
    #   --debug_train_size 100     → fast debugging run
    #
    # Examples:
    #   python train_classifier.py ... --shuffle_train_labels
    #   python train_classifier.py ... --debug_train_size 100
    #
    # Leave OFF for real experiments

    parser.add_argument("--shuffle_train_labels", action="store_true")
    parser.add_argument("--debug_train_size", type=int, default=None)

    # Logging / output
    parser.add_argument("--save_metrics", type=str, default=None)

    args = parser.parse_args()

    # RUN CONFIG
    print("\n===== RUN CONFIG =====")
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.lr}")
    print(f"Max length: {args.max_length}")
    print(f"Shuffle labels: {args.shuffle_train_labels}")
    print(f"Debug train size: {args.debug_train_size}")
    print("======================\n")

    set_seed(args.seed)

    print("Using device:", DEVICE)
    print("Model name:", args.model_name)
    print("Train file:", args.train_file)
    print("Dev file:  ", args.dev_file)
    print("Test file: ", args.test_file)
    print("Out dir:   ", args.out_dir)

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # LOAD DATA
    train_df = pd.read_csv(args.train_file)
    dev_df = pd.read_csv(args.dev_file)
    test_df = pd.read_csv(args.test_file)

    required_cols = {"text", "label"}
    for file_name, df in [
        (args.train_file, train_df),
        (args.dev_file, dev_df),
        (args.test_file, test_df),
    ]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {file_name}: {sorted(missing)}")

    overlap_train_dev = exact_text_overlap(train_df, dev_df)
    overlap_train_test = exact_text_overlap(train_df, test_df)
    print(f"Exact text overlap train↔dev: {overlap_train_dev}")
    print(f"Exact text overlap train↔test: {overlap_train_test}")

    # SANITY CHECKS
    if args.shuffle_train_labels:
        print("\nSANITY CHECK ENABLED: SHUFFLING TRAIN LABELS")
        train_df = train_df.copy()
        train_df["label"] = train_df["label"].sample(frac=1.0, random_state=args.seed).values

    if args.debug_train_size is not None:
        print(f"\nSANITY CHECK ENABLED: USING ONLY {args.debug_train_size} TRAIN EXAMPLES")
        sample_n = min(args.debug_train_size, len(train_df))
        train_df = train_df.sample(n=sample_n, random_state=args.seed)
    # END OF SANITY CHECKS

    print(f"Train size: {len(train_df)}, Dev size: {len(dev_df)}, Test size: {len(test_df)}")

    # TOKENIZER AND DATALOADERS
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    train_dataset = EssayDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        args.max_length,
    )
    dev_dataset = EssayDataset(
        dev_df["text"].tolist(),
        dev_df["label"].tolist(),
        tokenizer,
        args.max_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    # MODEL AND OPTIMIZER
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # TRAINING LOOP
    best_dev_acc = -1.0
    best_epoch = -1
    epoch_history = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1} completed. Average training loss: {avg_loss:.4f}")

        # Dev set validation
        dev_acc, dev_report = evaluate(model, dev_loader, device=DEVICE)
        print(f"Validation accuracy after epoch {epoch + 1}: {dev_acc:.4f}")
        print(dev_report)

        epoch_record = {
            "epoch": epoch + 1,
            "avg_train_loss": avg_loss,
            "dev_accuracy": dev_acc,
        }
        epoch_history.append(epoch_record)

        # Save best model
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_epoch = epoch + 1

            model.config.id2label = {0: "Human", 1: "AI"}
            model.config.label2id = {"Human": 0, "AI": 1}
            model.config.capstone_base_model = args.model_name

            model.save_pretrained(args.out_dir)
            tokenizer.save_pretrained(args.out_dir)

            print(f"New best model saved (epoch {best_epoch}, dev acc = {dev_acc:.4f})")

    print(f"\nBest model checkpoint saved to: {args.out_dir}")
    print(f"Best dev acc: {best_dev_acc:.4f} (epoch {best_epoch})")
    print("Run eval with:")
    print(f"  python scripts/eval_classifier.py --model_dir {args.out_dir} --data_file {args.test_file}")

    metrics = {
        "timestamp": datetime.now().isoformat(),
        "device": str(DEVICE),
        "model_name": args.model_name,
        "train_file": args.train_file,
        "dev_file": args.dev_file,
        "test_file": args.test_file,
        "out_dir": args.out_dir,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "shuffle_train_labels": args.shuffle_train_labels,
        "debug_train_size": args.debug_train_size,
        "train_size_final": len(train_df),
        "dev_size": len(dev_df),
        "test_size": len(test_df),
        "overlap_train_dev": overlap_train_dev,
        "overlap_train_test": overlap_train_test,
        "best_dev_acc": best_dev_acc,
        "best_epoch": best_epoch,
        "epoch_history": epoch_history,
    }

    metrics_path = args.save_metrics or os.path.join(args.out_dir, "train_metrics.json")
    save_json(metrics, metrics_path)
    print(f"Saved training metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
