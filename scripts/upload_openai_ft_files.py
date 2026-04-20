from openai import OpenAI
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_file", required=True, help="Path to training JSONL")
parser.add_argument("--valid_file", required=True, help="Path to validation JSONL")
args = parser.parse_args()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

with open(args.train_file, "rb") as f:
    train = client.files.create(file=f, purpose="fine-tune")

with open(args.valid_file, "rb") as f:
    valid = client.files.create(file=f, purpose="fine-tune")

print("TRAIN FILE ID:", train.id)
print("VALID FILE ID:", valid.id)