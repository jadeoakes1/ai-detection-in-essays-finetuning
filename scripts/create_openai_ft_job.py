from openai import OpenAI
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--training_file", required=True)
parser.add_argument("--validation_file", required=True)
args = parser.parse_args()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

job = client.fine_tuning.jobs.create(
    training_file=args.training_file,
    validation_file=args.validation_file,
    model="gpt-4.1-mini-2025-04-14",
    method={"type": "supervised"},
)

print("JOB ID:", job.id)
print("STATUS:", job.status)
print("FINE-TUNED MODEL:", job.fine_tuned_model)