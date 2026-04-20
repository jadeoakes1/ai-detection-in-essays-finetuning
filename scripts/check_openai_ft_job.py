from openai import OpenAI
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--job_id", required=True)
args = parser.parse_args()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

job = client.fine_tuning.jobs.retrieve(args.job_id)

print("JOB ID:", job.id)
print("STATUS:", job.status)
print("FINE-TUNED MODEL:", job.fine_tuned_model)
print("TRAINING FILE:", job.training_file)
print("VALIDATION FILE:", job.validation_file)
print("ERROR:", job.error)