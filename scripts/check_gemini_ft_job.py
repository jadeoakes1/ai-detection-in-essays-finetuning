import argparse
from google import genai
from google.genai.types import HttpOptions

def check_job(project_id, job_name, location="us-central1"):
    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
        http_options=HttpOptions(api_version="v1beta1"),
    )

    job = client.tunings.get(name=job_name)

    print(f"\nJob: {job.name}")
    print(f"State: {job.state}")

    # Optional: show more info if available
    if getattr(job, "error", None):
        print(f"Error: {job.error}")

    if getattr(job, "tuned_model", None):
        if getattr(job.tuned_model, "model", None):
            print(f"Tuned model: {job.tuned_model.model}")
        if getattr(job.tuned_model, "endpoint", None):
            print(f"Endpoint: {job.tuned_model.endpoint}")

    return job


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, help="Full tuning job name")
    parser.add_argument("--project", default="capstone-493619")
    parser.add_argument("--location", default="us-central1")
    args = parser.parse_args()

    check_job(
        project_id=args.project,
        job_name=args.job,
        location=args.location,
    )