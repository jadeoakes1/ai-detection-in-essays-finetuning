import argparse
import time
from google import genai
from google.genai.types import (
    HttpOptions,
    CreateTuningJobConfig,
    TuningDataset,
    TuningValidationDataset,
    AdapterSize,
)

def start_tuning(project_id, train_path, dev_path, model_name, location="us-central1"):
    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
        http_options=HttpOptions(api_version="v1beta1"),
    )

    print("Launching tuning job")
    print(f"Project: {project_id}")
    print(f"Location: {location}")
    print(f"Train: {train_path}")
    print(f"Dev:   {dev_path}")
    print(f"Name:  {model_name}")

    training_dataset = TuningDataset(gcs_uri=train_path)
    validation_dataset = TuningValidationDataset(gcs_uri=dev_path)

    tuning_job = client.tunings.tune(
        base_model="gemini-2.5-flash",
        training_dataset=training_dataset,
        config=CreateTuningJobConfig(
            tuned_model_display_name=model_name,
            validation_dataset=validation_dataset,
            epoch_count=3,
            adapter_size=AdapterSize.ADAPTER_SIZE_SIXTEEN,
            learning_rate_multiplier=1.0,
        ),
    )

    print(f"\nJob created: {tuning_job.name}")
    print(f"Initial state: {tuning_job.state}")
    print("Exiting — job will continue running in Vertex AI.")

    return tuning_job

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="GCS URI to training JSONL")
    parser.add_argument("--dev", required=True, help="GCS URI to validation JSONL")
    parser.add_argument("--name", required=True, help="Display name for tuned model")
    parser.add_argument("--project", default="capstone-493619")
    parser.add_argument("--location", default="us-central1")
    args = parser.parse_args()

    start_tuning(
        project_id=args.project,
        train_path=args.train,
        dev_path=args.dev,
        model_name=args.name,
        location=args.location,
    )