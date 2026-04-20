import argparse
import os
from google.cloud import storage

def upload_files(project_id, bucket_name, file_paths):
    # Initializes the client
    client = storage.Client(project=project_id)
    
    # Try to get or create the bucket
    try:
        bucket = client.get_bucket(bucket_name)
        print(f"Using existing bucket: {bucket_name}")
    except Exception:
        bucket = client.create_bucket(bucket_name, location="us-central1")
        print(f"Bucket {bucket_name} created.")

    # Loop through all files provided in the command line
    for local_path in file_paths:
        if not os.path.exists(local_path):
            print(f"Error: File {local_path} not found. Skipping...")
            continue
            
        destination_blob_name = os.path.basename(local_path)
        blob = bucket.blob(destination_blob_name)
        
        print(f"Uploading {local_path}...")
        blob.upload_from_filename(local_path)
        print(f"Successfully uploaded to gs://{bucket_name}/{destination_blob_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload multiple datasets to GCS")
    
    # Changed to nargs='+' to accept one or more files
    parser.add_argument("--files", nargs='+', required=True, help="List of local .jsonl files (e.g. train.jsonl dev.jsonl)")
    parser.add_argument("--bucket", default="capstone-2026-jade", help="GCS bucket name")
    parser.add_argument("--project", default="capstone-493619", help="Google Cloud Project ID")

    args = parser.parse_args()

    upload_files(args.project, args.bucket, args.files)