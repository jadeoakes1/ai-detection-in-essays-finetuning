#!/usr/bin/env python

import argparse
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", required=True, help="Fine-tuning job ID")
    parser.add_argument("--output_csv", default="results.csv", help="Where to save results CSV")
    parser.add_argument("--print_events", action="store_true", help="Print training events")
    parser.add_argument("--events_output", default=None, help="File to save events (txt)")

    args = parser.parse_args()

    client = OpenAI()

    # Get job info
    job = client.fine_tuning.jobs.retrieve(args.job_id)

    print("\n===== JOB INFO =====")
    print("Status:", job.status)
    print("Model:", job.fine_tuned_model)

    # Download results CSV
    result_files = job.result_files

    if result_files:
        file_id = result_files[0]
        print("\nDownloading result file:", file_id)

        content = client.files.content(file_id)

        with open(args.output_csv, "wb") as f:
            f.write(content.read())

        print(f"Saved metrics to: {args.output_csv}")
    else:
        print("\nNo result file found")

    # Handle events
    if args.print_events or args.events_output:
        events = client.fine_tuning.jobs.list_events(args.job_id)

        lines = []

        for e in events.data:
            msg = e.message
            lines.append(msg)

            if args.print_events:
                print(msg)

        # Save to file if requested
        if args.events_output:
            with open(args.events_output, "w") as f:
                for line in lines:
                    f.write(line + "\n")

            print(f"\nSaved events to: {args.events_output}")


if __name__ == "__main__":
    main()