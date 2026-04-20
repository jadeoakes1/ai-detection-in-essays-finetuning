#!/usr/bin/env python
"""
Convert OpenAI chat-format JSONL files into Gemini supervised tuning JSONL files.

Expected OpenAI format per line:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "Human" or "AI"}
  ]
}
"""

import argparse
import json
import os


def convert_record(openai_record: dict) -> dict:
    messages = openai_record.get("messages", [])
    if not messages:
        raise ValueError("Missing 'messages' field")

    system_text = None
    contents = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "system":
            system_text = content
        elif role == "user":
            contents.append({
                "role": "user",
                "parts": [{"text": content}]
            })
        elif role == "assistant":
            contents.append({
                "role": "model",
                "parts": [{"text": content}]
            })
        else:
            raise ValueError(f"Unexpected role: {role}")

    if system_text is None:
        raise ValueError("No system message found")

    return {
        "systemInstruction": {
            "parts": [{"text": system_text.strip()}]
        },
        "contents": contents,
    }


def convert_file(input_path: str, output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                gemini_record = convert_record(record)
                fout.write(json.dumps(gemini_record, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                raise ValueError(f"Error on line {line_num} of {input_path}: {e}") from e

    print(f"Saved Gemini JSONL to: {output_path}")
    print(f"Rows converted: {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input", required=True, help="Path to OpenAI train JSONL")
    parser.add_argument("--valid_input", required=True, help="Path to OpenAI valid/dev JSONL")
    parser.add_argument("--train_output", required=True, help="Path to Gemini train JSONL")
    parser.add_argument("--valid_output", required=True, help="Path to Gemini valid/dev JSONL")
    args = parser.parse_args()

    convert_file(args.train_input, args.train_output)
    convert_file(args.valid_input, args.valid_output)


if __name__ == "__main__":
    main()