#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict

import requests


ANTHROPIC_MODEL_DEFAULT = "claude-sonnet-4-6"


SYSTEM_PROMPT = """You are writing a short paragraph in English.

Follow these rules exactly:
- Write one natural-sounding paragraph.
- Length: 90 to 250 words.
- Respond directly to the assigned prompt.
- Use normal paragraph form.
- Do not use bullet points, headings, or numbered lists.
"""

USER_TEMPLATE = """Write one paragraph that follows the prompt exactly.

Prompt:
{prompt_text}
"""


@dataclass
class GenerationRecord:
    provider: str
    model: str
    prompt_id: str
    candidate_index: int
    timestamp_utc: str
    prompt_text: str
    system_prompt: str
    raw_text: str
    cleaned_text: str
    word_count: int
    api_parameters: Dict[str, object]
    response_metadata: Dict[str, object]


def utc_timestamp() -> str:
    import datetime as dt
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def clean_paragraph_text(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    text = re.sub(r"^\s*Paragraph:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*Here(?:'s| is)\s+(?:a\s+)?paragraph:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def ensure_ok(response: requests.Response) -> None:
    if not response.ok:
        raise RuntimeError(
            f"Anthropic API error {response.status_code}: {response.text[:2000]}"
        )


def generate_anthropic(api_key: str, prompt_text: str, model: str) -> Dict[str, object]:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "max_tokens": 500,
        "messages": [
            {
                "role": "user",
                "content": USER_TEMPLATE.format(prompt_text=prompt_text),
            }
        ],
    }

    response = requests.post(url, headers=headers, json=payload, timeout=300)
    ensure_ok(response)
    data = response.json()

    chunks = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            chunks.append(block.get("text", ""))

    text = "\n".join(chunks).strip()

    return {
        "text": text,
        "metadata": {
            "id": data.get("id"),
            "usage": data.get("usage"),
            "stop_reason": data.get("stop_reason"),
        },
        "api_parameters": {
            "max_tokens": 500,
        },
    }


def save_record(record: GenerationRecord, output_dir: Path) -> None:
    provider_dir = output_dir / record.provider / record.prompt_id
    provider_dir.mkdir(parents=True, exist_ok=True)

    stem = f"candidate_{record.candidate_index:02d}"
    txt_path = provider_dir / f"{stem}.txt"
    json_path = provider_dir / f"{stem}.json"

    txt_path.write_text(record.cleaned_text + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(asdict(record), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_prompts(prompts_file: Path) -> Dict[str, str]:
    with prompts_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not data:
        raise ValueError("Prompts file must be a non-empty JSON object.")

    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("Prompts file must map string IDs to string prompt texts.")

    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates-per-prompt", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--anthropic-model", default=ANTHROPIC_MODEL_DEFAULT)
    args = parser.parse_args()

    if args.candidates_per_prompt < 1:
        raise ValueError("--candidates-per-prompt must be >= 1")

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set.")

    prompts = load_prompts(args.prompts_file)
    prompt_items = list(prompts.items())

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Provider: anthropic")
    print(f"Model: {args.anthropic_model}")
    print(f"Prompts: {len(prompt_items)}")
    print(f"Candidates per prompt: {args.candidates_per_prompt}")
    print(f"Output directory: {args.output_dir}")
    print()

    failures = []

    for prompt_id, prompt_text in prompt_items:
        for candidate_index in range(1, args.candidates_per_prompt + 1):
            print(f"Generating anthropic | {prompt_id} | candidate {candidate_index} ...", flush=True)

            try:
                result = generate_anthropic(api_key, prompt_text, args.anthropic_model)
                cleaned = clean_paragraph_text(result["text"])

                record = GenerationRecord(
                    provider="anthropic",
                    model=args.anthropic_model,
                    prompt_id=prompt_id,
                    candidate_index=candidate_index,
                    timestamp_utc=utc_timestamp(),
                    prompt_text=prompt_text,
                    system_prompt=SYSTEM_PROMPT,
                    raw_text=result["text"],
                    cleaned_text=cleaned,
                    word_count=count_words(cleaned),
                    api_parameters=result["api_parameters"],
                    response_metadata=result["metadata"],
                )

                save_record(record, args.output_dir)
                print(f"  saved | words={record.word_count}")

            except Exception as e:
                failures.append({
                    "provider": "anthropic",
                    "model": args.anthropic_model,
                    "prompt_id": prompt_id,
                    "candidate_index": candidate_index,
                    "error": str(e),
                })
                print(f"  ERROR: {e}", file=sys.stderr)

            time.sleep(args.sleep_seconds)

    summary = {
        "provider": "anthropic",
        "model": args.anthropic_model,
        "prompts_count": len(prompt_items),
        "candidates_per_prompt": args.candidates_per_prompt,
        "failures": failures,
        "finished_at_utc": utc_timestamp(),
    }

    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Done.")

    if failures:
        print(f"There were {len(failures)} failures. See run_summary.json.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())