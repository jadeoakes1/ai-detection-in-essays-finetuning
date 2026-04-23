#!/usr/bin/env python3
"""
Generate candidate essays for six prompts using:
- OpenAI GPT-5.4
- Anthropic Claude Sonnet 4.6
- Google Gemini 3 Flash

Environment variables:
  OPENAI_API_KEY
  ANTHROPIC_API_KEY
  GEMINI_API_KEY

Example:
  python scripts/generate_essays.py \
      --prompts-file data/new_dataset/ai_essays/prompts.json \
      --output-dir outputs \
      --candidates-per-prompt 2 \
      --sleep-seconds 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict

import requests


OPENAI_MODEL_DEFAULT = "gpt-5.4"
ANTHROPIC_MODEL_DEFAULT = "claude-sonnet-4-6"
GEMINI_MODEL_DEFAULT = "gemini-3-flash-preview"


SYSTEM_PROMPT = """You are writing a college student argumentative essay.

Follow these rules exactly:
- Write a natural-sounding student essay in English.
- Length: 300 to 400 words.
- Respond directly to the assigned prompt and defend the stated position.
- Use clear reasoning and concrete examples.
- Use normal paragraph form.
- Do not use bullet points, headings, or numbered lists.
- Do not mention being an AI, language model, assistant, or chatbot.
- Do not include disclaimers, notes, or any text outside the essay itself.
"""

USER_TEMPLATE = """Write one essay that follows the prompt exactly.

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


def clean_essay_text(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    text = re.sub(r"^\s*Essay:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*Here(?:'s| is)\s+(?:a\s+)?(?:short\s+)?essay:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def ensure_ok(response: requests.Response, provider: str) -> None:
    if not response.ok:
        raise RuntimeError(
            f"{provider} API error {response.status_code}: {response.text[:2000]}"
        )


def generate_openai(api_key: str, prompt_text: str, model: str) -> Dict[str, object]:
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "reasoning": {"effort": "none"},
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(prompt_text=prompt_text)},
        ],
        "max_output_tokens": 900,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=300)
    ensure_ok(response, "OpenAI")
    data = response.json()

    text = data.get("output_text")
    if not text:
        chunks = []
        for item in data.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    chunks.append(content.get("text", ""))
        text = "\n".join(chunks).strip()

    return {
        "text": text or "",
        "metadata": {
            "id": data.get("id"),
            "usage": data.get("usage"),
        },
        "api_parameters": {
            "max_output_tokens": 900,
            "reasoning_effort": "none",
        },
    }


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
        "max_tokens": 900,
        "messages": [
            {
                "role": "user",
                "content": USER_TEMPLATE.format(prompt_text=prompt_text),
            }
        ],
    }
    response = requests.post(url, headers=headers, json=payload, timeout=300)
    ensure_ok(response, "Anthropic")
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
            "max_tokens": 900,
        },
    }


def generate_gemini(api_key: str, prompt_text: str, model: str) -> Dict[str, object]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": USER_TEMPLATE.format(prompt_text=prompt_text)}],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 900,
            "thinkingConfig": {"thinkingLevel": "MINIMAL"},
        },
    }
    response = requests.post(url, headers=headers, json=payload, timeout=300)
    ensure_ok(response, "Gemini")
    data = response.json()

    chunks = []
    for candidate in data.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            if "text" in part:
                chunks.append(part["text"])
    text = "\n".join(chunks).strip()

    return {
        "text": text,
        "metadata": {
            "usageMetadata": data.get("usageMetadata"),
            "modelVersion": data.get("modelVersion"),
        },
        "api_parameters": {
            "maxOutputTokens": 900,
            "thinkingLevel": "MINIMAL",
        },
    }


def save_record(record: GenerationRecord, output_dir: Path) -> None:
    provider_dir = output_dir / record.provider / record.prompt_id
    provider_dir.mkdir(parents=True, exist_ok=True)

    stem = f"candidate_{record.candidate_index:02d}"
    txt_path = provider_dir / f"{stem}.txt"
    json_path = provider_dir / f"{stem}.json"

    txt_path.write_text(record.cleaned_text + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(asdict(record), indent=2, ensure_ascii=False), encoding="utf-8")


def load_prompts(prompts_file: Path) -> Dict[str, str]:
    with prompts_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError("Prompts file must be a non-empty JSON object mapping prompt IDs to strings.")
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("Prompts file must map string IDs to string prompt texts.")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates-per-prompt", type=int, default=2)
    parser.add_argument("--providers", nargs="+", default=["openai", "anthropic", "gemini"])
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--shuffle-prompt-order", action="store_true")
    parser.add_argument("--openai-model", default=OPENAI_MODEL_DEFAULT)
    parser.add_argument("--anthropic-model", default=ANTHROPIC_MODEL_DEFAULT)
    parser.add_argument("--gemini-model", default=GEMINI_MODEL_DEFAULT)
    args = parser.parse_args()

    if args.candidates_per_prompt < 1:
        raise ValueError("--candidates-per-prompt must be >= 1")

    prompts = load_prompts(args.prompts_file)
    prompt_items = list(prompts.items())
    if args.shuffle_prompt_order:
        random.shuffle(prompt_items)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    openai_key = os.getenv("OPENAI_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")

    provider_fns = {
        "openai": (
            lambda prompt: generate_openai(openai_key, prompt, args.openai_model),
            args.openai_model,
            bool(openai_key),
        ),
        "anthropic": (
            lambda prompt: generate_anthropic(anthropic_key, prompt, args.anthropic_model),
            args.anthropic_model,
            bool(anthropic_key),
        ),
        "gemini": (
            lambda prompt: generate_gemini(gemini_key, prompt, args.gemini_model),
            args.gemini_model,
            bool(gemini_key),
        ),
    }

    requested = []
    for provider in args.providers:
        provider = provider.lower()
        if provider not in provider_fns:
            raise ValueError(f"Unknown provider: {provider}")
        requested.append(provider)

    print(f"Providers: {requested}")
    print(f"Prompts: {len(prompt_items)}")
    print(f"Candidates per prompt: {args.candidates_per_prompt}")
    print(f"Output directory: {args.output_dir}")
    print()

    failures = []

    for provider in requested:
        generator_fn, model_name, enabled = provider_fns[provider]
        if not enabled:
            print(f"[SKIP] {provider}: missing API key environment variable")
            continue

        print(f"=== {provider} | model={model_name} ===")
        for prompt_id, prompt_text in prompt_items:
            for candidate_index in range(1, args.candidates_per_prompt + 1):
                print(f"Generating {provider} | {prompt_id} | candidate {candidate_index} ...", flush=True)
                try:
                    result = generator_fn(prompt_text)
                    cleaned = clean_essay_text(result["text"])
                    record = GenerationRecord(
                        provider=provider,
                        model=model_name,
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
                    failures.append(
                        {
                            "provider": provider,
                            "model": model_name,
                            "prompt_id": prompt_id,
                            "candidate_index": candidate_index,
                            "error": str(e),
                        }
                    )
                    print(f"  ERROR: {e}", file=sys.stderr)

                time.sleep(args.sleep_seconds)
        print()

    summary = {
        "providers_requested": requested,
        "prompts_count": len(prompt_items),
        "candidates_per_prompt": args.candidates_per_prompt,
        "failures": failures,
        "finished_at_utc": utc_timestamp(),
        "models": {
            "openai": args.openai_model,
            "anthropic": args.anthropic_model,
            "gemini": args.gemini_model,
        },
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