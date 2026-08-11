#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
from pathlib import Path
from typing import Any, Dict
from urllib import error, request

from dotenv import load_dotenv

from direct_generation_common import (
    SYSTEM_PROMPT,
    build_prompt,
    pack_exemplars,
    read_jsonl,
    validate_batch,
    write_jsonl_line,
)


def response_json(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
    timeout_seconds: float,
    ca_bundle: str | None = None,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": max_output_tokens,
        "response_format": {"type": "json_object"},
    }
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    api_request = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    ssl_context = ssl.create_default_context(cafile=ca_bundle) if ca_bundle else ssl.create_default_context()
    try:
        with request.urlopen(api_request, timeout=timeout_seconds, context=ssl_context) as response:
            body = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        request_id = exc.headers.get("x-request-id", "unknown")
        raise RuntimeError(f"OpenAI request failed ({exc.code}, request_id={request_id}): {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenAI request failed: {exc.reason}") from exc

    try:
        raw = str(body["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected OpenAI response structure: {body}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"OpenAI did not return JSON: {raw}")
        return json.loads(raw[start : end + 1])


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate validated physics MCQs with OpenAI")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="generated/direct_generated_openai_v2.jsonl")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--target-count", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-prompt-chars", type=int, default=24000)
    parser.add_argument("--max-output-tokens", type=int, default=6000)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--ca-bundle", help="PEM CA bundle; defaults to SSL_CERT_FILE or certifi")
    args = parser.parse_args()
    if min(args.target_count, args.batch_size, args.max_attempts) <= 0:
        parser.error("target count, batch size, and max attempts must be positive")

    exemplars = pack_exemplars(read_jsonl(args.input), args.max_prompt_chars)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment or .env")
    ca_bundle = args.ca_bundle or os.getenv("SSL_CERT_FILE")
    if not ca_bundle:
        try:
            import certifi

            ca_bundle = certifi.where()
        except ImportError:
            ca_bundle = None
    if ca_bundle and not Path(ca_bundle).is_file():
        raise RuntimeError(f"CA bundle does not exist: {ca_bundle}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out_path.open("w", encoding="utf-8") as handle:
        while total < args.target_count:
            wanted = min(args.batch_size, args.target_count - total)
            retry_note = ""
            for attempt in range(1, args.max_attempts + 1):
                result = response_json(
                    api_key,
                    args.openai_base_url,
                    args.model,
                    build_prompt(exemplars, wanted, retry_note),
                    args.max_output_tokens,
                    args.timeout_seconds,
                    ca_bundle,
                )
                problems, errors = validate_batch(result, wanted)
                if problems:
                    break
                retry_note = " | ".join(errors)
                print(f"batch rejected (attempt {attempt}): {retry_note}", file=sys.stderr)
            else:
                raise RuntimeError(f"OpenAI failed schema validation after {args.max_attempts} attempts: {retry_note}")

            for problem in problems:
                total += 1
                problem["id"] = problem.get("id") or f"openai_physics_{total:03d}"
                problem["_meta"] = {
                    "source_input": args.input,
                    "generator": "07_direct_generate_openai_validated",
                    "model": args.model,
                    "record_index": total,
                }
                write_jsonl_line(handle, problem)
            print(f"wrote {total}/{args.target_count}", file=sys.stderr)
    print(f"Wrote {total} validated physics MCQs -> {out_path}")


if __name__ == "__main__":
    main()
