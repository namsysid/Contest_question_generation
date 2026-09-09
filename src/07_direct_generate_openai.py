#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import re
import secrets
import ssl
import sys
from pathlib import Path
from typing import Any, Dict
from urllib import error, request

from dotenv import load_dotenv

try:
    from direct_generation_common import (
        SYSTEM_PROMPT,
        build_prompt,
        pack_exemplars,
        read_jsonl,
        validate_batch,
        write_jsonl_line,
    )
except ModuleNotFoundError:  # Support import-by-path in tests and package tooling.
    from src.direct_generation_common import (
        SYSTEM_PROMPT,
        build_prompt,
        pack_exemplars,
        read_jsonl,
        validate_batch,
        write_jsonl_line,
    )


FMA_TOPIC_SPECS = (
    ("kinematics", "Kinematics", "one- and two-dimensional motion, including projectile motion"),
    ("forces", "Forces", "Newton's laws, free-body reasoning, friction, drag, and constrained motion"),
    ("energy", "Energy", "work, power, mechanical energy, and springs"),
    ("momentum", "Momentum", "linear momentum, impulse, center of mass, and collisions"),
    ("circular_gravity", "Circular & gravity", "centripetal dynamics, gravitation, and orbital mechanics"),
    ("rotation", "Rotation", "torque, angular momentum, rotational dynamics, and rolling"),
    ("oscillations", "Oscillations", "spring and pendulum oscillations"),
    ("fluids", "Fluids", "pressure, buoyancy, continuity, and Bernoulli flow"),
)


def question_fingerprint(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def closest_duplicate(candidate: str, previous: list[str]) -> tuple[float, str]:
    normalized = question_fingerprint(candidate)
    closest_ratio = 0.0
    closest_question = ""
    for prior in previous:
        old = question_fingerprint(prior)
        ratio = 1.0 if normalized == old else difflib.SequenceMatcher(None, normalized, old).ratio()
        if ratio > closest_ratio:
            closest_ratio, closest_question = ratio, prior
    return closest_ratio, closest_question


def topic_prompt(
    exemplar_blocks: list[str],
    target_count: int,
    topic_label: str,
    topic_description: str,
    prior_questions: list[str],
    retry_note: str = "",
) -> str:
    avoid = "\n".join(f"- {question[:240]}" for question in prior_questions[-30:]) or "(none yet)"
    prefix = f"""TOPIC-CONSTRAINED F=ma GENERATION REQUEST
Required category: {topic_label}
Required content: {topic_description}
Every item in this response must primarily test this category. Cross-topic reasoning is allowed,
but the requested category must be the central skill. Create materially distinct scenarios and
reasoning paths, including from the prior generated questions below.

PRIOR GENERATED QUESTIONS TO AVOID:
{avoid}

"""
    return prefix + build_prompt(exemplar_blocks, target_count, retry_note)


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
    parser.add_argument(
        "--per-topic",
        type=int,
        default=0,
        help="Generate this many questions for each of the eight standard F=ma topic categories",
    )
    parser.add_argument(
        "--topic-key",
        choices=[topic[0] for topic in FMA_TOPIC_SPECS],
        help="Limit --per-topic generation to one category (useful for retries or shards)",
    )
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-prompt-chars", type=int, default=24000)
    parser.add_argument("--max-output-tokens", type=int, default=6000)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--ca-bundle", help="PEM CA bundle; defaults to SSL_CERT_FILE or certifi")
    parser.add_argument("--resume", action="store_true", help="Continue a partial topic-balanced output")
    parser.add_argument("--run-seed", type=int, help="Reproducible exemplar rotation seed")
    args = parser.parse_args()
    if min(args.target_count, args.batch_size, args.max_attempts) <= 0:
        parser.error("target count, batch size, and max attempts must be positive")

    source_rows = read_jsonl(args.input)
    if not source_rows:
        raise ValueError("input contains no rows")
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
    existing = read_jsonl(str(out_path)) if args.resume and out_path.is_file() else []
    run_seed = args.run_seed
    if existing:
        stored_seed = (existing[0].get("_meta") or {}).get("run_seed")
        if stored_seed is None:
            raise RuntimeError("existing output has no run_seed; restart without --resume")
        if run_seed is not None and int(stored_seed) != run_seed:
            raise RuntimeError("--run-seed does not match the existing partial output")
        run_seed = int(stored_seed)
    if run_seed is None:
        run_seed = secrets.randbits(63)

    if args.per_topic:
        selected_topics = [topic for topic in FMA_TOPIC_SPECS if not args.topic_key or topic[0] == args.topic_key]
        schedule = [topic for topic in selected_topics for _ in range(args.per_topic)]
    else:
        schedule = [("mixed", "Mixed mechanics", "any in-scope F=ma mechanics topic")] * args.target_count
    if len(existing) > len(schedule):
        raise RuntimeError(f"existing output has {len(existing)} rows, above requested {len(schedule)}")
    for index, row in enumerate(existing):
        expected_key = schedule[index][0]
        actual_key = (row.get("_meta") or {}).get("topic_key", "mixed")
        if actual_key != expected_key:
            raise RuntimeError(f"existing row {index + 1} has topic {actual_key!r}; expected {expected_key!r}")

    eligible_offsets = [
        index for index, row in enumerate(source_rows)
        if pack_exemplars([row], args.max_prompt_chars, max_count=1)
    ]
    random.Random(run_seed).shuffle(eligible_offsets)
    if not eligible_offsets:
        raise RuntimeError("input contains no usable F=ma MCQ exemplars")

    total = len(existing)
    prior_questions = [str(row.get("question") or "") for row in existing]
    mode = "a" if args.resume else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        while total < len(schedule):
            topic_key, topic_label, topic_description = schedule[total]
            remaining_in_topic = sum(1 for item in schedule[total:] if item[0] == topic_key)
            wanted = min(args.batch_size, remaining_in_topic)
            retry_note = ""
            for attempt in range(1, args.max_attempts + 1):
                start = (total * 1009 + (attempt - 1) * 7) % len(eligible_offsets)
                selected = [
                    source_rows[eligible_offsets[(start + offset) % len(eligible_offsets)]]
                    for offset in range(min(7, len(eligible_offsets)))
                ]
                exemplars = pack_exemplars(selected, args.max_prompt_chars, max_count=7)
                result = response_json(
                    api_key,
                    args.openai_base_url,
                    args.model,
                    topic_prompt(
                        exemplars,
                        wanted,
                        topic_label,
                        topic_description,
                        prior_questions,
                        retry_note,
                    ),
                    args.max_output_tokens,
                    args.timeout_seconds,
                    ca_bundle,
                )
                problems, errors = validate_batch(result, wanted)
                if problems:
                    duplicates = []
                    comparison_pool = list(prior_questions)
                    for problem in problems:
                        ratio, closest = closest_duplicate(problem["question"], comparison_pool)
                        if ratio >= 0.82:
                            duplicates.append(
                                f"similarity {ratio:.3f} to {closest[:160]!r}"
                            )
                        comparison_pool.append(problem["question"])
                    if not duplicates:
                        break
                    errors.extend(duplicates)
                retry_note = " | ".join(errors)
                print(f"batch rejected (attempt {attempt}): {retry_note}", file=sys.stderr)
            else:
                raise RuntimeError(f"OpenAI failed schema validation after {args.max_attempts} attempts: {retry_note}")

            for problem in problems:
                total += 1
                problem["id"] = f"fma-{topic_key}-{sum(1 for row in schedule[:total] if row[0] == topic_key):03d}"
                problem["_meta"] = {
                    "source_input": args.input,
                    "generator": "07_direct_generate_openai_validated",
                    "model": args.model,
                    "record_index": total,
                    "topic_key": topic_key,
                    "topic": topic_label,
                    "topic_description": topic_description,
                    "run_seed": run_seed,
                }
                write_jsonl_line(handle, problem)
                prior_questions.append(problem["question"])
            print(f"wrote {total}/{len(schedule)} ({topic_label})", file=sys.stderr)
    print(f"Wrote {total} validated physics MCQs -> {out_path}")


if __name__ == "__main__":
    main()
