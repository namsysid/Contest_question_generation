#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import random
import re
import secrets
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from direct_generation_common import (
    SYSTEM_PROMPT,
    batch_json_schema,
    build_prompt,
    pack_exemplars,
    read_jsonl,
    validate_batch,
    write_jsonl_line,
)
from ollama_client import generate_json


def question_fingerprint(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def is_duplicate_question(candidate: str, previous: list[str]) -> bool:
    normalized = question_fingerprint(candidate)
    for prior in previous:
        old = question_fingerprint(prior)
        if normalized == old or difflib.SequenceMatcher(None, normalized, old).ratio() >= 0.82:
            return True
    return False


def prompt_for_request(
    source_rows: list[dict],
    request_index: int,
    run_seed: int,
    offset: int,
    max_prompt_chars: int,
    retry_note: str,
    prior_questions: list[str],
) -> str:
    rotated = source_rows[offset:] + source_rows[:offset]
    # A long list of examples causes small local models to copy a familiar pattern or
    # ignore the requested primary seed. One clean seed makes every call genuinely
    # independent and keeps the task unambiguous.
    exemplars = pack_exemplars(rotated, max_prompt_chars, max_count=1)
    if not exemplars:
        raise RuntimeError("no complete MCQ exemplars could be packed")
    avoid = "\n".join(f"- {question[:240]}" for question in prior_questions[-25:]) or "(none yet)"
    prefix = f"""UNIQUE GENERATION REQUEST {request_index + 1}
Run nonce: {run_seed}. Corpus rotation offset: {offset}.
Use the first exemplar as the primary topic seed and create a question materially different
from every prior generated question listed below. Change the physical setup, target quantity,
and required reasoning—not merely names or numbers.

PRIOR GENERATED QUESTIONS THAT MUST NOT BE REPEATED:
{avoid}

"""
    return prefix + build_prompt(exemplars, 1, retry_note)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate validated F=ma mechanics MCQs with Ollama/Qwen")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="generated/direct_generated_qwen_v2.jsonl")
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--target-count", type=int, default=25)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Problems per model call; 1 is recommended for 7B local models",
    )
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=9000,
        help="Exemplar prompt budget; keep this modest for 7B models",
    )
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--max-output-tokens", type=int, default=1200)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--ollama-base-url")
    parser.add_argument("--resume", action="store_true", help="append to an existing partial output")
    parser.add_argument("--run-seed", type=int, help="optional reproducible run seed; random by default")
    args = parser.parse_args()
    if min(args.target_count, args.batch_size, args.max_attempts) <= 0:
        parser.error("target count, batch size, and max attempts must be positive")
    if args.batch_size != 1:
        parser.error("Qwen direct generation requires --batch-size 1 so every problem gets a separate prompt")

    source_rows = read_jsonl(args.input)
    if not source_rows:
        raise ValueError("input contains no rows")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    previous_output = read_jsonl(str(out_path)) if out_path.is_file() else []
    existing = previous_output if args.resume else []
    total = len(existing)
    if total > args.target_count:
        raise RuntimeError(f"existing output has {total} rows, above target {args.target_count}")
    mode = "a" if args.resume else "w"
    # Even when starting a fresh run (and overwriting the file), retain the old
    # questions as a rejection/avoidance list. Previously this list was discarded,
    # so a duplicate from the immediately preceding run could be accepted.
    historical_questions = [str(row.get("question") or "") for row in previous_output]
    prior_questions = [str(row.get("question") or "") for row in existing]
    banned_questions = list(prior_questions if args.resume else historical_questions)
    if any(is_duplicate_question(question, prior_questions[:index]) for index, question in enumerate(prior_questions)):
        raise RuntimeError("existing output contains duplicate questions; restart without --resume")
    if existing:
        stored_seed = existing[0].get("_meta", {}).get("run_seed")
        if stored_seed is None:
            raise RuntimeError("existing output has no run seed; restart without --resume")
        if args.run_seed is not None and args.run_seed != stored_seed:
            raise RuntimeError(f"--run-seed {args.run_seed} does not match existing run seed {stored_seed}")
        run_seed = int(stored_seed)
    else:
        run_seed = args.run_seed if args.run_seed is not None else secrets.randbits(63)
    offsets = list(range(len(source_rows)))
    random.Random(run_seed).shuffle(offsets)
    print(f"run seed: {run_seed}", file=sys.stderr, flush=True)
    with out_path.open(mode, encoding="utf-8") as handle:
        while total < args.target_count:
            wanted = min(args.batch_size, args.target_count - total)
            retry_note = ""
            prompt_offset = 0
            for attempt in range(1, args.max_attempts + 1):
                print(
                    f"requesting {wanted} problem(s), attempt {attempt}/{args.max_attempts}, "
                    f"completed {total}/{args.target_count}",
                    file=sys.stderr,
                    flush=True,
                )
                started = time.monotonic()
                try:
                    prompt_offset = offsets[total % len(offsets)]
                    prompt = prompt_for_request(
                        source_rows,
                        total,
                        run_seed,
                        prompt_offset,
                        args.max_prompt_chars,
                        retry_note,
                        banned_questions,
                    )
                    result = generate_json(
                        args.model,
                        prompt,
                        system=SYSTEM_PROMPT,
                        temperature=args.temperature,
                        max_output_tokens=args.max_output_tokens,
                        json_schema=batch_json_schema(wanted),
                        base_url=args.ollama_base_url,
                        timeout=args.timeout_seconds,
                        seed=(run_seed + total * args.max_attempts + attempt) % 2147483647,
                    )
                except (RuntimeError, TimeoutError, ValueError) as exc:
                    retry_note = str(exc)
                    print(
                        f"request failed after {time.monotonic() - started:.1f}s: {retry_note}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                problems, errors = validate_batch(result, wanted)
                if problems:
                    duplicates = [problem for problem in problems if is_duplicate_question(problem["question"], banned_questions)]
                    if duplicates:
                        retry_note = "duplicate of a previously generated question; create a materially different problem"
                        print(f"batch rejected (attempt {attempt}): {retry_note}", file=sys.stderr)
                        continue
                    print(
                        f"response accepted after {time.monotonic() - started:.1f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
                retry_note = " | ".join(errors)
                print(f"batch rejected (attempt {attempt}): {retry_note}", file=sys.stderr)
            else:
                raise RuntimeError(f"Qwen failed schema validation after {args.max_attempts} attempts: {retry_note}")

            for problem in problems:
                model_id = problem.get("id")
                total += 1
                problem["id"] = f"qwen_direct_{total:03d}"
                problem["_meta"] = {
                    "source_input": args.input,
                    "generator": "07_direct_generate_qwen_validated",
                    "model": args.model,
                    "record_index": total,
                    "model_generated_id": model_id,
                    "prompt_rotation_offset": prompt_offset,
                    "run_seed": run_seed,
                }
                write_jsonl_line(handle, problem)
                prior_questions.append(problem["question"])
                banned_questions.append(problem["question"])
            print(f"wrote {total}/{args.target_count}", file=sys.stderr)
    print(f"Wrote {total} validated F=ma mechanics MCQs -> {out_path}")


if __name__ == "__main__":
    main()
