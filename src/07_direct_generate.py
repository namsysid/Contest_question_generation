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
import time
from pathlib import Path
from typing import Any
from urllib import error, request

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
from usnco_generation import (
    USNCO_SYSTEM_PROMPT,
    USNCO_TOPIC_BY_KEY,
    USNCO_TOPIC_SPECS,
    build_usnco_prompt,
    pack_usnco_exemplars,
    usnco_batch_json_schema,
    validate_usnco_batch,
)


TOPIC_SPECS = (
    "one-dimensional or two-dimensional kinematics; avoid springs, pulleys, and rolling objects",
    "Newton's laws with friction, drag, or constrained motion; avoid a plain block-on-incline setup",
    "linear momentum, impulse, center of mass, or collisions",
    "work and mechanical energy in a non-spring scenario",
    "statics, torque, or center of mass; avoid a pivoted rod with a mass attached at its end",
    "rotational dynamics or rolling motion; avoid a compound pendulum and a plain rolling incline",
    "gravitation or orbital mechanics",
    "oscillations involving a spring or pendulum; avoid a compound rod-and-ball pendulum",
    "fluid statics or fluid dynamics",
    "dimensional analysis, estimation, or elementary experimental data analysis",
)


def question_fingerprint(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def is_duplicate_question(candidate: str, previous: list[str]) -> bool:
    return closest_duplicate(candidate, previous)[0] >= 0.82


def closest_duplicate(candidate: str, previous: list[str]) -> tuple[float, str]:
    normalized = question_fingerprint(candidate)
    closest_ratio = 0.0
    closest_question = ""
    for prior in previous:
        old = question_fingerprint(prior)
        ratio = 1.0 if normalized == old else difflib.SequenceMatcher(None, normalized, old).ratio()
        if ratio > closest_ratio:
            closest_ratio = ratio
            closest_question = prior
    return closest_ratio, closest_question


def prompt_for_request(
    source_rows: list[dict],
    request_index: int,
    run_seed: int,
    exemplar_offsets: list[int],
    max_prompt_chars: int,
    exemplar_count: int,
    target_topic: str,
    retry_note: str,
    prior_questions: list[str],
) -> str:
    selected_rows = [source_rows[offset] for offset in exemplar_offsets]
    # A few independently sampled examples provide stylistic variety without burying
    # the generation instructions in the full corpus.
    exemplars = pack_exemplars(selected_rows, max_prompt_chars, max_count=exemplar_count)
    if not exemplars:
        raise RuntimeError("no complete MCQ exemplars could be packed")
    avoid = "\n".join(f"- {question[:240]}" for question in prior_questions[-25:]) or "(none yet)"
    prefix = f"""UNIQUE GENERATION REQUEST {request_index + 1}
Run nonce: {run_seed}. Corpus exemplar offsets: {exemplar_offsets[:len(exemplars)]}.
REQUIRED TOPIC FOR THIS REQUEST: {target_topic}.
The exemplars are references for contest style only, not topic seeds. Do not combine or
paraphrase their scenarios. Create a question materially different from every exemplar and
every prior generated question below. Change the physical setup, target quantity, and required
reasoning—not merely names or numbers. Reject your own draft and start over if it uses the same
kind of apparatus or opening sentence as a prior question.

PRIOR GENERATED QUESTIONS THAT MUST NOT BE REPEATED:
{avoid}

"""
    return prefix + build_prompt(exemplars, 1, retry_note)


def generate_openai_json(
    model: str,
    prompt: str,
    *,
    system: str,
    json_schema: dict[str, Any],
    max_output_tokens: int,
    timeout: float,
    base_url: str,
) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment or .env")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": max_output_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "contest_problem_batch", "strict": True, "schema": json_schema},
        },
    }
    api_request = request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(api_request, timeout=timeout, context=ssl.create_default_context()) as response:
            body = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI request failed ({exc.code}): {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenAI request failed: {exc.reason}") from exc
    try:
        raw = str(body["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"unexpected OpenAI response structure: {body}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"OpenAI did not return valid JSON: {raw[:500]}") from exc


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate validated F=ma or USNCO MCQs with Ollama/Qwen")
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--competition",
        choices=("fma", "usnco"),
        default="fma",
        help="Generation contract to use; defaults to the existing F=ma behavior",
    )
    parser.add_argument("--provider", choices=("ollama", "openai"), default="ollama")
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
        default=15000,
        help="Exemplar prompt budget; keep this modest for 7B models",
    )
    parser.add_argument(
        "--exemplar-count",
        type=int,
        default=3,
        help="Number of independently sampled corpus exemplars per request",
    )
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--max-output-tokens", type=int, default=1200)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--ollama-base-url")
    parser.add_argument(
        "--openai-base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    parser.add_argument("--resume", action="store_true", help="append to an existing partial output")
    parser.add_argument(
        "--allow-direct-ablation", action="store_true",
        help="Explicitly allow the non-research direct baseline; never use this for site question banks",
    )
    parser.add_argument(
        "--complex-reasoning-ablation", action="store_true",
        help="Direct F=ma experiment: demand one solution-first, reasoning-heavy competition problem",
    )
    parser.add_argument(
        "--exemplar-id", action="append", default=[],
        help="Restrict the direct experiment to an exact exemplar ID; may be repeated",
    )
    parser.add_argument(
        "--fma-topic", default="",
        help="Optional exact F=ma topic instruction for a direct ablation",
    )
    parser.add_argument("--run-seed", type=int, help="optional reproducible run seed; random by default")
    parser.add_argument(
        "--per-topic",
        type=int,
        default=0,
        help="USNCO only: generate this many questions in each selected topic",
    )
    parser.add_argument(
        "--topic-key",
        choices=[topic[0] for topic in USNCO_TOPIC_SPECS],
        help="USNCO only: restrict generation to one application topic key",
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        choices=range(1, 6),
        help="USNCO only: require a fixed difficulty from 1 through 5",
    )
    args = parser.parse_args()
    if min(args.target_count, args.batch_size, args.max_attempts, args.exemplar_count) <= 0:
        parser.error("target count, batch size, max attempts, and exemplar count must be positive")
    if args.provider == "ollama" and args.batch_size != 1:
        parser.error("Qwen direct generation requires --batch-size 1 so every problem gets a separate prompt")
    if args.competition != "usnco" and (args.per_topic or args.topic_key or args.difficulty):
        parser.error("--per-topic, --topic-key, and --difficulty require --competition usnco")
    if args.competition == "usnco" and (args.difficulty or 3) >= 3:
        parser.error(
            "competition-level USNCO generation must use chem/run_usnco_pipeline.py; "
            "stage 07 direct generation is restricted to difficulty 1-2"
        )
    if args.competition == "fma" and not args.allow_direct_ablation:
        parser.error(
            "F=ma bank generation must use fma/run_fma_pipeline.py; pass "
            "--allow-direct-ablation only for controlled baseline experiments"
        )
    if args.per_topic < 0:
        parser.error("--per-topic cannot be negative")

    source_rows = read_jsonl(args.input)
    if args.exemplar_id:
        wanted_ids = set(args.exemplar_id)
        source_rows = [row for row in source_rows if str(row.get("id") or "") in wanted_ids]
        missing_ids = wanted_ids - {str(row.get("id") or "") for row in source_rows}
        if missing_ids:
            raise ValueError(f"exemplar IDs not found: {sorted(missing_ids)}")
    if not source_rows:
        raise ValueError("input contains no rows")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    previous_output = read_jsonl(str(out_path)) if out_path.is_file() else []
    existing = previous_output if args.resume else []
    total = len(existing)
    if args.competition == "fma" and total > args.target_count:
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
    exemplar_packer = pack_usnco_exemplars if args.competition == "usnco" else pack_exemplars
    offsets = [
        offset for offset in offsets
        if exemplar_packer([source_rows[offset]], args.max_prompt_chars, max_count=1)
    ]
    if not offsets:
        raise RuntimeError(f"input contains no usable in-scope {args.competition.upper()} MCQ exemplars")
    print(
        f"eligible {args.competition.upper()} exemplars: {len(offsets)}/{len(source_rows)}",
        file=sys.stderr,
        flush=True,
    )
    if args.competition == "usnco":
        selected_topics = [
            topic for topic in USNCO_TOPIC_SPECS if not args.topic_key or topic[0] == args.topic_key
        ]
        if args.per_topic:
            schedule = [topic for topic in selected_topics for _ in range(args.per_topic)]
        elif args.topic_key:
            schedule = [selected_topics[0]] * args.target_count
        else:
            schedule = [USNCO_TOPIC_SPECS[index % len(USNCO_TOPIC_SPECS)] for index in range(args.target_count)]
        if len(existing) > len(schedule):
            raise RuntimeError(f"existing output has {len(existing)} rows, above requested {len(schedule)}")
        for index, row in enumerate(existing):
            actual = str((row.get("_meta") or {}).get("topic_key") or row.get("topic_key") or "")
            if actual != schedule[index][0]:
                raise RuntimeError(
                    f"existing row {index + 1} has topic {actual!r}; expected {schedule[index][0]!r}"
                )
    else:
        schedule = []
    print(f"run seed: {run_seed}", file=sys.stderr, flush=True)
    with out_path.open(mode, encoding="utf-8") as handle:
        target_total = len(schedule) if args.competition == "usnco" else args.target_count
        while total < target_total:
            if args.competition == "usnco":
                current_key = schedule[total][0]
                contiguous_topic_rows = 0
                for scheduled in schedule[total:]:
                    if scheduled[0] != current_key:
                        break
                    contiguous_topic_rows += 1
                wanted = min(args.batch_size, contiguous_topic_rows)
            else:
                wanted = min(args.batch_size, target_total - total)
            retry_note = ""
            prompt_offset = 0
            prompt_offsets: list[int] = []
            if args.competition == "usnco":
                topic_key, topic_label, topic_description = schedule[total]
                target_topic = topic_description
            else:
                topic_key, topic_label = "", ""
                target_topic = args.fma_topic or TOPIC_SPECS[total % len(TOPIC_SPECS)]
            for attempt in range(1, args.max_attempts + 1):
                print(
                    f"requesting {wanted} problem(s), attempt {attempt}/{args.max_attempts}, "
                    f"completed {total}/{target_total}",
                    file=sys.stderr,
                    flush=True,
                )
                started = time.monotonic()
                try:
                    # Move to a new shuffled slice on every attempt. The large prime
                    # separates adjacent output records so their initial slices do not
                    # simply reuse the final retry slice of the preceding record.
                    start = (total * 1009 + (attempt - 1) * args.exemplar_count) % len(offsets)
                    prompt_offsets = [
                        offsets[(start + index) % len(offsets)]
                        for index in range(min(args.exemplar_count, len(offsets)))
                    ]
                    prompt_offset = prompt_offsets[0]
                    selected_rows = [source_rows[offset] for offset in prompt_offsets]
                    if args.competition == "usnco":
                        exemplars = pack_usnco_exemplars(
                            selected_rows, args.max_prompt_chars, max_count=args.exemplar_count
                        )
                        prompt = build_usnco_prompt(
                            exemplars,
                            wanted,
                            topic_key,
                            topic_label,
                            topic_description,
                            banned_questions,
                            difficulty=args.difficulty,
                            retry_note=retry_note,
                        )
                        system_prompt = USNCO_SYSTEM_PROMPT
                        output_schema = usnco_batch_json_schema(
                            wanted, topic_key, topic_label, args.difficulty or 3
                        )
                    else:
                        prompt = prompt_for_request(
                            source_rows,
                            total,
                            run_seed,
                            prompt_offsets,
                            args.max_prompt_chars,
                            args.exemplar_count,
                            target_topic,
                            retry_note,
                            banned_questions,
                        )
                        system_prompt = SYSTEM_PROMPT
                        if args.complex_reasoning_ablation:
                            prompt = """DIRECT COMPLEXITY ABLATION.
Create one genuinely complex and physically sound F=ma competition problem. Work solution-first:
first lock a valid mechanism with all unknowns closed by explicit standard mechanics, then carry its
derivation through. Require at least one indispensable non-obvious modeling decision and at least four
dependent conceptual deductions; arithmetic length and a routine textbook template do not count.
Once a complex mechanism is physically valid, commit to it without second-guessing, weakening it, or
retreating to an easier problem. The final stem must be self-contained, uniquely solvable, and no closer
in scenario to any exemplar than necessary for authentic F=ma style. Verify the keyed choice independently.

""" + prompt
                            system_prompt += (
                                "\nThis is an intentionally complex solution-first experiment. Commit to a "
                                "sound difficult mechanism and do not simplify away its essential reasoning."
                            )
                        output_schema = batch_json_schema(wanted)
                    if args.provider == "openai":
                        result = generate_openai_json(
                            args.model,
                            prompt,
                            system=system_prompt,
                            json_schema=output_schema,
                            max_output_tokens=args.max_output_tokens,
                            timeout=args.timeout_seconds,
                            base_url=args.openai_base_url,
                        )
                    else:
                        result = generate_json(
                            args.model,
                            prompt,
                            system=system_prompt,
                            temperature=args.temperature,
                            max_output_tokens=args.max_output_tokens,
                            json_schema=output_schema,
                            base_url=args.ollama_base_url,
                            timeout=args.timeout_seconds,
                            seed=(run_seed + total * 1009 + attempt) % 2147483647,
                        )
                except (RuntimeError, TimeoutError, ValueError) as exc:
                    retry_note = str(exc)
                    print(
                        f"request failed after {time.monotonic() - started:.1f}s: {retry_note}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                if args.competition == "usnco":
                    problems, errors = validate_usnco_batch(
                        result, wanted, topic_key, topic_label, args.difficulty or 3
                    )
                else:
                    problems, errors = validate_batch(result, wanted)
                if problems:
                    comparison_pool = list(banned_questions)
                    duplicate = None
                    for problem in problems:
                        ratio, closest = closest_duplicate(problem["question"], comparison_pool)
                        if ratio >= 0.82:
                            duplicate = (ratio, closest, problem["question"])
                            break
                        comparison_pool.append(problem["question"])
                    if duplicate:
                        ratio, closest, candidate = duplicate
                        retry_note = (
                            f"duplicate similarity {ratio:.3f} to prior question: {closest[:220]!r}; "
                            f"stay in the required topic but use entirely different apparatus and reasoning"
                        )
                        print(
                            f"batch rejected (attempt {attempt}): {retry_note}\n"
                            f"candidate: {candidate[:300]}",
                            file=sys.stderr,
                        )
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
                if args.competition == "usnco":
                    topic_number = sum(1 for item in schedule[:total] if item[0] == topic_key)
                    problem["id"] = f"usnco-{topic_key}-{topic_number:03d}"
                else:
                    problem["id"] = f"qwen_direct_{total:03d}"
                problem["_meta"] = {
                    "source_input": args.input,
                    "generator": f"07_direct_generate_{args.provider}_{args.competition}_validated",
                    "model": args.model,
                    "provider": args.provider,
                    "competition": "USNCO" if args.competition == "usnco" else "F=ma",
                    "record_index": total,
                    "model_generated_id": model_id,
                    "prompt_rotation_offset": prompt_offset,
                    "prompt_exemplar_offsets": prompt_offsets,
                    "prompt_exemplar_count": len(prompt_offsets),
                    "target_topic": target_topic,
                    "run_seed": run_seed,
                }
                if args.competition == "usnco":
                    exemplar_sources = []
                    for row in selected_rows:
                        source = row.get("source") if isinstance(row.get("source"), dict) else {}
                        exemplar_sources.append(row.get("source_url") or source.get("url") or row.get("source_pdf"))
                    problem["_meta"].update({
                        "topic_key": topic_key,
                        "topic": topic_label,
                        "topic_description": USNCO_TOPIC_BY_KEY[topic_key][1],
                        "generation_mode": "solution_centric_direct",
                        "prompt_exemplar_ids": [row.get("id") for row in selected_rows],
                        "prompt_exemplar_sources": exemplar_sources,
                    })
                write_jsonl_line(handle, problem)
                prior_questions.append(problem["question"])
                banned_questions.append(problem["question"])
            print(f"wrote {total}/{target_total}", file=sys.stderr)
    label = "USNCO chemistry" if args.competition == "usnco" else "F=ma mechanics"
    print(f"Wrote {total} validated {label} MCQs -> {out_path}")


if __name__ == "__main__":
    main()
