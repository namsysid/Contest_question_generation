#!/usr/bin/env python3
"""Blindly solve and score generated physics MCQs using question text and choices only."""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib import error, request

from dotenv import load_dotenv


SYSTEM_TEMPLATE = """You are a strict evaluator of __COMPETITION_LABEL__ multiple-choice physics problems.

You receive only the rendered question and answer choices. Solve it cold. Never infer or use a
stored answer key, solution, generation metadata, or hidden structure.

First attempt to solve the problem. Use status `solved` only when one answer is defensible,
`partial` for meaningful progress without a confident answer, and `unclear` when the problem is
ambiguous, inconsistent, or underspecified. If no choice matches, use UNKNOWN; do not select a
merely nearby choice unless approximation is explicitly appropriate.

Score difficulty from 1 (trivial recall/direct substitution) to 5 (non-obvious olympiad insight).
Score each competition dimension from 1 (very weak) to 5 (strong official-contest quality):
- depth_reasoning: number and nontriviality of reasoning steps
- conceptual_richness: interaction of distinct physics concepts
- clarity: precision, completeness, uniqueness, and well-posedness
- olympiad_similarity: resemblance to genuine F=ma/Olympiad physics

Return strict JSON only:
{
  "solve_attempt": {
    "status": "solved|partial|unclear",
    "selected_answer": "A|B|C|D|E|UNKNOWN",
    "reasoning_summary": "at most 120 words",
    "issue_notes": ["short note"]
  },
  "difficulty_assessment": {"score": 1, "notes": "at most 60 words"},
  "competition_appropriateness": {
    "depth_reasoning": 1,
    "conceptual_richness": 1,
    "clarity": 1,
    "olympiad_similarity": 1,
    "notes": "at most 60 words"
  }
}
"""


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_choices(value: Any) -> Dict[str, str]:
    if isinstance(value, dict):
        choices = {str(key).upper(): str(body).strip() for key, body in value.items()}
    elif isinstance(value, list):
        choices = {chr(65 + index): str(body).strip() for index, body in enumerate(value[:5])}
    else:
        return {}
    return {key: choices[key] for key in "ABCDE" if key in choices and choices[key]}


def load_exemplars(paths: List[str], limit: int, max_chars: int) -> List[str]:
    exemplars: List[str] = []
    for path in paths:
        if not Path(path).is_file():
            continue
        for row in read_jsonl(path):
            text = str(row.get("question_text") or row.get("question") or "").strip()
            if text and len(text) <= max_chars and all(f"({key})" in text.upper() for key in "ABC"):
                exemplars.append(text)
                if len(exemplars) == limit:
                    return exemplars
    return exemplars


def build_system(subject: str, competition: str, exemplars: List[str]) -> str:
    label = " & ".join(part for part in (subject, competition) if part) or "Physics Olympiad"
    system = SYSTEM_TEMPLATE.replace("__COMPETITION_LABEL__", label)
    if exemplars:
        system += "\nReal competition exemplars for style calibration only:\n"
        system += "\n---\n".join(exemplars)
    return system


def api_json(
    api_key: str,
    base_url: str,
    ca_bundle: str | None,
    model: str,
    system: str,
    user: str,
    max_output_tokens: int,
    timeout_seconds: float,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_completion_tokens": max_output_tokens,
        "response_format": {"type": "json_object"},
    }
    api_request = request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    context = ssl.create_default_context(cafile=ca_bundle) if ca_bundle else ssl.create_default_context()
    try:
        with request.urlopen(api_request, timeout=timeout_seconds, context=context) as response:
            body = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        request_id = exc.headers.get("x-request-id", "unknown")
        raise RuntimeError(f"OpenAI request failed ({exc.code}, request_id={request_id}): {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenAI request failed: {exc.reason}") from exc
    try:
        raw = str(body["choices"][0]["message"]["content"] or "").strip()
        return json.loads(raw)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"OpenAI returned an invalid JSON grading response: {body}") from exc


def validate_score(score: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        solve = score["solve_attempt"]
        difficulty = score["difficulty_assessment"]
        competition = score["competition_appropriateness"]
        if solve["status"] not in {"solved", "partial", "unclear"}:
            return False, "invalid solve status"
        if solve["selected_answer"] not in {"A", "B", "C", "D", "E", "UNKNOWN"}:
            return False, "invalid selected answer"
        if not isinstance(solve.get("issue_notes"), list):
            return False, "issue_notes must be a list"
        values = [difficulty["score"]] + [
            competition[key]
            for key in ("depth_reasoning", "conceptual_richness", "clarity", "olympiad_similarity")
        ]
        if any(not isinstance(value, int) or not 1 <= value <= 5 for value in values):
            return False, "all scores must be integers from 1 to 5"
    except (KeyError, TypeError):
        return False, "missing required grading fields"
    return True, ""


def completed_count(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                json.loads(line)
                count += 1
    return count


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Blindly solve and score generated physics MCQs")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="scored/scored.jsonl")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--subject", default="Physics")
    parser.add_argument("--competition", default="F=ma")
    parser.add_argument("--exemplars", nargs="*", default=["data/txts/all_questions.jsonl"])
    parser.add_argument("--exemplar-limit", type=int, default=3)
    parser.add_argument("--exemplar-max-chars", type=int, default=1800)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=1600)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--ca-bundle", help="PEM CA bundle; defaults to SSL_CERT_FILE or certifi")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

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

    rows = read_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]
    for index, row in enumerate(rows, start=1):
        if not str(row.get("question") or "").strip():
            raise ValueError(f"input row {index} has no question")
        if set(normalize_choices(row.get("choices"))) != set("ABCDE"):
            raise ValueError(f"input row {index} does not have exactly choices A-E")

    exemplars = load_exemplars(args.exemplars, args.exemplar_limit, args.exemplar_max_chars)
    system = build_system(args.subject, args.competition, exemplars)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start = completed_count(out_path) if args.resume else 0
    if start > len(rows):
        raise RuntimeError(f"output contains {start} rows but input contains only {len(rows)}")
    mode = "a" if args.resume else "w"

    with out_path.open(mode, encoding="utf-8") as handle:
        for index, row in enumerate(rows[start:], start=start + 1):
            problem = {
                "id": row.get("id"),
                "question": row["question"],
                "choices": normalize_choices(row["choices"]),
            }
            retry_note = ""
            for attempt in range(1, args.max_attempts + 1):
                user = "PROBLEM_JSON:\n" + json.dumps(problem, ensure_ascii=False)
                if retry_note:
                    user += f"\nPrevious grading output was rejected: {retry_note}. Return the exact schema."
                score = api_json(
                    api_key,
                    args.openai_base_url,
                    ca_bundle,
                    args.model,
                    system,
                    user,
                    args.max_output_tokens,
                    args.timeout_seconds,
                )
                valid, retry_note = validate_score(score)
                if valid:
                    break
                if args.debug:
                    print(f"[{index}/{len(rows)}] rejected attempt {attempt}: {retry_note}", file=sys.stderr)
            else:
                raise RuntimeError(f"grader failed validation after {args.max_attempts} attempts: {retry_note}")

            out_row = {
                "id": row.get("id"),
                **score,
                "_meta": {
                    **(row.get("_meta") or {}),
                    "grader": "06_verify_and_score_rest",
                    "grader_model": args.model,
                    "grading_input_index": index,
                },
            }
            handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            handle.flush()
            if args.debug:
                print(f"[{index}/{len(rows)}] scored in {time.strftime('%H:%M:%S')}", file=sys.stderr)
    print(f"Scored {len(rows) - start} rows -> {out_path}")


if __name__ == "__main__":
    main()
