#!/usr/bin/env python3
"""
07_direct_generate_openai.py  (BULK DIRECT CONTEST PROBLEM GENERATION)

OpenAI-backed variant of stage 07. Reads an exemplar JSONL corpus, packs as
many problems as fit into the prompt budget, and asks the model to generate a
batch of new contest-style problems in one call.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()

SYSTEM_GEN = """You generate contest-faithful multiple-choice problems from a bank of exemplars.

Hard constraints:
- Produce novel scenarios and novel phrasing; do not paraphrase or lightly rewrite exemplars.
- Provide exactly 5 answer choices (A)-(E) with plausible, confusable distractors.
- Keep each problem self-contained and solvable without outside references.
- Maintain the style and difficulty distribution implied by the exemplars.
- Return strict JSON only (no markdown).
"""

INVALID_ESCAPE_RE = re.compile(r"\\(?![\"\\/bfnrtu])")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl_line(f, row: Dict[str, Any]) -> None:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    f.flush()


def _loads_with_escape_repair(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        repaired = INVALID_ESCAPE_RE.sub(r"\\\\", text)
        if repaired == text:
            raise exc
        return json.loads(repaired)


def parse_json_text(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Model returned empty content")
    try:
        return _loads_with_escape_repair(raw)
    except json.JSONDecodeError:
        i = raw.find("{")
        j = raw.rfind("}")
        if i < 0 or j < 0 or j <= i:
            raise ValueError(f"Model did not return JSON. Output was:\n{raw}")
        return _loads_with_escape_repair(raw[i : j + 1])


def llm_json(
    client: Any,
    model: str,
    system: str,
    user: str,
    max_output_tokens: int,
) -> Dict[str, Any]:
    if client is None:
        raise RuntimeError(
            "OpenAI client not available. Install the `openai` package and set OPENAI_API_KEY."
        )

    last_raw = ""
    prompts = [
        user,
        user
        + "\n\nIMPORTANT: If the previous answer was truncated, return the same JSON schema with shorter solutions and concise anti-copy fields. Return only JSON.",
    ]
    token_budgets = [max_output_tokens, max(max_output_tokens * 2, max_output_tokens + 1000)]

    for prompt, budget in zip(prompts, token_budgets):
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=budget,
        )

        text_parts: List[str] = []
        for out in getattr(resp, "output", []) or []:
            if getattr(out, "type", None) != "message":
                continue
            for content in getattr(out, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    text_parts.append(content.text)

        last_raw = "".join(text_parts).strip()
        try:
            return parse_json_text(last_raw)
        except Exception:
            continue

    raise ValueError(f"Model did not return parseable JSON. Output was:\n{last_raw}")


def extract_problems(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("problems", "generated_problems", "questions", "items", "results"):
        value = result.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    if all(k in result for k in ("question", "choices", "answer")):
        return [result]

    candidate_lists = [v for v in result.values() if isinstance(v, list)]
    if len(candidate_lists) == 1 and all(isinstance(item, dict) for item in candidate_lists[0]):
        return candidate_lists[0]

    return []


def stringify_choices(choices: Any) -> str:
    if isinstance(choices, dict):
        parts = []
        for key in ("A", "B", "C", "D", "E"):
            if key in choices:
                parts.append(f"({key}) {choices[key]}")
        return " | ".join(parts)
    if isinstance(choices, list):
        return " | ".join(str(x) for x in choices if x is not None)
    return ""


def exemplar_from_row(row: Dict[str, Any]) -> str:
    rid = str(row.get("id") or "unknown")

    if isinstance(row.get("problem"), dict):
        problem = row["problem"]
        stem = str(problem.get("stem") or "").strip()
        choices = stringify_choices(problem.get("choices"))
        answer = str(problem.get("answer_key") or "").strip()
        domain = str(row.get("domain") or "").strip()
        return (
            f"ID: {rid}\n"
            f"DOMAIN: {domain}\n"
            f"QUESTION:\n{stem}\n"
            f"CHOICES:\n{choices or '(none listed)'}\n"
            f"ANSWER: {answer or '(unknown)'}"
        ).strip()

    question = str(row.get("question") or row.get("question_text") or "").strip()
    prompt = str(row.get("prompt") or "").strip()
    constraints = str(row.get("constraints") or "").strip()
    choices = stringify_choices(row.get("choices"))
    answer = str(row.get("answer") or "").strip()

    if question:
        return (
            f"ID: {rid}\n"
            f"QUESTION:\n{question}\n"
            f"CHOICES:\n{choices or '(none listed)'}\n"
            f"ANSWER: {answer or '(unknown)'}"
        ).strip()

    return (
        f"ID: {rid}\n"
        f"PROMPT:\n{prompt}\n"
        f"CONSTRAINTS:\n{constraints or '(none)'}"
    ).strip()


def pack_exemplars(rows: List[Dict[str, Any]], max_prompt_chars: int) -> List[str]:
    packed: List[str] = []
    used = 0
    for row in rows:
        exemplar = exemplar_from_row(row)
        block = f"### EXEMPLAR {len(packed) + 1}\n{exemplar}\n"
        if packed and used + len(block) > max_prompt_chars:
            break
        if not packed and len(block) > max_prompt_chars:
            packed.append(block[:max_prompt_chars])
            break
        packed.append(block)
        used += len(block)
    return packed


def build_prompt(exemplar_blocks: List[str], target_count: int) -> str:
    exemplar_text = "\n".join(exemplar_blocks)
    return f"""Below is a bank of exemplar contest problems.

Use them only to infer style, scope, format, and difficulty profile.
Do not copy wording, numbers, or scenario structure too closely.

Generate {target_count} NEW problems in the same general style.

Return strict JSON with this exact top-level schema:
{{
  "problems": [
    {{
      "id": "<string>",
      "question": "<string stem>",
      "choices": {{"A":"...", "B":"...", "C":"...", "D":"...", "E":"..."}},
      "answer": "A|B|C|D|E",
      "solution": "<clear solution; may include equations>",
      "anti_copy_report": {{
        "novel_scenario_summary": "<1-2 sentences>",
        "differences_from_exemplars": ["...", "..."],
        "possible_overlap_risks": ["...", "..."]
      }}
    }}
  ]
}}

The `problems` array must contain exactly {target_count} items if possible.
If the context is too large or quality would degrade, return as many complete items as can be generated reliably.

EXEMPLARS:
{exemplar_text}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL corpus of exemplar problems or prompts")
    ap.add_argument("--out", default="direct_generated.jsonl")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--target_count", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--max_prompt_chars", type=int, default=24000)
    ap.add_argument("--max_output_tokens", type=int, default=5000)
    ap.add_argument("--timeout_seconds", type=float, default=180.0)
    ap.add_argument("--debug", action="store_true", help="print prompt packing progress to stderr")
    args = ap.parse_args()

    if args.target_count <= 0:
        raise ValueError("--target_count must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    rows = read_jsonl(args.input)
    exemplar_blocks = pack_exemplars(rows, args.max_prompt_chars)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")
    client = OpenAI(timeout=args.timeout_seconds) if OpenAI is not None else None

    if args.debug:
        print(f"[pack] source rows={len(rows)}", file=sys.stderr, flush=True)
        print(f"[pack] exemplar rows packed={len(exemplar_blocks)}", file=sys.stderr, flush=True)
        print(f"[pack] max_prompt_chars={args.max_prompt_chars}", file=sys.stderr, flush=True)
        print(f"[call] model={args.model} batch_size={args.batch_size} timeout={args.timeout_seconds}s", file=sys.stderr, flush=True)
        print(f"[write] initialized {args.out}", file=sys.stderr, flush=True)

    total_written = 0
    batch_num = 0
    with open(out_path, "a", encoding="utf-8") as out_f:
        while total_written < args.target_count:
            batch_num += 1
            remaining = args.target_count - total_written
            batch_target = min(args.batch_size, remaining)
            prompt = build_prompt(exemplar_blocks, batch_target)

            if args.debug:
                print(
                    f"[batch {batch_num}] request={batch_target} remaining={remaining} prompt_chars={len(prompt)}",
                    file=sys.stderr,
                    flush=True,
                )
                t0 = time.time()

            result = llm_json(
                client,
                args.model,
                SYSTEM_GEN,
                prompt,
                max_output_tokens=args.max_output_tokens,
            )

            if args.debug:
                dt = time.time() - t0
                print(f"[batch {batch_num}] model done in {dt:.2f}s", file=sys.stderr, flush=True)

            problems = extract_problems(result)
            if not problems:
                top_level_keys = sorted(result.keys())
                raise ValueError(
                    f"Model response missing a usable problems list in batch {batch_num}. "
                    f"Top-level keys were: {top_level_keys}"
                )

            wrote_this_batch = 0
            for cand in problems[:batch_target]:
                if not isinstance(cand, dict):
                    continue
                total_written += 1
                wrote_this_batch += 1
                out_row = {
                    **cand,
                    "id": cand.get("id") or f"generated_{total_written:03d}",
                    "_meta": {
                        "source_input": args.input,
                        "packed_exemplar_count": len(exemplar_blocks),
                        "requested_count": args.target_count,
                        "returned_count": total_written,
                        "generator": "07_direct_generate_openai",
                        "model": args.model,
                        "batch_index": batch_num,
                    },
                }
                write_jsonl_line(out_f, out_row)

            if wrote_this_batch == 0:
                raise ValueError(f"Batch {batch_num} returned no writable problems.")

            if args.debug:
                print(
                    f"[batch {batch_num}] wrote {wrote_this_batch} rows (total={total_written})",
                    file=sys.stderr,
                    flush=True,
                )

    if args.debug:
        print(f"[write] wrote {total_written} rows -> {args.out}", file=sys.stderr, flush=True)
    print(f"Wrote {total_written} rows -> {args.out}")


if __name__ == "__main__":
    main()
