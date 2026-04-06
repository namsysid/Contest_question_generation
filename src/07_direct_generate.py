#!/usr/bin/env python3
"""
07_direct_generate.py  (BULK DIRECT CONTEST PROBLEM GENERATION)

Reads an exemplar JSONL corpus, packs as many problems as fit into the prompt
budget, and asks the model to generate a batch of new contest-style problems
in one call.

Input JSONL can be either:
- enriched rows with `problem.stem`
- generated/direct rows with `question`
- prompt rows with `prompt` / `constraints`

Output JSONL schema (one generated problem per line):
{
  "id": "...",
  "question": "...",
  "choices": {"A":"...", "B":"...", "C":"...", "D":"...", "E":"..."},
  "answer": "A|B|C|D|E",
  "solution": "...",
  "anti_copy_report": {
    "novel_scenario_summary": "...",
    "differences_from_exemplars": ["...", "..."],
    "possible_overlap_risks": ["...", "..."]
  },
  "_meta": {...}
}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from ollama_client import generate_json, generate_text

load_dotenv()

SYSTEM_GEN = """You generate contest-faithful multiple-choice problems from a bank of exemplars.

Hard constraints:
- Produce novel scenarios and novel phrasing; do not paraphrase or lightly rewrite exemplars.
- Provide exactly 5 answer choices (A)-(E) with plausible, confusable distractors.
- Keep each problem self-contained and solvable without outside references.
- Maintain the style and difficulty distribution implied by the exemplars.
- Return strict JSON only (no markdown).
"""


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


def llm_json(model: str, system: str, user: str, temperature: float = 0.2) -> Dict[str, Any]:
    return generate_json(model, user, system=system, temperature=temperature)


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
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--target_count", type=int, default=25)
    ap.add_argument("--max_prompt_chars", type=int, default=120000)
    ap.add_argument("--temperature", type=float, default=0.35)
    ap.add_argument("--debug", action="store_true", help="print prompt packing progress to stderr")
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    exemplar_blocks = pack_exemplars(rows, args.max_prompt_chars)
    prompt = build_prompt(exemplar_blocks, args.target_count)

    if args.debug:
        print(f"[pack] source rows={len(rows)}", file=sys.stderr, flush=True)
        print(f"[pack] exemplar rows packed={len(exemplar_blocks)}", file=sys.stderr, flush=True)
        print(f"[pack] prompt chars={len(prompt)}", file=sys.stderr, flush=True)
        print(f"[call] model={args.model}", file=sys.stderr, flush=True)
        t0 = time.time()

    result = llm_json(args.model, SYSTEM_GEN, prompt, temperature=args.temperature)

    if args.debug:
        dt = time.time() - t0
        print(f"[call] model done in {dt:.2f}s", file=sys.stderr, flush=True)

    problems = extract_problems(result)
    if not problems:
        top_level_keys = sorted(result.keys())
        try:
            raw = generate_text(args.model, prompt, system=SYSTEM_GEN, temperature=args.temperature, format_json=True)
        except Exception:
            raw = None
        detail = f"Top-level keys were: {top_level_keys}"
        if raw:
            detail += f"\nRaw output was:\n{raw}"
        raise ValueError(f"Model response missing a usable problems list. {detail}")

    with open(args.out, "w", encoding="utf-8") as out_f:
        for idx, cand in enumerate(problems, start=1):
            if not isinstance(cand, dict):
                continue
            out_row = {
                **cand,
                "id": cand.get("id") or f"generated_{idx:03d}",
                "_meta": {
                    "source_input": args.input,
                    "packed_exemplar_count": len(exemplar_blocks),
                    "requested_count": args.target_count,
                    "returned_count": len(problems),
                    "generator": "07_direct_generate_bulk",
                },
            }
            write_jsonl_line(out_f, out_row)

    if args.debug:
        print(f"[write] wrote {len(problems)} rows -> {args.out}", file=sys.stderr, flush=True)
    print(f"Wrote {len(problems)} rows -> {args.out}")


if __name__ == "__main__":
    main()
