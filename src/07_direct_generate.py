#!/usr/bin/env python3
"""
07_direct_generate.py  (DIRECT PROMPTED CONTEST PROBLEM GENERATION)

Reads an input JSONL and directly prompts an LLM to generate contest-style
multiple-choice problems (straight prompting; no skeletons).

Input JSONL schema (per line; minimal):
{
  "id": "optional-id",
  "prompt": "what to generate (topic/constraints/style)",
  "constraints": "optional extra constraints"
}

Output JSONL schema:
{
  "id": "...",
  "question": "...",
  "choices": {"A":"...","B":"...","C":"...","D":"...","E":"..."},
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
import os
import sys
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not set (env or .env).")

SYSTEM_GEN = """You generate contest-faithful STEM multiple-choice problems (e.g., F=ma / USNCO style).

Hard constraints:
- You MUST produce a novel scenario and novel phrasing.
- Provide exactly 5 answer choices (A)-(E) with plausible, confusable distractors.
- Keep the problem self-contained and solvable without outside references.
Return strict JSON only (no markdown).
"""


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_jsonl_line(f, row: Dict[str, Any]) -> None:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    f.flush()


def llm_json(client: OpenAI, model: str, system: str, user: str, temperature: float = 0.2) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        response_format={"type":"json_object"},
    )
    return json.loads(resp.choices[0].message.content)

def build_prompt(row: Dict[str, Any]) -> str:
    prompt = row.get("prompt") or ""
    constraints = row.get("constraints") or ""

    return f"""You are given:

1) USER_PROMPT (topic/constraints/style):
{prompt}

2) EXTRA_CONSTRAINTS:
{constraints}

TASK:
Generate ONE NEW contest-style multiple-choice problem that fits the prompt.
Make the scenario and phrasing clearly novel.
Provide confusable distractors, and a clear solution that supports the answer.

OUTPUT strict JSON schema:
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
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL with prompt/constraints")
    ap.add_argument("--out", default="direct_generated.jsonl")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--debug", action="store_true", help="print per-row progress to stderr")
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    client = OpenAI()

    total = len(rows)
    if args.debug:
        print(f"[write] open (truncate): {args.out}", file=sys.stderr, flush=True)

    with open(args.out, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows, start=1):
            rid = row.get("id") or f"row_{i:05d}"
            prompt = build_prompt(row)

            if args.debug:
                print(f"[{i}/{total}] id={rid} calling model={args.model}", file=sys.stderr, flush=True)
                t0 = time.time()

            cand = llm_json(client, args.model, SYSTEM_GEN, prompt, temperature=0.25)

            if args.debug:
                dt = time.time() - t0
                print(f"[{i}/{total}] id={rid} model done in {dt:.2f}s", file=sys.stderr, flush=True)

            out_row = {
                **cand,
                "id": cand.get("id") or rid,
                "_meta": {
                    "input_id": rid,
                },
            }

            write_jsonl_line(out_f, out_row)
            if args.debug:
                print(f"[{i}/{total}] id={rid} wrote", file=sys.stderr, flush=True)

            if args.sleep:
                time.sleep(args.sleep)

    if args.debug:
        print(f"[write] done: {args.out}", file=sys.stderr, flush=True)
    print(f"Wrote {total} rows -> {args.out}")


if __name__ == "__main__":
    main()
