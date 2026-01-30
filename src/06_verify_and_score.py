#!/usr/bin/env python3
"""
06_verify_and_score.py  (PAPER-FAITHFUL VERIFICATION + RUBRIC SCORING)

Implements the paper's Verification stage as an *evaluation* artifact:
1) Answer and Question Validity (binary gatekeeper)
2) Competition Appropriateness (rubric scores 1-5)
3) Difficulty Assessment (rubric score 1-5)

Input:  generated_problems.jsonl (from 05_generate_questions.py)
Output: scored.jsonl with:
{
  "id": "...",
  "gatekeeper": {"pass": true/false, "reasons":[...]},
  "competition_appropriateness": {
    "depth_reasoning": 1-5,
    "conceptual_richness": 1-5,
    "clarity": 1-5,
    "olympiad_similarity": 1-5,
    "notes": "..."
  },
  "difficulty_assessment": {"score":1-5,"notes":"..."},
  "_meta": {...}
}
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not set (env or .env).")

SYSTEM = """You are an expert evaluator of Olympiad-style STEM multiple-choice problems.

You will receive a single problem JSON with: question, choices, answer, solution, skeleton_text.

Return ONLY strict JSON with keys:
{
  "gatekeeper": {
     "pass": true/false,
     "reasons": ["..."]
  },
  "competition_appropriateness": {
     "depth_reasoning": 1-5,
     "conceptual_richness": 1-5,
     "clarity": 1-5,
     "olympiad_similarity": 1-5,
     "notes": "..."
  },
  "difficulty_assessment": {
     "score": 1-5,
     "notes": "..."
  }
}

Scoring rubrics (1-5):
- depth_reasoning: steps/structure required; 5 = multi-step with nontrivial insight
- conceptual_richness: quality/interestingness of concepts; 5 = rich, not plug-and-chug
- clarity: statement & solution clarity; 5 = very clear, minimal ambiguity
- olympiad_similarity: 5 = indistinguishable from real contest problems

Difficulty score:
1 = very easy; 3 = medium; 5 = very hard (Olympiad-level for target contest).

Be conservative: if the solution does not convincingly support the answer, gatekeeper.pass=false.
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

def llm_json(client: OpenAI, model: str, system: str, user: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        response_format={"type":"json_object"},
    )
    return json.loads(resp.choices[0].message.content)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="generated_problems.jsonl from 05_generate_questions.py")
    ap.add_argument("--out", default="scored.jsonl")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    client = OpenAI()
    out_rows: List[Dict[str, Any]] = []

    for r in rows:
        user = "PROBLEM_JSON:\n" + json.dumps({
            "id": r.get("id"),
            "question": r.get("question"),
            "choices": r.get("choices"),
            "answer": r.get("answer"),
            "solution": r.get("solution"),
            "skeleton_text": r.get("skeleton_text"),
        }, ensure_ascii=False)

        score = llm_json(client, args.model, SYSTEM, user)
        out_rows.append({
            "id": r.get("id"),
            **score,
            "_meta": r.get("_meta") or {},
        })

    write_jsonl(args.out, out_rows)
    print(f"Wrote {len(out_rows)} scored rows -> {args.out}")

if __name__ == "__main__":
    main()
