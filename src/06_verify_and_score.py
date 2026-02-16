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
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not set (env or .env).")

SYSTEM_TEMPLATE = """You are an expert evaluator of Olympiad-style STEM multiple-choice problems.

You will receive a single problem JSON with: question, choices, and answer.
You also receive exemplars of real __COMPETITION_LABEL__ problems and solutions for comparison.

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

Competition Appropriateness Rubric (1-5) for holistic __COMPETITION_LABEL__ judgment:
1 - Not Appropriate: Poorly posed, unclear, trivial, gimmicky, or unlike contest problems.
2 - Weakly Appropriate: Coherent but lacks contest realism; feels like a textbook exercise or has awkward structure.
3 - Moderately Appropriate: Contest-like topic/structure but missing depth/elegance/polish; acceptable only as low-quality practice.
4 - Highly Appropriate: Matches real contest problems in structure/reasoning/clarity; fair, educational; high-quality practice.
5 - Excellent / Contest-Ready: Indistinguishable from real Olympiad problems; clean, fair, conceptually rich, elegant.
Notes: Focus on structure and reasoning over wording; multiple valid solution paths are fine; novelty is preferred over rehashed templates.

Difficulty Assessment Rubric (1-5):
1 - Very Easy: Single obvious idea or direct application of a basic fact.
2 - Easy: Basic reasoning beyond recall; quick once the main idea is found.
3 - Medium: Multiple steps or careful case analysis; mid-tier contest difficulty.
4 - Hard: Deep understanding or clever insight; upper-tier contest difficulty.
5 - Very Hard / Olympiad-Level: Sustained multi-step reasoning and significant insight; hardest major-contest level.
Notes: Judge minimum required reasoning, not solution length; ignore rare shortcuts unless they trivialize the problem;
assume a well-prepared contest participant.

Make sure to run through the problem and evaluate it holistically. Don't just delve into semantics. Run through the problems as if you were a competitor, evaluating the entire problem.
You must look at exemplars provided from the competition to base your judgement from.
"""

def load_exemplars(paths: Iterable[str], limit: int, max_chars: int) -> List[str]:
    exemplars: List[str] = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if len(exemplars) >= limit:
                        return exemplars
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = (obj.get("question_text") or "").strip()
                    if not text:
                        continue
                    # Keep only reasonably sized, choice-based items.
                    if len(text) > max_chars:
                        continue
                    if "(A)" not in text or "(B)" not in text or "(C)" not in text:
                        continue
                    exemplars.append(text)
        except FileNotFoundError:
            continue
    return exemplars

def build_system_with_exemplars(system_base: str, exemplars: List[str], competition_label: str) -> str:
    if not exemplars:
        return system_base
    chunks = [f"\n\nEXEMPLARS (real {competition_label} problems; use as style reference, not to copy):"]
    for i, ex in enumerate(exemplars, start=1):
        chunks.append(f"\n---\nEXEMPLAR {i}:\n{ex}")
    return system_base + "".join(chunks)

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
    ap.add_argument("--subject", default="Chem", help="subject label, e.g., Chem or Phys")
    ap.add_argument("--competition", default="USNCO", help="competition label, e.g., USNCO or F=ma")
    ap.add_argument(
        "--exemplars",
        nargs="*",
        default=["data/text/exam1-2015-1-8.jsonl"],
        help="one or more jsonl files with real competition problems (question_text)",
    )
    ap.add_argument("--exemplar-limit", type=int, default=3)
    ap.add_argument("--exemplar-max-chars", type=int, default=1200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--debug", action="store_true", help="print per-row progress to stderr")
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    client = OpenAI()
    total = len(rows)
    competition_label = " & ".join(p for p in [args.subject, args.competition] if p)
    if not competition_label:
        competition_label = "Olympiad"
    system_base = SYSTEM_TEMPLATE.replace("__COMPETITION_LABEL__", competition_label)
    exemplars = load_exemplars(args.exemplars, args.exemplar_limit, args.exemplar_max_chars)
    system = build_system_with_exemplars(system_base, exemplars, competition_label)

    if args.debug:
        print(f"[write] open (truncate): {args.out}", file=sys.stderr, flush=True)
    with open(args.out, "w", encoding="utf-8") as out_f:
        for i, r in enumerate(rows, start=1):
            if args.debug:
                rid = r.get("id", "?")
                print(f"[{i}/{total}] id={rid} calling model={args.model}", file=sys.stderr, flush=True)
                t0 = time.time()

            user = "PROBLEM_JSON:\n" + json.dumps({
                "id": r.get("id"),
                "question": r.get("question"),
                "choices": r.get("choices"),
                "answer": r.get("answer"),
                "solution": r.get("solution"),
                "skeleton_text": r.get("skeleton_text"),
            }, ensure_ascii=False)

            score = llm_json(client, args.model, system, user)

            if args.debug:
                dt = time.time() - t0
                print(f"[{i}/{total}] id={r.get('id','?')} model done in {dt:.2f}s", file=sys.stderr, flush=True)

            out_row = {
                "id": r.get("id"),
                **score,
                "_meta": r.get("_meta") or {},
            }
            write_jsonl_line(out_f, out_row)
            if args.debug:
                print(f"[{i}/{total}] id={r.get('id','?')} wrote", file=sys.stderr, flush=True)

    if args.debug:
        print(f"[write] done: {args.out}", file=sys.stderr, flush=True)
    print(f"Wrote {total} scored rows -> {args.out}")

if __name__ == "__main__":
    main()
