#!/usr/bin/env python3
"""
06_verify_and_score.py  (GRAPH-AWARE VERIFICATION + RUBRIC SCORING)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not set (env or .env).")

SYSTEM_TEMPLATE = """You are an expert evaluator of Olympiad-style STEM multiple-choice problems.

You will receive a single problem JSON with: question, choices, answer, solution, and graph_text.
You also receive exemplars of real __COMPETITION_LABEL__ problems for comparison.

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

Judge the problem holistically as a contestant would.
Use graph_text as evidence of intended hidden structure, but do not reward it if the rendered problem does not actually realize that structure.
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
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="scored.jsonl")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--subject", default="Chem")
    ap.add_argument("--competition", default="USNCO")
    ap.add_argument("--exemplars", nargs="*", default=["data/text/exam1-2015-1-8.jsonl"])
    ap.add_argument("--exemplar-limit", type=int, default=3)
    ap.add_argument("--exemplar-max-chars", type=int, default=1200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    client = OpenAI()
    total = len(rows)
    competition_label = " & ".join(p for p in [args.subject, args.competition] if p) or "Olympiad"
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
                "graph_text": r.get("graph_text") or r.get("skeleton_text"),
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
