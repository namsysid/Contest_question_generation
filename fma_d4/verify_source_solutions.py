#!/usr/bin/env python3
"""Independently re-solve regenerated source solutions and build usable seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from circuit_lab.model_client import generate_json  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-output-tokens", type=int, default=10000)
    args = parser.parse_args()
    rows = read_jsonl(args.solutions)
    # Deliberately hide the first solver's answer and derivation. Showing them
    # causes agreement-by-anchoring rather than independent verification.
    payload = [{"id": r["id"], "question": r["question"]} for r in rows]
    system = """Independently solve each F=ma source question from its stem and choices only.
Report the selected answer and a compact derivation sufficient to expose equations, signs, frames,
and threshold conditions. Judge shortest-path difficulty honestly. Return JSON."""
    prompt = "Assess:\n" + json.dumps(payload, ensure_ascii=False) + """
Return {"verdicts":[{"id":"...","independent_answer":"A|B|C|D|E|UNKNOWN",
"independent_solution":"...","difficulty":1,"problem_well_formed":true,
"answer_confident":true,"issues":["..."]}]}"""
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {"verdicts": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "id": {"type": "string"},
                "independent_answer": {"type": "string", "enum": [*"ABCDE", "UNKNOWN"]},
                "independent_solution": {"type": "string"},
                "difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
                "problem_well_formed": {"type": "boolean"},
                "answer_confident": {"type": "boolean"},
                "issues": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["id", "independent_answer", "independent_solution", "difficulty",
                         "problem_well_formed", "answer_confident", "issues"],
        }}},
        "required": ["verdicts"],
    }
    result = generate_json(args.model, prompt, provider=args.provider, system=system,
                           reasoning_effort=args.reasoning_effort,
                           max_output_tokens=args.max_output_tokens,
                           json_schema=schema)
    verdicts = {r["id"]: r for r in result.get("verdicts") or []}
    accepted, rejected = [], []
    for row in rows:
        verdict = verdicts.get(row["id"], {})
        okay = (
            verdict.get("independent_answer") == row.get("derived_answer")
            and verdict.get("problem_well_formed") is True
            and verdict.get("answer_confident") is True
        )
        combined = {**row, "independent_verdict": verdict}
        if okay:
            first_solver_difficulty = combined.get("difficulty")
            combined.update({
                "question_text": row["question"]["stem"] + "\n" + "\n".join(row["question"]["choices"]),
                "solution_signature": " | ".join([
                    "STEPS=" + " -> ".join(row["shortest_solution_steps"]),
                    "DECISIONS=" + "; ".join(row["non_obvious_decisions"]),
                    "MECHANISMS=" + ", ".join(row["mechanism_tags"]),
                    "PITFALLS=" + "; ".join(row["pitfalls"]),
                    "BLIND_DERIVATION=" + str(verdict.get("independent_solution") or ""),
                ]),
                "source_solver_difficulty": first_solver_difficulty,
                "difficulty": verdict.get("difficulty"),
                "verification_status": "blind_answer_agreement",
            })
            accepted.append(combined)
        else:
            rejected.append(combined)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in accepted))
    args.out.with_suffix(args.out.suffix + ".rejected").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected))
    print(json.dumps({"accepted": len(accepted), "rejected": len(rejected), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
