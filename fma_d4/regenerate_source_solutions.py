#!/usr/bin/env python3
"""Regenerate full source solutions from question text, without trusting graphs."""

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
    parser.add_argument("--enriched", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    args = parser.parse_args()

    wanted = set(args.ids)
    questions = []
    for row in read_jsonl(args.enriched):
        if row.get("id") not in wanted:
            continue
        problem = row.get("problem") or {}
        questions.append({
            "id": row["id"],
            "stem": problem.get("stem"),
            "choices": problem.get("choices"),
        })
    if {row["id"] for row in questions} != wanted:
        raise RuntimeError("one or more requested source ids were not found")

    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {"solutions": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "id": {"type": "string"},
                "derived_answer": {"type": "string", "enum": list("ABCDE")},
                "solution_text": {"type": "string"},
                "shortest_solution_steps": {"type": "array", "minItems": 2, "items": {"type": "string"}},
                "non_obvious_decisions": {"type": "array", "items": {"type": "string"}},
                "mechanism_tags": {"type": "array", "minItems": 2, "items": {"type": "string"}},
                "pitfalls": {"type": "array", "items": {"type": "string"}},
                "verification_methods": {"type": "array", "minItems": 2, "items": {"type": "string"}},
                "difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["id", "derived_answer", "solution_text", "shortest_solution_steps",
                         "non_obvious_decisions", "mechanism_tags", "pitfalls",
                         "verification_methods", "difficulty"],
        }}},
        "required": ["solutions"],
    }
    system = """Solve F=ma source questions from scratch. Do not infer from an existing graph or
claimed key. Give a complete algebraic derivation, including signs, frames, threshold conditions,
and choice mapping. shortest_solution_steps must list only indispensable conceptual deductions;
do not inflate difficulty with algebra or checks. Return strict JSON."""
    result = generate_json(
        args.model,
        "Solve these source questions independently:\n" + json.dumps(questions, ensure_ascii=False),
        provider=args.provider, system=system, reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens, json_schema=schema,
    )
    solutions = result.get("solutions") or []
    if {row.get("id") for row in solutions} != wanted:
        raise RuntimeError("model did not return exactly the requested source ids")
    question_by_id = {row["id"]: row for row in questions}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps({**row, "question": question_by_id[row["id"]]}, ensure_ascii=False) + "\n" for row in solutions))
    print(json.dumps({"solutions": len(solutions), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
