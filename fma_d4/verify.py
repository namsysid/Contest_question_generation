#!/usr/bin/env python3
"""Independently solve and quality-check generated D4 candidates in batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from circuit_lab.model_client import generate_json


def accepted(candidate: dict, verdict: dict) -> bool:
    return (
        verdict.get("selected_answer") == candidate.get("answer")
        and verdict.get("physics_valid") is True
        and verdict.get("unique_answer") is True
        and int(verdict.get("difficulty", 0)) >= 4
        and int(verdict.get("conceptual_deductions", 0)) >= 4
        and int(verdict.get("non_obvious_decisions", 0)) >= 1
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--reasoning-effort", default="high")
    args = parser.parse_args()

    candidates = [json.loads(line) for line in args.candidates.read_text().splitlines() if line.strip()]
    accepted_rows: list[dict] = []
    rejected_rows: list[dict] = []
    system = """You are an independent F=ma contest editor. Solve each problem
from scratch. You will not be shown a claimed answer or solution. Judge actual
shortest-solution difficulty, not wording or algebra length. D4 means the
hardest quartile of modern F=ma: at least four linked conceptual deductions and
at least one non-obvious modeling decision. A merely familiar template with
changed numbers is not D4. Return JSON only."""

    for start in range(0, len(candidates), args.batch_size):
        batch = candidates[start : start + args.batch_size]
        compact = [{
            "id": row["id"], "question": row["question"],
            "choices": row["choices"],
        } for row in batch]
        prompt = "Independently assess this batch:\n" + json.dumps(compact, ensure_ascii=False) + """

Return {"assessments":[{"id":"...","selected_answer":"A|B|C|D|E|UNKNOWN",
"independent_solution":"...",
"physics_valid":true,"unique_answer":true,"difficulty":1,
"conceptual_deductions":0,"non_obvious_decisions":0,
"familiar_template":false,"issues":["..."]}]}"""
        result = generate_json(
            args.model, prompt, provider=args.provider, system=system,
            reasoning_effort=args.reasoning_effort, max_output_tokens=8000,
        )
        by_id = {item["id"]: item for item in result.get("assessments", [])}
        for candidate in batch:
            verdict = by_id.get(candidate["id"], {"id": candidate["id"], "issues": ["missing verifier verdict"]})
            combined = {**candidate, "independent_verdict": verdict}
            (accepted_rows if accepted(candidate, verdict) else rejected_rows).append(combined)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row) + "\n" for row in accepted_rows))
    rejected_path = args.out.with_suffix(args.out.suffix + ".rejected")
    rejected_path.write_text("".join(json.dumps(row) + "\n" for row in rejected_rows))
    print(json.dumps({"accepted": len(accepted_rows), "rejected": len(rejected_rows), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
