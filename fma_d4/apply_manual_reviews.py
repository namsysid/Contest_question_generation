#!/usr/bin/env python3
"""Attach explicit manual-equation reviews to source-solution records."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


p = argparse.ArgumentParser()
p.add_argument("--solutions", nargs="+", type=Path, required=True)
p.add_argument("--reviews", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
solutions = {r["id"]: r for path in a.solutions for r in rows(path)}
reviews = rows(a.reviews)
output = []
for review in reviews:
    source = solutions.get(review["id"])
    if source is None:
        raise RuntimeError(f"missing solution for {review['id']}")
    if source.get("derived_answer") != review.get("independent_answer"):
        raise RuntimeError(f"manual answer disagreement for {review['id']}")
    verdict = {k: v for k, v in review.items() if k not in {"verification_status"}}
    record = {**source, "independent_verdict": verdict}
    record.update({
        "question_text": source["question"]["stem"] + "\n" + "\n".join(source["question"]["choices"]),
        "solution_signature": " | ".join([
            "STEPS=" + " -> ".join(source["shortest_solution_steps"]),
            "DECISIONS=" + "; ".join(source["non_obvious_decisions"]),
            "MECHANISMS=" + ", ".join(source["mechanism_tags"]),
            "PITFALLS=" + "; ".join(source["pitfalls"]),
            "INDEPENDENT_REVIEW=" + review["independent_solution"],
        ]),
        "source_solver_difficulty": source.get("difficulty"),
        "difficulty": review.get("difficulty"),
        "verification_status": review["verification_status"],
    })
    output.append(record)
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output))
print(json.dumps({"manual_verified": len(output), "out": str(a.out)}, indent=2))

