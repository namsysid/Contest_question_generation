#!/usr/bin/env python3
"""Join official questions to fail-closed, independently verified D4/D5 solutions."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REQUIRED = (
    "id", "official_answer", "derived_answer", "answer_key_source", "solution_text",
    "shortest_solution_steps", "non_obvious_decisions", "mechanism_tags", "pitfalls",
    "verification_methods", "difficulty", "verified",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def year_for(row: dict[str, Any]) -> int:
    match = re.search(r"(?:19|20)\d{2}", str(row.get("source_pdf") or row.get("id") or ""))
    return int(match.group()) if match else 0


def validate(solution: dict[str, Any]) -> list[str]:
    errors = [f"missing {key}" for key in REQUIRED if solution.get(key) in (None, "", [], False)]
    official = str(solution.get("official_answer") or "").strip().upper()
    derived = str(solution.get("derived_answer") or "").strip().upper()
    if official not in "ABCDE" or len(official) != 1:
        errors.append("official_answer must be A-E")
    if derived != official:
        errors.append("derived_answer does not agree with official_answer")
    if solution.get("difficulty") not in (4, 5):
        errors.append("difficulty must be 4 or 5")
    if len(solution.get("shortest_solution_steps") or []) < 4:
        errors.append("fewer than four indispensable solution steps")
    if len(solution.get("non_obvious_decisions") or []) < 1:
        errors.append("no non-obvious modeling decision")
    if len(solution.get("mechanism_tags") or []) < 2:
        errors.append("fewer than two mechanism tags")
    if len(solution.get("verification_methods") or []) < 2:
        errors.append("fewer than two verification methods")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="*", default=[2024, 2025, 2026])
    args = parser.parse_args()
    questions = {str(row.get("id")): row for row in read_jsonl(args.questions)}
    solutions = read_jsonl(args.solutions)
    output: list[dict[str, Any]] = []
    failures: list[str] = []
    for solution in solutions:
        item_id = str(solution.get("id") or "")
        errors = validate(solution)
        question = questions.get(item_id)
        if not question:
            errors.append("no matching source question")
        elif question.get("has_diagram"):
            errors.append("diagram problem excluded")
        elif year_for(question) not in args.years:
            errors.append("source year excluded")
        if errors:
            failures.append(f"{item_id or '<missing id>'}: {', '.join(errors)}")
            continue
        steps = [str(value).strip() for value in solution["shortest_solution_steps"]]
        decisions = [str(value).strip() for value in solution["non_obvious_decisions"]]
        tags = [str(value).strip() for value in solution["mechanism_tags"]]
        signature = " | ".join([
            "STEPS=" + " -> ".join(steps),
            "DECISIONS=" + "; ".join(decisions),
            "MECHANISMS=" + ", ".join(tags),
            "PITFALLS=" + "; ".join(str(x).strip() for x in solution["pitfalls"]),
        ])
        output.append({
            "id": item_id,
            "question_text": question["question_text"],
            "question_number": question.get("question_number"),
            "source_pdf": question.get("source_pdf"),
            "source_year": year_for(question),
            **{key: solution[key] for key in REQUIRED if key != "id"},
            "solution_signature": signature,
        })
    if failures:
        raise RuntimeError("verified seed bank rejected records:\n" + "\n".join(failures))
    if not output:
        raise RuntimeError("no verified D4/D5 seeds; refusing to build an empty hard-source bank")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(output)} verified D4/D5 seeds -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

