#!/usr/bin/env python3
"""Apply independent cold-solve results to an F=ma set and reject unresolved items."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def effective_selected_answer(solve: dict[str, Any]) -> str:
    reasoning = str(solve.get("reasoning_summary") or "")
    conclusions = re.findall(
        r"(?:answer|choice)\s*(?:is|:|=)?\s*(?:choice\s*)?([A-E])\b|"
        r"\b([A-E])\s+is\s+(?:the\s+)?correct\b|"
        r"\bmatches\s+(?:choice\s+)?([A-E])\b",
        reasoning,
        flags=re.IGNORECASE,
    )
    letters = [letter.upper() for groups in conclusions for letter in groups if letter]
    return letters[-1] if letters else str(solve.get("selected_answer") or "").upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out-reports", type=Path,
                        help="Write reports with answer fields reconciled to their derivations")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    reports = {str(row.get("id")): row for row in read_jsonl(args.reports)}
    corrected = 0
    report_corrections = 0
    output: list[dict[str, Any]] = []
    normalized_reports: list[dict[str, Any]] = []
    for row in rows:
        record_id = str(row.get("id") or "")
        report = reports.get(record_id)
        if not report:
            raise ValueError(f"missing verification report for {record_id}")
        solve = report.get("solve_attempt") or {}
        topic = report.get("topic_assessment") or {}
        selected = effective_selected_answer(solve)
        if solve.get("status") != "solved" or selected not in "ABCDE" or len(selected) != 1:
            raise ValueError(f"independent verifier could not solve {record_id}: {solve}")
        if topic.get("matches_required") is not True:
            raise ValueError(f"independent verifier found topic drift for {record_id}: {topic}")
        updated = dict(row)
        if selected != row.get("answer"):
            corrected += 1
            updated["answer"] = selected
            reasoning = str(solve.get("reasoning_summary") or "").strip()
            if reasoning:
                updated["solution"] = reasoning
        updated["_meta"] = {
            **(row.get("_meta") or {}),
            "independent_verifier": (report.get("_meta") or {}).get("grader"),
            "independent_verifier_model": (report.get("_meta") or {}).get("grader_model"),
            "answer_corrected_by_verifier": selected != row.get("answer"),
        }
        output.append(updated)
        normalized_report = dict(report)
        normalized_report["solve_attempt"] = {**solve, "selected_answer": selected}
        if selected != str(solve.get("selected_answer") or "").upper():
            report_corrections += 1
            normalized_report["_meta"] = {
                **(report.get("_meta") or {}), "selected_answer_reconciled_to_reasoning": True,
            }
        normalized_reports.append(normalized_report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.out_reports:
        args.out_reports.parent.mkdir(parents=True, exist_ok=True)
        with args.out_reports.open("w", encoding="utf-8") as handle:
            for report in normalized_reports:
                handle.write(json.dumps(report, ensure_ascii=False) + "\n")
    print(f"Wrote {len(output)} verified questions ({corrected} answer corrections) -> {args.out}")
    if args.out_reports:
        print(f"Wrote normalized reports ({report_corrections} field corrections) -> {args.out_reports}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
