#!/usr/bin/env python3
"""Merge replacement verification reports into a base report set in question order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--replacements", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    questions = read_jsonl(args.questions)
    reports = {str(row.get("id")): row for row in read_jsonl(args.base)}
    replacement_rows = read_jsonl(args.replacements)
    reports.update({str(row.get("id")): row for row in replacement_rows})
    ordered = []
    for question in questions:
        record_id = str(question.get("id") or "")
        if record_id not in reports:
            raise ValueError(f"missing report for {record_id}")
        ordered.append(reports[record_id])
    with args.out.open("w", encoding="utf-8") as handle:
        for report in ordered:
            handle.write(json.dumps(report, ensure_ascii=False) + "\n")
    print(f"Merged {len(replacement_rows)} replacement reports into {len(ordered)} records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
