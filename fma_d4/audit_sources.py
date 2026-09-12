#!/usr/bin/env python3
"""Report whether a question corpus actually contains answers and worked solutions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.questions)
    report = {
        "input": str(args.questions),
        "rows": len(rows),
        "text_only": sum(not row.get("has_diagram") for row in rows),
        "with_answer": sum(bool(row.get("answer") or row.get("answer_key")) for row in rows),
        "with_solution": sum(bool(row.get("solution") or row.get("solution_text")) for row in rows),
        "warning": (
            "analysis/problem_graph is model metadata, not a verified solution or answer key"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

