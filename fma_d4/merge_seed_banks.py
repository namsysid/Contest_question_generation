#!/usr/bin/env python3
"""Merge independently accepted seed banks while rejecting duplicate ids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows, seen = [], set()
    for path in args.inputs:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in seen:
                raise RuntimeError(f"duplicate seed id: {row['id']}")
            verdict = row.get("independent_verdict") or {}
            row["source_solver_difficulty"] = row.get("difficulty")
            row["difficulty"] = verdict.get("difficulty", row.get("difficulty"))
            row["verification_status"] = "blind_answer_agreement"
            blind = str(verdict.get("independent_solution") or "").strip()
            if blind and "BLIND_DERIVATION=" not in str(row.get("solution_signature") or ""):
                row["solution_signature"] = str(row.get("solution_signature") or "") + " | BLIND_DERIVATION=" + blind
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise RuntimeError("no accepted seeds to merge")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(json.dumps({"seeds": len(rows), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
