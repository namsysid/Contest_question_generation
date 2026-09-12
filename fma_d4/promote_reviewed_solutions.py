#!/usr/bin/env python3
"""Promote explicitly reviewed regenerated solutions into retrieval seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--reviewed-ids", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reviewed = set(args.reviewed_ids)
    rows = [json.loads(line) for line in args.solutions.read_text().splitlines() if line.strip()]
    if {row["id"] for row in rows if row["id"] in reviewed} != reviewed:
        raise RuntimeError("reviewed id missing from regenerated solutions")
    seeds = []
    for row in rows:
        if row["id"] not in reviewed:
            continue
        row.update({
            "question_text": row["question"]["stem"] + "\n" + "\n".join(row["question"]["choices"]),
            "solution_signature": " | ".join([
                "STEPS=" + " -> ".join(row["shortest_solution_steps"]),
                "DECISIONS=" + "; ".join(row["non_obvious_decisions"]),
                "MECHANISMS=" + ", ".join(row["mechanism_tags"]),
                "PITFALLS=" + "; ".join(row["pitfalls"]),
            ]),
            "manual_reviewed": True,
        })
        seeds.append(row)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in seeds))
    print(json.dumps({"seeds": len(seeds), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
