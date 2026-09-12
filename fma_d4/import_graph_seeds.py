#!/usr/bin/env python3
"""Adapt enriched problem graphs into provisional D4 retrieval seeds.

This is an experimental compatibility path. Graphs describe a proposed solution
topology but do not carry independently checked answers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enriched", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-difficulty", type=int, default=4)
    args = parser.parse_args()

    seeds = []
    for row in read_jsonl(args.enriched):
        analysis = row.get("analysis") or {}
        problem = row.get("problem") or {}
        profile = analysis.get("graph_profile") or {}
        difficulty = int(analysis.get("difficulty") or 0)
        if difficulty < args.min_difficulty or analysis.get("diagram_required"):
            continue
        signature_parts = [
            "TARGET=" + str(profile.get("target_summary") or ""),
            "STATE_EVOLUTION=" + " -> ".join(profile.get("state_evolution") or []),
            "HIDDEN_STATES=" + "; ".join(profile.get("hidden_states") or []),
            "COUPLING=" + "; ".join(profile.get("coupling_points") or []),
            "INSIGHTS=" + "; ".join(profile.get("insight_type") or []),
            "TRAPS=" + "; ".join(profile.get("trap_profile") or []),
        ]
        choices = problem.get("choices") or []
        seeds.append({
            "id": row["id"],
            "question_text": str(problem.get("stem") or "") + "\n" + "\n".join(choices),
            "solution_text": analysis.get("graph_text") or analysis.get("skeleton") or "",
            "solution_signature": " | ".join(signature_parts),
            "difficulty": min(5, difficulty),
            "mechanism_tags": analysis.get("structure_tags") or analysis.get("concepts") or [],
            "graph_provisional": True,
        })

    if len(seeds) < 2:
        raise RuntimeError("need at least two graph-derived hard seeds")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in seeds))
    print(json.dumps({"seeds": len(seeds), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
