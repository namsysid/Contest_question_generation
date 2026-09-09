#!/usr/bin/env python3
"""Independently grade finished contest questions on the shared 1-5 difficulty rubric."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from circuit_lab.model_client import generate_json
from difficulty_rubric import DIFFICULTY_RUBRIC


SYSTEM = """You are a strict contest difficulty auditor. Rate the shortest correct solve a prepared
student could use. Ignore the supplied label, solution length, prose length, and graph complexity.
Do not reward routine arithmetic. You must actually solve every original problem before rating it.
Return strict JSON only."""

AUDIT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "assessments": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"}, "derived_answer": {"type": "string"},
                    "shortest_solution": {"type": "array", "items": {"type": "string"}},
                    "difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
                    "competition_level": {"type": "boolean"},
                    "reasoning_steps": {"type": "integer", "minimum": 1},
                    "non_obvious_decisions": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                },
                "required": [
                    "id", "derived_answer", "shortest_solution", "difficulty",
                    "competition_level", "reasoning_steps", "non_obvious_decisions", "notes",
                ],
            },
        },
    },
    "required": ["assessments"],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt_for(rows: list[dict[str, Any]], competition: str) -> str:
    items = []
    for row in rows:
        content = row.get("content") if isinstance(row.get("content"), dict) else {}
        items.append({
            "id": row.get("id"),
            "question": row.get("question") or row.get("question_text") or content.get("question_text"),
            "choices": row.get("choices") or content.get("choices"),
        })
    return f"""Competition: {competition}

{DIFFICULTY_RUBRIC}

For every item, infer the shortest valid solution and compare it with authentic {competition}
expectations. A score of 3 means normal {competition} difficulty—not merely a correct high-school
exercise. Return exactly one assessment per supplied id:
{{"assessments":[{{"id":"...","derived_answer":"...","shortest_solution":["step"],
"difficulty":1,"competition_level":false,"reasoning_steps":1,
"non_obvious_decisions":["decision"],"notes":"concise concrete reason"}}]}}

ITEMS:
{json.dumps(items, ensure_ascii=False)}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--competition", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="qwen3:14b-q4_K_M")
    parser.add_argument("--provider", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.ids:
        selected = set(args.ids)
        rows = [row for row in rows if str(row.get("id")) in selected]
        missing = selected - {str(row.get("id")) for row in rows}
        if missing:
            raise RuntimeError(f"requested ids not found: {sorted(missing)}")
    completed = read_jsonl(args.out) if args.resume and args.out.exists() else []
    completed_ids = {str(row["id"]) for row in completed}
    pending = [row for row in rows if str(row.get("id")) not in completed_ids]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if completed else "w"
    with args.out.open(mode, encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            expected = [str(row.get("id")) for row in batch]
            last_error = ""
            for attempt in range(1, 4):
                try:
                    result = generate_json(
                        args.model, prompt_for(batch, args.competition), provider=args.provider,
                        system=SYSTEM, temperature=0.0, max_output_tokens=4000,
                        json_schema=AUDIT_SCHEMA,
                    )
                    assessments = result.get("assessments")
                    if not isinstance(assessments, list):
                        raise ValueError("missing assessments array")
                    by_id = {str(item.get("id")): item for item in assessments if isinstance(item, dict)}
                    if set(by_id) != set(expected):
                        raise ValueError("assessment ids do not exactly match request")
                    for item_id in expected:
                        item = by_id[item_id]
                        score = item.get("difficulty")
                        if not isinstance(score, int) or isinstance(score, bool) or score not in range(1, 6):
                            raise ValueError(f"invalid difficulty for {item_id}")
                        if not isinstance(item.get("shortest_solution"), list) or not item["shortest_solution"]:
                            raise ValueError(f"missing shortest solution for {item_id}")
                        item["competition_level"] = bool(item.get("competition_level"))
                        item["competition"] = args.competition
                        item["model"] = args.model
                    break
                except (RuntimeError, ValueError) as exc:
                    last_error = str(exc)
                    if attempt < 3:
                        time.sleep(attempt)
            else:
                raise RuntimeError(f"audit batch failed: {last_error}")
            for item_id in expected:
                handle.write(json.dumps(by_id[item_id], ensure_ascii=False) + "\n")
            handle.flush()
            print(f"audited {len(completed) + min(start + len(batch), len(pending))}/{len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
