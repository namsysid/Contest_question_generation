#!/usr/bin/env python3
"""Convert verified derivations into transferable, surface-free mechanisms."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from circuit_lab.model_client import generate_json  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-bank", type=Path, required=True)
    p.add_argument("--min-difficulty", type=int, default=4)
    p.add_argument("--ids", nargs="*", default=[])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="gpt-5")
    p.add_argument("--provider", default="openai")
    args = p.parse_args()
    requested = set(args.ids)
    selected = [r for r in read_jsonl(args.seed_bank)
                if (r.get("id") in requested if requested else int(r.get("difficulty") or 0) >= args.min_difficulty)]
    if not selected:
        raise RuntimeError("no verified seeds meet the difficulty floor")
    payload = [{"id": r["id"], "question": r["question_text"],
                "derivation": r["solution_text"],
                "blind_derivation": (r.get("independent_verdict") or {}).get("independent_solution")}
               for r in selected]
    item = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "id": {"type": "string"}, "deep_insight": {"type": "string"},
            "invariant_chain": {"type": "array", "minItems": 4, "items": {"type": "string"}},
            "generic_equations": {"type": "array", "items": {"type": "string"}},
            "indispensable_decisions": {"type": "array", "minItems": 1, "items": {"type": "string"}},
            "transfer_patterns": {"type": "array", "minItems": 2, "items": {"type": "string"}},
            "forbidden_objects": {"type": "array", "minItems": 1, "items": {"type": "string"}},
            "forbidden_geometry": {"type": "array", "items": {"type": "string"}},
            "forbidden_target_forms": {"type": "array", "minItems": 1, "items": {"type": "string"}},
            "forbidden_event_sequence": {"type": "string"},
        },
        "required": ["id", "deep_insight", "invariant_chain", "generic_equations",
                     "indispensable_decisions", "transfer_patterns", "forbidden_objects",
                     "forbidden_geometry", "forbidden_target_forms", "forbidden_event_sequence"],
    }
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"abstractions": {"type": "array", "items": item}},
              "required": ["abstractions"]}
    system = """Extract the transferable reasoning invariant from verified hard F=ma solutions.
Separate deep physics from surface form. The invariant chain must use generic roles and dependencies,
not the source's objects, numerical values, named shapes, or requested comparison. Record every
recognizable source feature as forbidden. Do not invent additional physics. Return strict JSON."""
    result = generate_json(args.model, "Abstract:\n" + json.dumps(payload, ensure_ascii=False),
                           provider=args.provider, system=system, reasoning_effort="medium",
                           max_output_tokens=8000, json_schema=schema)
    output = result.get("abstractions") or []
    if {r["id"] for r in output} != {r["id"] for r in selected}:
        raise RuntimeError("abstraction ids do not match selected seeds")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output))
    print(json.dumps({"abstractions": len(output), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
