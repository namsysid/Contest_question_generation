#!/usr/bin/env python3
"""Generate one diagram-free F=ma candidate from one surface-free abstraction."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from circuit_lab.model_client import generate_json  # noqa: E402


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def forbidden_hits(question: str, abstraction: dict) -> list[str]:
    text = question.casefold()
    terms = [*(abstraction.get("forbidden_objects") or []),
             *(abstraction.get("forbidden_geometry") or [])]
    terms = [t.strip() for t in terms if len(t.strip()) >= 4]
    return [t for t in terms if re.search(r"\b" + re.escape(t.casefold()) + r"\b", text)]


def local_rejections(candidate: dict, abstraction: dict) -> list[str]:
    errors: list[str] = []
    question = str(candidate.get("question") or "")
    if len(question.split()) > 180:
        errors.append("stem exceeds 180 words")
    if any(len(str(c).split()) > 14 for c in (candidate.get("choices") or {}).values()):
        errors.append("an answer choice exceeds 14 words")
    hits = forbidden_hits(question, abstraction)
    if hits:
        errors.append("forbidden source features: " + ", ".join(hits))
    return errors


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--abstractions", type=Path, required=True)
    p.add_argument("--seed-bank", type=Path, required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--provider", default="openai")
    p.add_argument("--reasoning-effort", default="medium")
    args = p.parse_args()

    abstraction = next(r for r in rows(args.abstractions) if r["id"] == args.id)
    seed = next(r for r in rows(args.seed_bank) if r["id"] == args.id)
    source_difficulty = int(seed.get("difficulty") or 2)
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "id": {"type": "string"}, "question": {"type": "string"},
            "choices": {"type": "object", "additionalProperties": False,
                        "properties": {k: {"type": "string"} for k in "ABCDE"},
                        "required": list("ABCDE")},
            "answer": {"type": "string", "enum": list("ABCDE")},
            "solution": {"type": "string"},
            "estimated_difficulty": {"type": "integer", "minimum": 1, "maximum": 4},
            "conceptual_steps": {"type": "array", "items": {"type": "string"}},
            "dimensional_check": {"type": "string"},
            "limiting_case_check": {"type": "string"},
            "anti_copy_audit": {"type": "string"},
        },
        "required": ["id", "question", "choices", "answer", "solution",
                     "estimated_difficulty", "conceptual_steps", "dimensional_check",
                     "limiting_case_check", "anti_copy_audit"],
    }
    system = """You create reliable, diagram-free F=ma multiple-choice practice problems.
Use the supplied abstract physics mechanism, but invent a genuinely different physical realization.
Match the source mechanism's natural difficulty: do not simplify away its central insight and do not
add gratuitous complexity merely to raise difficulty. Use only standard high-school contest mechanics.
The student-facing stem must be self-contained and need no diagram. Avoid every forbidden apparatus,
geometry, event sequence, and target form. Solve the problem completely before writing five plausible,
compact choices. Return strict JSON."""
    prompt = f"""SOURCE DIFFICULTY LABEL: D{source_difficulty}
ABSTRACT MECHANISM (contains no source wording):
{json.dumps(abstraction, ensure_ascii=False)}

Create exactly one new question. Preserve the real conceptual dependency of the mechanism while
changing its surface experience. The answer and solution must follow from the stem alone."""
    candidate = generate_json(
        args.model, prompt, provider=args.provider, system=system,
        reasoning_effort=args.reasoning_effort, max_output_tokens=8000, json_schema=schema,
    )
    candidate["_meta"] = {
        "source_abstraction_id": args.id,
        "source_difficulty": source_difficulty,
        "constructor_model": args.model,
        "local_rejections": local_rejections(candidate, abstraction),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(candidate, ensure_ascii=False) + "\n")
    print(json.dumps({"id": args.id, "out": str(args.out),
                      "local_rejections": candidate["_meta"]["local_rejections"]}, indent=2))


if __name__ == "__main__":
    main()

