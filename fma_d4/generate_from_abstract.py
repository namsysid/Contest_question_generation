#!/usr/bin/env python3
"""Generate a solved D4 candidate without exposing source wording to the constructor."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from circuit_lab.model_client import generate_json  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def forbidden_hits(question: str, abstraction: dict) -> list[str]:
    text = question.casefold()
    terms = [*(abstraction.get("forbidden_objects") or []), *(abstraction.get("forbidden_geometry") or [])]
    terms = [term for term in terms if not re.fullmatch(r"mass\s+[a-z].*", term.strip(), re.IGNORECASE)]
    return [term for term in terms if len(term.strip()) >= 4 and re.search(r"\b" + re.escape(term.casefold()) + r"\b", text)]


def local_rejections(result: dict, primary: dict) -> list[str]:
    errors = []
    question = str(result.get("question") or "")
    if len(question.split()) > 180:
        errors.append("stem exceeds 180 words")
    if any(len(str(choice).split()) > 14 for choice in (result.get("choices") or {}).values()):
        errors.append("an answer choice exceeds 14 words")
    hits = forbidden_hits(question, primary)
    if hits:
        errors.append("forbidden source features: " + ", ".join(hits))
    if len(result.get("conceptual_deductions") or []) < 4:
        errors.append("fewer than four deductions")
    return errors


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--abstractions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="gpt-5")
    p.add_argument("--provider", default="openai")
    p.add_argument("--attempt", type=int, default=1)
    p.add_argument("--feedback", default="")
    args = p.parse_args()
    abstractions = read_jsonl(args.abstractions)
    if not abstractions:
        raise RuntimeError("no abstractions")
    primary = abstractions[(args.attempt - 1) % len(abstractions)]
    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "id": {"type": "string"}, "question": {"type": "string"},
            "choices": {"type": "object", "additionalProperties": False,
                        "properties": {k: {"type": "string"} for k in "ABCDE"}, "required": list("ABCDE")},
            "answer": {"type": "string", "enum": list("ABCDE")},
            "solution": {"type": "string"}, "difficulty": {"type": "integer", "const": 4},
            "non_obvious_decision": {"type": "string"},
            "conceptual_deductions": {"type": "array", "minItems": 4, "items": {"type": "string"}},
            "causal_chain_audit": {"type": "array", "minItems": 4, "items": {"type": "string"}},
            "dimensional_check": {"type": "string"}, "limiting_case_check": {"type": "string"},
            "forbidden_feature_audit": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["id", "question", "choices", "answer", "solution", "difficulty",
                     "non_obvious_decision", "conceptual_deductions", "causal_chain_audit",
                     "dimensional_check", "limiting_case_check", "forbidden_feature_audit"],
    }
    system = """Construct one diagram-free hardest-quartile F=ma multiple-choice problem from an
abstract reasoning invariant. You do not know the source wording. Use standard pre-university
classical mechanics; do not require named graduate-level formulas. Difficulty must arise from a
non-obvious model/regime choice plus a necessary causal chain, not calculus length. Use a completely
different apparatus, geometry, observable, target form, and event sequence from every forbidden
feature. Merely replacing a named shape by another axial solid is a parallel form, not a transfer.
Solve exactly before choosing five compact answers of at most 14 words each. Prefer a numerical value,
dimensionless ratio, scaling law, ordering, or short expression; never use five full formulas.
Return strict JSON."""
    prompt = f"""ATTEMPT {args.attempt}. TRANSFERABLE INVARIANT:
{json.dumps(primary, ensure_ascii=False)}

Create a new physical realization whose student-facing setup would not suggest the source problem.
Every primitive given must be sufficient, and no derived quantity
may be leaked. The correct answer must follow from the written stem alone. Audit dimensions and at
least one limiting case. State how each forbidden category was avoided.

REVISION FEEDBACK FROM EARLIER ATTEMPTS:
{args.feedback or 'None; create a fresh realization.'}"""
    result = generate_json(args.model, prompt, provider=args.provider, system=system,
                           reasoning_effort="medium", max_output_tokens=14000, json_schema=schema)
    errors = local_rejections(result, primary)
    result["_meta"] = {"attempt": args.attempt, "abstraction_id": primary["id"],
                       "local_rejections": errors}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(args.out), "local_rejections": errors}, indent=2))


if __name__ == "__main__":
    main()
