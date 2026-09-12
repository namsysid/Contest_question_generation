#!/usr/bin/env python3
"""Generate D4 candidates from verified solution mechanisms, not raw problem graphs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from circuit_lab.model_client import generate_json  # noqa: E402


SYSTEM = """You create one hardest-quartile F=ma multiple-choice problem from verified solution
mechanisms. Preserve the primary source's genuinely hard causal dependency chain; never rename or
cosmetically reskin it. Combine a contrast mechanism only when you can name a shared physical state
through which the two laws causally interact. Never bolt unrelated mechanisms together.
First solve the new problem privately, then write a concise diagram-free stem and five short choices.
Difficulty must come from an indispensable model/regime decision and at least four dependent conceptual
deductions, not excessive algebra, obscure formalism, or extra requested outputs. Use only standard
F=ma classical mechanics. Treat velocity, force, impulse, and angular momentum as vectors when their
directions differ. Do not assume a recurrence, return condition, collision geometry, or active regime
that should instead be derived from primitive givens. Return strict JSON only."""

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "id": {"type": "string"},
        "question": {"type": "string", "maxLength": 1800},
        "choices": {
            "type": "object", "additionalProperties": False,
            "properties": {key: {"type": "string", "maxLength": 180} for key in "ABCDE"},
            "required": list("ABCDE"),
        },
        "answer": {"type": "string", "enum": list("ABCDE")},
        "solution": {"type": "string", "maxLength": 5000},
        "difficulty": {"type": "integer", "const": 4},
        "topic": {"type": "string"},
        "mechanism_mapping": {"type": "array", "minItems": 2, "items": {"type": "string"}},
        "non_obvious_decision": {"type": "string"},
        "conceptual_deductions": {"type": "array", "minItems": 4, "items": {"type": "string"}},
        "novelty_explanation": {"type": "string"},
        "causal_interface": {"type": "string"},
        "vector_audit": {"type": "string"},
        "assumption_audit": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "id", "question", "choices", "answer", "solution", "difficulty", "topic",
        "mechanism_mapping", "non_obvious_decision", "conceptual_deductions",
        "novelty_explanation",
        "causal_interface", "vector_audit", "assumption_audit",
    ],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def words(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value).casefold()))


def overlap(candidate: str, sources: list[str]) -> float:
    candidate_words = words(candidate)
    return max((len(candidate_words & words(source)) / max(1, len(candidate_words | words(source))) for source in sources), default=0.0)


def prompt(bundle: dict[str, Any]) -> str:
    seeds = [bundle["primary_seed"], *(bundle.get("contrast_seeds") or [])]
    mechanisms = [{
        "id": seed["id"],
        "verified_solution": seed["solution_text"],
        "solution_signature": seed["solution_signature"],
        "difficulty": seed["difficulty"],
    } for seed in seeds]
    return f"""VERIFIED HARD-SOURCE MECHANISMS:
{json.dumps(mechanisms, ensure_ascii=False)}

Create one new D4 problem whose shortest solution preserves the primary record's hard dependency chain.
Use another record only if its mechanism has a physically necessary causal interface with that chain. Change the
physical setup, target relation, variables, and numerical structure. Do not reproduce a source's event
sequence. The stem must be at most 180 words, contain no diagram dependency, and ask for one result.
Every supplied quantity must be primitive; do not leak a force, speed, mode, load, impulse, or transition
state that the student is supposed to derive. In causal_interface, identify the shared state connecting
mechanisms, or state that the contrast was used only for traps/verification. Explicitly audit vector
directions and every simplifying assumption. Return one solved candidate."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--provider", choices=("openai", "ollama"), default="openai")
    args = parser.parse_args()
    bundles = read_jsonl(args.bundles)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for bundle in bundles:
        candidate = generate_json(
            args.model, prompt(bundle), provider=args.provider, system=SYSTEM,
            temperature=0.2, max_output_tokens=10000, reasoning_effort="medium",
            json_schema=SCHEMA,
        )
        seeds = [bundle["primary_seed"], *(bundle.get("contrast_seeds") or [])]
        source_questions = [str(seed.get("question_text") or "") for seed in seeds]
        errors = []
        if len(str(candidate.get("question") or "").split()) > 180:
            errors.append("stem exceeds 180 words")
        if overlap(str(candidate.get("question") or ""), source_questions) >= 0.55:
            errors.append("surface overlap with a source is too high")
        if len(candidate.get("conceptual_deductions") or []) < 4:
            errors.append("fewer than four claimed deductions")
        candidate["_meta"] = {
            "bundle_id": bundle.get("bundle_id"), "model": args.model,
            "requires_independent_verification": True,
        }
        if errors:
            rejected.append({**candidate, "_rejection_reasons": errors})
        else:
            accepted.append(candidate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    rejected_path = args.out.with_suffix(args.out.suffix + ".rejected")
    with rejected_path.open("w", encoding="utf-8") as handle:
        for row in rejected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Generated {len(accepted)} candidates; locally rejected {len(rejected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
