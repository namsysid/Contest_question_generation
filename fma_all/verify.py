#!/usr/bin/env python3
"""Blind-solve general F=ma candidates; difficulty is metadata, not a gate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from circuit_lab.model_client import generate_json  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--provider", default="openai")
    p.add_argument("--reasoning-effort", default="high")
    args = p.parse_args()
    candidates = [json.loads(line) for line in args.candidates.read_text().splitlines() if line.strip()]
    compact = [{"id": r["id"], "question": r["question"], "choices": r["choices"]}
               for r in candidates]
    system = """You are an independent F=ma contest editor. You are not shown the constructor's
answer or solution. Solve every question from scratch. Check sufficiency, physical consistency,
uniqueness, and answer choices. Rate shortest-path difficulty from D1 (routine) through D4
(hardest quartile). Do not reject a valid problem because it is below D4. Return strict JSON."""
    prompt = "Independently solve and assess:\n" + json.dumps(compact, ensure_ascii=False) + """

Return {"assessments":[{"id":"...","selected_answer":"A|B|C|D|E|UNKNOWN",
"independent_solution":"...","physics_valid":true,"unique_answer":true,
"difficulty":1,"issues":["..."]}]}"""
    result = generate_json(args.model, prompt, provider=args.provider, system=system,
                           reasoning_effort=args.reasoning_effort, max_output_tokens=12000)
    by_id = {r["id"]: r for r in result.get("assessments", [])}
    output = []
    for candidate in candidates:
        verdict = by_id.get(candidate["id"], {"id": candidate["id"],
                                              "issues": ["missing verifier verdict"]})
        passed = (verdict.get("selected_answer") == candidate.get("answer")
                  and verdict.get("physics_valid") is True
                  and verdict.get("unique_answer") is True)
        output.append({**candidate, "blind_verdict": verdict, "correctness_pass": passed})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output))
    print(json.dumps({"checked": len(output),
                      "correctness_pass": sum(bool(r["correctness_pass"]) for r in output)}, indent=2))


if __name__ == "__main__":
    main()

