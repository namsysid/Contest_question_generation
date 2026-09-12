#!/usr/bin/env python3
"""Compare generated candidates against actual source surfaces."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from circuit_lab.model_client import generate_json  # noqa: E402


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def is_surface_parallel(verdict: dict) -> bool:
    concrete = ("same_apparatus", "same_geometry", "same_event_sequence", "same_target_form")
    return any(verdict.get(key) is True for key in concrete)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--source-bank", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--provider", default="openai")
    args = p.parse_args()
    candidate = rows(args.candidate)[0]
    sources = [{"id": r["id"], "question": r["question_text"]} for r in rows(args.source_bank)]
    system = """Act as a contest parallel-form editor. Compare surface experience, not shared
physics vocabulary. Reject when a student familiar with a source would recognize the same apparatus,
geometry, event sequence, target transformation, or choice pattern despite renamed symbols.
Return JSON only."""
    prompt = "CANDIDATE:\n" + json.dumps({"id": candidate["id"], "question": candidate["question"], "choices": candidate["choices"]}) + "\nSOURCES:\n" + json.dumps(sources)
    prompt += '\nReturn {"parallel":true,"closest_source_id":"...|NONE","same_apparatus":false,"same_geometry":false,"same_event_sequence":false,"same_target_form":false,"explanation":"..."}'
    verdict = generate_json(args.model, prompt, provider=args.provider, system=system,
                            reasoning_effort="low", max_output_tokens=3000)
    verdict["surface_parallel"] = is_surface_parallel(verdict)
    verdict["novelty_pass"] = not verdict["surface_parallel"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(verdict, ensure_ascii=False) + "\n")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
