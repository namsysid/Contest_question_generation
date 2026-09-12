#!/usr/bin/env python3
"""Batch-check candidate surface novelty against the verified source questions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from circuit_lab.model_client import generate_json  # noqa: E402


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--source-bank", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--provider", default="openai")
    args = p.parse_args()
    candidates = [{"id": r["id"], "question": r["question"], "choices": r["choices"]}
                  for r in rows(args.candidates)]
    sources = [{"id": r["id"], "question": r["question_text"]} for r in rows(args.source_bank)]
    system = """Act as a contest parallel-form editor. Shared physics is allowed and expected.
Flag a candidate only when a student would recognize the same surface problem: the same apparatus,
geometry, event sequence, or target form. Renaming variables is not novelty. Return strict JSON."""
    prompt = ("CANDIDATES:\n" + json.dumps(candidates, ensure_ascii=False)
              + "\nSOURCES:\n" + json.dumps(sources, ensure_ascii=False)
              + """\nReturn {"assessments":[{"id":"...","closest_source_id":"...|NONE",
"same_apparatus":false,"same_geometry":false,"same_event_sequence":false,
"same_target_form":false,"explanation":"..."}]}""")
    result = generate_json(args.model, prompt, provider=args.provider, system=system,
                           reasoning_effort="low", max_output_tokens=6000)
    output = []
    for verdict in result.get("assessments", []):
        keys = ("same_apparatus", "same_geometry", "same_event_sequence", "same_target_form")
        verdict["novelty_pass"] = not any(verdict.get(k) is True for k in keys)
        output.append(verdict)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output))
    print(json.dumps({"checked": len(output),
                      "novelty_pass": sum(bool(r["novelty_pass"]) for r in output)}, indent=2))


if __name__ == "__main__":
    main()

