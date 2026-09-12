#!/usr/bin/env python3
"""Promote a candidate only after local, blind-solve, and surface-novelty gates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def one(path: Path) -> dict:
    return json.loads(next(line for line in path.read_text().splitlines() if line.strip()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--blind-verdict-row", type=Path, required=True)
    p.add_argument("--novelty", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    candidate, checked, novelty = one(args.candidate), one(args.blind_verdict_row), one(args.novelty)
    verdict = checked.get("independent_verdict") or {}
    correctness = (
        verdict.get("selected_answer") == candidate.get("answer")
        and verdict.get("physics_valid") is True
        and verdict.get("unique_answer") is True
        and int(verdict.get("difficulty") or 0) >= 4
        and int(verdict.get("conceptual_deductions") or 0) >= 4
        and int(verdict.get("non_obvious_decisions") or 0) >= 1
    )
    concrete = ("same_apparatus", "same_geometry", "same_event_sequence", "same_target_form")
    surface_novel = not any(novelty.get(key) is True for key in concrete)
    local = not (candidate.get("_meta") or {}).get("local_rejections")
    if not (correctness and surface_novel and local):
        raise RuntimeError(
            f"promotion failed: correctness={correctness}, "
            f"surface_novel={surface_novel}, local={local}"
        )
    output = {**candidate, "blind_verdict": verdict, "novelty_verdict": {
        **novelty, "surface_parallel": False, "novelty_pass": True,
    }, "status": "D4_ACCEPTED"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False) + "\n")
    print(json.dumps({"status": output["status"], "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
