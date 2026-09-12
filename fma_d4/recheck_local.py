#!/usr/bin/env python3
"""Re-run deterministic local gates after gate logic changes; no model call."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from generate_from_abstract import local_rejections


def first(path: Path) -> dict:
    return json.loads(next(line for line in path.read_text().splitlines() if line.strip()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--abstractions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    candidate = first(args.candidate)
    abstractions = {r["id"]: r for r in
                    (json.loads(line) for line in args.abstractions.read_text().splitlines() if line.strip())}
    abstraction = abstractions[(candidate.get("_meta") or {})["abstraction_id"]]
    candidate.setdefault("_meta", {})["local_rejections"] = local_rejections(candidate, abstraction)
    args.out.write_text(json.dumps(candidate, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(args.out), "local_rejections": candidate["_meta"]["local_rejections"]}, indent=2))


if __name__ == "__main__":
    main()
