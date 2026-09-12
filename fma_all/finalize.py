#!/usr/bin/env python3
"""Combine deterministic, blind-correctness, and novelty results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--verified", type=Path, required=True)
    p.add_argument("--novelty", type=Path, required=True)
    p.add_argument("--accepted", type=Path, required=True)
    p.add_argument("--rejected", type=Path, required=True)
    args = p.parse_args()
    novelty = {r["id"]: r for r in rows(args.novelty)}
    accepted, rejected = [], []
    for candidate in rows(args.verified):
        n = novelty.get(candidate["id"], {"novelty_pass": False,
                                          "explanation": "missing novelty verdict"})
        local = not (candidate.get("_meta") or {}).get("local_rejections")
        passed = bool(candidate.get("correctness_pass") and n.get("novelty_pass") and local)
        result = {**candidate, "novelty_verdict": n,
                  "status": "ACCEPTED" if passed else "REJECTED"}
        (accepted if passed else rejected).append(result)
    args.accepted.parent.mkdir(parents=True, exist_ok=True)
    args.accepted.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in accepted))
    args.rejected.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected))
    print(json.dumps({"accepted": len(accepted), "rejected": len(rejected)}, indent=2))


if __name__ == "__main__":
    main()
