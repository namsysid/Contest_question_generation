#!/usr/bin/env python3
"""Retrieve a primary verified D4 mechanism and diverse contrast mechanisms."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tag_set(row: dict[str, Any]) -> set[str]:
    return {str(tag).strip().casefold() for tag in row.get("mechanism_tags") or [] if str(tag).strip()}


def diversity_score(candidate: dict[str, Any], chosen: list[dict[str, Any]]) -> tuple[float, int]:
    tags = tag_set(candidate)
    max_overlap = max((len(tags & tag_set(row)) / max(1, len(tags | tag_set(row))) for row in chosen), default=0.0)
    return (1.0 - max_overlap, int(candidate.get("difficulty") or 0))


def cosine(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    denom = math.sqrt(sum(x * x for x in left) * sum(x * x for x in right))
    return sum(x * y for x, y in zip(left, right)) / denom if denom else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-bank", type=Path, required=True)
    parser.add_argument("--anchor-id", required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument(
        "--embedding-index", type=Path,
        help="Optional JSONL of verified solution-signature embeddings keyed by id",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.seed_bank)
    by_id = {str(row.get("id")): row for row in rows}
    if args.anchor_id not in by_id:
        raise RuntimeError(f"verified anchor not found: {args.anchor_id}")
    embeddings: dict[str, list[float]] = {}
    if args.embedding_index:
        embeddings = {
            str(row.get("id")): row.get("embedding") or []
            for row in read_jsonl(args.embedding_index)
        }
        missing = set(by_id) - set(embeddings)
        if missing:
            raise RuntimeError(f"solution-signature embeddings missing for {sorted(missing)}")
    chosen = [by_id[args.anchor_id]]
    remaining = [row for row in rows if str(row.get("id")) != args.anchor_id]
    while remaining and len(chosen) < max(1, args.count):
        def score(row: dict[str, Any]) -> tuple[float, int]:
            diversity, difficulty = diversity_score(row, chosen)
            relevance = cosine(
                embeddings.get(args.anchor_id), embeddings.get(str(row.get("id")))
            ) if embeddings else 0.5
            return (0.65 * relevance + 0.35 * diversity, difficulty)
        best = max(remaining, key=score)
        chosen.append(best)
        remaining.remove(best)
    if len(chosen) < 2:
        raise RuntimeError("D4 synthesis needs at least two verified, distinct mechanisms")
    bundle = {
        "bundle_id": f"d4_{args.anchor_id}",
        "primary_seed": chosen[0],
        "contrast_seeds": chosen[1:],
        "retrieval_policy": "verified primary plus mechanism-diverse contrasts",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote one D4 bundle with {len(chosen)} verified mechanisms -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
