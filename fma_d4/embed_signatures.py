#!/usr/bin/env python3
"""Embed verified, readable solution signatures for D4 retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from circuit_lab.model_client import embed_texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.seed_bank.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit("Seed bank is empty; refusing to create a meaningless index.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            vectors = embed_texts(
                args.model,
                [row["solution_signature"] for row in batch],
                provider=args.provider,
            )
            for row, vector in zip(batch, vectors, strict=True):
                handle.write(json.dumps({
                    "id": row["id"],
                    "embedding_model": args.model,
                    "embedding": vector,
                }) + "\n")

    print(json.dumps({"embedded": len(rows), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
