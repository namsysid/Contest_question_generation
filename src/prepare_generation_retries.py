#!/usr/bin/env python3
"""Create aligned stage-05 retry inputs for bundles without a gatekeeper PASS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic-dir", required=True)
    parser.add_argument("--accepted", action="append", default=[])
    parser.add_argument("--bundles-out", required=True)
    parser.add_argument("--skeletons-out", required=True)
    args = parser.parse_args()

    topic_dir = Path(args.topic_dir)
    accepted_paths = [topic_dir / "generated_problems.jsonl", *(Path(p) for p in args.accepted)]
    passed = {
        row.get("_meta", {}).get("bundle_id")
        for path in accepted_paths
        for row in read_jsonl(path)
        if row.get("_gatekeeper", {}).get("verdict") == "PASS"
    }
    bundles = read_jsonl(topic_dir / "retrieval_bundles.jsonl")
    skeletons = read_jsonl(topic_dir / "generated_skeletons.jsonl")
    retry_ids = {row.get("bundle_id") for row in bundles} - passed
    write_jsonl(Path(args.bundles_out), [row for row in bundles if row.get("bundle_id") in retry_ids])
    write_jsonl(Path(args.skeletons_out), [row for row in skeletons if row.get("bundle_id") in retry_ids])
    print(f"retry={len(retry_ids)} passed={len(passed)}")


if __name__ == "__main__":
    main()
