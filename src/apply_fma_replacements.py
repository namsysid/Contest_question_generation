#!/usr/bin/env python3
"""Replace audited F=ma records from category-specific generated shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def apply_replacements(
    rows: list[dict[str, Any]], manifest: dict[str, list[str]], replacement_dir: Path
) -> list[dict[str, Any]]:
    replacements: dict[str, dict[str, Any]] = {}
    for topic_key, target_ids in manifest.items():
        candidates = read_jsonl(replacement_dir / f"{topic_key}.jsonl")
        if len(candidates) != len(target_ids):
            raise ValueError(f"{topic_key}: expected {len(target_ids)} replacements, found {len(candidates)}")
        for target_id, candidate in zip(target_ids, candidates):
            candidate_topic = str((candidate.get("_meta") or {}).get("topic_key") or "")
            if candidate_topic != topic_key:
                raise ValueError(f"replacement for {target_id} has topic {candidate_topic!r}")
            replacements[target_id] = candidate

    existing_ids = {str(row.get("id") or "") for row in rows}
    missing = sorted(set(replacements) - existing_ids)
    if missing:
        raise ValueError("replacement targets not present: " + ", ".join(missing))

    output: list[dict[str, Any]] = []
    for row in rows:
        target_id = str(row.get("id") or "")
        candidate = replacements.get(target_id)
        if candidate is None:
            output.append(row)
            continue
        old_meta = row.get("_meta") or {}
        replacement_meta = candidate.get("_meta") or {}
        candidate["id"] = target_id
        candidate["_meta"] = {
            **replacement_meta,
            "record_index": old_meta.get("record_index"),
            "replaces_audited_item": target_id,
            "replacement_model": replacement_meta.get("model"),
        }
        output.append(candidate)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--replacement-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replacement-out", type=Path,
                        help="Also write only replacement records with their final target IDs")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output = apply_replacements(read_jsonl(args.input), manifest, args.replacement_dir)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.replacement_out:
        target_ids = {target for targets in manifest.values() for target in targets}
        with args.replacement_out.open("w", encoding="utf-8") as handle:
            for row in output:
                if row.get("id") in target_ids:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Applied {sum(map(len, manifest.values()))} replacements -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
