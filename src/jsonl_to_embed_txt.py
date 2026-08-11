#!/usr/bin/env python3
"""
skeleton_to_embedding_text.py

Convert solution-skeleton JSONL files into canonical,
embedding-friendly text representations.

Usage:
  python skeleton_to_embedding_text.py input.jsonl output.jsonl
"""

import json
import sys
from pathlib import Path


OP_WHITELIST = {
    "ESTABLISH_FRAMEWORK",
    "IDENTIFY_RELATION",
    "APPLY_RELATION",
    "SAVE_RESULT",
    "APPLY_RESULT",
}


def normalize_text(s: str) -> str:
    """Normalize whitespace and punctuation for embedding stability."""
    if not s:
        return ""
    return " ".join(s.strip().split())


def serialize_step(step: dict) -> str:
    op = step.get("op", "").strip()
    if op not in OP_WHITELIST:
        return None

    # Prefer semantic fields over raw math / numbers
    note = normalize_text(step.get("note", ""))

    out = step.get("out", {})
    relation = out.get("relation_name")
    result = out.get("result_name")

    if op == "IDENTIFY_RELATION" and relation:
        return f"{op}: {relation}"

    if op == "SAVE_RESULT" and result:
        return f"{op}: {result}"

    if note:
        return f"{op}: {note}"

    return op


def skeleton_to_embedding_text(record: dict) -> str:
    lines = []
    pid = record.get("id", "unknown")

    lines.append(f"PROBLEM_ID: {pid}")

    for step in record.get("skeleton", []):
        line = serialize_step(step)
        if line:
            lines.append(line)

    return "\n".join(lines)


def main():
    if len(sys.argv) != 3:
        print("Usage: python skeleton_to_embedding_text.py input.jsonl output.jsonl")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    with input_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue

            record = json.loads(line)
            embedding_text = skeleton_to_embedding_text(record)

            fout.write(
                json.dumps(
                    {
                        "id": record.get("id"),
                        "embedding_text": embedding_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
