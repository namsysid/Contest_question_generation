#!/usr/bin/env python3
"""Merge provenance-bearing verified solution records without relabeling them."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--inputs", nargs="+", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
seen: set[str] = set()
output = []
for path in a.inputs:
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in seen:
            raise RuntimeError(f"duplicate id: {row['id']}")
        if not row.get("verification_status"):
            raise RuntimeError(f"missing verification provenance: {row['id']}")
        if not row.get("solution_signature"):
            raise RuntimeError(f"missing structured solution signature: {row['id']}")
        seen.add(row["id"])
        output.append(row)
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output))
print(json.dumps({"reference_records": len(output), "out": str(a.out)}, indent=2))
