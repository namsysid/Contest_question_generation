#!/usr/bin/env python3
"""Select JSONL records by id without modifying their contents."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--infile", type=Path, required=True)
p.add_argument("--ids", nargs="+", required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
wanted = set(a.ids)
rows = [json.loads(line) for line in a.infile.read_text().splitlines() if line.strip()]
selected = [r for r in rows if r.get("id") in wanted]
if {r.get("id") for r in selected} != wanted:
    raise RuntimeError("one or more selected ids are missing")
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
print(json.dumps({"selected": len(selected), "out": str(a.out)}, indent=2))
