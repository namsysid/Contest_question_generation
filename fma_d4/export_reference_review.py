#!/usr/bin/env python3
"""Export the verified reference bank as a readable Markdown review file."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--bank", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
rows = [json.loads(line) for line in a.bank.read_text().splitlines() if line.strip()]
parts = [
    "# F=ma 2024-2026 verified source-solution reference bank\n",
    "These records are retained as readable generator context and are not re-embedded. "
    "`blind_answer_agreement` means GPT-5 generated the derivation and GPT-5-mini independently "
    "selected the same answer from the question alone. `manual_equation_review` records an explicit "
    "equation audit after verifier service failures.\n",
]
for row in rows:
    verdict = row.get("independent_verdict") or {}
    parts.extend([
        f"## {row['id']}\n",
        f"- Answer: **{row['derived_answer']}**\n",
        f"- Difficulty: **D{row.get('difficulty', '?')}**\n",
        f"- Verification: `{row.get('verification_status')}`\n",
        "### Source question\n",
        str(row.get("question_text") or row.get("question", {}).get("stem") or "") + "\n",
        "### Structured solution\n",
        str(row.get("solution_text") or "") + "\n",
        "### Indispensable steps\n",
        "\n".join(f"{i}. {step}" for i, step in enumerate(row.get("shortest_solution_steps") or [], 1)) + "\n",
        "### Independent check\n",
        str(verdict.get("independent_solution") or "") + "\n",
    ])
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text("\n".join(parts))
print(json.dumps({"records": len(rows), "out": str(a.out)}, indent=2))
