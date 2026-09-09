#!/usr/bin/env python3
"""Filter the raw corpus to text-complete questions from official AAPT F=ma PDFs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

OFFICIAL_RE = re.compile(r"^\d{4}_fnet_ma_exam(?:_[AB])?\.pdf$", re.IGNORECASE)
CHOICE_MARKER_RE = re.compile(r"\(\s*([A-E])\s*\)")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def usable(row: dict[str, Any], pdf_dir: Path, years: set[int] | None = None) -> bool:
    source_pdf = Path(str(row.get("source_pdf") or "")).name
    year_match = re.match(r"^(\d{4})_", source_pdf)
    question_text = str(row.get("question_text") or "").strip()
    choice_order = CHOICE_MARKER_RE.findall(question_text)
    return bool(
        OFFICIAL_RE.fullmatch(source_pdf)
        and (not years or (year_match and int(year_match.group(1)) in years))
        and (pdf_dir / source_pdf).is_file()
        and question_text
        and choice_order == list("ABCDE")
        and not row.get("has_diagram")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/txts/all_questions.jsonl"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/raw_pdfs"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--years", nargs="+", type=int)
    args = parser.parse_args()
    selected_years = set(args.years or [])
    rows = [row for row in read_jsonl(args.input) if usable(row, args.pdf_dir, selected_years)]
    if not rows:
        raise RuntimeError("No complete text-only questions matched official AAPT F=ma PDFs")
    ids = [str(row.get("id") or "") for row in rows]
    if len(set(ids)) != len(ids) or any(not item_id for item_id in ids):
        raise RuntimeError("Official F=ma source corpus contains missing or duplicate IDs")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Prepared {len(rows)} official text-only F=ma research rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
