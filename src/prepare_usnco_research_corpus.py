#!/usr/bin/env python3
"""Convert complete official USNCO MCQs into stage-01 research-pipeline input."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from usnco_generation import USNCO_TOPIC_BY_KEY, infer_usnco_topic


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(row: dict[str, Any]) -> dict[str, Any] | None:
    content = row.get("content") if isinstance(row.get("content"), dict) else {}
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    question = str(row.get("question") or content.get("question_text") or "").strip()
    choices = row.get("choices") or content.get("choices") or {}
    if not question or not isinstance(choices, dict):
        return None
    if re.search(r"\b(?:figure|diagram|shown)\b", question, flags=re.IGNORECASE):
        return None
    choices = {letter: str(choices.get(letter) or "").strip() for letter in "ABCD"}
    if any(not choices[letter] for letter in "ABCD"):
        return None
    number = int(source.get("question_number") or 0)
    topic_key = str(row.get("topic_key") or infer_usnco_topic(number, question))
    if topic_key not in USNCO_TOPIC_BY_KEY:
        return None
    answer_obj = row.get("answer")
    answer = str(answer_obj.get("letter") if isinstance(answer_obj, dict) else answer_obj or "").upper()
    choice_text = "\n".join(f"({letter}) {choices[letter]}" for letter in "ABCD")
    provenance = source.get("url") or f"USNCO {source.get('year', '')} {source.get('exam_type', '')}".strip()
    return {
        "id": str(row.get("id") or ""),
        "domain": "usnco",
        "question_text": f"{question}\n{choice_text}",
        "answer_key": answer if len(answer) == 1 and answer in "ABCD" else None,
        "source_pdf": provenance,
        "has_diagram": False,
        "diagram_files": [],
        "topic_key": topic_key,
        "topic": USNCO_TOPIC_BY_KEY[topic_key][0],
        "source": source,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--topic-key", choices=USNCO_TOPIC_BY_KEY)
    parser.add_argument("--years", nargs="+", type=int)
    args = parser.parse_args()
    rows = [normalized for row in read_jsonl(args.input) if (normalized := normalize(row))]
    if args.years:
        selected_years = set(args.years)
        rows = [row for row in rows if (row.get("source") or {}).get("year") in selected_years]
    if args.topic_key:
        rows = [row for row in rows if row["topic_key"] == args.topic_key]
    if not rows:
        raise RuntimeError("No complete USNCO source questions matched the requested topic")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Prepared {len(rows)} official USNCO research rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
