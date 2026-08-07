#!/usr/bin/env python3
"""Convert a plain-text worksheet/problem bank into pipeline input JSONL.

The all-in-one pipeline expects JSONL rows with `id`, `domain`, and
`question_text`. This script splits plain text on numbered problem markers such
as `1)`, `1.`, including markers that appear mid-line.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


QUESTION_MARKER_RE = re.compile(r"(?m)^\s*(\d{1,4})[\).]\s+")
SECTION_HEADING_RE = re.compile(
    r"^(?:write|determine|use|find|solve|simplify|evaluate|choose|select|complete|circle|"
    r"answer|show|calculate|compute)\b",
    re.IGNORECASE,
)


def clean_block(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_questions(text: str, *, domain: str, id_prefix: str, include_context: bool) -> List[Dict[str, object]]:
    text = clean_block(text)
    matches = list(QUESTION_MARKER_RE.finditer(text))
    if not matches:
        return [
            {
                "id": f"{id_prefix}_001",
                "domain": domain,
                "question_number": 1,
                "question_text": text,
            }
        ] if text else []

    rows: List[Dict[str, object]] = []
    context = clean_block(text[: matches[0].start()])

    for idx, match in enumerate(matches):
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        qnum = int(match.group(1))
        body = clean_block(text[match.end() : next_start])
        if not body:
            continue

        lines = [line.strip() for line in body.splitlines() if line.strip()]
        if len(lines) == 1 and SECTION_HEADING_RE.match(lines[0]) and idx + 1 < len(matches):
            context = lines[0]
            continue

        next_context = ""
        while lines and SECTION_HEADING_RE.match(lines[-1]):
            next_context = lines.pop()
        if next_context:
            body = clean_block("\n".join(lines))
            if not body:
                context = next_context
                continue

        question_text = f"{qnum}) {body}".strip()
        if include_context and context:
            question_text = f"{context}\n\n{question_text}"

        rows.append(
            {
                "id": f"{id_prefix}_{len(rows) + 1:03d}",
                "domain": domain,
                "question_number": qnum,
                "question_text": question_text,
            }
        )

        if next_context:
            context = next_context

    return rows


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert numbered plain-text problems to JSONL.")
    ap.add_argument("--input", required=True, help="Input .txt file")
    ap.add_argument("--out", required=True, help="Output .jsonl file")
    ap.add_argument("--domain", default="math")
    ap.add_argument("--id-prefix", default="", help="Default: derived from --domain")
    ap.add_argument(
        "--include-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prepend section heading text to each question when available.",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)
    id_prefix = args.id_prefix or args.domain
    rows = split_questions(
        in_path.read_text(encoding="utf-8"),
        domain=args.domain,
        id_prefix=id_prefix,
        include_context=args.include_context,
    )
    write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
