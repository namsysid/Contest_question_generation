#!/usr/bin/env python3
"""Export generated problem JSONL to readable text and HTML files."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def choices_lines(choices: Any) -> List[str]:
    if isinstance(choices, dict):
        return [f"  {key}. {choices[key]}" for key in ("A", "B", "C", "D", "E") if key in choices]
    if isinstance(choices, list):
        labels = ["A", "B", "C", "D", "E"]
        return [f"  {labels[i]}. {value}" for i, value in enumerate(choices[:5])]
    return []


def gatekeeper_status(row: Dict[str, Any]) -> str:
    gate = row.get("_gatekeeper")
    if isinstance(gate, dict):
        verdict = str(gate.get("verdict") or "").strip()
        if verdict:
            return verdict
    return ""


def iter_selected(rows: Iterable[Dict[str, Any]], include_failed: bool) -> Iterable[Dict[str, Any]]:
    for row in rows:
        if include_failed or gatekeeper_status(row) != "FAIL":
            yield row


def write_text(path: Path, rows: List[Dict[str, Any]], *, include_answers: bool) -> None:
    lines: List[str] = ["Generated Math Questions", "=" * 24, ""]
    answer_lines: List[str] = []

    for idx, row in enumerate(rows, start=1):
        verdict = gatekeeper_status(row)
        suffix = f" [{verdict}]" if verdict else ""
        lines.append(f"{idx}. {row.get('question', '').strip()}{suffix}")
        choice_lines = choices_lines(row.get("choices"))
        for choice in choice_lines:
            lines.append(choice)
        lines.append("")
        answer = str(row.get("answer") or "").strip()
        if answer:
            answer_lines.append(f"{idx}. {answer}")

    if include_answers and answer_lines:
        lines.extend(["", "Answer Key", "=" * 10, *answer_lines, ""])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_html(path: Path, rows: List[Dict[str, Any]], *, include_answers: bool) -> None:
    parts: List[str] = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Generated Math Questions</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;line-height:1.45;max-width:820px;margin:32px auto;padding:0 20px;color:#111}",
        "h1{font-size:24px;margin-bottom:24px}",
        ".q{break-inside:avoid;margin:0 0 24px 0}",
        ".stem{font-weight:600;margin-bottom:10px}",
        ".choices{margin:0;padding-left:24px}",
        ".choices li{margin:4px 0}",
        ".fail{color:#8a1f11;font-size:12px;font-weight:700;margin-left:6px}",
        ".answers{margin-top:36px;border-top:1px solid #bbb;padding-top:16px}",
        "@media print{body{margin:18mm auto}.q{page-break-inside:avoid}}",
        "</style></head><body>",
        "<h1>Generated Math Questions</h1>",
    ]

    answer_parts: List[str] = []
    for idx, row in enumerate(rows, start=1):
        verdict = gatekeeper_status(row)
        verdict_html = f" <span class='fail'>[{html.escape(verdict)}]</span>" if verdict else ""
        parts.append("<section class='q'>")
        parts.append(f"<div class='stem'>{idx}. {html.escape(str(row.get('question') or '').strip())}{verdict_html}</div>")
        choice_rows = choices_lines(row.get("choices"))
        if choice_rows:
            parts.append("<ol class='choices' type='A'>")
        choices = row.get("choices")
        if isinstance(choices, dict):
            for key in ("A", "B", "C", "D", "E"):
                if key in choices:
                    parts.append(f"<li>{html.escape(str(choices[key]))}</li>")
        elif isinstance(choices, list):
            for choice in choices[:5]:
                parts.append(f"<li>{html.escape(str(choice))}</li>")
        if choice_rows:
            parts.append("</ol>")
        parts.append("</section>")
        answer = str(row.get("answer") or "").strip()
        if answer:
            answer_parts.append(f"<li>{idx}. {html.escape(answer)}</li>")

    if include_answers and answer_parts:
        parts.append("<section class='answers'><h2>Answer Key</h2><ol>")
        parts.extend(answer_parts)
        parts.append("</ol></section>")

    parts.append("</body></html>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export generated questions to readable formats.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-txt", required=True)
    ap.add_argument("--out-html", default="")
    ap.add_argument("--include-answers", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include-failed", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    rows = list(iter_selected(read_jsonl(Path(args.input)), include_failed=args.include_failed))
    write_text(Path(args.out_txt), rows, include_answers=args.include_answers)
    if args.out_html:
        write_html(Path(args.out_html), rows, include_answers=args.include_answers)
    print(f"Wrote {len(rows)} questions -> {args.out_txt}")
    if args.out_html:
        print(f"Wrote HTML -> {args.out_html}")


if __name__ == "__main__":
    main()
