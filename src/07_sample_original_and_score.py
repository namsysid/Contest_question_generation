#!/usr/bin/env python3
"""Sample original source questions and score them through stage 06."""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent

CHOICE_MARKER_RE = re.compile(r"(?:^|\n)\s*\(([A-Ea-e])\)\s*", re.MULTILINE)
LEADING_NUMBER_RE = re.compile(r"^\s*\d+\.\s*")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_question_and_choices(text: str) -> Tuple[str, Dict[str, str]]:
    raw = text or ""
    matches = list(CHOICE_MARKER_RE.finditer(raw))
    if len(matches) < 4:
        return clean_text(LEADING_NUMBER_RE.sub("", raw, count=1)), {}

    stem = raw[: matches[0].start()]
    stem = clean_text(LEADING_NUMBER_RE.sub("", stem, count=1))

    choices: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        key = match.group(1).upper()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        value = clean_text(raw[start:end])
        if value:
            choices[key] = value

    ordered_keys = [k for k in ("A", "B", "C", "D", "E") if k in choices]
    normalized = {k: choices[k] for k in ordered_keys}
    return stem, normalized


def normalize_row(row: Dict[str, Any], input_path: str) -> Dict[str, Any] | None:
    rid = str(row.get("id") or "").strip()
    if not rid:
        return None

    source_text = ""
    if isinstance(row.get("question_text"), str):
        source_text = row["question_text"]
    elif isinstance(row.get("problem"), dict):
        problem = row["problem"]
        source_text = str(problem.get("stem") or "")

    if not source_text.strip():
        return None

    question, choices = parse_question_and_choices(source_text)
    if len(choices) < 4 or not question:
        return None

    return {
        "id": rid,
        "question": question,
        "choices": choices,
        "_meta": {
            "source_input": input_path,
            "source_kind": "original_dataset_sample",
            "source_pdf": row.get("source_pdf") or (row.get("source") or {}).get("pdf"),
            "has_diagram": bool(row.get("has_diagram")) if "has_diagram" in row else None,
            "generator": "07_sample_original_and_score",
        },
    }


def build_sample(rows: List[Dict[str, Any]], input_path: str, count: int, seed: int) -> List[Dict[str, Any]]:
    normalized = []
    for row in rows:
        norm = normalize_row(row, input_path)
        if norm is not None:
            normalized.append(norm)

    if len(normalized) < count:
        raise ValueError(f"Only {len(normalized)} rows could be normalized, fewer than requested count={count}.")

    rng = random.Random(seed)
    rng.shuffle(normalized)
    return normalized[:count]


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample original questions and score them with stage 06.")
    ap.add_argument("--input", required=True, help="Original JSONL dataset, such as data/txts/all_questions.jsonl")
    ap.add_argument("--sampled-out", default=str(ROOT / "sampled_original_as_generated.jsonl"))
    ap.add_argument("--scored-out", default=str(ROOT / "scored" / "sampled_original_scored.jsonl"))
    ap.add_argument("--count", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--subject", default="Physics")
    ap.add_argument("--competition", default="f=ma")
    ap.add_argument("--exemplars", nargs="*", default=["data/txts/all_questions.jsonl"])
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    sampled = build_sample(rows, args.input, args.count, args.seed)

    sampled_out = Path(args.sampled_out)
    scored_out = Path(args.scored_out)
    write_jsonl(sampled_out, sampled)

    if args.debug:
        print(f"[sample] input rows={len(rows)}", file=sys.stderr, flush=True)
        print(f"[sample] sampled rows={len(sampled)}", file=sys.stderr, flush=True)
        print(f"[sample] seed={args.seed}", file=sys.stderr, flush=True)
        print(f"[write] wrote sample -> {sampled_out}", file=sys.stderr, flush=True)

    cmd = [
        sys.executable,
        str(SRC / "06_verify_and_score.py"),
        "--input",
        str(sampled_out),
        "--out",
        str(scored_out),
        "--model",
        args.model,
        "--subject",
        args.subject,
        "--competition",
        args.competition,
        "--exemplars",
    ] + list(args.exemplars)
    if args.debug:
        cmd.append("--debug")

    print("[run]", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=str(ROOT))
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "Stage 06 scoring failed. The sampled file was still written to "
            f"{sampled_out}. Check the Python environment for missing dependencies "
            f"or API configuration. Underlying exit code: {exc.returncode}"
        ) from exc

    print("Done.")
    print(f"- Sampled: {sampled_out}")
    print(f"- Scored: {scored_out}")


if __name__ == "__main__":
    main()
