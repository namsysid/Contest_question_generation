#!/usr/bin/env python3
"""
pdf_fma_to_jsonl.py

Parse an F=ma-style solutions PDF into JSONL:
{"id":"p1","question":"...","solution":"..."}

This is heuristic-based. Expect to tweak patterns per packet.

Examples:
  python pdf_fma_to_jsonl.py input.pdf -o out.jsonl
  python pdf_fma_to_jsonl.py input.pdf -o out.jsonl --debug --debug-pages 1-3
  python pdf_fma_to_jsonl.py input.pdf -o out.jsonl --prefer-pymupdf

Dependencies:
  pip install pdfplumber pymupdf

Notes:
- Works best on text PDFs (not scanned images).
- If your PDF explicitly labels "Problem X" and "Solution", you'll get good results.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ----------------------------
# Text extraction
# ----------------------------

def extract_text_pdfplumber(pdf_path: str) -> List[str]:
    import pdfplumber  # type: ignore
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            # tweak tolerances if needed
            text = p.extract_text(x_tolerance=2, y_tolerance=2) or ""
            pages.append(text)
    return pages


def extract_text_pymupdf(pdf_path: str) -> List[str]:
    import fitz  # PyMuPDF  # type: ignore
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text") or ""
        pages.append(text)
    doc.close()
    return pages


# ----------------------------
# Heuristics / parsing
# ----------------------------

@dataclass
class Problem:
    pid: str
    question: str
    solution: str


def normalize_whitespace(s: str) -> str:
    # normalize line endings, collapse excessive blank lines, keep paragraphs
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # remove trailing spaces
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    # collapse 3+ newlines to 2
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def remove_common_headers_footers(page_text: str) -> str:
    """
    Very conservative cleanup; adjust as needed.
    - Removes lone page numbers
    - Removes obvious repeated 'F=ma' header lines, if present
    """
    lines = [ln.strip() for ln in page_text.split("\n")]
    cleaned = []
    for ln in lines:
        if not ln:
            cleaned.append("")
            continue
        # lone page number
        if re.fullmatch(r"\d{1,3}", ln):
            continue
        # common header-ish noise (tweak for your PDFs)
        if re.search(r"\bF\s*=\s*ma\b", ln) and len(ln) <= 40:
            # if it's short and contains F=ma, likely header
            continue
        cleaned.append(ln)
    return normalize_whitespace("\n".join(cleaned))


# Problem start patterns commonly seen:
# - "1." at start of line
# - "Problem 1"
# - "1)" or "1 :"
PROBLEM_START_RE = re.compile(
    r"""
    (?mx)                           # multiline, verbose
    ^\s*(?:                         # start of line
        Problem\s+(\d{1,3})\b       # "Problem 12"
        |
        (\d{1,3})\s*[\.\)]\s+       # "12." or "12)"
    )
    """,
    re.VERBOSE,
)

# Solution markers:
SOLUTION_MARK_RE = re.compile(
    r"(?im)^\s*(?:Solution|Answer|Solution\s*:)\b"
)

# Sometimes PDFs include "Solution to Problem X"
SOLUTION_TO_PROBLEM_RE = re.compile(
    r"(?im)^\s*Solution\s*(?:to|for)\s*(?:Problem\s*)?(\d{1,3})\b"
)


def find_problem_blocks(full_text: str) -> List[Tuple[int, int, Optional[int]]]:
    """
    Returns list of (start_idx, end_idx, problem_number_or_None)
    based on PROBLEM_START_RE.
    """
    matches = list(PROBLEM_START_RE.finditer(full_text))
    blocks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        # group(1) for "Problem N", group(2) for "N."
        n = m.group(1) or m.group(2)
        blocks.append((start, end, int(n) if n else None))
    return blocks


def split_question_solution(block_text: str) -> Tuple[str, str]:
    """
    Split within a problem block into question and solution.
    Heuristics:
      - Look for "Solution ..." line; first occurrence splits.
      - If not found, try to split on a big separator like multiple newlines after the statement.
      - Fallback: entire block as solution, empty question.
    """
    block_text = normalize_whitespace(block_text)

    # Prefer "Solution to Problem X" or "Solution"
    m = SOLUTION_TO_PROBLEM_RE.search(block_text)
    if m:
        idx = m.start()
        q = block_text[:idx].strip()
        s = block_text[idx:].strip()
        return q, s

    m = SOLUTION_MARK_RE.search(block_text)
    if m:
        idx = m.start()
        q = block_text[:idx].strip()
        s = block_text[idx:].strip()
        return q, s

    # Try a softer split: first double-blank-line after some text (often after problem statement)
    # but only if it seems plausible (question shorter than solution, etc.).
    parts = re.split(r"\n\s*\n", block_text, maxsplit=1)
    if len(parts) == 2:
        q_candidate, rest = parts[0].strip(), parts[1].strip()
        # If "rest" is much larger, assume it's solution-ish
        if len(rest) > 1.5 * len(q_candidate):
            return q_candidate, rest

    # Fallback: treat as solution-only
    return "", block_text


def parse_fma_pdf_to_problems(pages_text: List[str], debug: bool = False) -> List[Problem]:
    # Clean per-page, then join with hard page breaks
    cleaned_pages = [remove_common_headers_footers(t) for t in pages_text]
    full_text = "\n\n<<PAGE_BREAK>>\n\n".join(cleaned_pages)
    full_text = normalize_whitespace(full_text)

    blocks = find_problem_blocks(full_text)

    if debug:
        print(f"[debug] total text chars: {len(full_text)}", file=sys.stderr)
        print(f"[debug] detected problem blocks: {len(blocks)}", file=sys.stderr)

    problems: List[Problem] = []
    auto_id = 1

    for (start, end, n) in blocks:
        raw_block = full_text[start:end].strip()

        # strip the leading "Problem N" or "N." header from the block for cleaner question text
        raw_block = PROBLEM_START_RE.sub("", raw_block, count=1).strip()
        q, s = split_question_solution(raw_block)

        pid_num = n if n is not None else auto_id
        pid = f"p{pid_num}"

        # If question is empty and solution starts with "Solution", try to salvage by taking
        # first paragraph as question (some packets omit explicit question label).
        if not q and s:
            # take first paragraph as question-ish statement if short enough
            paras = re.split(r"\n\s*\n", s, maxsplit=2)
            if len(paras) >= 2 and len(paras[0]) < 800:
                # But don't steal if it clearly starts with "Solution"
                if not SOLUTION_MARK_RE.match(paras[0]):
                    q = paras[0].strip()
                    s = "\n\n".join(paras[1:]).strip()

        problems.append(
            Problem(
                pid=pid,
                question=normalize_whitespace(q),
                solution=normalize_whitespace(s),
            )
        )
        auto_id += 1

    # If we found nothing, fallback: try splitting by "Solution to Problem X"
    if not problems:
        if debug:
            print("[debug] No PROBLEM_START blocks found; trying Solution-to-Problem splitting.", file=sys.stderr)
        # Make blocks using "Solution to Problem X" as anchors, then backtrack some text for question
        sol_matches = list(SOLUTION_TO_PROBLEM_RE.finditer(full_text))
        for i, m in enumerate(sol_matches):
            sol_start = m.start()
            sol_end = sol_matches[i + 1].start() if i + 1 < len(sol_matches) else len(full_text)
            n = int(m.group(1))
            sol_block = full_text[sol_start:sol_end].strip()

            # look backward for a preceding "Problem N" occurrence (best-effort)
            back_window_start = max(0, sol_start - 5000)
            back_window = full_text[back_window_start:sol_start]
            pm = list(PROBLEM_START_RE.finditer(back_window))
            q_text = ""
            if pm:
                last = pm[-1]
                q_text = back_window[last.end():].strip()

            problems.append(
                Problem(
                    pid=f"p{n}",
                    question=normalize_whitespace(q_text),
                    solution=normalize_whitespace(sol_block),
                )
            )

    return problems


# ----------------------------
# JSONL output
# ----------------------------

def write_jsonl(problems: List[Problem], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for pr in problems:
            rec = {"id": pr.pid, "question": pr.question, "solution": pr.solution}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ----------------------------
# CLI
# ----------------------------

def parse_page_range(spec: str, n_pages: int) -> List[int]:
    """
    "1-3,7,10-12" (1-indexed) -> list of 0-indexed page indices
    """
    idxs = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a_i = int(a)
            b_i = int(b)
            for p in range(a_i, b_i + 1):
                if 1 <= p <= n_pages:
                    idxs.add(p - 1)
        else:
            p = int(part)
            if 1 <= p <= n_pages:
                idxs.add(p - 1)
    return sorted(idxs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="Input PDF")
    ap.add_argument("-o", "--out", default="out.jsonl", help="Output JSONL path")
    ap.add_argument("--prefer-pymupdf", action="store_true", help="Use PyMuPDF for extraction (often better)")
    ap.add_argument("--debug", action="store_true", help="Debug logs to stderr")
    ap.add_argument("--debug-pages", default="", help="Only parse a subset of pages: e.g. '1-3,7'")
    args = ap.parse_args()

    # Extract text
    if args.prefer_pymupdf:
        pages = extract_text_pymupdf(args.pdf)
    else:
        pages = extract_text_pdfplumber(args.pdf)

    if args.debug_pages:
        keep = parse_page_range(args.debug_pages, len(pages))
        pages = [pages[i] for i in keep]
        if args.debug:
            print(f"[debug] restricting to pages (0-indexed): {keep}", file=sys.stderr)

    problems = parse_fma_pdf_to_problems(pages, debug=args.debug)

    if args.debug:
        print(f"[debug] parsed problems: {len(problems)}", file=sys.stderr)
        # show quick preview
        for pr in problems[:3]:
            print(f"[debug] {pr.pid}: q={len(pr.question)} chars, s={len(pr.solution)} chars", file=sys.stderr)

    write_jsonl(problems, args.out)

    if args.debug:
        print(f"[debug] wrote: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
