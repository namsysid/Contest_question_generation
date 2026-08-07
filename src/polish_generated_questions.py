#!/usr/bin/env python3
"""Polish a generated question set by filtering weak, repetitive, or off-level items.

This is a deterministic final-pass cleaner. It does not call an LLM. It reads the
structured generated JSONL, keeps higher-quality rows, removes near-duplicate
questions/scenarios, and optionally writes readable TXT/HTML exports.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
from difflib import SequenceMatcher
from pathlib import Path
from shutil import which
from typing import Any, Dict, Iterable, List, Sequence, Tuple


WORD_RE = re.compile(r"[a-z0-9]+")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
QUESTION_START_RE = re.compile(r"^\s*(\d+)[\).]\s+(.*)$")
CHOICE_LINE_RE = re.compile(r"^\s*([A-E])[\).]\s+(.*)$")
ANSWER_LINE_RE = re.compile(r"^\s*(\d+)[\).]\s+(.+?)\s*$")
STATUS_RE = re.compile(r"\s*\[(PASS|FAIL)\]\s*$", re.IGNORECASE)
MATH_EXPR_RE = re.compile(
    r"\\frac|sqrt|quadratic|rational expression|domain restriction|denominator(?:s)? must|"
    r"\([a-z]\s*[+\-*/]\s*\d|\d\s*[+\-*/]\s*[a-z]\)|[a-z]\s*\^\s*\d",
    re.IGNORECASE,
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "if", "in", "is", "it", "of", "on", "or", "that", "the", "these", "this", "to",
    "two", "which", "with", "what", "when", "where", "who", "why", "how", "given",
    "ratio", "ratios", "equivalent", "find", "determine", "value", "number",
    "numbers", "sequence", "sequences", "pattern", "patterns", "term", "terms",
    "starts", "begins", "start", "begin", "next",
}

SCENARIO_GROUPS = {
    "cooking": {"baker", "bakes", "bakery", "chef", "recipe", "flour", "sugar", "cookies", "sauce", "spices", "herbs"},
    "containers": {"container", "containers", "box", "boxes", "beads", "balls", "marbles", "red", "blue", "green", "yellow"},
    "school": {"class", "classes", "students", "school", "club", "team", "teams"},
    "parks": {"park", "parks", "trees", "maple", "planted"},
    "cars": {"car", "cars", "miles", "gallons", "gas"},
    "paint": {"paint", "painter", "blue", "red"},
    "science": {"chemical", "chemist", "biology", "biologist", "experiment", "solution", "substance", "concentration"},
    "abstract": {"expression", "expressions", "variable", "variables", "equation", "equations"},
}

DEFAULT_FORBIDDEN = {
    "engineer", "chemist", "chemical", "biologist", "biology", "experiment",
    "quadratic", "rational expression", "domain restriction", "square root", "sqrt",
    "triangle", "trigonometry",
}

POLISH_LEVELS: Dict[str, Dict[str, Any]] = {
    "basic": {
        "keep_failed": True,
        "similarity_threshold": 1.01,
        "max_per_scenario": 0,
    },
    "medium": {
        "keep_failed": False,
        "similarity_threshold": 0.84,
        "max_per_scenario": 0,
    },
    "strict": {
        "keep_failed": False,
        "similarity_threshold": 0.72,
        "max_per_scenario": 2,
    },
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_text_questions(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    main_text, answer_text = split_answer_key(raw)
    answer_key = parse_answer_key(answer_text)

    rows: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    current_question_lines: List[str] = []

    def finish_current() -> None:
        nonlocal current, current_question_lines
        if current is None:
            return
        question = "\n".join(line.rstrip() for line in current_question_lines).strip()
        question = re.sub(r"\n{3,}", "\n\n", question)
        status_match = STATUS_RE.search(question)
        if status_match:
            verdict = status_match.group(1).upper()
            question = STATUS_RE.sub("", question).rstrip()
            current["_gatekeeper"] = {"verdict": verdict}
        current["question"] = question
        qnum = current.get("question_number")
        if qnum in answer_key:
            current["answer"] = answer_key[qnum]
        rows.append(current)
        current = None
        current_question_lines = []

    for raw_line in main_text.splitlines():
        line = raw_line.rstrip()
        q_match = QUESTION_START_RE.match(line)
        choice_match = CHOICE_LINE_RE.match(line)

        if q_match:
            finish_current()
            qnum = int(q_match.group(1))
            first_line = q_match.group(2).strip()
            status_match = STATUS_RE.search(first_line)
            verdict = ""
            if status_match:
                verdict = status_match.group(1).upper()
                first_line = STATUS_RE.sub("", first_line).rstrip()

            current = {
                "id": f"{path.stem}_{qnum:03d}",
                "question_number": qnum,
            }
            if verdict:
                current["_gatekeeper"] = {"verdict": verdict}
            current_question_lines = [first_line] if first_line else []
            continue

        if current is None:
            continue

        if choice_match:
            choices = current.setdefault("choices", {})
            if isinstance(choices, dict):
                choices[choice_match.group(1)] = choice_match.group(2).strip()
            continue

        stripped = line.strip()
        if stripped or current_question_lines:
            current_question_lines.append(line)

    finish_current()
    return rows


def split_answer_key(text: str) -> Tuple[str, str]:
    match = re.search(r"(?im)^\s*answer\s+key\s*:?\s*$", text)
    if not match:
        return text, ""
    return text[: match.start()], text[match.end() :]


def parse_answer_key(text: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for line in text.splitlines():
        match = ANSWER_LINE_RE.match(line)
        if match:
            out[int(match.group(1))] = match.group(2).strip()
    return out


def read_questions(path: Path, input_format: str) -> List[Dict[str, Any]]:
    fmt = input_format.lower()
    if fmt == "auto":
        fmt = "jsonl" if path.suffix.lower() in {".jsonl", ".json"} else "txt"
    if fmt == "jsonl":
        return read_jsonl(path)
    if fmt == "txt":
        return read_text_questions(path)
    raise ValueError(f"Unsupported input format: {input_format}")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return " ".join(WORD_RE.findall(text.lower()))


def content_words(text: str) -> set[str]:
    return {w for w in WORD_RE.findall(text.lower()) if len(w) > 2 and w not in STOPWORDS and not w.isdigit()}


def question_text(row: Dict[str, Any]) -> str:
    return str(row.get("question") or row.get("question_text") or "").strip()


def choices_lines(choices: Any) -> List[str]:
    if isinstance(choices, dict):
        return [f"  {key}. {choices[key]}" for key in ("A", "B", "C", "D", "E") if key in choices]
    if isinstance(choices, list):
        labels = ["A", "B", "C", "D", "E"]
        return [f"  {labels[i]}. {value}" for i, value in enumerate(choices[:5])]
    return []


def gatekeeper_verdict(row: Dict[str, Any]) -> str:
    gate = row.get("_gatekeeper")
    if isinstance(gate, dict):
        return str(gate.get("verdict") or "").strip().upper()
    return ""


def scenario_group(text: str) -> str:
    words = content_words(text)
    best = ("other", 0)
    for group, vocab in SCENARIO_GROUPS.items():
        score = len(words & vocab)
        if score > best[1]:
            best = (group, score)
    return best[0]


def similarity(a: str, b: str) -> float:
    na = normalize_text(NUMBER_RE.sub("#", a))
    nb = normalize_text(NUMBER_RE.sub("#", b))
    if not na or not nb:
        return 0.0
    seq = SequenceMatcher(None, na, nb).ratio()
    aw = content_words(na)
    bw = content_words(nb)
    jac = len(aw & bw) / max(1, len(aw | bw))
    return max(seq, jac)


def sequence_signature(text: str) -> str:
    lower = text.lower()
    if "sequence" not in lower and "pattern" not in lower:
        return ""

    task = "other"
    if re.search(r"\bnext\b|comes next|should come", lower):
        task = "next"
    else:
        nth_match = re.search(r"\b(\d+)(?:st|nd|rd|th)\s+(?:number|term)\b", lower)
        if nth_match:
            task = f"nth:{nth_match.group(1)}"
        elif "what is the pattern" in lower or "identify" in lower:
            task = "rule"

    rule = "unknown"
    if "sum of the previous two" in lower or "adding the two" in lower or "two before" in lower:
        rule = "sum_previous_two"
    elif "twice" in lower or "double" in lower or "times the one before" in lower:
        rule = "geometric"
    elif re.search(r"\bmore than\b|\badd\s+\d+|\badding\s+\d+", lower):
        rule = "arithmetic"
    else:
        nums = [float(x) for x in NUMBER_RE.findall(lower)]
        if len(nums) >= 3:
            diffs = [round(nums[i + 1] - nums[i], 8) for i in range(len(nums) - 1)]
            if len(set(diffs)) == 1:
                rule = "arithmetic"
            elif all(nums[i] != 0 for i in range(len(nums) - 1)):
                ratios = [round(nums[i + 1] / nums[i], 8) for i in range(len(nums) - 1)]
                if len(set(ratios)) == 1:
                    rule = "geometric"
            if rule == "unknown" and len(nums) >= 4 and all(
                round(nums[i] + nums[i + 1] - nums[i + 2], 8) == 0
                for i in range(len(nums) - 2)
            ):
                rule = "sum_previous_two"

    return f"sequence:{task}:{rule}"


def has_single_answer(row: Dict[str, Any]) -> bool:
    answer = str(row.get("answer") or "").strip()
    if not answer:
        return False
    if "|" in answer:
        return False
    choices = row.get("choices")
    if isinstance(choices, dict) and choices:
        return answer in {"A", "B", "C", "D", "E"}
    return True


def is_off_level(text: str, forbidden_terms: set[str]) -> bool:
    lower = text.lower()
    if any(term in lower for term in forbidden_terms):
        return True
    return bool(MATH_EXPR_RE.search(text))


def reject_reason(
    row: Dict[str, Any],
    *,
    keep_failed: bool,
    forbidden_terms: set[str],
) -> str:
    q = question_text(row)
    if not q:
        return "missing question"
    if not keep_failed and gatekeeper_verdict(row) == "FAIL":
        return "gatekeeper failed"
    if not has_single_answer(row):
        return "missing or non-single answer"
    if is_off_level(q, forbidden_terms):
        return "off-level wording/content"
    return ""


def polish_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    keep_failed: bool,
    max_questions: int,
    similarity_threshold: float,
    max_per_scenario: int,
    forbidden_terms: set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    scenario_counts: Dict[str, int] = {}
    seen_sequence_signatures: Dict[str, str] = {}

    for row in rows:
        q = question_text(row)
        reason = reject_reason(row, keep_failed=keep_failed, forbidden_terms=forbidden_terms)
        group = scenario_group(q)

        if not reason and max_per_scenario > 0 and scenario_counts.get(group, 0) >= max_per_scenario:
            reason = f"too many {group} scenarios"

        if not reason:
            sig = sequence_signature(q)
            if sig and sig in seen_sequence_signatures:
                reason = f"near duplicate sequence pattern of {seen_sequence_signatures[sig]} ({sig})"

        if not reason:
            for kept_row in kept:
                sim = similarity(q, question_text(kept_row))
                if sim >= similarity_threshold:
                    reason = f"near duplicate of {kept_row.get('id', 'kept item')} (similarity={sim:.2f})"
                    break

        if reason:
            rejected.append({"id": row.get("id"), "reason": reason, "scenario_group": group, "question": q})
            continue

        out = dict(row)
        out["_polish"] = {
            "scenario_group": group,
            "source_gatekeeper": gatekeeper_verdict(row) or None,
        }
        kept.append(out)
        scenario_counts[group] = scenario_counts.get(group, 0) + 1
        sig = sequence_signature(q)
        if sig:
            seen_sequence_signatures[sig] = str(row.get("id") or "kept item")
        if max_questions > 0 and len(kept) >= max_questions:
            break

    return kept, rejected


def write_reject_report(path: Path, rejected: Sequence[Dict[str, Any]]) -> None:
    lines = ["Rejected Questions", "==================", ""]
    for idx, row in enumerate(rejected, start=1):
        lines.append(f"{idx}. {row.get('id') or '(no id)'} - {row['reason']} [{row['scenario_group']}]")
        lines.append(str(row.get("question") or "").strip())
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_text(path: Path, rows: Sequence[Dict[str, Any]], *, include_answers: bool) -> None:
    lines = ["Polished Math Questions", "=======================", ""]
    answers: List[str] = []
    for idx, row in enumerate(rows, start=1):
        lines.append(f"{idx}. {question_text(row)}")
        for choice in choices_lines(row.get("choices")):
            lines.append(choice)
        lines.append("")
        answer = str(row.get("answer") or "").strip()
        if answer:
            answers.append(f"{idx}. {answer}")
    if include_answers and answers:
        lines.extend(["", "Answer Key", "==========", *answers, ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_html(path: Path, rows: Sequence[Dict[str, Any]], *, include_answers: bool) -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Polished Math Questions</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;line-height:1.45;max-width:820px;margin:32px auto;padding:0 20px;color:#111}",
        "h1{font-size:24px;margin-bottom:24px}.q{break-inside:avoid;margin-bottom:24px}.stem{font-weight:600;margin-bottom:10px}",
        ".choices{margin:0;padding-left:24px}.choices li{margin:4px 0}.answers{margin-top:36px;border-top:1px solid #bbb;padding-top:16px}",
        "@media print{body{margin:18mm auto}.q{page-break-inside:avoid}}",
        "</style></head><body><h1>Polished Math Questions</h1>",
    ]
    answer_parts: List[str] = []
    for idx, row in enumerate(rows, start=1):
        parts.append("<section class='q'>")
        parts.append(f"<div class='stem'>{idx}. {html.escape(question_text(row))}</div>")
        choices = row.get("choices")
        if choices_lines(choices):
            parts.append("<ol class='choices' type='A'>")
            if isinstance(choices, dict):
                for key in ("A", "B", "C", "D", "E"):
                    if key in choices:
                        parts.append(f"<li>{html.escape(str(choices[key]))}</li>")
            elif isinstance(choices, list):
                for choice in choices[:5]:
                    parts.append(f"<li>{html.escape(str(choice))}</li>")
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


def maybe_write_pdf(txt_path: Path, pdf_path: Path) -> bool:
    if which("cupsfilter") is None:
        return False
    with pdf_path.open("wb") as out_f:
        subprocess.run(["cupsfilter", str(txt_path)], stdout=out_f, stderr=subprocess.DEVNULL, check=True)
    return True


def apply_polish_level(args: argparse.Namespace) -> None:
    settings = POLISH_LEVELS[args.polish_level]
    if args.keep_failed is None:
        args.keep_failed = bool(settings["keep_failed"])
    if args.similarity_threshold is None:
        args.similarity_threshold = float(settings["similarity_threshold"])
    if args.max_per_scenario is None:
        args.max_per_scenario = int(settings["max_per_scenario"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Final-pass polish generated questions.")
    ap.add_argument("--input", required=True, help="Generated problems JSONL or readable text file")
    ap.add_argument("--input-format", choices=["auto", "jsonl", "txt"], default="auto")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-txt", default="")
    ap.add_argument("--out-html", default="")
    ap.add_argument("--out-pdf", default="")
    ap.add_argument("--reject-report", default="")
    ap.add_argument("--max-questions", type=int, default=0, help="0 = no cap")
    ap.add_argument(
        "--polish-level",
        choices=["basic", "medium", "strict"],
        default="medium",
        help="basic keeps anything non-weird; medium removes clear duplicates; strict is harsh",
    )
    ap.add_argument("--similarity-threshold", type=float, default=None, help="Override the polish-level duplicate threshold")
    ap.add_argument("--max-per-scenario", type=int, default=None, help="Override the polish-level scenario cap; 0 disables")
    ap.add_argument("--keep-failed", action=argparse.BooleanOptionalAction, default=None, help="Override whether gatekeeper FAIL rows are kept")
    ap.add_argument("--include-answers", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--forbid-term", action="append", default=[], help="Additional lowercase term to reject")
    ap.add_argument("--allow-default-forbidden", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    apply_polish_level(args)

    forbidden = set(DEFAULT_FORBIDDEN) if args.allow_default_forbidden else set()
    forbidden.update(term.lower() for term in args.forbid_term)

    rows = read_questions(Path(args.input), args.input_format)
    kept, rejected = polish_rows(
        rows,
        keep_failed=args.keep_failed,
        max_questions=args.max_questions,
        similarity_threshold=args.similarity_threshold,
        max_per_scenario=args.max_per_scenario,
        forbidden_terms=forbidden,
    )

    write_jsonl(Path(args.out_jsonl), kept)
    if args.out_txt:
        write_text(Path(args.out_txt), kept, include_answers=args.include_answers)
    if args.out_html:
        write_html(Path(args.out_html), kept, include_answers=args.include_answers)
    if args.reject_report:
        write_reject_report(Path(args.reject_report), rejected)
    if args.out_pdf:
        if not args.out_txt:
            raise SystemExit("--out-pdf requires --out-txt so there is a text source to convert.")
        wrote_pdf = maybe_write_pdf(Path(args.out_txt), Path(args.out_pdf))
        if not wrote_pdf:
            print("PDF skipped: cupsfilter not found.")

    print(f"Kept {len(kept)} of {len(rows)} questions -> {args.out_jsonl}")
    print(f"Rejected {len(rejected)} questions")


if __name__ == "__main__":
    main()
