#!/usr/bin/env python3
"""
01_enrich_problem_schema_paper.py  (PAPER-FAITHFUL SCHEMA ENRICHMENT + OPERATOR LIMITING)

Input:  raw JSONL from 00_pdf-to-txt.py
Output: enriched JSONL with:
  - problem.stem + problem.choices
  - analysis.skeleton: structured object
  - analysis.* tags (concepts/skills/difficulty/structure_tags)
  - IMPORTANT: operator set is LIMITED + normalized to a fixed ontology

Why this exists:
- Your draft paper claims a predefined operator set and skeletons that are compact, comparable,
  and retrieval-friendly. This script enforces that contract.

Operator ontology (default):
  IDENTIFY_GIVENS
  IDENTIFY_RELATION
  APPLY_RELATION
  SAVE_RESULT
  CHECK

You can adjust:
  --allowed_ops IDENTIFY_GIVENS,IDENTIFY_RELATION,APPLY_RELATION,SAVE_RESULT,CHECK
  --max_steps 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set. Please set it in .env or environment variables.")


# ----------------------------
# Heuristics / parsing
# ----------------------------

CHOICE_RE = re.compile(
    r"\(\s*([A-E])\s*\)\s*(.+?)(?=(?:\n\s*\(\s*[A-E]\s*\)\s)|\Z)",
    re.DOTALL,
)
LEADING_QNUM_RE = re.compile(r"^\s*\d+\.\s*", re.DOTALL)
FOOTER_JUNK_RE = re.compile(
    r"(Copyright.*?$|F\s*=\s*ma Exam.*?$|\n\s*\d+\s*$)",
    re.IGNORECASE | re.MULTILINE,
)
FIGURE_HINT_RE = re.compile(r"(figure|diagram|shown|shown below)", re.IGNORECASE)


def clean_question_text(qtext: str) -> str:
    s = (qtext or "").replace("\r\n", "\n").replace("\r", "\n")
    s = FOOTER_JUNK_RE.sub("", s)
    return s.strip()


def split_stem_and_choices(qtext: str) -> Tuple[str, List[str], Optional[str]]:
    s = clean_question_text(qtext)
    s = LEADING_QNUM_RE.sub("", s).strip()

    matches = list(CHOICE_RE.finditer(s))
    if not matches:
        return s.strip(), [], None

    first_choice_start = matches[0].start()
    stem = s[:first_choice_start].strip()

    choices = []
    for m in matches:
        letter = m.group(1).strip()
        body = re.sub(r"\s+", " ", m.group(2)).strip()
        choices.append(f"({letter}) {body}")

    return stem, choices, None


def infer_domain(raw: Dict[str, Any]) -> str:
    if raw.get("domain"):
        d = str(raw["domain"]).lower()
        if d in ("fma", "usnco"):
            return d
    src = (raw.get("source_pdf") or "").lower()
    if "f" in src and "ma" in src:
        return "fma"
    return "fma"


# ----------------------------
# Operator limiting / normalization
# ----------------------------

def _kw(s: str) -> str:
    return re.sub(r"[^a-z]+", " ", (s or "").lower()).strip()

def normalize_op(op: str, allowed: List[str]) -> str:
    """
    Map arbitrary model ops -> allowed ontology.
    Conservative, keyword-based.
    """
    a = set(allowed)
    o = (op or "").strip()

    if o in a:
        return o

    k = _kw(o)
    if any(w in k for w in ["given", "known", "assume", "define", "set up", "setup", "identify given"]):
        return "IDENTIFY_GIVENS" if "IDENTIFY_GIVENS" in a else allowed[0]
    if any(w in k for w in ["law", "relation", "equation", "principle", "use", "identify relation", "choose equation"]):
        return "IDENTIFY_RELATION" if "IDENTIFY_RELATION" in a else allowed[0]
    if any(w in k for w in ["apply", "compute", "solve", "substitute", "plug", "derive", "combine", "manipulate"]):
        return "APPLY_RELATION" if "APPLY_RELATION" in a else allowed[0]
    if any(w in k for w in ["result", "final", "conclude", "answer", "save", "select choice"]):
        return "SAVE_RESULT" if "SAVE_RESULT" in a else allowed[0]
    if any(w in k for w in ["check", "units", "sanity", "dimension", "limit", "sign"]):
        return "CHECK" if "CHECK" in a else allowed[0]

    # fallback: safest "APPLY_RELATION" if present, else first allowed
    return "APPLY_RELATION" if "APPLY_RELATION" in a else allowed[0]


def compress_steps(steps: List[Dict[str, Any]], allowed_ops: List[str], max_steps: int) -> List[Dict[str, Any]]:
    """
    - normalize op values into a fixed operator set
    - collapse consecutive duplicate ops (keep the first, concatenate text)
    - truncate to max_steps
    """
    cleaned: List[Dict[str, Any]] = []
    prev_op: Optional[str] = None

    for st in steps or []:
        if not isinstance(st, dict):
            continue
        op = normalize_op(str(st.get("op", "")), allowed_ops)
        txt = str(st.get("text", "")).strip()
        if not txt:
            continue

        if cleaned and prev_op == op:
            cleaned[-1]["text"] = (cleaned[-1]["text"].rstrip(".") + "; " + txt).strip()
        else:
            cleaned.append({"op": op, "text": txt})
            prev_op = op

        if len(cleaned) >= max_steps:
            break

    # Ensure at least 3 steps when possible (paper-friendly)
    return cleaned


# ----------------------------
# LLM schema enrichment
# ----------------------------

SYSTEM = """You are an expert STEM competition solution analyst.

Return ONLY strict JSON.

You MUST use ONLY the allowed operator set for each step's `op`.
You MUST keep the number of steps short and abstract (no full arithmetic, no long derivations).
"""

USER_TMPL = """Domain: {domain}

Problem stem:
{stem}

Choices:
{choices_block}

Diagrams present: {has_diagram}
Diagram files: {diagram_files}

ALLOWED_OPS (you MUST use one of these exact strings for every step.op):
{allowed_ops_block}

Return JSON with keys:
- skeleton: object with keys:
  - givens: list of strings (0-8)
  - target: string
  - laws: list of strings (0-6) (very short names like "work-energy", "ideal gas", "torque balance")
  - steps: list of objects {{op: string, text: string}} (3-{max_steps} steps)
  - final_form: string (final expression or computation plan; keep abstract)
- concepts: list of short strings (2-8)
- skills: list of short strings (0-8)
- difficulty: integer 1-10
- structure_tags: list of short strings (0-8)
- units_expected: boolean
- diagram_required: boolean
- quick_checks: list of short strings (0-6)

Guidelines:
- Do not compute numeric answers; keep it symbolic/structural.
- steps.text should be <= 25 words each on average.
- diagram_required true ONLY if information is missing without the figure.
"""

client = OpenAI(timeout=20.0, max_retries=3)


def llm_enrich(
    domain: str,
    stem: str,
    choices: List[str],
    has_diagram: bool,
    diagram_files: List[str],
    model: str,
    allowed_ops: List[str],
    max_steps: int,
) -> Dict[str, Any]:
    choices_block = "\n".join(choices) if choices else "(none)"
    user = USER_TMPL.format(
        domain=domain,
        stem=stem,
        choices_block=choices_block,
        has_diagram=str(bool(has_diagram)).lower(),
        diagram_files=json.dumps(diagram_files),
        allowed_ops_block="\n".join(f"- {o}" for o in allowed_ops),
        max_steps=max_steps,
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    text = resp.choices[0].message.content
    if text is None:
        raise ValueError("Model returned empty content")
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j < 0 or j <= i:
        raise ValueError(f"Model did not return JSON. Output was:\n{text}")
    return json.loads(text[i : j + 1])


# ----------------------------
# Schema builder
# ----------------------------

def build_schema(raw: Dict[str, Any], corpus_name: str = "unknown", year: Optional[int] = None, variant: Optional[str] = None) -> Dict[str, Any]:
    domain = infer_domain(raw)
    stem, choices, answer_key = split_stem_and_choices(raw.get("question_text", ""))

    diagrams = []
    for f in raw.get("diagram_files", []) or []:
        diagrams.append(
            {
                "id": Path(f).stem,
                "page": None,
                "file": f,
                "bbox": None,
                "alt_text": None,
                "hash": None,
            }
        )

    return {
        "id": raw["id"],
        "domain": domain,
        "source": {
            "corpus": corpus_name,
            "year": year,
            "variant": variant,
            "pdf": raw.get("source_pdf"),
            "page_range": None,
        },
        "problem": {
            "stem": stem,
            "choices": choices,
            "answer_key": answer_key,
            "units_expected": None,
            "diagrams": diagrams,
        },
        "analysis": {
            "skeleton": None,
            "concepts": [],
            "skills": [],
            "difficulty": None,
            "structure_tags": [],
            "diagram_required": bool(raw.get("has_diagram")) or bool(FIGURE_HINT_RE.search(stem)),
        },
        "checks": {"quick_checks": [], "validators": [], "flags": []},
        "retrieval": {
            "embed_views": {
                "question": "stem + choices",
                "skeleton": "analysis.skeleton (operator-limited, linearizable)",
            },
            "embedding_ref": None,
        },
    }


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_jsonl_line(f, row: Dict[str, Any]) -> None:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    f.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="raw JSONL from 00_pdf-to-txt.py")
    ap.add_argument("--out", default="enriched.jsonl")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--corpus", default="unknown")
    ap.add_argument("--year", type=int, default=0)
    ap.add_argument("--variant", default="")
    ap.add_argument("--max_steps", type=int, default=8)
    ap.add_argument("--allowed_ops", default="IDENTIFY_GIVENS,IDENTIFY_RELATION,APPLY_RELATION,SAVE_RESULT,CHECK")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--debug", action="store_true", help="print per-row progress to stderr")
    args = ap.parse_args()

    allowed_ops = [s.strip() for s in args.allowed_ops.split(",") if s.strip()]
    if not allowed_ops:
        raise ValueError("allowed_ops must be a non-empty comma-separated list.")

    rows = read_jsonl(args.input)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    total = len(rows)
    if args.debug:
        print(f"[write] open (truncate): {args.out}", file=sys.stderr, flush=True)
    with open(args.out, "w", encoding="utf-8") as out_f:
        for i, raw in enumerate(rows, start=1):
            schema = build_schema(raw, corpus_name=args.corpus, year=(args.year or None), variant=(args.variant or None))
            domain = schema["domain"]
            stem = schema["problem"]["stem"]
            choices = schema["problem"]["choices"]
            has_diagram = bool(raw.get("has_diagram"))
            diagram_files = raw.get("diagram_files", []) or []

            if args.debug:
                rid = raw.get("id", "?")
                print(f"[{i}/{total}] id={rid} calling model={args.model}", file=sys.stderr, flush=True)
                t0 = time.time()

            if args.debug:
                print(f"[{i}/{total}] id={raw.get('id','?')} parsing model response", file=sys.stderr, flush=True)

            enrich = llm_enrich(
                domain=domain,
                stem=stem,
                choices=choices,
                has_diagram=has_diagram,
                diagram_files=diagram_files,
                model=args.model,
                allowed_ops=allowed_ops,
                max_steps=args.max_steps,
            )

            if args.debug:
                dt = time.time() - t0
                print(f"[{i}/{total}] id={raw.get('id','?')} model done in {dt:.2f}s", file=sys.stderr, flush=True)
                print(f"[{i}/{total}] id={raw.get('id','?')} post-process start", file=sys.stderr, flush=True)

            sk = enrich.get("skeleton") or {}
            steps = sk.get("steps") or []
            sk["steps"] = compress_steps(steps, allowed_ops=allowed_ops, max_steps=args.max_steps)
            # enforce laws short list
            laws = sk.get("laws") or []
            if isinstance(laws, list):
                sk["laws"] = [str(x).strip()[:80] for x in laws[:6] if str(x).strip()]
            else:
                sk["laws"] = []

            schema["analysis"]["skeleton"] = sk
            schema["analysis"]["concepts"] = enrich.get("concepts") or []
            schema["analysis"]["skills"] = enrich.get("skills") or []
            schema["analysis"]["difficulty"] = enrich.get("difficulty")
            schema["analysis"]["structure_tags"] = enrich.get("structure_tags") or []
            schema["problem"]["units_expected"] = enrich.get("units_expected")
            schema["analysis"]["diagram_required"] = bool(enrich.get("diagram_required"))
            schema["checks"]["quick_checks"] = enrich.get("quick_checks") or []

            write_jsonl_line(out_f, schema)
            if args.debug:
                print(f"[{i}/{total}] id={raw.get('id','?')} wrote", file=sys.stderr, flush=True)

    if args.debug:
        print(f"[write] done: {args.out}", file=sys.stderr, flush=True)
    print(f"Wrote {total} enriched rows -> {args.out}")


if __name__ == "__main__":
    main()
