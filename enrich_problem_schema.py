#!/usr/bin/env python3
"""
enrich_problem_schema.py

Given "raw" problem JSON objects like:
{
  "id": "...",
  "question_number": 2,
  "question_text": "2. ... (A) ...",
  "has_diagram": false,
  "diagram_files": [],
  "source_pdf": "exam1-2015-1-8.pdf"
}

This script:
- parses stem + choices (MCQ)
- builds the normalized schema (problem + analysis + checks-lite)
- calls an LLM to produce a structured solution skeleton + concepts/tags/difficulty
- (optionally) produces a short "final form" expression and quick validation flags

It does NOT store embeddings in the JSON (by design).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Check that API key is set
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set. Please set it in .env file or environment variable.")

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
    # remove common footer/header junk and trailing page numbers
    s = qtext.replace("\r\n", "\n").replace("\r", "\n")
    s = FOOTER_JUNK_RE.sub("", s)
    s = s.strip()
    return s


def split_stem_and_choices(qtext: str) -> Tuple[str, List[str], Optional[str]]:
    """
    Returns: (stem, choices_list, answer_key_if_in_text)
    We assume answer key is NOT present in the question_text; so answer_key is None.
    """
    s = clean_question_text(qtext)
    s = LEADING_QNUM_RE.sub("", s).strip()

    matches = list(CHOICE_RE.finditer(s))
    if not matches:
        # not MCQ or parsing failed; treat entire thing as stem
        return s.strip(), [], None

    first_choice_start = matches[0].start()
    stem = s[:first_choice_start].strip()

    # Build choice strings in original order A..E as present
    choices = []
    for m in matches:
        letter = m.group(1).strip()
        body = re.sub(r"\s+", " ", m.group(2)).strip()
        choices.append(f"({letter}) {body}")

    return stem, choices, None


def infer_domain(raw: Dict[str, Any]) -> str:
    # minimal heuristic: you can override by adding raw["domain"]
    if "domain" in raw and raw["domain"]:
        d = str(raw["domain"]).lower()
        if d in ("fma", "usnco"):
            return d
    src = (raw.get("source_pdf") or "").lower()
    if "f" in src and "ma" in src:
        return "fma"
    # fallback default
    return "fma"


# ----------------------------
# LLM schema enrichment
# ----------------------------

SYSTEM = """You are an expert STEM competition solution analyst.

Given a problem stem and choices, produce a STRUCTURED solution skeleton (no full arithmetic),
plus concept tags and structural tags. Keep it concise, but complete enough to uniquely define
the solution pathway.

Return ONLY valid JSON matching the requested keys.
"""

USER_TMPL = """Domain: {domain}

Problem stem:
{stem}

Choices:
{choices_block}

Diagrams present: {has_diagram}
Diagram files: {diagram_files}

Return JSON with keys:
- skeleton: object with keys:
  - givens: list of strings
  - target: string
  - steps: list of objects {{op: string, text: string}}
  - final_form: string (final expression or computation plan; no plugging numbers if avoidable)
- concepts: list of short strings (2-8)
- skills: list of short strings (0-8)
- difficulty: integer 1-10
- structure_tags: list of short strings (0-8)
- units_expected: boolean
- diagram_required: boolean
- quick_checks: list of short strings (0-6)  (e.g., "avg speed must be < 80 and > 50")

Guidelines:
- If domain=fma: use physics language (kinematics/dynamics/energy/etc.).
- For multiple-choice numeric: units_expected=true.
- diagram_required should be true ONLY if the solution depends on diagram information not fully described in text.
"""

client = OpenAI()


def llm_enrich(
    domain: str,
    stem: str,
    choices: List[str],
    has_diagram: bool,
    diagram_files: List[str],
    model: str,
) -> Dict[str, Any]:
    choices_block = "\n".join(choices) if choices else "(none)"
    user = USER_TMPL.format(
        domain=domain,
        stem=stem,
        choices_block=choices_block,
        has_diagram=str(bool(has_diagram)).lower(),
        diagram_files=json.dumps(diagram_files),
    )
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    text = resp.output_text

    # tolerant JSON parse: take first {...} block
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j < 0 or j <= i:
        raise ValueError(f"Model did not return JSON. Output was:\n{text}")
    return json.loads(text[i : j + 1])


# ----------------------------
# Schema builder
# ----------------------------

def build_schema(
    raw: Dict[str, Any],
    *,
    corpus_name: str = "unknown",
    year: Optional[int] = None,
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    domain = infer_domain(raw)
    stem, choices, answer_key = split_stem_and_choices(raw.get("question_text", ""))

    # diagram object list (we don't have bbox yet in your raw JSON, so store file + page=None)
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

    out = {
        "id": raw["id"],
        "domain": domain,
        "source": {
            "corpus": corpus_name,
            "year": year,
            "variant": variant,
            "pdf": raw.get("source_pdf"),
            "page_range": None,  # fill later if you capture page numbers
        },
        "problem": {
            "stem": stem,
            "choices": choices,
            "answer_key": answer_key,  # usually None at this stage
            "units_expected": None,    # filled by LLM
            "diagrams": diagrams,
        },
        "analysis": {
            "skeleton": None,          # filled by LLM (structured object)
            "concepts": [],
            "skills": [],
            "difficulty": None,
            "structure_tags": [],
            "diagram_required": bool(raw.get("has_diagram")) or bool(FIGURE_HINT_RE.search(stem)),
        },
        # keep only lightweight checks/flags here
        "checks": {
            "quick_checks": [],
            "validators": [],
            "flags": [],
        },
        # pointers only; embeddings live elsewhere
        "retrieval": {
            "embed_views": {
                "question": "stem + choices",
                "skeleton": "analysis.skeleton (string-joined)",
                "diagram_alt": "problem.diagrams[].alt_text (joined)",
            },
            "embedding_ref": None,
        },
    }
    return out


def skeleton_to_joined_text(skel_obj: Dict[str, Any]) -> str:
    # Useful later for embedding the skeleton in a stable way
    givens = skel_obj.get("givens", [])
    target = skel_obj.get("target", "")
    steps = skel_obj.get("steps", [])
    final_form = skel_obj.get("final_form", "")
    lines = []
    if givens:
        lines.append("GIVENS: " + "; ".join(givens))
    if target:
        lines.append("TARGET: " + target)
    if steps:
        for s in steps:
            op = s.get("op", "").strip()
            tx = s.get("text", "").strip()
            if op and tx:
                lines.append(f"{op}: {tx}")
            elif tx:
                lines.append(tx)
    if final_form:
        lines.append("FINAL_FORM: " + final_form)
    return "\n".join(lines).strip()


def enrich_one(raw: Dict[str, Any], model: str, corpus_name: str) -> Dict[str, Any]:
    doc = build_schema(raw, corpus_name=corpus_name)

    llm = llm_enrich(
        domain=doc["domain"],
        stem=doc["problem"]["stem"],
        choices=doc["problem"]["choices"],
        has_diagram=bool(doc["problem"]["diagrams"]),
        diagram_files=[d["file"] for d in doc["problem"]["diagrams"]],
        model=model,
    )

    # Fill schema from LLM
    doc["analysis"]["skeleton"] = llm["skeleton"]
    doc["analysis"]["concepts"] = llm.get("concepts", [])
    doc["analysis"]["skills"] = llm.get("skills", [])
    doc["analysis"]["difficulty"] = llm.get("difficulty", None)
    doc["analysis"]["structure_tags"] = llm.get("structure_tags", [])
    doc["problem"]["units_expected"] = llm.get("units_expected", None)
    doc["analysis"]["diagram_required"] = llm.get("diagram_required", doc["analysis"]["diagram_required"])
    doc["checks"]["quick_checks"] = llm.get("quick_checks", [])

    # Light validators suggestions based on domain
    if doc["domain"] == "fma":
        doc["checks"]["validators"] = ["dimensional", "sanity_bounds"]
    else:
        doc["checks"]["validators"] = ["sanity_bounds"]

    # Optional: store a canonical skeleton text for embedding later (still not embeddings)
    doc["analysis"]["_skeleton_text"] = skeleton_to_joined_text(doc["analysis"]["skeleton"])

    return doc


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSONL or single JSON file")
    ap.add_argument("--out", dest="out_path", required=True, help="Output JSONL")
    ap.add_argument("--model", default="gpt-4.1-mini", help="LLM model for skeleton/tagging")
    ap.add_argument("--corpus", default="unknown", help="Corpus name to store in source.corpus")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load input: either a single JSON object or JSONL
    raws: List[Dict[str, Any]] = []
    txt = in_path.read_text(encoding="utf-8").strip()
    lines = txt.splitlines()
    
    # If there's only one line, try parsing as single JSON
    # Otherwise, treat as JSONL (one JSON object per line)
    if len(lines) == 1:
        try:
            raws = [json.loads(txt)]
        except json.JSONDecodeError:
            # If single JSON parse fails, treat as JSONL anyway
            for line in lines:
                line = line.strip()
                if line:
                    raws.append(json.loads(line))
    else:
        # Multiple lines = JSONL format
        for line in lines:
            line = line.strip()
            if line:
                raws.append(json.loads(line))

    with out_path.open("w", encoding="utf-8") as f:
        for raw in raws:
            enriched = enrich_one(raw, model=args.model, corpus_name=args.corpus)
            f.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    print(f"Done. Wrote: {out_path}")


if __name__ == "__main__":
    main()
