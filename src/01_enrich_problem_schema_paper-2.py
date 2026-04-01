#!/usr/bin/env python3
"""
01_enrich_problem_schema_paper.py  (GRAPH-FIRST SCHEMA ENRICHMENT)

Input:  raw JSONL from 00_pdf-to-txt.py
Output: enriched JSONL with:
  - problem.stem + problem.choices
  - analysis.problem_graph: typed latent problem graph
  - analysis.graph_text: stable string form of the graph for retrieval
  - analysis.* tags (concepts/skills/difficulty/structure_tags)

This stage aligns the corpus with the newer graph-first retrieval/generation pipeline.
Legacy `analysis.skeleton` is still populated as a compatibility alias, but the graph is
the authoritative structural representation.
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
from ollama_client import generate_json, generate_text

load_dotenv()


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


NODE_TYPES = ["Given", "Target", "Law", "State", "Constraint", "Auxiliary", "Trap", "Distractor"]
EDGE_TYPES = ["supports", "depends_on", "derived_from", "couples", "rules_out", "produces_distractor"]


# ----------------------------
# LLM schema enrichment
# ----------------------------

SYSTEM = """You are an expert STEM competition solution analyst.

Return ONLY strict JSON.

Represent the problem as a typed latent problem graph, not as a step-by-step derivation.
Keep it abstract: no full arithmetic, no long derivations.
"""

USER_TMPL = """Domain: {domain}

Problem stem:
{stem}

Choices:
{choices_block}

Diagrams present: {has_diagram}
Diagram files: {diagram_files}

Allowed node types:
{node_types_block}

Allowed edge types:
{edge_types_block}

Return JSON with keys:
- problem_graph: object with keys:
  - nodes: list of objects {{id: string, type: string, label: string, importance: "primary"|"secondary"|"optional"}}
  - edges: list of objects {{src: string, dst: string, type: string, note: string}}
- graph_profile: object with keys:
  - target_summary: short string
  - state_evolution: list of short strings (1-6)
  - hidden_states: list of short strings (1-6)
  - coupling_points: list of short strings (1-6)
  - insight_type: list of short strings (1-4)
  - trap_profile: list of short strings (1-5)
  - reasoning_depth: integer 1-5
  - why_naive_method_fails: list of short strings (0-3)
- concepts: list of short strings (2-8)
- skills: list of short strings (0-8)
- difficulty: integer 1-10
- structure_tags: list of short strings (0-8)
- units_expected: boolean
- diagram_required: boolean
- quick_checks: list of short strings (0-6)

Guidelines:
- Do not compute numeric answers; keep it symbolic/structural.
- Include at least 1 Target node, at least 2 Law/Constraint nodes total, and at least 2 State nodes.
- Prefer including at least one nontrivial feature: Auxiliary or Trap or nontrivial Constraint coupling.
- Use graph_profile to capture contest complexity, especially hidden conditions, coupled equations, thresholds, and likely wrong paths.
- Make graph_profile concise and retrieval-friendly, not essay-like.
- diagram_required true ONLY if information is missing without the figure.
"""

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
        node_types_block="\n".join(f"- {o}" for o in NODE_TYPES),
        edge_types_block="\n".join(f"- {o}" for o in EDGE_TYPES),
    )

    try:
        return generate_json(model, user, system=SYSTEM, temperature=0.2)
    except Exception as exc:
        try:
            raw = generate_text(model, user, system=SYSTEM, temperature=0.2)
        except Exception:
            raise exc
        raise ValueError(f"Model returned invalid JSON. Raw output was:\n{raw}") from exc


# ----------------------------
# Schema builder
# ----------------------------

def normalize_graph(graph: Any) -> Dict[str, Any]:
    if not isinstance(graph, dict):
        return {"graph_type": "typed_problem_graph", "nodes": [], "edges": []}

    nodes: List[Dict[str, str]] = []
    seen: set[str] = set()
    for idx, node in enumerate(graph.get("nodes") or [], start=1):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or f"n{idx}").strip()[:40]
        if not node_id or node_id in seen:
            node_id = f"n{idx}"
        seen.add(node_id)

        node_type = str(node.get("type") or "State").strip()
        if node_type not in NODE_TYPES:
            node_type = "State"

        label = " ".join(str(node.get("label") or "").split())[:160]
        if not label:
            continue

        importance = str(node.get("importance") or "primary").strip().lower()
        if importance not in {"primary", "secondary", "optional"}:
            importance = "primary"

        nodes.append(
            {
                "id": node_id,
                "type": node_type,
                "label": label,
                "importance": importance,
            }
        )

    valid_ids = {node["id"] for node in nodes}
    edges: List[Dict[str, str]] = []
    for edge in graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        src = str(edge.get("src") or "").strip()[:40]
        dst = str(edge.get("dst") or "").strip()[:40]
        edge_type = str(edge.get("type") or "supports").strip()
        if src not in valid_ids or dst not in valid_ids:
            continue
        if edge_type not in EDGE_TYPES:
            edge_type = "supports"

        row = {"src": src, "dst": dst, "type": edge_type}
        note = " ".join(str(edge.get("note") or "").split())[:160]
        if note:
            row["note"] = note
        edges.append(row)

    return {"graph_type": "typed_problem_graph", "nodes": nodes, "edges": edges}



def normalize_profile(profile: Any) -> Dict[str, Any]:
    if not isinstance(profile, dict):
        profile = {}

    def _clean_list(key: str, limit: int) -> List[str]:
        vals = profile.get(key) or []
        out: List[str] = []
        if isinstance(vals, list):
            for v in vals:
                txt = " ".join(str(v).split())[:160]
                if txt:
                    out.append(txt)
        return out[:limit]

    target_summary = " ".join(str(profile.get("target_summary") or "").split())[:160]
    reasoning_depth = profile.get("reasoning_depth", 0)
    try:
        reasoning_depth = int(reasoning_depth)
    except Exception:
        reasoning_depth = 0
    if reasoning_depth < 1 or reasoning_depth > 5:
        reasoning_depth = 2

    return {
        "target_summary": target_summary,
        "state_evolution": _clean_list("state_evolution", 6),
        "hidden_states": _clean_list("hidden_states", 6),
        "coupling_points": _clean_list("coupling_points", 6),
        "insight_type": _clean_list("insight_type", 4),
        "trap_profile": _clean_list("trap_profile", 5),
        "reasoning_depth": reasoning_depth,
        "why_naive_method_fails": _clean_list("why_naive_method_fails", 3),
    }


def profile_to_text(profile: Dict[str, Any]) -> str:
    bits: List[str] = []
    if profile.get("target_summary"):
        bits.append(f"TARGET_SUMMARY: {profile['target_summary']}")
    if profile.get("state_evolution"):
        bits.append("STATE_EVOLUTION: " + " | ".join(profile["state_evolution"]))
    if profile.get("hidden_states"):
        bits.append("HIDDEN_STATES: " + " | ".join(profile["hidden_states"]))
    if profile.get("coupling_points"):
        bits.append("COUPLING_POINTS: " + " | ".join(profile["coupling_points"]))
    if profile.get("insight_type"):
        bits.append("INSIGHT_TYPE: " + " | ".join(profile["insight_type"]))
    if profile.get("trap_profile"):
        bits.append("TRAP_PROFILE: " + " | ".join(profile["trap_profile"]))
    bits.append(f"REASONING_DEPTH: {profile.get('reasoning_depth', 2)}")
    if profile.get("why_naive_method_fails"):
        bits.append("WHY_NAIVE_FAILS: " + " | ".join(profile["why_naive_method_fails"]))
    return "\n".join(bits).strip()

def graph_to_text(graph: Dict[str, Any]) -> str:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    groups = {node_type: [] for node_type in NODE_TYPES}
    labels: Dict[str, str] = {}

    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = str(node.get("type") or "State")
        node_id = str(node.get("id") or "").strip()
        label = str(node.get("label") or "").strip()
        if not node_id or not label or node_type not in groups:
            continue
        groups[node_type].append(f"{node_id}:{label}")
        labels[node_id] = label

    parts: List[str] = []
    for node_type in NODE_TYPES:
        if groups[node_type]:
            parts.append(f"{node_type.upper()}S: " + " | ".join(groups[node_type]))

    if edges:
        edge_bits = []
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            src = str(edge.get("src") or "").strip()
            dst = str(edge.get("dst") or "").strip()
            edge_type = str(edge.get("type") or "supports").strip()
            if not src or not dst:
                continue
            bit = f"{src}[{labels.get(src, src)}] -{edge_type}-> {dst}[{labels.get(dst, dst)}]"
            note = str(edge.get("note") or "").strip()
            if note:
                bit += f" ({note})"
            edge_bits.append(bit)
        if edge_bits:
            parts.append("EDGES: " + " | ".join(edge_bits))

    return "\n".join(parts).strip()

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
            "problem_graph": None,
            "graph_profile": None,
            "graph_text": "",
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
                "problem_graph": "analysis.problem_graph / analysis.graph_text",
                "skeleton": "analysis.graph_text (legacy alias)",
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
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--corpus", default="unknown")
    ap.add_argument("--year", type=int, default=0)
    ap.add_argument("--variant", default="")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--debug", action="store_true", help="print per-row progress to stderr")
    args = ap.parse_args()

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
            )

            if args.debug:
                dt = time.time() - t0
                print(f"[{i}/{total}] id={raw.get('id','?')} model done in {dt:.2f}s", file=sys.stderr, flush=True)
                print(f"[{i}/{total}] id={raw.get('id','?')} post-process start", file=sys.stderr, flush=True)

            problem_graph = normalize_graph(enrich.get("problem_graph") or {})
            graph_profile = normalize_profile(enrich.get("graph_profile") or {})
            graph_text_core = graph_to_text(problem_graph)
            graph_text_profile = profile_to_text(graph_profile)
            graph_text = "\n".join(x for x in [graph_text_core, graph_text_profile] if x).strip()

            schema["analysis"]["problem_graph"] = problem_graph
            schema["analysis"]["graph_profile"] = graph_profile
            schema["analysis"]["graph_text"] = graph_text
            schema["analysis"]["skeleton"] = graph_text
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
