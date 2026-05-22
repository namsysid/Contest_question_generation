#!/usr/bin/env python3
"""
01_enrich_problem_schema_paper.py  (GRAPH-FIRST SCHEMA ENRICHMENT + GRAPH GRAMMAR)

Input:  raw JSONL from 00_pdf-to-txt.py
Output: enriched JSONL with:
  - problem.stem + problem.choices
  - analysis.problem_graph: typed latent problem graph
  - analysis.graph_text: stable string form of the graph for retrieval
  - analysis.graph_metrics: computed structural metrics
  - analysis.graph_constraints: grammar + budget parameters used
  - analysis.* tags (concepts/skills/difficulty/structure_tags)

This version adds:
  - explicit graph grammar (allowed node/edge arrangements)
  - configurable structural budgets (max depth / max branching / max nodes / max edges)
  - grammar enforcement modes: off | warn | strict
  - computed graph metrics for downstream bucketing experiments

Legacy `analysis.skeleton` is still populated as a compatibility alias, but the graph is
the authoritative structural representation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
LEADING_QNUM_RE = re.compile(r"^\s*\d+[\).]\s*", re.DOTALL)
FOOTER_JUNK_RE = re.compile(
    r"(Copyright.*?$|F\s*=\s*ma Exam.*?$)",
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
        d = str(raw["domain"]).strip().lower()
        if d:
            return d
    src = (raw.get("source_pdf") or "").lower()
    if "f" in src and "ma" in src:
        return "fma"
    if "usnco" in src or "chem" in src:
        return "chem"
    return "fma"


def _choice_list(raw_choices: Any) -> List[str]:
    if isinstance(raw_choices, dict):
        out = []
        for key in ("A", "B", "C", "D", "E"):
            if key in raw_choices:
                out.append(f"({key}) {raw_choices[key]}")
        return out
    if isinstance(raw_choices, list):
        return [str(choice).strip() for choice in raw_choices if str(choice).strip()]
    return []


def _raw_question_text(raw: Dict[str, Any]) -> str:
    if isinstance(raw.get("question_text"), str):
        return raw["question_text"]
    if isinstance(raw.get("question"), str):
        return raw["question"]
    problem = raw.get("problem")
    if isinstance(problem, dict):
        return str(problem.get("stem") or "")
    if isinstance(raw.get("prompt"), str):
        return raw["prompt"]
    return ""


NODE_TYPES = ["Given", "Target", "Law", "State", "Constraint", "Auxiliary", "Trap", "Distractor"]
EDGE_TYPES = ["supports", "depends_on", "derived_from", "couples", "rules_out", "produces_distractor"]
NODE_TYPE_SET = set(NODE_TYPES)
EDGE_TYPE_SET = set(EDGE_TYPES)

# Graph grammar: allowed (src_type, dst_type) pairs per edge type.
GRAPH_GRAMMAR: Dict[str, Set[Tuple[str, str]]] = {
    "supports": {
        ("Given", "Law"), ("Given", "State"), ("Given", "Constraint"), ("Given", "Auxiliary"), ("Given", "Target"),
        ("State", "Law"), ("State", "Constraint"), ("State", "Auxiliary"), ("State", "Target"),
        ("Law", "State"), ("Law", "Constraint"), ("Law", "Target"), ("Law", "Auxiliary"),
        ("Constraint", "Law"), ("Constraint", "State"), ("Constraint", "Target"),
        ("Auxiliary", "Law"), ("Auxiliary", "State"), ("Auxiliary", "Target"),
    },
    "depends_on": {
        ("Target", "Law"), ("Target", "State"), ("Target", "Constraint"), ("Target", "Auxiliary"), ("Target", "Given"),
        ("State", "Law"), ("State", "Constraint"), ("State", "Auxiliary"), ("State", "Given"),
        ("Law", "Constraint"), ("Law", "Given"), ("Law", "Auxiliary"),
        ("Auxiliary", "Law"), ("Auxiliary", "Constraint"), ("Auxiliary", "Given"),
    },
    "derived_from": {
        ("State", "Given"), ("State", "Law"), ("State", "Constraint"), ("State", "Auxiliary"), ("State", "State"),
        ("Auxiliary", "Given"), ("Auxiliary", "Law"), ("Auxiliary", "Constraint"), ("Auxiliary", "State"),
        ("Target", "State"), ("Target", "Law"), ("Target", "Auxiliary"),
    },
    "couples": {
        ("Constraint", "State"), ("Constraint", "Law"), ("Constraint", "Auxiliary"), ("Constraint", "Target"),
        ("State", "Constraint"), ("State", "State"), ("State", "Law"),
        ("Law", "Constraint"), ("Law", "State"),
        ("Auxiliary", "State"), ("Auxiliary", "Law"),
    },
    "rules_out": {
        ("Constraint", "Trap"), ("Law", "Trap"), ("State", "Trap"), ("Given", "Trap"), ("Auxiliary", "Trap"),
        ("Constraint", "Distractor"), ("Law", "Distractor"), ("State", "Distractor"),
    },
    "produces_distractor": {
        ("Trap", "Distractor"), ("Law", "Distractor"), ("State", "Distractor"), ("Auxiliary", "Distractor"),
    },
}

DEFAULT_LIMITS = {
    "max_depth": 10,
    "max_branching": 10,
    "max_nodes": 100,
    "max_edges": 100,
}


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

Graph grammar (treat as hard in spirit; strict postprocessing may enforce it):
{grammar_block}

Structural budgets:
- max depth: {max_depth}
- max outgoing branching per node: {max_branching}
- max nodes: {max_nodes}
- max edges: {max_edges}

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
- Include at least 1 Target node, at least 2 Law/Constraint nodes total, and at least 2 State nodes unless the problem is truly degenerate.
- Prefer including at least one nontrivial feature: Auxiliary or Trap or nontrivial Constraint coupling.
- Keep the graph compact and grammar-consistent.
- Favor a single coherent dependency structure over many loose nodes.
- Use graph_profile to capture contest complexity, especially hidden conditions, coupled equations, thresholds, and likely wrong paths.
- Make graph_profile concise and retrieval-friendly, not essay-like.
- diagram_required true ONLY if information is missing without the figure.
"""


def format_grammar_block() -> str:
    lines: List[str] = []
    for edge_type in EDGE_TYPES:
        allowed = sorted(GRAPH_GRAMMAR[edge_type])
        pairs = ", ".join(f"{src}->{dst}" for src, dst in allowed)
        lines.append(f"- {edge_type}: {pairs}")
    return "\n".join(lines)


def llm_enrich(
    domain: str,
    stem: str,
    choices: List[str],
    has_diagram: bool,
    diagram_files: List[str],
    model: str,
    limits: Dict[str, Optional[int]],
) -> Dict[str, Any]:
    def budget_text(value: Optional[int]) -> str:
        return "unbounded" if value is None else str(value)

    choices_block = "\n".join(choices) if choices else "(none)"
    user = USER_TMPL.format(
        domain=domain,
        stem=stem,
        choices_block=choices_block,
        has_diagram=str(bool(has_diagram)).lower(),
        diagram_files=json.dumps(diagram_files),
        node_types_block="\n".join(f"- {o}" for o in NODE_TYPES),
        edge_types_block="\n".join(f"- {o}" for o in EDGE_TYPES),
        grammar_block=format_grammar_block(),
        max_depth=budget_text(limits["max_depth"]),
        max_branching=budget_text(limits["max_branching"]),
        max_nodes=budget_text(limits["max_nodes"]),
        max_edges=budget_text(limits["max_edges"]),
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
# Graph grammar / metrics
# ----------------------------


def edge_respects_grammar(src_type: str, dst_type: str, edge_type: str) -> bool:
    return (src_type, dst_type) in GRAPH_GRAMMAR.get(edge_type, set())



def compute_graph_metrics(graph: Dict[str, Any]) -> Dict[str, Any]:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    id_to_type = {n["id"]: n["type"] for n in nodes if isinstance(n, dict) and "id" in n and "type" in n}

    out_adj: Dict[str, List[str]] = defaultdict(list)
    in_adj: Dict[str, List[str]] = defaultdict(list)
    edge_type_counts: Counter[str] = Counter()
    for e in edges:
        if not isinstance(e, dict):
            continue
        src, dst, et = e.get("src"), e.get("dst"), e.get("type")
        if src in id_to_type and dst in id_to_type:
            out_adj[src].append(dst)
            in_adj[dst].append(src)
            edge_type_counts[str(et)] += 1

    node_type_counts: Counter[str] = Counter(str(n.get("type")) for n in nodes if isinstance(n, dict))
    max_out = max((len(v) for v in out_adj.values()), default=0)
    avg_out = round(sum(len(v) for v in out_adj.values()) / max(len(nodes), 1), 3)

    sources = [n["id"] for n in nodes if n.get("type") == "Given"]
    targets = [n["id"] for n in nodes if n.get("type") == "Target"]

    # Longest source-to-target depth in DAG-like approximation via BFS from all sources.
    max_depth = 0
    if sources and targets:
        for s in sources:
            dq = deque([(s, 0)])
            seen_depth: Dict[str, int] = {s: 0}
            while dq:
                cur, d = dq.popleft()
                if cur in targets:
                    max_depth = max(max_depth, d)
                if d > len(nodes):
                    continue
                for nxt in out_adj.get(cur, []):
                    nd = d + 1
                    if nd > seen_depth.get(nxt, -1):
                        seen_depth[nxt] = nd
                        dq.append((nxt, nd))

    # Weakly connected components count.
    undirected: Dict[str, Set[str]] = defaultdict(set)
    for e in edges:
        if not isinstance(e, dict):
            continue
        src, dst = e.get("src"), e.get("dst")
        if src in id_to_type and dst in id_to_type:
            undirected[src].add(dst)
            undirected[dst].add(src)

    components = 0
    visited: Set[str] = set()
    for nid in id_to_type:
        if nid in visited:
            continue
        components += 1
        dq = deque([nid])
        visited.add(nid)
        while dq:
            cur = dq.popleft()
            for nxt in undirected.get(cur, set()):
                if nxt not in visited:
                    visited.add(nxt)
                    dq.append(nxt)

    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_type_counts": dict(node_type_counts),
        "edge_type_counts": dict(edge_type_counts),
        "max_branching": max_out,
        "avg_branching": avg_out,
        "target_depth": max_depth,
        "connected_components": components,
    }



def prune_to_limits(graph: Dict[str, Any], limits: Dict[str, Optional[int]]) -> Tuple[Dict[str, Any], List[str]]:
    """Prune softly toward useful structure: keep core node types, then important nodes, then connected edges."""
    notes: List[str] = []
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])

    # Rank nodes so target/state/law/constraint survive first.
    type_rank = {
        "Target": 0,
        "State": 1,
        "Law": 2,
        "Constraint": 3,
        "Given": 4,
        "Auxiliary": 5,
        "Trap": 6,
        "Distractor": 7,
    }
    importance_rank = {"primary": 0, "secondary": 1, "optional": 2}
    nodes_sorted = sorted(
        nodes,
        key=lambda n: (
            importance_rank.get(str(n.get("importance", "primary")), 0),
            type_rank.get(str(n.get("type", "State")), 99),
            str(n.get("id", "")),
        ),
    )

    max_nodes = limits.get("max_nodes")
    if max_nodes is not None and len(nodes_sorted) > max_nodes:
        kept_nodes = nodes_sorted[: max_nodes]
        notes.append(f"pruned_nodes:{len(nodes_sorted) - len(kept_nodes)}")
    else:
        kept_nodes = nodes_sorted

    kept_ids = {n["id"] for n in kept_nodes}
    edges = [e for e in edges if e.get("src") in kept_ids and e.get("dst") in kept_ids]

    # Enforce branching cap.
    out_count: Dict[str, int] = defaultdict(int)
    pruned_edges: List[Dict[str, Any]] = []
    removed_branch = 0
    max_branching = limits.get("max_branching")
    for e in edges:
        src = str(e.get("src") or "")
        if max_branching is not None and out_count[src] >= max_branching:
            removed_branch += 1
            continue
        out_count[src] += 1
        pruned_edges.append(e)
    edges = pruned_edges
    if removed_branch:
        notes.append(f"pruned_branching_edges:{removed_branch}")

    # Enforce edge cap.
    max_edges = limits.get("max_edges")
    if max_edges is not None and len(edges) > max_edges:
        notes.append(f"pruned_edges:{len(edges) - max_edges}")
        edges = edges[: max_edges]

    graph = {"graph_type": "typed_problem_graph", "nodes": kept_nodes, "edges": edges}

    # Enforce depth cap by iteratively dropping edges that push farthest target paths.
    max_depth = limits.get("max_depth")
    if max_depth is not None and max_depth >= 0:
        safety = 0
        while safety < 50:
            metrics = compute_graph_metrics(graph)
            if metrics["target_depth"] <= max_depth:
                break
            if not graph["edges"]:
                break
            removed = graph["edges"].pop()  # deterministic last-edge trim after prior sorting/preservation
            notes.append(f"pruned_for_depth:{removed.get('src')}->{removed.get('dst')}:{removed.get('type')}")
            safety += 1

    return graph, notes



def normalize_graph(
    graph: Any,
    grammar_mode: str = "warn",
    limits: Optional[Dict[str, Optional[int]]] = None,
) -> Tuple[Dict[str, Any], List[str], Dict[str, Any]]:
    if limits is None:
        limits = dict(DEFAULT_LIMITS)
    if not isinstance(graph, dict):
        empty = {"graph_type": "typed_problem_graph", "nodes": [], "edges": []}
        return empty, ["invalid_graph_object"], compute_graph_metrics(empty)

    nodes: List[Dict[str, str]] = []
    seen: Set[str] = set()
    notes: List[str] = []

    for idx, node in enumerate(graph.get("nodes") or [], start=1):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or f"n{idx}").strip()[:40]
        if not node_id or node_id in seen:
            node_id = f"n{idx}"
        seen.add(node_id)

        node_type = str(node.get("type") or "State").strip()
        if node_type not in NODE_TYPE_SET:
            notes.append(f"coerced_node_type:{node_id}:{node_type}->State")
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
    id_to_type = {node["id"]: node["type"] for node in nodes}
    edges: List[Dict[str, str]] = []

    for edge in graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        src = str(edge.get("src") or "").strip()[:40]
        dst = str(edge.get("dst") or "").strip()[:40]
        edge_type = str(edge.get("type") or "supports").strip()
        if src not in valid_ids or dst not in valid_ids:
            notes.append(f"dropped_edge_missing_node:{src}->{dst}:{edge_type}")
            continue
        if edge_type not in EDGE_TYPE_SET:
            notes.append(f"coerced_edge_type:{src}->{dst}:{edge_type}->supports")
            edge_type = "supports"

        if not edge_respects_grammar(id_to_type[src], id_to_type[dst], edge_type):
            msg = f"grammar_violation:{src}[{id_to_type[src]}]-{edge_type}->{dst}[{id_to_type[dst]}]"
            if grammar_mode == "strict":
                notes.append(f"dropped_{msg}")
                continue
            if grammar_mode == "warn":
                notes.append(msg)

        row = {"src": src, "dst": dst, "type": edge_type}
        note = " ".join(str(edge.get("note") or "").split())[:160]
        if note:
            row["note"] = note
        edges.append(row)

    norm = {"graph_type": "typed_problem_graph", "nodes": nodes, "edges": edges}
    norm, prune_notes = prune_to_limits(norm, limits)
    notes.extend(prune_notes)
    metrics = compute_graph_metrics(norm)
    return norm, notes, metrics



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



def graph_to_text(graph: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None) -> str:
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

    if metrics:
        parts.append(
            "GRAPH_METRICS: "
            f"nodes={metrics.get('node_count', 0)} | "
            f"edges={metrics.get('edge_count', 0)} | "
            f"depth={metrics.get('target_depth', 0)} | "
            f"max_branching={metrics.get('max_branching', 0)} | "
            f"components={metrics.get('connected_components', 0)}"
        )

    return "\n".join(parts).strip()


# ----------------------------
# Schema builder
# ----------------------------


def build_schema(raw: Dict[str, Any], corpus_name: str = "unknown", year: Optional[int] = None, variant: Optional[str] = None) -> Dict[str, Any]:
    domain = infer_domain(raw)
    stem, choices, answer_key = split_stem_and_choices(_raw_question_text(raw))
    if not choices:
        structured_choices = raw.get("choices")
        if structured_choices is None and isinstance(raw.get("problem"), dict):
            structured_choices = raw["problem"].get("choices")
        choices = _choice_list(structured_choices)
    problem_answer = None
    if isinstance(raw.get("problem"), dict):
        problem_answer = raw["problem"].get("answer_key")
    answer_key = answer_key or raw.get("answer") or raw.get("answer_key") or problem_answer

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
            "graph_metrics": None,
            "graph_constraints": None,
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
                "problem_graph": "analysis.problem_graph / analysis.graph_text / analysis.graph_metrics",
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
    ap.add_argument("--grammar-mode", choices=["off", "warn", "strict"], default="strict")
    ap.add_argument("--max-depth", type=int, default=None, help="optional cap; omit for unbounded")
    ap.add_argument("--max-branching", type=int, default=None, help="optional cap; omit for unbounded")
    ap.add_argument("--max-nodes", type=int, default=None, help="optional cap; omit for unbounded")
    ap.add_argument("--max-edges", type=int, default=None, help="optional cap; omit for unbounded")
    ap.add_argument("--debug", action="store_true", help="print per-row progress to stderr")
    args = ap.parse_args()

    limits: Dict[str, Optional[int]] = {
        "max_depth": (max(0, args.max_depth) if args.max_depth is not None else None),
        "max_branching": (max(1, args.max_branching) if args.max_branching is not None else None),
        "max_nodes": (max(1, args.max_nodes) if args.max_nodes is not None else None),
        "max_edges": (max(0, args.max_edges) if args.max_edges is not None else None),
    }

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

            enrich = llm_enrich(
                domain=domain,
                stem=stem,
                choices=choices,
                has_diagram=has_diagram,
                diagram_files=diagram_files,
                model=args.model,
                limits=limits,
            )

            if args.debug:
                dt = time.time() - t0
                print(f"[{i}/{total}] id={raw.get('id','?')} model done in {dt:.2f}s", file=sys.stderr, flush=True)
                print(f"[{i}/{total}] id={raw.get('id','?')} post-process start", file=sys.stderr, flush=True)

            problem_graph, validator_notes, metrics = normalize_graph(
                enrich.get("problem_graph") or {},
                grammar_mode=args.grammar_mode,
                limits=limits,
            )
            graph_profile = normalize_profile(enrich.get("graph_profile") or {})
            graph_text_core = graph_to_text(problem_graph, metrics=metrics)
            graph_text_profile = profile_to_text(graph_profile)
            graph_text = "\n".join(x for x in [graph_text_core, graph_text_profile] if x).strip()

            schema["analysis"]["problem_graph"] = problem_graph
            schema["analysis"]["graph_profile"] = graph_profile
            schema["analysis"]["graph_text"] = graph_text
            schema["analysis"]["graph_metrics"] = metrics
            schema["analysis"]["graph_constraints"] = {
                "grammar_mode": args.grammar_mode,
                "limits": limits,
                "grammar_version": "v1",
            }
            schema["analysis"]["skeleton"] = graph_text
            schema["analysis"]["concepts"] = enrich.get("concepts") or []
            schema["analysis"]["skills"] = enrich.get("skills") or []
            schema["analysis"]["difficulty"] = enrich.get("difficulty")
            schema["analysis"]["structure_tags"] = enrich.get("structure_tags") or []
            schema["problem"]["units_expected"] = enrich.get("units_expected")
            schema["analysis"]["diagram_required"] = bool(enrich.get("diagram_required"))
            schema["checks"]["quick_checks"] = enrich.get("quick_checks") or []
            schema["checks"]["validators"] = validator_notes

            write_jsonl_line(out_f, schema)
            if args.debug:
                print(f"[{i}/{total}] id={raw.get('id','?')} wrote", file=sys.stderr, flush=True)

    if args.debug:
        print(f"[write] done: {args.out}", file=sys.stderr, flush=True)
    print(f"Wrote {total} enriched rows -> {args.out}")


if __name__ == "__main__":
    main()
