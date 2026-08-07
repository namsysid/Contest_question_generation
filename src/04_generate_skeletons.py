#!/usr/bin/env python3
"""
04_generate_skeletons.py  (STAGE-1 GRAPH GENERATION)

Generates one new typed latent problem graph per retrieval bundle.
Backward compatibility:
- output file still named like generated_skeletons.jsonl
- includes `skeleton_text` alias equal to `graph_text`
"""
from __future__ import annotations
import argparse, json, os, sys, time
from typing import Any, Dict, List
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None
from ollama_client import generate_json

load_dotenv()

NODE_TYPES = ["Given", "Target", "Law", "State", "Constraint", "Auxiliary", "Trap", "Distractor"]
EDGE_TYPES = ["supports", "depends_on", "derived_from", "couples", "rules_out", "produces_distractor"]

DIFFICULTY_GUIDANCE = {
    "easy": """DIFFICULTY TARGET: easy
- Aim for level 1 difficulty.
- For math, use elementary to early-middle-school tasks with one direct idea.
- Prefer small whole numbers and one clear pattern, rule, or missing value.
- Keep the graph minimal: one target, one law/constraint, one state, and only necessary edges.""",
    "medium": """DIFFICULTY TARGET: medium
- Aim for level 2 difficulty.
- For math, use middle-school to early-high-school tasks with one or two linked ideas.
- Use a moderate dependency structure that matches the topic shown in the seed and exemplars.
- Keep the graph moderate without adding advanced algebra.""",
    "hard": """DIFFICULTY TARGET: hard
- Aim for level 3 difficulty.
- For math, use high-school-accessible challenge problems while staying within the seed/exemplar topic.
- Multi-step reasoning and carefully designed distractor traps are appropriate when the exemplars support them.
- Do not introduce unrelated advanced topics just to make the problem harder.""",
}
DIFFICULTY_LEVEL = {"easy": 1, "medium": 2, "hard": 3}

TOPIC_GUIDANCE_DEFAULT = """TOPIC GUIDANCE:
- Infer the topic from SEED_GRAPH_TEXT and SOLUTION EXEMPLARS.
- Preserve the topic family and dependency backbone; do not switch to another math topic.
- For math, avoid unrelated advanced topics unless they appear in the seed/exemplars."""

TOPIC_GUIDANCE_RATIO = """TOPIC GUIDANCE:
- Topic family: ratios, rates, proportions, equivalent ratios, ratio tables, or simple scaling.
- Keep the graph concrete: equivalent ratios, rate/unit-rate comparison, one missing value, or multi-step rate reasoning if hard mode asks for it.
- Do not switch to sequence, geometry, trigonometry, or unrelated algebra."""

TOPIC_GUIDANCE_RATIO_HARD = """TOPIC GUIDANCE:
- Topic family: rates, ratios, proportions, and rate-based algebraic modeling.
- Generate a contest-style challenge graph, not a longer worksheet computation.
- Prefer a non-obvious setup with variables, coupled linear equations, weighted averages, mixture/work/speed-rate constraints, or a hidden invariant.
- Include at least one Constraint/Auxiliary/Trap node that captures the modeling insight or a plausible wrong path.
- Algebra is allowed when it serves the rate/proportion topic; avoid unrelated advanced topics and avoid brute-force arithmetic-only graphs.
- The givens must be mutually consistent and must not already reveal the requested target value.
- Use exactly one primary target. Do not include extra targets unless they are explicitly needed to support the main target.
- Sanity-check the modeled equation before returning: it should have a valid positive solution and no given should make the scenario impossible."""

TOPIC_GUIDANCE_SEQUENCE = """TOPIC GUIDANCE:
- Topic family: number sequences and pattern rules.
- Keep the graph centered on identifying and applying the rule shown by the seed/exemplars.
- Preserve the specific sequence type in the seed/exemplars: arithmetic, geometric, constant, Fibonacci-like/add-previous-terms, alternating, or another explicit pattern.
- Do not force every sequence to be Fibonacci-like. Use a sum-of-previous-terms law only when the seed/exemplars support that rule.
- Do not switch to ratios, rates, proportions, geometry, or unrelated algebra."""

TOPIC_GUIDANCE_INEQUALITY = """TOPIC GUIDANCE:
- Topic family: linear inequalities in one variable, especially multi-step inequalities with variables on both sides.
- Keep the graph centered on isolating the variable while preserving/reversing the inequality direction correctly.
- Difficulty 2 should use one or two linked algebra steps: combine like terms, distribute a small integer or simple fraction, move variable terms across the inequality, then isolate the variable.
- Include a constraint or trap for the inequality sign when multiplying or dividing by a negative coefficient when the generated graph uses that step.
- The target should be an inequality solution in notation such as x > 4, y <= -3, or all real numbers/no solution when the structure supports it.
- Do not switch to ratios, rates, proportions, sequences, geometry, trigonometry, quadratic inequalities, compound inequalities, absolute value inequalities, systems, or word-problem modeling."""

SYSTEM = """You generate STRUCTURED typed latent problem graphs for STEM problems.

Hard constraints:
- Do NOT copy any exemplar graph verbatim.
- Keep it abstract: no full arithmetic, no long derivations.
- Include at least 1 Target node.
- For advanced STEM domains, include at least 2 Law/Constraint nodes total, at least 2 State nodes, and at least one Auxiliary, Trap, or Constraint coupling.
- For grade-school math, keep the graph smaller: one Law/Constraint and one State is enough when the source task is simple.
Return strict JSON only.
"""

USER_TMPL = """Domain: {domain}

SEED_GRAPH_TEXT (style prior; do not copy; be at least as complex):
{seed_graph}

SOLUTION EXEMPLARS (do not copy):
{solution_exemplars_block}

Allowed node types:
{node_types_block}

Allowed edge types:
{edge_types_block}

TASK:
Generate ONE NEW typed latent problem graph that is faithful to the source domain and exemplars.

{difficulty_guidance}

{topic_guidance}

Return strict JSON with keys:
- problem_graph: object with keys:
  - nodes: list of objects {{id: string, type: string, label: string, importance: "primary"|"secondary"|"optional"}}
  - edges: list of objects {{src: string, dst: string, type: string, note: string}}
- topic: short string or null
- difficulty: integer 1, 2, or 3 matching the requested difficulty target
"""


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl_line(f, row: Dict[str, Any]) -> None:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    f.flush()


def llm_json(model: str, system: str, user: str, temperature: float=0.25) -> Dict[str, Any]:
    return generate_json(model, user, system=system, temperature=temperature)


def llm_json_with_retries(
    model: str,
    system: str,
    user: str,
    *,
    temperature: float,
    retries: int,
    debug: bool,
    debug_prefix: str,
) -> Dict[str, Any]:
    last_exc: Exception | None = None
    max_attempts = max(1, retries + 1)
    for attempt in range(1, max_attempts + 1):
        try:
            return llm_json(model, system, user, temperature=temperature)
        except Exception as exc:
            last_exc = exc
            if debug:
                print(
                    f"{debug_prefix} JSON call failed attempt {attempt}/{max_attempts}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            if attempt < max_attempts:
                time.sleep(0.6 * attempt)
    assert last_exc is not None
    raise last_exc


def bundle_topic_text(bundle: Dict[str, Any], seed_graph: str, exemplars: List[Dict[str, Any]]) -> str:
    parts: List[str] = [
        str(bundle.get("domain") or ""),
        str(bundle.get("seed_topic") or ""),
        seed_graph,
    ]
    for key in ("question_exemplars", "paired_exemplars"):
        for ex in bundle.get(key) or []:
            if isinstance(ex, dict):
                parts.extend([
                    str(ex.get("topic") or ""),
                    str(ex.get("question_text") or ""),
                    str(ex.get("graph_text") or ex.get("skeleton_text") or ""),
                ])
    for ex in exemplars:
        if isinstance(ex, dict):
            parts.extend([
                str(ex.get("topic") or ""),
                str(ex.get("graph_text") or ex.get("skeleton_text") or ""),
            ])
    return " ".join(parts).lower()


def topic_guidance_for_bundle(bundle: Dict[str, Any], seed_graph: str, exemplars: List[Dict[str, Any]], difficulty: str) -> str:
    text = bundle_topic_text(bundle, seed_graph, exemplars)
    inequality_terms = (
        "inequality",
        "inequality notation",
        "linear inequality",
        "solve for",
        "variables on both sides",
        "greater than",
        "less than",
        "≥",
        "≤",
        ">=",
        "<=",
    )
    sequence_terms = ("fibonacci", "sequence", "recursive", "recurrence", "next term", "nth term", "a6", "geometric sequence", "arithmetic sequence")
    ratio_terms = ("ratio", "proportion", "unit rate", "scaling", "equivalent ratios")
    if any(term in text for term in inequality_terms):
        return TOPIC_GUIDANCE_INEQUALITY
    if any(term in text for term in sequence_terms):
        return TOPIC_GUIDANCE_SEQUENCE
    if any(term in text for term in ratio_terms):
        if difficulty == "hard":
            return TOPIC_GUIDANCE_RATIO_HARD
        return TOPIC_GUIDANCE_RATIO
    return TOPIC_GUIDANCE_DEFAULT


def normalize_graph(graph: Any) -> Dict[str, Any]:
    if not isinstance(graph, dict):
        return {"nodes": [], "edges": []}
    nodes=[]; seen=set()
    for idx, n in enumerate(graph.get("nodes") or [], start=1):
        if not isinstance(n, dict):
            continue
        nid = str(n.get("id") or f"n{idx}").strip()[:40]
        if not nid or nid in seen:
            nid = f"n{idx}"
        seen.add(nid)
        ntype = str(n.get("type") or "State").strip()
        if ntype not in NODE_TYPES:
            ntype = "State"
        label = " ".join(str(n.get("label") or "").split())[:160]
        if not label:
            continue
        importance = str(n.get("importance") or "primary").strip().lower()
        if importance not in {"primary", "secondary", "optional"}:
            importance = "primary"
        nodes.append({"id": nid, "type": ntype, "label": label, "importance": importance})
    valid = {n["id"] for n in nodes}
    edges=[]
    for e in graph.get("edges") or []:
        if not isinstance(e, dict):
            continue
        src = str(e.get("src") or "").strip()[:40]
        dst = str(e.get("dst") or "").strip()[:40]
        etype = str(e.get("type") or "supports").strip()
        if src not in valid or dst not in valid:
            continue
        if etype not in EDGE_TYPES:
            etype = "supports"
        row={"src": src, "dst": dst, "type": etype}
        note = " ".join(str(e.get("note") or "").split())[:160]
        if note:
            row["note"] = note
        edges.append(row)
    return {"nodes": nodes, "edges": edges}


def graph_text(graph: Dict[str, Any]) -> str:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    groups = {t: [] for t in NODE_TYPES}
    labels = {}
    for n in nodes:
        groups[n["type"]].append(f"{n['id']}:{n['label']}")
        labels[n["id"]] = n["label"]
    parts=[]
    for t in NODE_TYPES:
        if groups[t]:
            parts.append(f"{t.upper()}S: " + " | ".join(groups[t]))
    if edges:
        parts.append("EDGES: " + " | ".join(
            f"{e['src']}[{labels.get(e['src'], e['src'])}] -{e['type']}-> {e['dst']}[{labels.get(e['dst'], e['dst'])}]" + (f" ({e['note']})" if e.get("note") else "")
            for e in edges
        ))
    return "\n".join(parts).strip()


def graph_ok(graph: Dict[str, Any], domain: str = "") -> bool:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    counts = {t: 0 for t in NODE_TYPES}
    for n in nodes:
        if isinstance(n, dict) and n.get("type") in counts:
            counts[n["type"]] += 1
    if domain.strip().lower() == "math":
        return (
            counts["Target"] >= 1 and
            (counts["Law"] + counts["Constraint"]) >= 1 and
            counts["State"] >= 1 and
            len(edges) >= 2
        )
    return (
        counts["Target"] >= 1 and
        (counts["Law"] + counts["Constraint"]) >= 2 and
        counts["State"] >= 2 and
        (counts["Auxiliary"] >= 1 or counts["Trap"] >= 1 or counts["Constraint"] >= 2) and
        len(edges) >= 4
    )


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bundles", required=True)
    ap.add_argument("--out", default="./generated/generated_skeletons.jsonl")
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--domain", default="fma")
    ap.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json_retries", type=int, default=3, help="Retry count for malformed/non-JSON model responses")
    ap.add_argument("--debug", action="store_true", help="print per-row progress to stderr")
    args=ap.parse_args()

    rows=read_jsonl(args.bundles)
    if args.limit and args.limit>0:
        rows=rows[:args.limit]

    total=len(rows)

    if args.debug:
        print(f"[write] open (truncate): {args.out}", file=sys.stderr, flush=True)
    with open(args.out, "w", encoding="utf-8") as out_f:
        for i, b in enumerate(rows, start=1):
            domain=b.get("domain") or args.domain
            seed_graph=b.get("seed_graph_text") or b.get("seed_skeleton_text") or ""
            sol_ex=b.get("solution_exemplars") or []
            ex_blocks=[]
            for j,ex in enumerate(sol_ex[:8], start=1):
                ex_blocks.append(f"EX {j}:\n{ex.get('graph_text') or ex.get('skeleton_text','')}")
            sol_block="\n\n".join(ex_blocks) if ex_blocks else "(none)"

            user=USER_TMPL.format(
                domain=domain,
                seed_graph=seed_graph,
                solution_exemplars_block=sol_block,
                node_types_block="\n".join(f"- {o}" for o in NODE_TYPES),
                edge_types_block="\n".join(f"- {o}" for o in EDGE_TYPES),
                difficulty_guidance=DIFFICULTY_GUIDANCE[args.difficulty],
                topic_guidance=topic_guidance_for_bundle(b, seed_graph, sol_ex, args.difficulty),
            )

            if args.debug:
                bid=b.get("bundle_id","?")
                print(f"[{i}/{total}] bundle_id={bid} calling model={args.model}", file=sys.stderr, flush=True)
                t0=time.time()

            best=None
            for k in range(3):
                js=llm_json_with_retries(
                    args.model,
                    SYSTEM,
                    user,
                    temperature=0.25,
                    retries=args.json_retries,
                    debug=args.debug,
                    debug_prefix=f"[{i}/{total}] bundle_id={b.get('bundle_id','?')}",
                )
                graph = normalize_graph(js.get("problem_graph") or {})
                gt = graph_text(graph)
                ok = graph_ok(graph, domain=domain)
                best=(js, graph, gt, ok)
                if ok:
                    break
                if args.debug:
                    print(f"[{i}/{total}] bundle_id={b.get('bundle_id','?')} retry {k+1} constraints_satisfied={bool(ok)}", file=sys.stderr, flush=True)

            js, graph, gt, ok = best

            if args.debug:
                dt=time.time()-t0
                print(f"[{i}/{total}] bundle_id={b.get('bundle_id','?')} model done in {dt:.2f}s", file=sys.stderr, flush=True)

            out_row={
                "bundle_id": b.get("bundle_id"),
                "generated_graph": graph,
                "problem_graph": graph,
                "graph_text": gt,
                "skeleton_text": gt,
                "topic": js.get("topic"),
                "difficulty": DIFFICULTY_LEVEL[args.difficulty],
                "_meta": {"seed_id": b.get("seed_id"), "mode": b.get("mode"), "anchor_id": b.get("anchor_id"), "anchor_distance": b.get("anchor_distance")},
                "_diagnostics": {"constraints_satisfied": bool(ok)}
            }
            write_jsonl_line(out_f, out_row)
            if args.debug:
                print(f"[{i}/{total}] bundle_id={b.get('bundle_id','?')} wrote", file=sys.stderr, flush=True)

    if args.debug:
        print(f"[write] done: {args.out}", file=sys.stderr, flush=True)
    print(f"Wrote {total} generated graphs -> {args.out}")

if __name__=="__main__":
    main()
