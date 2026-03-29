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
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not set (env or .env).")

NODE_TYPES = ["Given", "Target", "Law", "State", "Constraint", "Auxiliary", "Trap", "Distractor"]
EDGE_TYPES = ["supports", "depends_on", "derived_from", "couples", "rules_out", "produces_distractor"]

SYSTEM = """You generate STRUCTURED typed latent problem graphs for Olympiad-style STEM problems.

Hard constraints:
- Do NOT copy any exemplar graph verbatim.
- Keep it abstract: no full arithmetic, no long derivations.
- The graph MUST be Olympiad-like and nontrivial, not one-equation.
- Include at least 1 Target node, at least 2 Law/Constraint nodes total, at least 2 State nodes.
- Include at least one nontrivial feature: Auxiliary or Trap or Constraint coupling.
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
Generate ONE NEW typed latent problem graph that is contest-faithful and nontrivial.

Return strict JSON with keys:
- problem_graph: object with keys:
  - nodes: list of objects {{id: string, type: string, label: string, importance: "primary"|"secondary"|"optional"}}
  - edges: list of objects {{src: string, dst: string, type: string, note: string}}
- topic: short string or null
- difficulty: integer 1-10 or null
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


def llm_json(client: OpenAI, model: str, system: str, user: str, temperature: float=0.25) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        response_format={"type":"json_object"},
    )
    return json.loads(resp.choices[0].message.content)


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


def graph_ok(graph: Dict[str, Any]) -> bool:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    counts = {t: 0 for t in NODE_TYPES}
    for n in nodes:
        if isinstance(n, dict) and n.get("type") in counts:
            counts[n["type"]] += 1
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
    ap.add_argument("--out", default="generated_skeletons.jsonl")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--domain", default="fma")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--debug", action="store_true", help="print per-row progress to stderr")
    args=ap.parse_args()

    rows=read_jsonl(args.bundles)
    if args.limit and args.limit>0:
        rows=rows[:args.limit]

    client=OpenAI()
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
            )

            if args.debug:
                bid=b.get("bundle_id","?")
                print(f"[{i}/{total}] bundle_id={bid} calling model={args.model}", file=sys.stderr, flush=True)
                t0=time.time()

            best=None
            for k in range(3):
                js=llm_json(client, args.model, SYSTEM, user, temperature=0.25)
                graph = normalize_graph(js.get("problem_graph") or {})
                gt = graph_text(graph)
                ok = graph_ok(graph)
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
                "difficulty": js.get("difficulty"),
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
