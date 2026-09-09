#!/usr/bin/env python3
"""
02_embed_and_index.py  (ANCHOR-BASED, DUAL-SPACE, GRAPH-FIRST)

Outputs:
  - skeleton_embedded.jsonl  (legacy filename; now stores graph-first structural view)
  - question_embedded.jsonl  (id + question fields + embedding)
  - anchors.jsonl            (top-density anchor ids; NOT to be shown to the model)

This stage now treats the structural retrieval space as a TYPED PROBLEM GRAPH space.
For backward compatibility, it still writes `skeleton` / `skeleton_text` aliases so
older downstream code can continue to run.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
from dotenv import load_dotenv
from circuit_lab.model_client import embed_texts as model_embed_texts

load_dotenv()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _string_list(values: Any, limit: int | None = None) -> List[str]:
    if not isinstance(values, list):
        return []
    out = [str(v).strip() for v in values if str(v).strip()]
    return out[:limit] if limit is not None else out


def _compact_node(node: Dict[str, Any]) -> str:
    nid = str(node.get("id", "")).strip()
    ntype = str(node.get("type", "")).strip()
    label = str(node.get("label", "")).strip()
    bits = [b for b in [nid, ntype, label] if b]
    return ":".join(bits)


def _compact_edge(edge: Dict[str, Any]) -> str:
    src = str(edge.get("src", "")).strip()
    etype = str(edge.get("type", "")).strip()
    dst = str(edge.get("dst", "")).strip()
    if src and etype and dst:
        return f"{src}-{etype}->{dst}"
    return ""


def safe_get_graph_text(r: Dict[str, Any]) -> str:
    """
    Convert typed problem graph structure into a stable embedding text.

    Expected preferred inputs:
      - r["graph_text"] as string
      - r["problem_graph"] as dict
      - r["analysis"]["graph_text"] / r["analysis"]["problem_graph"]

    Fallbacks:
      - legacy skeleton text / skeleton object
    """
    direct_graph_text = r.get("graph_text")
    if isinstance(direct_graph_text, str) and direct_graph_text.strip():
        return direct_graph_text.strip()

    analysis = r.get("analysis") or {}
    if isinstance(analysis, dict):
        analysis_graph_text = analysis.get("graph_text")
        if isinstance(analysis_graph_text, str) and analysis_graph_text.strip():
            return analysis_graph_text.strip()

    graph = r.get("problem_graph")
    if graph is None and isinstance(analysis, dict):
        graph = analysis.get("problem_graph")

    if isinstance(graph, dict):
        parts: List[str] = []
        graph_type = str(graph.get("graph_type", "typed_problem_graph")).strip()
        if graph_type:
            parts.append(f"GRAPH_TYPE: {graph_type}")

        target = str(graph.get("target", "")).strip()
        if target:
            parts.append(f"TARGET: {target}")

        laws = _string_list(graph.get("laws"), limit=8)
        if laws:
            parts.append("LAWS: " + ", ".join(laws))

        nodes = graph.get("nodes") or []
        if isinstance(nodes, list) and nodes:
            node_bits = [_compact_node(n) for n in nodes if isinstance(n, dict)]
            node_bits = [b for b in node_bits if b]
            if node_bits:
                parts.append("NODES: " + " | ".join(node_bits[:32]))

        edges = graph.get("edges") or []
        if isinstance(edges, list) and edges:
            edge_bits = [_compact_edge(e) for e in edges if isinstance(e, dict)]
            edge_bits = [b for b in edge_bits if b]
            if edge_bits:
                parts.append("EDGES: " + " | ".join(edge_bits[:48]))

        traps = _string_list(graph.get("traps"), limit=8)
        if traps:
            parts.append("TRAPS: " + " | ".join(traps))

        distractors = _string_list(graph.get("distractor_causes"), limit=8)
        if distractors:
            parts.append("DISTRACTORS: " + " | ".join(distractors))

        if parts:
            return "\n".join(parts).strip()

    return safe_get_skeleton_text(r)


def safe_get_skeleton_text(r: Dict[str, Any]) -> str:
    """
    Backward-compatible structural text extraction.
    Used as a fallback and also written as an alias for downstream scripts.
    """
    sk = r.get("skeleton", "")
    if isinstance(sk, str):
        return sk.strip()

    if isinstance(sk, dict):
        optypes = sk.get("optypes", []) or []
        laws = sk.get("laws", []) or []
        steps = sk.get("steps", []) or []
        parts = []
        if optypes:
            parts.append("OPTYPES: " + ", ".join(map(str, optypes)))
        if laws:
            parts.append("LAWS: " + ", ".join(map(str, laws)))
        if steps:
            steps_txt = " | ".join(map(str, steps))
            parts.append("STEPS: " + steps_txt)
        return "\n".join(parts).strip()

    analysis = r.get("analysis") or {}
    if isinstance(analysis, dict):
        legacy = analysis.get("skeleton")
        if isinstance(legacy, str):
            return legacy.strip()
        if isinstance(legacy, dict):
            laws = _string_list(legacy.get("laws"))
            steps = legacy.get("steps") or []
            parts = []
            if laws:
                parts.append("LAWS: " + ", ".join(laws))
            if isinstance(steps, list) and steps:
                bits = []
                for st in steps:
                    if isinstance(st, dict):
                        op = str(st.get("op", "")).strip()
                        tx = str(st.get("text", "")).strip()
                        if op and tx:
                            bits.append(f"{op}:{tx}")
                if bits:
                    parts.append("STEPS: " + " | ".join(bits))
            if parts:
                return "\n".join(parts).strip()

    return str(sk).strip()


def safe_get_question_text(r: Dict[str, Any]) -> str:
    q = r.get("question", "")
    if q:
        return str(q).strip()

    problem = r.get("problem", {})
    if isinstance(problem, dict):
        stem = problem.get("stem", "") or ""
        choices = problem.get("choices", []) or []
        parts: List[str] = []
        if stem:
            parts.append(stem.strip())
        if choices:
            choices_str = " ".join(str(c).strip() for c in choices if c)
            if choices_str:
                parts.append(choices_str)
        return " ".join(parts).strip()

    return ""


def embed_texts(model: str, texts: List[str], batch: int = 64, provider: str = "ollama") -> List[List[float]]:
    embedding_dim = None
    embs: List[List[float]] = []

    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        non_empty_chunk = []
        non_empty_indices = []
        for j, text in enumerate(chunk):
            if text and text.strip():
                non_empty_chunk.append(text)
                non_empty_indices.append(j)

        if non_empty_chunk:
            resp = model_embed_texts(model, non_empty_chunk, provider=provider)
            if embedding_dim is None and resp:
                embedding_dim = len(resp[0])

            chunk_embs = [None] * len(chunk)
            for idx, emb in zip(non_empty_indices, resp):
                chunk_embs[idx] = emb
            zero_vec = [0.0] * embedding_dim if embedding_dim else []
            for j in range(len(chunk_embs)):
                if chunk_embs[j] is None:
                    chunk_embs[j] = zero_vec
            embs.extend(chunk_embs)
        else:
            if embedding_dim is None:
                test_resp = model_embed_texts(model, ["test"], provider=provider)
                embedding_dim = len(test_resp[0])
            zero_vec = [0.0] * embedding_dim
            embs.extend([zero_vec for _ in chunk])
    return embs


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / denom


def knn_density_scores(X: np.ndarray, k: int = 20) -> np.ndarray:
    Xn = l2_normalize(X.astype(np.float32))
    sims = Xn @ Xn.T
    np.fill_diagonal(sims, -1.0)
    k = max(1, min(k, max(1, X.shape[0] - 1)))
    topk = np.partition(sims, -k, axis=1)[:, -k:]
    dists = 1.0 - topk
    mean_dist = dists.mean(axis=1)
    return -mean_dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Enriched problems JSONL (has question + graph/skeleton)")
    ap.add_argument("--embed_model", default="qwen3-embedding")
    ap.add_argument("--provider", choices=["ollama", "openai"], default="ollama")
    ap.add_argument("--anchor_frac", type=float, default=0.18, help="15–20% recommended (e.g. 0.15–0.20)")
    ap.add_argument("--k_density", type=int, default=20, help="kNN size for density proxy")
    ap.add_argument("--out_skel", default="./data/skeleton_embedded.jsonl")
    ap.add_argument("--out_q", default="./data/question_embedded.jsonl")
    ap.add_argument("--out_anchors", default="./data/anchors.jsonl")
    args = ap.parse_args()

    load_dotenv()
    rows = read_jsonl(args.input)
    if not rows:
        raise RuntimeError("No rows found in input JSONL.")

    graph_texts = [safe_get_graph_text(r) for r in rows]
    q_texts = [safe_get_question_text(r) for r in rows]

    graph_embs = embed_texts(args.embed_model, graph_texts, provider=args.provider)
    q_embs = embed_texts(args.embed_model, q_texts, provider=args.provider)

    sk_out: List[Dict[str, Any]] = []
    q_out: List[Dict[str, Any]] = []
    for r, e_graph, e_q, graph_txt, q_txt in zip(rows, graph_embs, q_embs, graph_texts, q_texts):
        rid = r.get("id")
        if not rid:
            raise RuntimeError("Missing id in a row; please ensure ids exist.")

        analysis = r.get("analysis") or {}
        problem_graph = r.get("problem_graph")
        if problem_graph is None and isinstance(analysis, dict):
            problem_graph = analysis.get("problem_graph")

        sk_row: Dict[str, Any] = {
            "id": rid,
            "problem_graph": problem_graph,
            "graph_text": graph_txt,
            # compatibility aliases for downstream code still expecting skeleton fields
            "skeleton": problem_graph,
            "skeleton_text": graph_txt,
            "embedding": e_graph,
        }
        for k in ["topic", "subtopic", "difficulty", "source"]:
            if k in r:
                sk_row[k] = r[k]
        if isinstance(analysis, dict):
            concepts = analysis.get("concepts") or []
            difficulty = analysis.get("difficulty")
            if "topic" not in sk_row and isinstance(concepts, list) and concepts:
                sk_row["topic"] = concepts[0]
            if "difficulty" not in sk_row and difficulty is not None:
                sk_row["difficulty"] = difficulty
        sk_out.append(sk_row)

        q_row: Dict[str, Any] = {
            "id": rid,
            "question": r.get("question"),
            "question_text": q_txt,
            "embedding": e_q,
        }
        for k in ["topic", "subtopic", "difficulty", "source"]:
            if k in r:
                q_row[k] = r[k]
        if isinstance(analysis, dict):
            concepts = analysis.get("concepts") or []
            difficulty = analysis.get("difficulty")
            if "topic" not in q_row and isinstance(concepts, list) and concepts:
                q_row["topic"] = concepts[0]
            if "difficulty" not in q_row and difficulty is not None:
                q_row["difficulty"] = difficulty
        q_out.append(q_row)

    write_jsonl(args.out_skel, sk_out)
    write_jsonl(args.out_q, q_out)

    X = np.array(graph_embs, dtype=np.float32)
    dens = knn_density_scores(X, k=args.k_density)

    n = len(rows)
    n_anchor = max(1, int(round(args.anchor_frac * n)))
    anchor_idx = np.argsort(-dens)[:n_anchor]

    anchors: List[Dict[str, Any]] = []
    for i in anchor_idx:
        anchors.append({
            "id": rows[i]["id"],
            "density": float(dens[i]),
            "anchor_space": "problem_graph",
        })

    write_jsonl(args.out_anchors, anchors)

    print(f"Wrote: {args.out_skel}, {args.out_q}, {args.out_anchors}")
    print(f"Anchors: {n_anchor}/{n} ({args.anchor_frac:.2%})")


if __name__ == "__main__":
    main()
