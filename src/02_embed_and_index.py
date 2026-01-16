#!/usr/bin/env python3
"""
02_embed_and_index.py  (ANCHOR-BASED, DUAL-SPACE)

Outputs:
  - skeleton_embedded.jsonl  (id + skeleton fields + embedding)
  - question_embedded.jsonl  (id + question fields + embedding)
  - anchors.json             (top-density anchor ids; NOT to be shown to the model)

Anchor idea:
  High-density skeletons define the manifold center.
  They are used only to guide sampling (centers), never pasted into prompts.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def read_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_get_skeleton_text(r: Dict) -> str:
    """
    Convert your skeleton structure into a stable embedding text.

    Expect either:
      - r["skeleton"] as string
      - or dict like {"optypes":[...], "laws":[...], "steps":[...]}
    """
    sk = r.get("skeleton", "")
    if isinstance(sk, str):
        return sk.strip()

    if isinstance(sk, dict):
        optypes = sk.get("optypes", []) or []
        laws = sk.get("laws", []) or []
        steps = sk.get("steps", []) or []
        # keep it abstract + stable
        parts = []
        if optypes:
            parts.append("OPTYPES: " + ", ".join(map(str, optypes)))
        if laws:
            parts.append("LAWS: " + ", ".join(map(str, laws)))
        if steps:
            # steps can be long; keep short but informative
            steps_txt = " | ".join(map(str, steps))
            parts.append("STEPS: " + steps_txt)
        return "\n".join(parts).strip()

    return str(sk).strip()


def safe_get_question_text(r: Dict) -> str:
    """
    Extract question text from enriched schema.
    Expects either:
      - r["question"] as string (legacy)
      - or r["problem"]["stem"] + r["problem"]["choices"] (enriched schema)
    """
    # Try legacy format first
    q = r.get("question", "")
    if q:
        return str(q).strip()
    
    # Try enriched schema format
    problem = r.get("problem", {})
    if isinstance(problem, dict):
        stem = problem.get("stem", "") or ""
        choices = problem.get("choices", []) or []
        parts = []
        if stem:
            parts.append(stem.strip())
        if choices:
            choices_str = " ".join(str(c).strip() for c in choices if c)
            if choices_str:
                parts.append(choices_str)
        return " ".join(parts).strip()
    
    return ""


def embed_texts(client: OpenAI, model: str, texts: List[str], batch: int = 64) -> List[List[float]]:
    """
    Embed texts, handling empty strings by filtering them out before API calls.
    Returns embeddings in the same order as input texts, with zero vectors for empty strings.
    """
    embedding_dim = None
    embs: List[List[float]] = []
    
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        # Filter out empty strings and track their positions
        non_empty_chunk = []
        non_empty_indices = []
        for j, text in enumerate(chunk):
            if text and text.strip():  # Non-empty string
                non_empty_chunk.append(text)
                non_empty_indices.append(j)
        
        if non_empty_chunk:
            resp = client.embeddings.create(model=model, input=non_empty_chunk)
            
            # Get embedding dimension from first response if not already known
            if embedding_dim is None and resp.data:
                embedding_dim = len(resp.data[0].embedding)
            
            # Map embeddings back to original positions
            chunk_embs = [None] * len(chunk)
            for idx, emb in zip(non_empty_indices, resp.data):
                chunk_embs[idx] = emb.embedding
            # Fill empty positions with zero vectors
            zero_vec = [0.0] * embedding_dim if embedding_dim else []
            for j in range(len(chunk_embs)):
                if chunk_embs[j] is None:
                    chunk_embs[j] = zero_vec
            embs.extend(chunk_embs)
        else:
            # All texts in chunk were empty - need dimension for zero vectors
            if embedding_dim is None:
                # Make a test call to get dimension
                test_resp = client.embeddings.create(model=model, input=["test"])
                embedding_dim = len(test_resp.data[0].embedding)
            zero_vec = [0.0] * embedding_dim
            embs.extend([zero_vec for _ in chunk])
    return embs


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / denom


def knn_density_scores(X: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Density proxy: negative mean distance to k nearest neighbors (cosine distance).
    Higher score => denser region.

    We do brute-force cosine for simplicity (fine for few hundred/thousand).
    """
    Xn = l2_normalize(X.astype(np.float32))
    # cosine distance = 1 - cosine similarity
    sims = Xn @ Xn.T  # (N,N)
    # exclude self by setting diag to -inf similarity (=> +inf distance)
    np.fill_diagonal(sims, -1.0)

    # take top-k most similar => lowest distances
    topk = np.partition(sims, -k, axis=1)[:, -k:]  # (N,k) similarities
    # convert to distances
    dists = 1.0 - topk
    mean_dist = dists.mean(axis=1)
    # density score: higher is denser
    return -mean_dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Enriched problems JSONL (has question + skeleton)")
    ap.add_argument("--embed_model", default="text-embedding-3-large")
    ap.add_argument("--anchor_frac", type=float, default=0.18, help="15–20% recommended (e.g. 0.15–0.20)")
    ap.add_argument("--k_density", type=int, default=20, help="kNN size for density proxy")
    ap.add_argument("--out_skel", default="skeleton_embedded.jsonl")
    ap.add_argument("--out_q", default="question_embedded.jsonl")
    ap.add_argument("--out_anchors", default="anchors.json")
    args = ap.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set (env or .env).")

    client = OpenAI()

    rows = read_jsonl(args.input)
    if not rows:
        raise RuntimeError("No rows found in input JSONL.")

    # ---- build embedding texts (separate spaces)
    sk_texts = [safe_get_skeleton_text(r) for r in rows]
    q_texts = [safe_get_question_text(r) for r in rows]

    # ---- embed
    sk_embs = embed_texts(client, args.embed_model, sk_texts)
    q_embs = embed_texts(client, args.embed_model, q_texts)

    # ---- write embedded outputs
    sk_out = []
    q_out = []
    for r, e_sk, e_q, sk_txt, q_txt in zip(rows, sk_embs, q_embs, sk_texts, q_texts):
        rid = r.get("id")
        if not rid:
            raise RuntimeError("Missing id in a row; please ensure ids exist.")

        sk_row = {
            "id": rid,
            "skeleton": r.get("skeleton"),
            "skeleton_text": sk_txt,
            "embedding": e_sk,
        }
        # keep anything else useful
        for k in ["topic", "subtopic", "difficulty", "source"]:
            if k in r:
                sk_row[k] = r[k]
        sk_out.append(sk_row)

        q_row = {
            "id": rid,
            "question": r.get("question"),
            "question_text": q_txt,
            "embedding": e_q,
        }
        for k in ["topic", "subtopic", "difficulty", "source"]:
            if k in r:
                q_row[k] = r[k]
        q_out.append(q_row)

    write_jsonl(args.out_skel, sk_out)
    write_jsonl(args.out_q, q_out)

    # ---- anchor selection in skeleton space
    X = np.array(sk_embs, dtype=np.float32)
    dens = knn_density_scores(X, k=args.k_density)

    n = len(rows)
    n_anchor = max(1, int(round(args.anchor_frac * n)))
    anchor_idx = np.argsort(-dens)[:n_anchor]  # top density

    anchors = []
    for i in anchor_idx:
        anchors.append({
            "id": rows[i]["id"],
            "density": float(dens[i]),
        })

    with open(args.out_anchors, "w", encoding="utf-8") as f:
        json.dump({
            "anchor_frac": args.anchor_frac,
            "k_density": args.k_density,
            "n_total": n,
            "n_anchors": n_anchor,
            "anchors": anchors
        }, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {args.out_skel}, {args.out_q}, {args.out_anchors}")
    print(f"Anchors: {n_anchor}/{n} ({args.anchor_frac:.2%})")


if __name__ == "__main__":
    main()
