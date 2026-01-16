#!/usr/bin/env python3
"""
03_retrieve.py  (ANCHOR-CENTERED SAMPLING)

Goal:
  Anchors define centers of the skeleton manifold.
  Retrieval samples near-but-not-too-near (annulus) for faithfulness without copying,
  plus optional tail exploration for rare/hard/weird-but-valid.

Outputs JSONL "targets" that will be used for generation.
Important: We do NOT output exemplar question text. We only output skeleton specs + control metadata.
"""

import argparse
import json
from typing import Dict, List, Tuple

import numpy as np


def read_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / denom


def cosine_dist_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # assumes both normalized
    sims = A @ B.T
    return 1.0 - sims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeleton_embedded", required=True, help="skeleton_embedded.jsonl from step 02")
    ap.add_argument("--anchors", required=True, help="anchors.json from step 02")
    ap.add_argument("-n", type=int, default=25, help="How many target skeletons to sample")
    ap.add_argument("--p_tail", type=float, default=0.20, help="Probability of tail (rare/hard) sampling")
    ap.add_argument("--band_lo", type=float, default=0.10, help="Lower distance quantile from anchor (avoid near-duplicates)")
    ap.add_argument("--band_hi", type=float, default=0.35, help="Upper distance quantile from anchor (keep faithfulness)")
    ap.add_argument("--strip_embedding", action="store_true", help="Do not include embedding vectors in output")
    args = ap.parse_args()

    sk = read_jsonl(args.skeleton_embedded)
    with open(args.anchors, "r", encoding="utf-8") as f:
        anchor_blob = json.load(f)

    id_to_idx = {r["id"]: i for i, r in enumerate(sk)}

    anchor_ids = [a["id"] for a in anchor_blob["anchors"]]
    anchor_dens = np.array([a["density"] for a in anchor_blob["anchors"]], dtype=np.float64)

    # density-weighted anchor pick (softmax-ish)
    # (shift for numerical stability)
    w = anchor_dens - anchor_dens.max()
    w = np.exp(w)
    w = w / (w.sum() + 1e-12)

    # embeddings
    X = np.array([r["embedding"] for r in sk], dtype=np.float32)
    Xn = l2_normalize(X)

    # global density proxy for tail sampling: use mean kNN distance approx via similarities
    # brute force: use average distance to top 20 neighbors
    sims = Xn @ Xn.T
    np.fill_diagonal(sims, -1.0)
    k = min(20, len(sk) - 1)
    topk = np.partition(sims, -k, axis=1)[:, -k:]
    mean_dist = (1.0 - topk).mean(axis=1)
    # low-density = large mean_dist
    tail_rank = np.argsort(-mean_dist)  # descending mean_dist => rarer

    rng = np.random.default_rng()

    selected_ids = []
    used = set()

    for _ in range(args.n):
        do_tail = (rng.random() < args.p_tail)

        if do_tail:
            # pick one from top of tail_rank, with mild randomness
            # sample from first ~25% of tail list
            cap = max(5, int(0.25 * len(tail_rank)))
            idx = int(rng.integers(0, cap))
            cand_i = int(tail_rank[idx])
            rid = sk[cand_i]["id"]
            if rid in used:
                # fallback linear scan
                for cand_i in tail_rank[:cap]:
                    rid = sk[int(cand_i)]["id"]
                    if rid not in used:
                        break
            mode = "tail"
            anchor_id = None
            anchor_dist = None

        else:
            # manifold mode: pick anchor (weighted), then sample candidate in an annulus around it
            anchor_id = rng.choice(anchor_ids, p=w)
            a_i = id_to_idx[anchor_id]

            d = 1.0 - (Xn @ Xn[a_i])  # cosine distance to anchor (vector)
            # exclude anchor itself
            d[a_i] = np.inf

            # choose distance band by quantiles
            lo = float(np.quantile(d[np.isfinite(d)], args.band_lo))
            hi = float(np.quantile(d[np.isfinite(d)], args.band_hi))
            band = np.where((d >= lo) & (d <= hi))[0]
            if len(band) == 0:
                # fallback: nearest non-self
                cand_i = int(np.argmin(d))
            else:
                cand_i = int(rng.choice(band))

            rid = sk[cand_i]["id"]
            if rid in used:
                # quick fallback: pick another in band
                for _try in range(20):
                    cand_i = int(rng.choice(band)) if len(band) else int(np.argmin(d))
                    rid = sk[cand_i]["id"]
                    if rid not in used:
                        break

            mode = "manifold"
            anchor_dist = float(d[cand_i])

        used.add(rid)
        selected_ids.append((rid, mode, anchor_id, anchor_dist))

    # output targets
    for rid, mode, anchor_id, anchor_dist in selected_ids:
        r = sk[id_to_idx[rid]]
        out = {
            "id": r["id"],
            "mode": mode,
            "anchor_id": anchor_id,
            "anchor_distance": anchor_dist,
            "skeleton": r.get("skeleton"),
            "skeleton_text": r.get("skeleton_text"),
            # include optional metadata
            "topic": r.get("topic"),
            "difficulty": r.get("difficulty"),
        }
        if not args.strip_embedding:
            out["embedding"] = r.get("embedding")
        print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
