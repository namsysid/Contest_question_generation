#!/usr/bin/env python3
"""
03_retrieve_diverse.py (updated)

Seedless retrieval with:
  1) operator-signature diversity (if optypes/laws exist)
  2) max-min (farthest-first) embedding fill

OUTPUT:
- Prints one JSON object per line (JSONL)
- Strips the raw embedding vector by default to keep output small
- Optionally keeps only a subset of fields
"""

import json
import argparse
import random
from collections import defaultdict
import numpy as np
import faiss

def operator_signature(rec):
    # Use empty tuples if not present; still works but collapses into one bucket
    return (
        tuple(sorted(rec.get("optypes", []))),
        tuple(sorted(rec.get("laws", []))),
    )

def maxmin_fill(embs, selected, k):
    X = np.array(embs, dtype=np.float32)
    faiss.normalize_L2(X)
    sel = list(selected)
    n = X.shape[0]

    while len(sel) < k and len(sel) < n:
        sims = X @ X[sel].T            # (n, |S|)
        max_sim = sims.max(axis=1)     # similarity to closest selected
        nxt = int(np.argmin(max_sim))  # least similar to current set
        if nxt in sel:
            break
        sel.append(nxt)
    return sel

def strip_fields(rec, keep_fields):
    out = dict(rec)
    # Always drop embedding unless explicitly kept
    out.pop("embedding", None)

    if keep_fields:
        filtered = {}
        for k in keep_fields:
            if k in out:
                filtered[k] = out[k]
        # Always keep id if it exists
        if "id" in out and "id" not in filtered:
            filtered["id"] = out["id"]
        return filtered
    return out

def main(inp, k, keep_fields):
    with open(inp) as f:
        records = [json.loads(l) for l in f if l.strip()]

    if not records:
        return

    # Ensure embeddings exist for max-min stage
    embs = []
    for r in records:
        if "embedding" not in r:
            raise RuntimeError("Input records must contain 'embedding'. Did you run 02_embed_and_index_diverse.py?")
        embs.append(r["embedding"])

    # Bucket by abstract structure if fields exist
    buckets = defaultdict(list)
    for i, r in enumerate(records):
        buckets[operator_signature(r)].append(i)

    sigs = list(buckets.keys())
    random.shuffle(sigs)

    selected = []
    for s in sigs:
        selected.append(random.choice(buckets[s]))
        if len(selected) >= k:
            break

    # Fill remainder by max-min if needed
    if len(selected) < k:
        selected = maxmin_fill(embs, selected, k)

    for i in selected[:k]:
        print(json.dumps(strip_fields(records[i], keep_fields), ensure_ascii=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL produced by 02_embed_and_index_diverse.py (contains embeddings)")
    ap.add_argument("-k", type=int, default=10, help="Number of anchors to output")
    ap.add_argument("--keep", default="", help="Comma-separated list of fields to keep (embedding is always dropped). Example: id,optypes,laws,target,embedding_text")
    args = ap.parse_args()

    keep_fields = [s.strip() for s in args.keep.split(",") if s.strip()] if args.keep else []
    main(args.input, args.k, keep_fields)
