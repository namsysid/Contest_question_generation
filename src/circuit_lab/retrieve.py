from __future__ import annotations

import argparse
import random
from typing import Any

import numpy as np

from .common import read_jsonl, write_jsonl


def norm(matrix: np.ndarray) -> np.ndarray:
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)


def mmr(query: np.ndarray, indices: list[int], matrix: np.ndarray, count: int, weight: float) -> list[int]:
    selected: list[int] = []
    remaining = set(indices)
    while remaining and len(selected) < count:
        best = max(remaining, key=lambda i: weight * float(matrix[i] @ query) - (1 - weight) * max([float(matrix[i] @ matrix[j]) for j in selected] or [0.0]))
        selected.append(best)
        remaining.remove(best)
    return selected


def build_bundles(enriched: list[dict[str, Any]], question_rows: list[dict[str, Any]], structure_rows: list[dict[str, Any]], anchors: list[dict[str, Any]], count: int, k_structure: int, k_question: int, weight: float, seed: int) -> list[dict[str, Any]]:
    random.seed(seed)
    by_id = {row["id"]: row for row in enriched}
    q_by_id = {row["id"]: row for row in question_rows}
    s_by_id = {row["id"]: row for row in structure_rows}
    ids = [row["id"] for row in structure_rows if row["id"] in q_by_id and row["id"] in by_id]
    qmat = norm(np.asarray([q_by_id[i]["embedding"] for i in ids], dtype=np.float32))
    smat = norm(np.asarray([s_by_id[i]["embedding"] for i in ids], dtype=np.float32))
    index = {item_id: i for i, item_id in enumerate(ids)}
    anchor_ids = [a["id"] for a in anchors if a.get("id") in index] or ids
    bundles = []
    for number in range(count):
        anchor_id = anchor_ids[number % len(anchor_ids)]
        anchor = by_id[anchor_id]
        fmt = anchor.get("response_type")
        topics = set(anchor.get("topics") or [])
        candidates = [i for i, item_id in enumerate(ids) if item_id != anchor_id and by_id[item_id].get("response_type") == fmt and topics.intersection(by_id[item_id].get("topics") or [])]
        if not candidates:
            candidates = [i for i, item_id in enumerate(ids) if item_id != anchor_id and by_id[item_id].get("response_type") == fmt]
        si = mmr(smat[index[anchor_id]], candidates, smat, k_structure, weight)
        qi = mmr(qmat[index[anchor_id]], candidates, qmat, k_question, weight)
        bundles.append({
            "bundle_id": f"circuit-bundle-{number:05d}", "anchor_id": anchor_id,
            "response_type": fmt, "topics": list(topics), "seed_graph_text": s_by_id[anchor_id]["graph_text"],
            "anchor_item": anchor,
            "structure_exemplars": [by_id[ids[i]] for i in si],
            "question_exemplars": [by_id[ids[i]] for i in qi],
        })
    return bundles


def main() -> None:
    parser = argparse.ArgumentParser(description="Build graph-first dual-retrieval Circuit Lab bundles")
    parser.add_argument("--enriched", required=True)
    parser.add_argument("--question-index", required=True)
    parser.add_argument("--structure-index", required=True)
    parser.add_argument("--anchors", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--k-structure", type=int, default=4)
    parser.add_argument("--k-question", type=int, default=4)
    parser.add_argument("--mmr-weight", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    bundles = build_bundles(read_jsonl(args.enriched), read_jsonl(args.question_index), read_jsonl(args.structure_index), read_jsonl(args.anchors), args.count, args.k_structure, args.k_question, args.mmr_weight, args.seed)
    write_jsonl(args.out, bundles)
    print(f"Wrote {len(bundles)} retrieval bundles")


if __name__ == "__main__":
    main()
