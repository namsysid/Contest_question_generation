#!/usr/bin/env python3
"""
03_retrieve.py  (DUAL-RAG, SEMI-CONVERGENT)

Goal:
  1) Sample TARGET SKELETONS from the skeleton manifold using anchors:
     - annulus sampling (near-but-not-too-near) for faithfulness without copying
     - optional tail sampling for rare/hard/weird-but-valid regions
  2) For each target skeleton, retrieve EXEMPLARS from the QUESTION/CHOICE manifold:
     - pick neighbors in question-embedding space using the *corresponding* question embedding (same id),
       which is the simplest "semi-convergent" bridge without learning a cross-space mapper.
     - attach exemplar question text + *compressed solution traces* (skeleton_text) for mapping.

Output JSONL "targets" with:
  {
    id, mode, anchor_id, anchor_distance,
    skeleton_text, skeleton (optional),
    topic, difficulty,
    exemplars: [{id, question_text, skeleton_text, topic, difficulty}, ...]
  }
"""

import argparse
import json
import random
from typing import Any, Dict, List, Tuple

import numpy as np


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / denom


def topk_cosine_sim(query: np.ndarray, mat: np.ndarray, k: int) -> List[int]:
    """
    query: (d,) normalized
    mat: (n,d) normalized
    returns indices of top-k by similarity
    """
    sims = mat @ query
    if k >= len(sims):
        return list(np.argsort(-sims))
    idx = np.argpartition(-sims, k)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx.tolist()


def mmr_select(
    query: np.ndarray,
    cand_idx: List[int],
    mat: np.ndarray,
    k: int,
    lambda_mult: float = 0.7,
) -> List[int]:
    """
    Maximal Marginal Relevance selection on cosine similarity.
    query, mat assumed normalized.
    """
    if not cand_idx:
        return []
    selected: List[int] = []
    cand_set = set(cand_idx)

    sims_q = {i: float(mat[i] @ query) for i in cand_idx}

    while len(selected) < k and cand_set:
        if not selected:
            best = max(cand_set, key=lambda i: sims_q[i])
            selected.append(best)
            cand_set.remove(best)
            continue

        def score(i: int) -> float:
            sim_to_query = sims_q[i]
            sim_to_sel = max(float(mat[i] @ mat[j]) for j in selected)
            return lambda_mult * sim_to_query - (1 - lambda_mult) * sim_to_sel

        best = max(cand_set, key=score)
        selected.append(best)
        cand_set.remove(best)

    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeleton_embedded", required=True, help="skeleton_embedded.jsonl from step 02")
    ap.add_argument("--question_embedded", required=True, help="question_embedded.jsonl from step 02")
    ap.add_argument("--anchors", required=True, help="anchors.json (list of skeleton ids or {id:...} objects)")
    ap.add_argument("-n", "--num_targets", type=int, default=25)
    ap.add_argument("--annulus_min", type=float, default=0.10, help="min cosine distance from anchor")
    ap.add_argument("--annulus_max", type=float, default=0.28, help="max cosine distance from anchor")
    ap.add_argument("--tail_frac", type=float, default=0.20, help="fraction sampled from far tail")
    ap.add_argument("--tail_min", type=float, default=0.35, help="min cosine distance for tail sampling")
    ap.add_argument("--k_exemplars", type=int, default=4, help="question exemplars per target")
    ap.add_argument("--mmr_lambda", type=float, default=0.7, help="MMR lambda for exemplar selection")
    ap.add_argument("--match_topic", action="store_true", help="filter exemplars to same topic when possible")
    ap.add_argument("--match_difficulty", action="store_true", help="filter exemplars to same difficulty when possible")
    ap.add_argument("--strip_embedding", action="store_true", help="omit embeddings in output targets")
    ap.add_argument("--seed", type=int, default=0, help="random seed (0 means random)")
    args = ap.parse_args()

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    skel_rows = read_jsonl(args.skeleton_embedded)
    q_rows = read_jsonl(args.question_embedded)

    # Build id maps
    skel_by_id: Dict[str, Dict[str, Any]] = {r["id"]: r for r in skel_rows if "id" in r}
    q_by_id: Dict[str, Dict[str, Any]] = {r["id"]: r for r in q_rows if "id" in r}

    # Matrices
    skel_ids = [r["id"] for r in skel_rows]
    skel_mat = np.array([r["embedding"] for r in skel_rows], dtype=np.float32)
    skel_mat = l2_normalize(skel_mat)

    q_ids = [r["id"] for r in q_rows]
    q_mat = np.array([r["embedding"] for r in q_rows], dtype=np.float32)
    q_mat = l2_normalize(q_mat)

    # Index lookups
    skel_index_by_id = {sid: i for i, sid in enumerate(skel_ids)}
    q_index_by_id = {qid: i for i, qid in enumerate(q_ids)}

    # Load anchors as list of ids (preferred). If anchors contain objects, accept {"id":...}
    # Supports both JSON (single list) and JSONL (one object per line) formats
    try:
        with open(args.anchors, "r", encoding="utf-8") as f:
            anchors_obj = json.load(f)
        # Try JSON format first (single object/list)
        if not isinstance(anchors_obj, list):
            raise ValueError("anchors file must contain a list or be JSONL format")
    except json.JSONDecodeError:
        # If JSON parsing fails, try JSONL format
        anchors_obj = read_jsonl(args.anchors)
    
    anchor_ids = []
    for a in anchors_obj:
        if isinstance(a, str):
            anchor_ids.append(a)
        elif isinstance(a, dict) and "id" in a:
            anchor_ids.append(a["id"])
    anchor_ids = [aid for aid in anchor_ids if aid in skel_by_id]

    if not anchor_ids:
        raise RuntimeError("No valid anchors found (ids not present in skeleton index).")

    # Precompute anchor vectors in skeleton space
    anchor_vecs: List[Tuple[str, np.ndarray]] = []
    for aid in anchor_ids:
        anchor_vecs.append((aid, skel_mat[skel_index_by_id[aid]]))

    def sample_from_anchor(aid: str, avec: np.ndarray, mode: str) -> Tuple[str, float]:
        """
        Returns (picked_skeleton_id, anchor_distance)
        """
        sims = skel_mat @ avec
        dists = 1.0 - sims

        if mode == "annulus":
            mask = (dists >= args.annulus_min) & (dists <= args.annulus_max)
        else:  # tail
            mask = dists >= args.tail_min

        idxs = np.where(mask)[0].tolist()
        if not idxs:
            # fallback: nearest non-self
            idxs = np.argsort(dists).tolist()
            idxs = [i for i in idxs if skel_ids[i] != aid][:50]

        pick_i = random.choice(idxs)
        pick_id = skel_ids[pick_i]
        return pick_id, float(dists[pick_i])

    def pick_exemplars_for_target(target_id: str, k: int) -> List[Dict[str, Any]]:
        """
        Semi-convergent mapping:
          use the target's *own* question embedding (same id) to find neighboring question stems.
        """
        if k <= 0:
            return []

        target_skel = skel_by_id.get(target_id, {})
        topic = target_skel.get("topic")
        diff = target_skel.get("difficulty")

        # Query embedding in question space: same id if available
        if target_id in q_index_by_id:
            q_query = q_mat[q_index_by_id[target_id]]
        else:
            q_query = q_mat[random.randrange(len(q_mat))]

        # Candidate pool indices with optional filters
        cand = list(range(len(q_ids)))
        if args.match_topic and topic is not None:
            cand2 = [i for i in cand if q_rows[i].get("topic") == topic]
            if len(cand2) >= max(k * 2, 10):
                cand = cand2
        if args.match_difficulty and diff is not None:
            cand2 = [i for i in cand if q_rows[i].get("difficulty") == diff]
            if len(cand2) >= max(k * 2, 10):
                cand = cand2

        # Top candidates by similarity, exclude self, then MMR for diversity
        cand_top_local = topk_cosine_sim(q_query, q_mat[cand], k=min(50, len(cand)))
        cand_top_idx = [cand[i] for i in cand_top_local]
        cand_top_idx = [i for i in cand_top_idx if q_ids[i] != target_id]

        chosen = mmr_select(q_query, cand_top_idx, q_mat, k=k, lambda_mult=args.mmr_lambda)

        exemplars: List[Dict[str, Any]] = []
        for i in chosen:
            ex_id = q_ids[i]
            ex_q = q_rows[i]
            ex_s = skel_by_id.get(ex_id, {})
            exemplars.append(
                {
                    "id": ex_id,
                    "question_text": ex_q.get("question_text") or ex_q.get("question") or "",
                    "skeleton_text": ex_s.get("skeleton_text") or "",
                    "topic": ex_q.get("topic"),
                    "difficulty": ex_q.get("difficulty"),
                }
            )
        return exemplars

    used_targets: set = set()

    for _ in range(args.num_targets):
        mode = "tail" if random.random() < args.tail_frac else "annulus"
        aid, avec = random.choice(anchor_vecs)
        picked_id, adist = sample_from_anchor(aid, avec, mode=mode)

        # Avoid duplicates if possible
        tries = 0
        while picked_id in used_targets and tries < 10:
            aid, avec = random.choice(anchor_vecs)
            picked_id, adist = sample_from_anchor(aid, avec, mode=mode)
            tries += 1
        used_targets.add(picked_id)

        r = skel_by_id[picked_id]
        out: Dict[str, Any] = {
            "id": picked_id,
            "mode": mode,
            "anchor_id": aid,
            "anchor_distance": adist,
            "skeleton": r.get("skeleton"),
            "skeleton_text": r.get("skeleton_text"),
            "topic": r.get("topic"),
            "difficulty": r.get("difficulty"),
            # Attach exemplars (question/choice manifold) + compressed traces (mapping)
            "exemplars": pick_exemplars_for_target(picked_id, args.k_exemplars),
        }

        if not args.strip_embedding:
            out["embedding"] = r.get("embedding")

        print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
