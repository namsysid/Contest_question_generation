#!/usr/bin/env python3
"""
03_retrieve_paper.py  (PAPER-FAITHFUL DUAL RETRIEVAL + MAPPED COMBINATION)

Matches the paper:
- Separate embedding spaces:
  * skeleton space (solution skeletons)
  * question space (question text + choices)
- Dual retrieval:
  * solution_exemplars are retrieved ONLY from skeleton space
  * question_exemplars are retrieved ONLY from question space
- Mapped combination:
  * paired_exemplars demonstrate the relation between question exemplars and their skeletons

Also keeps your anchor-based annulus sampling.

Output JSONL bundles:
{
  "bundle_id": "<string>",
  "seed_id": "<corpus id used as seed>",
  "mode": "annulus"|"tail",
  "anchor_id": "<anchor id>",
  "anchor_distance": <float>,

  "seed_skeleton_text": "...",
  "seed_topic": "...",
  "seed_difficulty": <int|None>,

  "solution_exemplars": [{"id","skeleton_text","topic","difficulty"}, ...],
  "question_exemplars":  [{"id","question_text","topic","difficulty"}, ...],
  "paired_exemplars":    [{"id","question_text","skeleton_text","topic","difficulty"}, ...]
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


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / denom


def topk_cosine_sim(query: np.ndarray, mat: np.ndarray, k: int) -> List[int]:
    sims = mat @ query
    if k >= len(sims):
        return list(np.argsort(-sims))
    idx = np.argpartition(-sims, k)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx.tolist()


def mmr_select(query: np.ndarray, cand_idx: List[int], mat: np.ndarray, k: int, lambda_mult: float = 0.7) -> List[int]:
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
    ap.add_argument("--anchors", required=True, help="anchors.json or anchors.jsonl from step 02")
    ap.add_argument("-n", "--num_bundles", type=int, default=25)
    ap.add_argument("--annulus_min", type=float, default=0.10)
    ap.add_argument("--annulus_max", type=float, default=0.28)
    ap.add_argument("--tail_frac", type=float, default=0.20)
    ap.add_argument("--tail_min", type=float, default=0.35)

    ap.add_argument("--k_solution", type=int, default=4, help="solution exemplars per bundle (skeleton space)")
    ap.add_argument("--k_question", type=int, default=4, help="question exemplars per bundle (question space)")
    ap.add_argument("--k_paired", type=int, default=3, help="paired exemplars (subset of question exemplars)")

    ap.add_argument("--mmr_lambda_solution", type=float, default=0.7)
    ap.add_argument("--mmr_lambda_question", type=float, default=0.7)

    ap.add_argument("--match_topic", action="store_true")
    ap.add_argument("--match_difficulty", action="store_true")

    ap.add_argument("--out", default="retrieval_bundles.jsonl")
    ap.add_argument("--seed", type=int, default=0, help="random seed (0 means random)")
    args = ap.parse_args()

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    skel_rows = read_jsonl(args.skeleton_embedded)
    q_rows = read_jsonl(args.question_embedded)

    skel_by_id: Dict[str, Dict[str, Any]] = {r["id"]: r for r in skel_rows if "id" in r}
    q_by_id: Dict[str, Dict[str, Any]] = {r["id"]: r for r in q_rows if "id" in r}

    skel_ids = [r["id"] for r in skel_rows]
    skel_mat = l2_normalize(np.array([r["embedding"] for r in skel_rows], dtype=np.float32))

    q_ids = [r["id"] for r in q_rows]
    q_mat = l2_normalize(np.array([r["embedding"] for r in q_rows], dtype=np.float32))

    skel_index_by_id = {sid: i for i, sid in enumerate(skel_ids)}
    q_index_by_id = {qid: i for i, qid in enumerate(q_ids)}

    # anchors: accept json list or jsonl
    try:
        with open(args.anchors, "r", encoding="utf-8") as f:
            anchors_obj = json.load(f)
        if not isinstance(anchors_obj, list):
            raise ValueError("anchors file must contain a list or be JSONL format")
    except json.JSONDecodeError:
        anchors_obj = read_jsonl(args.anchors)

    anchor_ids: List[str] = []
    for a in anchors_obj:
        if isinstance(a, str):
            anchor_ids.append(a)
        elif isinstance(a, dict) and "id" in a:
            anchor_ids.append(a["id"])
    anchor_ids = [aid for aid in anchor_ids if aid in skel_by_id]
    if not anchor_ids:
        raise RuntimeError("No valid anchors found (ids not present in skeleton index).")

    anchor_vecs: List[Tuple[str, np.ndarray]] = [(aid, skel_mat[skel_index_by_id[aid]]) for aid in anchor_ids]

    def sample_seed(aid: str, avec: np.ndarray, mode: str) -> Tuple[str, float]:
        sims = skel_mat @ avec
        dists = 1.0 - sims
        if mode == "annulus":
            mask = (dists >= args.annulus_min) & (dists <= args.annulus_max)
        else:
            mask = dists >= args.tail_min
        idxs = np.where(mask)[0].tolist()
        if not idxs:
            idxs = np.argsort(dists).tolist()
            idxs = [i for i in idxs if skel_ids[i] != aid][:80]
        pick_i = random.choice(idxs)
        return skel_ids[pick_i], float(dists[pick_i])

    def _filter_candidates(ids: List[str], rows: List[Dict[str, Any]], topic: Any, diff: Any) -> List[int]:
        cand = list(range(len(ids)))
        if args.match_topic and topic is not None:
            cand2 = [i for i in cand if rows[i].get("topic") == topic]
            if len(cand2) >= 10:
                cand = cand2
        if args.match_difficulty and diff is not None:
            cand2 = [i for i in cand if rows[i].get("difficulty") == diff]
            if len(cand2) >= 10:
                cand = cand2
        return cand

    def retrieve_solution_exemplars(seed_id: str, k: int) -> List[Dict[str, Any]]:
        if k <= 0:
            return []
        seed = skel_by_id[seed_id]
        topic = seed.get("topic")
        diff = seed.get("difficulty")
        qv = skel_mat[skel_index_by_id[seed_id]]

        cand = _filter_candidates(skel_ids, skel_rows, topic, diff)
        # top similar in skeleton space (exclude seed)
        top_local = topk_cosine_sim(qv, skel_mat[cand], k=min(80, len(cand)))
        top_idx = [cand[i] for i in top_local if skel_ids[cand[i]] != seed_id]
        chosen = mmr_select(qv, top_idx, skel_mat, k=k, lambda_mult=args.mmr_lambda_solution)

        out = []
        for i in chosen:
            sid = skel_ids[i]
            r = skel_by_id[sid]
            out.append({"id": sid, "skeleton_text": r.get("skeleton_text") or "", "topic": r.get("topic"), "difficulty": r.get("difficulty")})
        return out

    def retrieve_question_exemplars(seed_id: str, k: int) -> List[Dict[str, Any]]:
        if k <= 0:
            return []
        seed = skel_by_id[seed_id]
        topic = seed.get("topic")
        diff = seed.get("difficulty")

        # Query in question space: same-id bridge if available; else random
        if seed_id in q_index_by_id:
            qv = q_mat[q_index_by_id[seed_id]]
        else:
            qv = q_mat[random.randrange(len(q_mat))]

        cand = _filter_candidates(q_ids, q_rows, topic, diff)
        top_local = topk_cosine_sim(qv, q_mat[cand], k=min(80, len(cand)))
        top_idx = [cand[i] for i in top_local if q_ids[cand[i]] != seed_id]
        chosen = mmr_select(qv, top_idx, q_mat, k=k, lambda_mult=args.mmr_lambda_question)

        out = []
        for i in chosen:
            qid = q_ids[i]
            r = q_by_id[qid]
            out.append({"id": qid, "question_text": r.get("question_text") or r.get("question") or "", "topic": r.get("topic"), "difficulty": r.get("difficulty")})
        return out

    bundles: List[Dict[str, Any]] = []
    used_seeds: set = set()

    for b in range(args.num_bundles):
        mode = "tail" if random.random() < args.tail_frac else "annulus"
        aid, avec = random.choice(anchor_vecs)
        seed_id, adist = sample_seed(aid, avec, mode=mode)

        tries = 0
        while seed_id in used_seeds and tries < 10:
            aid, avec = random.choice(anchor_vecs)
            seed_id, adist = sample_seed(aid, avec, mode=mode)
            tries += 1
        used_seeds.add(seed_id)

        seed_row = skel_by_id[seed_id]
        sol_ex = retrieve_solution_exemplars(seed_id, args.k_solution)
        q_ex = retrieve_question_exemplars(seed_id, args.k_question)

        # Build paired exemplars from question exemplars (subset) with attached skeleton_text
        paired: List[Dict[str, Any]] = []
        for ex in q_ex[: max(0, args.k_paired)]:
            sid = ex["id"]
            sk = skel_by_id.get(sid, {})
            paired.append({
                "id": sid,
                "question_text": ex.get("question_text") or "",
                "skeleton_text": sk.get("skeleton_text") or "",
                "topic": ex.get("topic"),
                "difficulty": ex.get("difficulty"),
            })

        bundles.append(
            {
                "bundle_id": f"bundle_{b:05d}",
                "seed_id": seed_id,
                "mode": mode,
                "anchor_id": aid,
                "anchor_distance": adist,
                "seed_skeleton_text": seed_row.get("skeleton_text") or "",
                "seed_topic": seed_row.get("topic"),
                "seed_difficulty": seed_row.get("difficulty"),
                "solution_exemplars": sol_ex,
                "question_exemplars": q_ex,
                "paired_exemplars": paired,
            }
        )

    write_jsonl(args.out, bundles)
    print(f"Wrote {len(bundles)} bundles -> {args.out}")


if __name__ == "__main__":
    main()
