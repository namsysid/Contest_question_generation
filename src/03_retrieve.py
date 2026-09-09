#!/usr/bin/env python3
"""
03_retrieve.py  (GRAPH-FIRST DUAL RETRIEVAL)

Retrieves bundles using graph_text as the primary structural representation.
Backward compatible with legacy skeleton_text fields.

python3 src/03_retrieve.py --skeleton_embedded data/skeleton_embedded.jsonl --question_embedded da
ta/question_embedded.jsonl --anchors data/anchors.jsonl 

"""
import argparse, json, random, re
from typing import Any, Dict, List, Tuple
import numpy as np

GRAPH_FEATURE_TERMS = {
    "auxiliary": ("AUXILIARYS:", "AUXILIARY:", "TYPE: AUXILIARY"),
    "trap": ("TRAPS:", "TRAP:", "TYPE: TRAP"),
    "distractor": ("DISTRACTORS:", "DISTRACTOR:", "TYPE: DISTRACTOR"),
    "constraint": ("CONSTRAINTS:", "CONSTRAINT:", "TYPE: CONSTRAINT"),
    "law": ("LAWS:", "LAW:", "TYPE: LAW"),
    "state": ("STATES:", "STATE:", "TYPE: STATE"),
}


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path,"w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    denom=np.linalg.norm(mat, axis=1, keepdims=True)+1e-12
    return mat/denom


def mmr_select(query: np.ndarray, cand_idx: List[int], mat: np.ndarray, k: int, lambda_mult: float=0.7) -> List[int]:
    if not cand_idx:
        return []
    selected=[]
    cand_set=set(cand_idx)
    sims_q={i: float(mat[i]@query) for i in cand_idx}
    while len(selected)<k and cand_set:
        if not selected:
            best=max(cand_set, key=lambda i: sims_q[i])
            selected.append(best); cand_set.remove(best); continue
        def score(i:int)->float:
            sim_q=sims_q[i]
            sim_sel=max(float(mat[i]@mat[j]) for j in selected)
            return lambda_mult*sim_q - (1-lambda_mult)*sim_sel
        best=max(cand_set, key=score)
        selected.append(best); cand_set.remove(best)
    return selected


def safe_graph_text_from_enriched(enriched_row: Dict[str, Any]) -> str:
    analysis = enriched_row.get("analysis") or {}
    txt = analysis.get("graph_text") or analysis.get("skeleton") or ""
    if isinstance(txt, str) and txt.strip():
        return txt.strip()
    graph = analysis.get("problem_graph") or {}
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    if not isinstance(nodes, list):
        return ""
    parts=[]
    by_type = {}
    for n in nodes:
        if not isinstance(n, dict):
            continue
        by_type.setdefault(str(n.get("type", "State")).upper(), []).append(f"{n.get('id','')}:{n.get('label','')}")
    for t in ["GIVEN", "TARGET", "LAW", "STATE", "CONSTRAINT", "AUXILIARY", "TRAP", "DISTRACTOR"]:
        k = t + "S"
        if by_type.get(t):
            parts.append(f"{k}: " + " | ".join(by_type[t]))
    if isinstance(edges, list) and edges:
        parts.append("EDGES: " + " | ".join(
            f"{e.get('src','')} -{e.get('type','supports')}-> {e.get('dst','')}" for e in edges if isinstance(e, dict)
        ))
    return "\n".join(parts).strip()


def get_any_text(row: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        v=row.get(k)
        if isinstance(v,str) and v.strip():
            return v.strip()
    return ""


def graph_feature_counts(graph_text: str) -> Dict[str, int]:
    s = (graph_text or "").upper()
    out = {k: 0 for k in GRAPH_FEATURE_TERMS}
    out["edges"] = s.count("->")
    for key, tokens in GRAPH_FEATURE_TERMS.items():
        out[key] = sum(s.count(tok.upper()) for tok in tokens)
    return out


def complexity_score(graph_text: str) -> int:
    c = graph_feature_counts(graph_text)
    return c["law"] + c["state"] + c["constraint"] + 2*c["auxiliary"] + 2*c["trap"] + c["distractor"] + c["edges"]


def has_insight(graph_text: str) -> bool:
    c = graph_feature_counts(graph_text)
    return c["auxiliary"] > 0 or c["trap"] > 0 or c["constraint"] > 0


def source_year(problem_id: str) -> int | None:
    """Extract a four-digit source year from canonical exam IDs."""
    match = re.search(r"(?:^|[-_])(20\d{2})(?:[-_]|$)", str(problem_id or ""))
    return int(match.group(1)) if match else None


def source_question_number(problem_id: str) -> int | None:
    match = re.search(r"(?:_Q|-)(\d+)$", str(problem_id or ""), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def split_preferred_indices(
    indices: List[int], ids: List[str], preferred_years: set[int], min_question_number: int = 0,
) -> Tuple[List[int], List[int]]:
    """Keep similarity order within recent and fallback source pools."""
    if not preferred_years:
        return list(indices), []
    def preferred_source(index: int) -> bool:
        number = source_question_number(ids[index])
        return source_year(ids[index]) in preferred_years and (
            min_question_number <= 0 or (number is not None and number >= min_question_number)
        )
    preferred = [i for i in indices if preferred_source(i)]
    fallback = [i for i in indices if not preferred_source(i)]
    return preferred, fallback


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--skeleton_embedded", required=True)
    ap.add_argument("--question_embedded", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--enriched", default="")
    ap.add_argument("--require_skeleton_text", action="store_true")
    ap.add_argument("--require_insight_seed", action="store_true")
    ap.add_argument("--require_insight_solution_exemplars", action="store_true")
    ap.add_argument(
        "--preferred_years", nargs="*", type=int, default=[],
        help="Prefer sources from these years; older sources are used only when the preferred pool cannot fill a bundle.",
    )
    ap.add_argument(
        "--min_preferred_question_number", type=int, default=0,
        help="Within preferred years, prioritize later exam questions at or above this number.",
    )
    ap.add_argument(
        "--strict_preferred_sources", action="store_true",
        help="Never backfill seeds or exemplars outside the preferred year/question-number pool.",
    )
    ap.add_argument("--min_seed_complexity", type=int, default=0)
    ap.add_argument("--min_sol_ex_complexity", type=int, default=0)
    ap.add_argument("-n","--num_bundles", type=int, default=25)
    ap.add_argument("--annulus_min", type=float, default=0.08)
    ap.add_argument("--annulus_max", type=float, default=0.40)
    ap.add_argument("--tail_frac", type=float, default=0.35)
    ap.add_argument("--tail_min", type=float, default=0.30)
    ap.add_argument("--k_solution", type=int, default=4)
    ap.add_argument("--k_question", type=int, default=4)
    ap.add_argument("--k_paired", type=int, default=3)
    ap.add_argument("--mmr_lambda_solution", type=float, default=0.7)
    ap.add_argument("--mmr_lambda_question", type=float, default=0.7)
    ap.add_argument("--out", default="retrieval_bundles.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args=ap.parse_args()
    preferred_years = set(args.preferred_years)

    if args.seed:
        random.seed(args.seed); np.random.seed(args.seed)

    skel_rows=read_jsonl(args.skeleton_embedded)
    q_rows=read_jsonl(args.question_embedded)
    skel_by_id={r["id"]: r for r in skel_rows if "id" in r}
    q_by_id={r["id"]: r for r in q_rows if "id" in r}

    enriched_by_id={}
    if args.enriched:
        for r in read_jsonl(args.enriched):
            if "id" in r:
                enriched_by_id[r["id"]] = r

    skel_ids=[r["id"] for r in skel_rows]
    skel_mat=l2_normalize(np.array([r["embedding"] for r in skel_rows], dtype=np.float32))
    q_ids=[r["id"] for r in q_rows]
    q_mat=l2_normalize(np.array([r["embedding"] for r in q_rows], dtype=np.float32))

    skel_index_by_id={sid:i for i,sid in enumerate(skel_ids)}
    q_index_by_id={qid:i for i,qid in enumerate(q_ids)}

    available_preferred_years = sorted(
        {year for sid in skel_ids if (year := source_year(sid)) in preferred_years},
        reverse=True,
    )
    if preferred_years and not available_preferred_years:
        print(
            "WARNING: none of the preferred source years are present in the structural index; "
            "retrieval will use older fallback sources."
        )

    try:
        with open(args.anchors,"r",encoding="utf-8") as f:
            anchors_obj=json.load(f)
        if not isinstance(anchors_obj,list):
            raise ValueError
    except Exception:
        anchors_obj=read_jsonl(args.anchors)

    anchor_ids=[]
    for a in anchors_obj:
        if isinstance(a,str): anchor_ids.append(a)
        elif isinstance(a,dict) and "id" in a: anchor_ids.append(a["id"])
    anchor_ids=[aid for aid in anchor_ids if aid in skel_by_id]
    if not anchor_ids:
        raise RuntimeError("No valid anchors found.")

    preferred_anchor_ids = [aid for aid in anchor_ids if source_year(aid) in preferred_years]
    if preferred_anchor_ids:
        anchor_ids = preferred_anchor_ids

    anchor_vecs=[(aid, skel_mat[skel_index_by_id[aid]]) for aid in anchor_ids]

    def get_graph_text(pid: str) -> str:
        r=skel_by_id.get(pid,{})
        txt=get_any_text(r, ["graph_text","skeleton_text","schema_text","skeleton","solution_skeleton_text"])
        if txt: return txt
        enr=enriched_by_id.get(pid)
        if enr: return safe_graph_text_from_enriched(enr)
        return ""

    def get_question_text(pid: str) -> str:
        r=q_by_id.get(pid,{})
        txt=get_any_text(r, ["question_text","question","qa_text"])
        if txt: return txt
        enr=enriched_by_id.get(pid)
        if enr:
            stem=((enr.get("problem") or {}).get("stem") or "").strip()
            choices=((enr.get("problem") or {}).get("choices") or [])
            if isinstance(choices,list) and choices:
                return (stem+"\n"+"\n".join(choices)).strip()
            return stem
        return ""

    def sample_seed(aid: str, avec: np.ndarray, mode: str) -> Tuple[str,float]:
        sims=skel_mat @ avec
        dists=1.0 - sims
        if mode=="annulus":
            mask=(dists>=args.annulus_min) & (dists<=args.annulus_max)
        else:
            mask=(dists>=args.tail_min)
        idxs=np.where(mask)[0].tolist()
        if not idxs:
            idxs=np.argsort(dists).tolist()
            idxs=[i for i in idxs if skel_ids[i]!=aid][:160]
        preferred, _ = split_preferred_indices(
            idxs, skel_ids, preferred_years, args.min_preferred_question_number
        )
        if preferred_years and not preferred and available_preferred_years:
            preferred = [
                i for i in np.argsort(dists).tolist()
                if skel_ids[i] != aid
                and source_year(skel_ids[i]) in preferred_years
                and (
                    args.min_preferred_question_number <= 0
                    or (source_question_number(skel_ids[i]) or 0) >= args.min_preferred_question_number
                )
            ][:160]
        if args.strict_preferred_sources and not preferred:
            raise RuntimeError("No seed satisfies the strict preferred-source cutoff")
        pick=random.choice(preferred or idxs)
        return skel_ids[pick], float(dists[pick])

    def retrieve_solution_exemplars(seed_id: str, k:int) -> List[Dict[str,Any]]:
        qv=skel_mat[skel_index_by_id[seed_id]]
        sims = skel_mat @ qv
        idx = np.argsort(-sims).tolist()
        idx = [i for i in idx if skel_ids[i] != seed_id][:220]
        preferred, fallback = split_preferred_indices(
            idx, skel_ids, preferred_years, args.min_preferred_question_number
        )
        # Oversample each tier because graph-complexity and insight checks happen below.
        chosen = mmr_select(qv, preferred, skel_mat, k=k*6, lambda_mult=args.mmr_lambda_solution)
        if not args.strict_preferred_sources:
            chosen += mmr_select(qv, fallback, skel_mat, k=k*6, lambda_mult=args.mmr_lambda_solution)
        out=[]
        for i in chosen:
            sid=skel_ids[i]
            gt=get_graph_text(sid)
            if args.min_sol_ex_complexity and complexity_score(gt) < args.min_sol_ex_complexity:
                continue
            if args.require_insight_solution_exemplars and not has_insight(gt):
                continue
            out.append({
                "id": sid, "graph_text": gt, "skeleton_text": gt,
                "question_text": get_question_text(sid),
                "topic": skel_by_id.get(sid,{}).get("topic"),
                "difficulty": skel_by_id.get(sid,{}).get("difficulty"),
            })
            if len(out) >= k:
                break
        return out

    def retrieve_question_exemplars(seed_id: str, k:int) -> List[Dict[str,Any]]:
        qv=q_mat[q_index_by_id[seed_id]] if seed_id in q_index_by_id else q_mat[random.randrange(len(q_mat))]
        sims = q_mat @ qv
        idx = np.argsort(-sims).tolist()
        idx = [i for i in idx if q_ids[i] != seed_id][:220]
        preferred, fallback = split_preferred_indices(
            idx, q_ids, preferred_years, args.min_preferred_question_number
        )
        chosen = mmr_select(qv, preferred, q_mat, k=k, lambda_mult=args.mmr_lambda_question)
        if len(chosen) < k and not args.strict_preferred_sources:
            chosen += mmr_select(qv, fallback, q_mat, k=k-len(chosen), lambda_mult=args.mmr_lambda_question)
        return [{"id": q_ids[i], "question_text": get_question_text(q_ids[i]), "topic": q_by_id.get(q_ids[i],{}).get("topic"), "difficulty": q_by_id.get(q_ids[i],{}).get("difficulty")} for i in chosen]

    bundles=[]
    used=set()
    attempts=0
    while len(bundles) < args.num_bundles and attempts < args.num_bundles*40:
        attempts += 1
        mode = "tail" if random.random() < args.tail_frac else "annulus"
        aid, avec = random.choice(anchor_vecs)
        seed_id, adist = sample_seed(aid, avec, mode)

        if seed_id in used:
            continue

        seed_graph_txt = get_graph_text(seed_id)
        if args.require_skeleton_text and not seed_graph_txt:
            continue
        if args.min_seed_complexity and complexity_score(seed_graph_txt) < args.min_seed_complexity:
            continue
        if args.require_insight_seed and not has_insight(seed_graph_txt):
            continue

        sol_ex = retrieve_solution_exemplars(seed_id, args.k_solution)
        if args.k_solution and len(sol_ex) < args.k_solution:
            continue

        q_ex = retrieve_question_exemplars(seed_id, args.k_question)
        paired=[]
        for ex in q_ex[:max(0,args.k_paired)]:
            sid=ex["id"]
            gt = get_graph_text(sid)
            paired.append({"id":sid,"question_text":ex.get("question_text") or "", "graph_text": gt, "skeleton_text": gt,
                           "topic": ex.get("topic"), "difficulty": ex.get("difficulty")})

        if args.require_skeleton_text:
            if any(not (ex.get("graph_text") or "").strip() for ex in sol_ex):
                continue
            if any(not (ex.get("graph_text") or "").strip() for ex in paired):
                continue

        used.add(seed_id)
        bundles.append({
            "bundle_id": f"bundle_{len(bundles):05d}",
            "seed_id": seed_id,
            "seed_source_year": source_year(seed_id),
            "mode": mode,
            "anchor_id": aid,
            "anchor_distance": adist,
            "seed_graph_text": seed_graph_txt,
            "seed_skeleton_text": seed_graph_txt,
            "seed_question_text": get_question_text(seed_id),
            "seed_topic": skel_by_id.get(seed_id,{}).get("topic"),
            "seed_difficulty": skel_by_id.get(seed_id,{}).get("difficulty"),
            "solution_exemplars": sol_ex,
            "question_exemplars": q_ex,
            "paired_exemplars": paired,
            "_diagnostics": {
                "seed_complexity": complexity_score(seed_graph_txt),
                "seed_graph_features": graph_feature_counts(seed_graph_txt),
                "preferred_source_years": args.preferred_years,
                "available_preferred_source_years": available_preferred_years,
                "min_preferred_question_number": args.min_preferred_question_number,
                "preferred_solution_exemplar_count": sum(
                    source_year(ex.get("id", "")) in preferred_years for ex in sol_ex
                ),
                "preferred_question_exemplar_count": sum(
                    source_year(ex.get("id", "")) in preferred_years for ex in q_ex
                ),
            }
        })

    write_jsonl(args.out, bundles)
    print(f"Wrote {len(bundles)} bundles -> {args.out}")

if __name__=="__main__":
    main()
