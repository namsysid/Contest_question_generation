#!/usr/bin/env python3
"""
03_retrieve_paper_v4.py

Paper-faithful dual retrieval + mapped combination, but fixes "textbook collapse" by requiring
seeds (and optionally solution exemplars) to contain at least one INSIGHT operator.

Backfills skeleton_text from --enriched.
Adds diagnostics fields.

Usage:
python3 03_retrieve_paper_v4.py \
  --skeleton_embedded index/skeleton_embedded.jsonl \
  --question_embedded index/question_embedded.jsonl \
  --anchors index/anchors.jsonl \
  --enriched data/enriched_schemae/enriched.jsonl \
  --require_skeleton_text \
  --require_insight_seed \
  --out retrieval_bundles.jsonl
"""
import argparse, json, random, re
from typing import Any, Dict, List, Tuple
import numpy as np

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

def safe_skeleton_text_from_enriched(enriched_row: Dict[str, Any]) -> str:
    sk = ((enriched_row.get("analysis") or {}).get("skeleton") or {})
    if isinstance(sk, str):
        return sk.strip()
    if not isinstance(sk, dict):
        return ""
    laws = sk.get("laws") or []
    steps = sk.get("steps") or []
    parts=[]
    if isinstance(laws, list) and laws:
        parts.append("LAWS: " + ", ".join(str(x) for x in laws if str(x).strip()))
    if isinstance(steps, list) and steps:
        bits=[]
        for st in steps:
            if isinstance(st, dict):
                op=str(st.get("op","")).strip()
                tx=str(st.get("text","")).strip()
                if op and tx:
                    bits.append(f"{op}:{tx}")
        if bits:
            parts.append("STEPS: " + " | ".join(bits))
    return "\n".join(parts).strip()

def get_any_text(row: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        v=row.get(k)
        if isinstance(v,str) and v.strip():
            return v.strip()
    return ""

def parse_ops(skeleton_text: str) -> List[str]:
    m = re.search(r"STEPS:\s*(.+)", skeleton_text or "", re.DOTALL)
    if not m:
        return []
    parts=[p.strip() for p in m.group(1).split("|") if p.strip()]
    ops=[]
    for p in parts:
        if ":" in p:
            ops.append(p.split(":",1)[0].strip())
    return ops

def complexity_score(skeleton_text: str) -> int:
    s=(skeleton_text or "").strip()
    if not s: return 0
    laws=0; steps=0; ops=set()
    m=re.search(r"LAWS:\s*(.+)", s)
    if m:
        laws=len([x for x in m.group(1).split(",") if x.strip()])
    m2=re.search(r"STEPS:\s*(.+)", s, re.DOTALL)
    if m2:
        parts=[p.strip() for p in m2.group(1).split("|") if p.strip()]
        steps=len(parts)
        for p in parts:
            if ":" in p: ops.add(p.split(":",1)[0].strip())
    return laws + steps + len(ops)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--skeleton_embedded", required=True)
    ap.add_argument("--question_embedded", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--enriched", default="")
    ap.add_argument("--require_skeleton_text", action="store_true")

    ap.add_argument("--insight_ops", default="INTRODUCE_AUX,CONSTRAINT_COUPLING,INVARIANT_SYMMETRY,CASEWORK_REGIME,CHECK")
    ap.add_argument("--require_insight_seed", action="store_true")
    ap.add_argument("--require_insight_solution_exemplars", action="store_true")
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

    insight_ops=set(s.strip() for s in args.insight_ops.split(",") if s.strip())

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

    # anchors
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

    anchor_vecs=[(aid, skel_mat[skel_index_by_id[aid]]) for aid in anchor_ids]

    def get_skeleton_text(pid: str) -> str:
        r=skel_by_id.get(pid,{})
        txt=get_any_text(r, ["skeleton_text","schema_text","skeleton","solution_skeleton_text"])
        if txt: return txt
        enr=enriched_by_id.get(pid)
        if enr: return safe_skeleton_text_from_enriched(enr)
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
        pick=random.choice(idxs)
        return skel_ids[pick], float(dists[pick])

    def retrieve_solution_exemplars(seed_id: str, k:int) -> List[Dict[str,Any]]:
        qv=skel_mat[skel_index_by_id[seed_id]]
        sims = skel_mat @ qv
        idx = np.argsort(-sims).tolist()
        idx = [i for i in idx if skel_ids[i] != seed_id][:220]
        chosen = mmr_select(qv, idx, skel_mat, k=k*3, lambda_mult=args.mmr_lambda_solution)
        out=[]
        for i in chosen:
            sid=skel_ids[i]
            st=get_skeleton_text(sid)
            if args.min_sol_ex_complexity and complexity_score(st) < args.min_sol_ex_complexity:
                continue
            if args.require_insight_solution_exemplars:
                ops=set(parse_ops(st))
                if not any(op in insight_ops for op in ops):
                    continue
            out.append({"id":sid,"skeleton_text":st,"topic":skel_by_id.get(sid,{}).get("topic"),"difficulty":skel_by_id.get(sid,{}).get("difficulty")})
            if len(out) >= k:
                break
        return out

    def retrieve_question_exemplars(seed_id: str, k:int) -> List[Dict[str,Any]]:
        qv=q_mat[q_index_by_id[seed_id]] if seed_id in q_index_by_id else q_mat[random.randrange(len(q_mat))]
        sims = q_mat @ qv
        idx = np.argsort(-sims).tolist()
        idx = [i for i in idx if q_ids[i] != seed_id][:220]
        chosen = mmr_select(qv, idx, q_mat, k=k, lambda_mult=args.mmr_lambda_question)
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

        seed_skel_txt = get_skeleton_text(seed_id)
        if args.require_skeleton_text and not seed_skel_txt:
            continue

        if args.min_seed_complexity and complexity_score(seed_skel_txt) < args.min_seed_complexity:
            continue

        if args.require_insight_seed:
            ops=set(parse_ops(seed_skel_txt))
            if not any(op in insight_ops for op in ops):
                continue

        sol_ex = retrieve_solution_exemplars(seed_id, args.k_solution)
        if args.k_solution and len(sol_ex) < args.k_solution:
            continue

        q_ex = retrieve_question_exemplars(seed_id, args.k_question)
        paired=[]
        for ex in q_ex[:max(0,args.k_paired)]:
            sid=ex["id"]
            paired.append({"id":sid,"question_text":ex.get("question_text") or "", "skeleton_text": get_skeleton_text(sid),
                           "topic": ex.get("topic"), "difficulty": ex.get("difficulty")})

        if args.require_skeleton_text:
            if any(not (ex.get("skeleton_text") or "").strip() for ex in sol_ex):
                continue
            if any(not (ex.get("skeleton_text") or "").strip() for ex in paired):
                continue

        used.add(seed_id)
        bundles.append({
            "bundle_id": f"bundle_{len(bundles):05d}",
            "seed_id": seed_id,
            "mode": mode,
            "anchor_id": aid,
            "anchor_distance": adist,
            "seed_skeleton_text": seed_skel_txt,
            "seed_topic": skel_by_id.get(seed_id,{}).get("topic"),
            "seed_difficulty": skel_by_id.get(seed_id,{}).get("difficulty"),
            "solution_exemplars": sol_ex,
            "question_exemplars": q_ex,
            "paired_exemplars": paired,
            "_diagnostics": {
                "seed_complexity": complexity_score(seed_skel_txt),
                "seed_ops": parse_ops(seed_skel_txt),
            }
        })

    write_jsonl(args.out, bundles)
    print(f"Wrote {len(bundles)} bundles -> {args.out}")

if __name__=="__main__":
    main()
