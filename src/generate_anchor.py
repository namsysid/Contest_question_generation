#!/usr/bin/env python3
from __future__ import annotations

"""
generate.py (anchor-sampling, no "types")

Generates many new problems without requiring a user query/spec.

Approach:
- Sample random "anchor" problems from your corpus.
- For each anchor, retrieve a diverse neighborhood via FAISS + MMR (using the anchor's skeleton embedding as the query).
- Feed exemplar (question, skeleton) pairs to an LLM to generate a novel problem.
- Output JSONL: one generated schema record per line.

Requirements:
- enriched.jsonl (your enriched corpus)
- index/{view}_ids.txt
- index/{view}_embeddings.npy  (normalized float32)
- index/faiss_{view}.index

Env:
- OPENAI_API_KEY set (dotenv supported)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import faiss

from dotenv import load_dotenv
from ollama_client import generate_text
load_dotenv()

# ----------------------------
# IO utils
# ----------------------------

def load_jsonl_map(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["id"]] = r
    return out

def load_ids(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()

# ----------------------------
# Views
# ----------------------------

def question_view(doc: Dict[str, Any]) -> str:
    stem = doc.get("problem", {}).get("stem", "")
    choices = doc.get("problem", {}).get("choices", [])
    return (stem + "\n" + "\n".join(choices)).strip()

def skeleton_view(doc: Dict[str, Any]) -> str:
    # canonical skeleton string (you already store analysis._skeleton_text)
    return (doc.get("analysis", {}).get("_skeleton_text") or "").strip()

# ----------------------------
# Retrieval (FAISS + MMR)
# ----------------------------

def mmr_select(query_vec: np.ndarray,
               cand_vecs: np.ndarray,
               cand_global_idx: np.ndarray,
               k: int,
               lambda_: float) -> np.ndarray:
    """
    Maximal Marginal Relevance selection on cosine vectors (normalized).
    """
    sim_q = cand_vecs @ query_vec               # (N,)
    sim_pair = cand_vecs @ cand_vecs.T          # (N,N)

    selected_local: List[int] = []
    remaining = list(range(len(cand_global_idx)))

    while remaining and len(selected_local) < k:
        best_score, best_i = -1e9, None
        for i in remaining:
            if not selected_local:
                score = float(sim_q[i])
            else:
                max_sim_sel = max(float(sim_pair[i, j]) for j in selected_local)
                score = float(lambda_ * sim_q[i] - (1.0 - lambda_) * max_sim_sel)
            if score > best_score:
                best_score, best_i = score, i
        selected_local.append(best_i)
        remaining.remove(best_i)

    return cand_global_idx[np.array(selected_local, dtype=int)]

def retrieve_exemplars_by_anchor(anchor_idx: int,
                                *,
                                faiss_index,
                                X: np.ndarray,
                                k: int,
                                candidates: int,
                                lambda_: float) -> List[int]:
    """
    Use the anchor's embedding vector as the query; retrieve top-N candidates via FAISS,
    then MMR-select k exemplars.
    """
    qvec = X[anchor_idx]  # already normalized
    sims, I = faiss_index.search(qvec[None, :], candidates)
    cand_global = I[0].astype(int)

    cand_vecs = X[cand_global]
    selected = mmr_select(qvec, cand_vecs, cand_global, k=k, lambda_=lambda_)
    return selected.tolist()

# ----------------------------
# Auto spec builder (no "types")
# ----------------------------

def auto_spec_from_exemplars(exemplars: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """
    Make a lightweight spec string (for the generator prompt) from what we already have.
    No clustering/types; just summarize domain + common concepts + target difficulty band.
    """
    domain = (exemplars[0].get("domain") or "fma").lower()

    # difficulty: median of available
    diffs = [e.get("analysis", {}).get("difficulty") for e in exemplars]
    diffs = [d for d in diffs if isinstance(d, int)]
    if diffs:
        diffs_sorted = sorted(diffs)
        diff = diffs_sorted[len(diffs_sorted)//2]
        diff_band = f"{max(1, diff-1)}-{min(10, diff+1)}"
    else:
        diff_band = "5-7"

    # top concepts (by frequency)
    from collections import Counter
    c = Counter()
    for e in exemplars:
        for tag in e.get("analysis", {}).get("concepts", []) or []:
            if isinstance(tag, str) and tag.strip():
                c[tag.strip().lower()] += 1
    top = [t for t,_ in c.most_common(4)]
    concept_part = ", ".join(top) if top else "(unspecified)"

    spec = f"{domain.upper()} auto; difficulty {diff_band}; concepts: {concept_part}; contest-style; novel numbers/wording."

    target = {
        "domain": domain,
        "desired_concepts": top,
        "difficulty_band": diff_band,
    }
    return spec, target

# ----------------------------
# Generation
# ----------------------------

def json_loads_loose(s: str) -> Dict[str, Any]:
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j < 0 or j <= i:
        raise ValueError("No JSON found in model output")
    return json.loads(s[i:j+1])

GEN_SYSTEM = """You generate a novel STEM competition problem using exemplar structures.

Hard rules:
- Do NOT copy numbers/wording from exemplars.
- Must be solvable and unambiguous.
- Keep the question in the same general contest style as exemplars.
- Output ONLY valid JSON with the requested keys.
"""

GEN_USER = """Spec (auto-derived):
{spec}

Anchor/target skeleton (structure intent):
{target_skeleton_text}

Exemplars (question view + skeleton):
{exemplar_block}

Return JSON with keys:
- new_problem:
  - stem
  - choices (list) (if MCQ; otherwise empty list)
- new_analysis:
  - skeleton (structured object like your pipeline uses)
  - concepts (list)
  - skills (list)
  - difficulty (1-10)
  - structure_tags (list)
  - diagram_required (bool)
- solution:
  - full_solution (string, complete)
  - final_answer (string)
  - quick_checks (list)
"""

def generate_from_exemplars(spec: str,
                            target_skeleton_text: str,
                            exemplars: List[Dict[str, Any]],
                            model: str = "qwen2.5:7b-instruct") -> Dict[str, Any]:
    exemplar_block = "\n\n".join(
        f"EX {i+1}\nQ:\n{question_view(d)}\n\nSKEL:\n{json.dumps(d.get('analysis', {}).get('skeleton', {}), ensure_ascii=False)}"
        for i, d in enumerate(exemplars)
    )

    user = GEN_USER.format(
        spec=spec,
        target_skeleton_text=target_skeleton_text,
        exemplar_block=exemplar_block,
    )

    return json_loads_loose(
        generate_text(model, user, system=GEN_SYSTEM, temperature=0.6)
    )

# ----------------------------
# Final schema assembly
# ----------------------------

def assemble_schema(out_id: str,
                    domain: str,
                    source_note: str,
                    generated: Dict[str, Any]) -> Dict[str, Any]:
    stem = generated["new_problem"]["stem"]
    choices = generated["new_problem"].get("choices", [])

    doc = {
        "id": out_id,
        "domain": domain,
        "source": {
            "corpus": source_note,
            "year": None,
            "variant": None,
            "pdf": None,
            "page_range": None,
        },
        "problem": {
            "stem": stem,
            "choices": choices,
            "answer_key": None,
            "units_expected": True if domain == "fma" else None,
            "diagrams": [],
        },
        "analysis": {
            "skeleton": generated["new_analysis"]["skeleton"],
            "concepts": generated["new_analysis"].get("concepts", []),
            "skills": generated["new_analysis"].get("skills", []),
            "difficulty": generated["new_analysis"].get("difficulty", None),
            "structure_tags": generated["new_analysis"].get("structure_tags", []),
            "diagram_required": generated["new_analysis"].get("diagram_required", False),
        },
        "checks": {
            "quick_checks": generated["solution"].get("quick_checks", []),
            "validators": ["dimensional", "sanity_bounds"] if domain == "fma" else ["sanity_bounds"],
            "flags": [],
        },
        "solution": {
            "full_solution": generated["solution"].get("full_solution", ""),
            "final_answer": generated["solution"].get("final_answer", ""),
        },
        "retrieval": {
            "embed_views": {
                "question": "stem + choices",
                "skeleton": "analysis.skeleton (canonical join)",
                "diagram_alt": "problem.diagrams[].alt_text (joined)",
            },
            "embedding_ref": None,
        },
    }
    return doc

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="enriched.jsonl", help="Enriched corpus JSONL")
    ap.add_argument("--index-dir", default="index")
    ap.add_argument("--view", default="skeleton", choices=["skeleton", "question"])
    ap.add_argument("--num", type=int, default=20, help="How many problems to generate")
    ap.add_argument("--k", type=int, default=8, help="MMR-selected exemplars per generation")
    ap.add_argument("--candidates", type=int, default=60, help="Initial FAISS top-N per anchor")
    ap.add_argument("--lambda", dest="lambda_", type=float, default=0.7, help="MMR tradeoff (0-1)")
    ap.add_argument("--gen-model", default="qwen2.5:7b-instruct")
    ap.add_argument("--out", default="generated.jsonl", help="Output JSONL")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--id-prefix", default="gen", help="ID prefix for generated problems")
    ap.add_argument("--min-anchor-skel-chars", type=int, default=80,
                    help="Skip anchors whose skeleton_view is shorter than this")
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    index_dir = Path(args.index_dir)

    corpus = load_jsonl_map(corpus_path)
    ordered_ids = load_ids(index_dir / f"{args.view}_ids.txt")

    X = np.load(index_dir / f"{args.view}_embeddings.npy").astype("float32")
    faiss_index = faiss.read_index(str(index_dir / f"faiss_{args.view}.index"))

    rng = np.random.default_rng(args.seed)

    # Build a pool of valid anchors (must have a non-trivial skeleton_view)
    valid_anchor_idx = []
    for i, rid in enumerate(ordered_ids):
        doc = corpus.get(rid)
        if not doc:
            continue
        sk = skeleton_view(doc)
        if len(sk) >= args.min_anchor_skel_chars:
            valid_anchor_idx.append(i)

    if not valid_anchor_idx:
        raise RuntimeError("No valid anchors found (check corpus ids alignment or skeleton lengths).")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_lines: List[str] = []
    generated_count = 0
    tried = 0
    max_tries = max(args.num * 3, 50)

    while generated_count < args.num and tried < max_tries:
        tried += 1
        anchor_idx = int(rng.choice(valid_anchor_idx))
        anchor_id = ordered_ids[anchor_idx]
        anchor_doc = corpus[anchor_id]

        # Retrieve neighborhood exemplars around anchor
        sel_idx = retrieve_exemplars_by_anchor(
            anchor_idx,
            faiss_index=faiss_index,
            X=X,
            k=args.k,
            candidates=args.candidates,
            lambda_=args.lambda_,
        )
        exemplar_ids = [ordered_ids[i] for i in sel_idx]
        exemplars = [corpus[eid] for eid in exemplar_ids if eid in corpus]
        if len(exemplars) < max(3, args.k // 2):
            continue

        # Auto spec from exemplars; target skeleton is the anchor's skeleton text
        spec, _ = auto_spec_from_exemplars(exemplars)
        target_skel = skeleton_view(anchor_doc)
        if not target_skel:
            continue

        # Generate
        try:
            gen = generate_from_exemplars(spec, target_skel, exemplars, model=args.gen_model)
        except Exception:
            continue

        out_id = f"{args.id_prefix}_{generated_count:05d}"
        domain = (anchor_doc.get("domain") or "fma").lower()
        final = assemble_schema(out_id, domain, "generated_seedless_anchor_rag", gen)
        final["provenance"] = {
            "auto": True,
            "anchor_id": anchor_id,
            "exemplar_ids": exemplar_ids,
            "spec": spec,
            "anchor_skeleton_text": target_skel,
        }

        out_lines.append(json.dumps(final, ensure_ascii=False))
        generated_count += 1

    out_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    print(f"Wrote {out_path} ({generated_count} problems; tried {tried} anchors)")

if __name__ == "__main__":
    main()
