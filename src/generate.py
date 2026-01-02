#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import faiss
from openai import OpenAI
import os

from dotenv import load_dotenv
load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set. Please set it in .env file or environment variable.")

client = OpenAI()

# ----------------------------
# Utilities
# ----------------------------

def load_jsonl_map(path: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["id"]] = r
    return out

def load_ids(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()

def embed_one(text: str, model: str = "text-embedding-3-large") -> np.ndarray:
    resp = client.embeddings.create(model=model, input=[text])
    v = np.array(resp.data[0].embedding, dtype="float32")[None, :]
    faiss.normalize_L2(v)
    return v[0]  # (d,)

def mmr_select(query_vec: np.ndarray,
               cand_vecs: np.ndarray,
               cand_global_idx: np.ndarray,
               k: int,
               lambda_: float) -> np.ndarray:
    sim_q = cand_vecs @ query_vec               # (N,)
    sim_pair = cand_vecs @ cand_vecs.T          # (N,N)

    selected_local = []
    remaining = list(range(len(cand_global_idx)))

    while remaining and len(selected_local) < k:
        best_score, best_i = -1e9, None
        for i in remaining:
            if not selected_local:
                score = sim_q[i]
            else:
                max_sim_sel = max(sim_pair[i, j] for j in selected_local)
                score = lambda_ * sim_q[i] - (1.0 - lambda_) * max_sim_sel
            if score > best_score:
                best_score, best_i = score, i
        selected_local.append(best_i)
        remaining.remove(best_i)

    return cand_global_idx[np.array(selected_local, dtype=int)]

def question_view(doc: Dict[str, Any]) -> str:
    stem = doc.get("problem", {}).get("stem", "")
    choices = doc.get("problem", {}).get("choices", [])
    return (stem + "\n" + "\n".join(choices)).strip()

def skeleton_view(doc: Dict[str, Any]) -> str:
    # you already store canonical text as analysis._skeleton_text
    return (doc.get("analysis", {}).get("_skeleton_text") or "").strip()

def json_loads_loose(s: str) -> Dict[str, Any]:
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j < 0 or j <= i:
        raise ValueError("No JSON found in model output")
    return json.loads(s[i:j+1])

# ----------------------------
# Seedless target skeleton
# ----------------------------

SPEC_TO_TARGET_SYSTEM = """You convert a generation specification into a compact target solution skeleton.
Return only JSON. No extra text."""

SPEC_TO_TARGET_USER = """Spec:
{spec}

Return JSON with keys:
- domain: "fma" or "usnco"
- target_skeleton_text: a compact skeleton query string (not prose; equations/plan ok)
- desired_concepts: list of short strings
- difficulty: integer 1-10
"""

def spec_to_target(spec: str, model: str = "gpt-4.1-mini") -> Dict[str, Any]:
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SPEC_TO_TARGET_SYSTEM},
            {"role": "user", "content": SPEC_TO_TARGET_USER.format(spec=spec)},
        ],
        temperature=0.2,
    )
    return json_loads_loose(resp.output_text)

# ----------------------------
# Retrieval (FAISS + MMR)
# ----------------------------

def retrieve_exemplars(target_skeleton_text: str,
                       index_dir: Path,
                       view: str,
                       k: int,
                       candidates: int,
                       lambda_: float,
                       emb_model: str) -> List[int]:
    idx = faiss.read_index(str(index_dir / f"faiss_{view}.index"))
    X = np.load(index_dir / f"{view}_embeddings.npy").astype("float32")  # normalized
    qvec = embed_one(target_skeleton_text, model=emb_model)

    sims, I = idx.search(qvec[None, :], candidates)
    cand_global = I[0].astype(int)
    cand_vecs = X[cand_global]

    selected = mmr_select(qvec, cand_vecs, cand_global, k=k, lambda_=lambda_)
    return selected.tolist()

# ----------------------------
# Generation
# ----------------------------

GEN_SYSTEM = """You generate a novel STEM competition problem consistent with the given spec and exemplar structures.

Hard rules:
- Do NOT copy numbers/wording from exemplars.
- Must be solvable and unambiguous.
- Output ONLY valid JSON with the requested keys.
"""

GEN_USER = """Spec (human):
{spec}

Target skeleton (seedless query rep):
{target_skeleton_text}

Use these exemplars as structural guidance (question view + skeleton):
{exemplar_block}

Return JSON with keys:
- new_problem:
  - stem
  - choices (list)  (if MCQ)
- new_analysis:
  - skeleton (structured object like your pipeline uses)
  - concepts (list)
  - skills (list)
  - difficulty (1-10)
  - structure_tags (list)
  - diagram_required (bool)
- solution:
  - full_solution (string, can be concise but complete)
  - final_answer (string)
  - quick_checks (list)
"""

def generate_from_exemplars(spec: str,
                            target: Dict[str, Any],
                            exemplars: List[Dict[str, Any]],
                            model: str = "gpt-4.1") -> Dict[str, Any]:
    exemplar_block = "\n\n".join(
        f"EX {i+1}\nQ:\n{question_view(d)}\n\nSKEL:\n{json.dumps(d.get('analysis', {}).get('skeleton', {}), ensure_ascii=False)}"
        for i, d in enumerate(exemplars)
    )

    user = GEN_USER.format(
        spec=spec,
        target_skeleton_text=target["target_skeleton_text"],
        exemplar_block=exemplar_block,
    )

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.6,
    )
    return json_loads_loose(resp.output_text)

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
            "answer_key": None,  # you can fill if you want letter mapping later
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
    # If --spec is omitted, the script will automatically discover "types" via
    # clustering in embedding space and generate many questions (seedless).
    ap.add_argument("--spec", default=None, help="Optional seedless spec string. If omitted, auto-generate many.")
    ap.add_argument("--corpus", default="enriched.jsonl", help="Enriched corpus JSONL")
    ap.add_argument("--index-dir", default="index")
    ap.add_argument("--view", default="skeleton", choices=["skeleton", "question"])
    ap.add_argument("--k", type=int, default=8, help="MMR-selected exemplars")
    ap.add_argument("--candidates", type=int, default=50, help="Initial FAISS top-N (single mode)")
    ap.add_argument("--lambda", dest="lambda_", type=float, default=0.7, help="MMR tradeoff")
    ap.add_argument("--emb-model", default="text-embedding-3-large")
    ap.add_argument("--spec-model", default="gpt-4.1-mini", help="Only used when --spec is provided")
    ap.add_argument("--gen-model", default="gpt-4.1")
    ap.add_argument("--out", default="generated.jsonl", help="Output file (.json for single, .jsonl for batch)")
    ap.add_argument("--id", default="gen_0001", help="ID for generated problem (single mode)")
    # Auto mode options
    ap.add_argument("--num", type=int, default=20, help="How many problems to generate in auto mode")
    ap.add_argument("--types", type=int, default=12, help="How many discovered 'types' (clusters) in auto mode")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (auto mode clustering)")
    ap.add_argument("--id-prefix", default="gen", help="ID prefix for auto mode")
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    index_dir = Path(args.index_dir)

    corpus = load_jsonl_map(corpus_path)
    ordered_ids = load_ids(index_dir / f"{args.view}_ids.txt")
    X = np.load(index_dir / f"{args.view}_embeddings.npy").astype("float32")  # normalized

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # SINGLE MODE: user-provided spec (seedless, but guided)
    # ----------------------------
    if args.spec is not None:
        target = spec_to_target(args.spec, model=args.spec_model)
        sel_global_idx = retrieve_exemplars(
            target["target_skeleton_text"],
            index_dir=index_dir,
            view=args.view,
            k=args.k,
            candidates=args.candidates,
            lambda_=args.lambda_,
            emb_model=args.emb_model,
        )
        exemplar_ids = [ordered_ids[i] for i in sel_global_idx]
        exemplars = [corpus[eid] for eid in exemplar_ids]
        generated = generate_from_exemplars(args.spec, target, exemplars, model=args.gen_model)
        final = assemble_schema(args.id, target["domain"], "generated_seedless_rag", generated)
        final["provenance"] = {"spec": args.spec, "target": target, "exemplar_ids": exemplar_ids}
        out_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")
        print("Exemplars:", exemplar_ids)
        return

    # ----------------------------
    # AUTO MODE: no query/spec.
    # Discover 'types' and generate many by sampling each cluster.
    # ----------------------------
    cluster_ids = discover_types_via_kmeans(X, num_types=args.types, seed=args.seed)
    num_clusters = int(cluster_ids.max()) + 1

    # Precompute member lists
    cluster_members: List[np.ndarray] = [np.where(cluster_ids == c)[0] for c in range(num_clusters)]
    cluster_sizes = [int(m.size) for m in cluster_members]
    cluster_order = sorted(range(num_clusters), key=lambda c: cluster_sizes[c], reverse=True)

    def doc_by_index(i: int) -> Dict[str, Any]:
        return corpus[ordered_ids[i]]

    # Choose a representative target skeleton per cluster: nearest-to-centroid member.
    cluster_target_idx: Dict[int, int] = {}
    for c in range(num_clusters):
        members = cluster_members[c]
        if members.size == 0:
            continue
        centroid = X[members].mean(axis=0).astype("float32")
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        sims = X[members] @ centroid
        cluster_target_idx[c] = int(members[int(np.argmax(sims))])

    generated_count = 0
    out_lines: List[str] = []

    # Round-robin through largest clusters to cover common types.
    c_cursor = 0
    while generated_count < args.num and cluster_order:
        c = cluster_order[c_cursor % len(cluster_order)]
        c_cursor += 1
        members = cluster_members[c]
        if members.size == 0:
            continue

        # MMR-select exemplars within the cluster (no external query needed)
        sel_idx = select_exemplars_for_cluster(X, cluster_ids, c, k=args.k, lambda_=args.lambda_)
        if not sel_idx:
            continue
        exemplars = [doc_by_index(i) for i in sel_idx]

        # Auto spec + target (no LLM): use representative skeleton from cluster
        spec, target = auto_spec_from_exemplars(exemplars)
        rep_i = cluster_target_idx.get(c, sel_idx[0])
        rep_doc = doc_by_index(rep_i)
        target["target_skeleton_text"] = skeleton_view(rep_doc)
        if not target["target_skeleton_text"]:
            # fallback: join exemplars' skeleton views
            target["target_skeleton_text"] = "\n".join(skeleton_view(d) for d in exemplars if skeleton_view(d))

        # Generate
        gen = generate_from_exemplars(spec, target, exemplars, model=args.gen_model)
        out_id = f"{args.id_prefix}_{generated_count:04d}_c{c:02d}"
        final = assemble_schema(out_id, target["domain"], "generated_seedless_rag_auto", gen)
        final["provenance"] = {
            "auto": True,
            "cluster": int(c),
            "spec": spec,
            "target": target,
            "exemplar_ids": [ordered_ids[i] for i in sel_idx],
        }
        out_lines.append(json.dumps(final, ensure_ascii=False))
        generated_count += 1

    out_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    print(f"Wrote {out_path} ({generated_count} problems)")

if __name__ == "__main__":
    main()
