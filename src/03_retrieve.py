#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import faiss
from openai import OpenAI
import os

from dotenv import load_dotenv
load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set. Please set it in .env file or environment variable.")

client = OpenAI()

def embed_one(text, model="text-embedding-3-large"):
    resp = client.embeddings.create(model=model, input=[text])
    v = np.array(resp.data[0].embedding, dtype="float32")[None, :]
    faiss.normalize_L2(v)
    return v[0]  # (d,)

def load_ids(path: Path):
    return path.read_text(encoding="utf-8").splitlines()

def load_corpus(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    return rows

def mmr_select(query_vec: np.ndarray,
               cand_vecs: np.ndarray,
               cand_idx: np.ndarray,
               k: int,
               lambda_: float) -> np.ndarray:
    """
    Maximal Marginal Relevance selection on cosine-sim vectors.
    query_vec: (d,) normalized
    cand_vecs: (N,d) normalized
    cand_idx: (N,) indices (into global id list)
    returns: (k,) selected cand_idx subset
    """
    # similarity to query
    sim_q = cand_vecs @ query_vec  # (N,)

    # pairwise similarity among candidates (N,N)
    sim_pair = cand_vecs @ cand_vecs.T

    selected_local = []
    remaining = list(range(len(cand_idx)))

    while remaining and len(selected_local) < k:
        best_score = -1e9
        best_i = None
        for i in remaining:
            if not selected_local:
                score = sim_q[i]
            else:
                max_sim_sel = max(sim_pair[i, j] for j in selected_local)
                score = lambda_ * sim_q[i] - (1.0 - lambda_) * max_sim_sel
            if score > best_score:
                best_score = score
                best_i = i
        selected_local.append(best_i)
        remaining.remove(best_i)

    return cand_idx[np.array(selected_local, dtype=int)]

def main(query_text: str,
         index_dir="index",
         view="skeleton",
         corpus_jsonl="data/enriched_schemae/2015-schemae.jsonl",
         k=10,
         candidates=50,
         lambda_=0.7,
         embedding_model="text-embedding-3-large"):

    index_dir = Path(index_dir)

    idx = faiss.read_index(str(index_dir / f"faiss_{view}.index"))
    ids = load_ids(index_dir / f"{view}_ids.txt")
    corpus = load_corpus(Path(corpus_jsonl))

    # Needed for MMR
    emb_path = index_dir / f"{view}_embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Missing {emb_path}. Run your embedding/index step first."
        )
    X = np.load(emb_path).astype("float32")  # (M,d), normalized already if you saved it that way

    qvec = embed_one(query_text, model=embedding_model)  # (d,)

    # initial retrieval (top-N)
    sims, I = idx.search(qvec[None, :], candidates)
    cand_global = I[0].astype(int)  # indices into ids / X

    # candidate vectors
    cand_vecs = X[cand_global]

    # MMR rerank/select
    sel_global = mmr_select(qvec, cand_vecs, cand_global, k=k, lambda_=lambda_)

    # print results (with original similarity to query for reference)
    for gi in sel_global:
        rid = ids[gi]
        score = float(X[gi] @ qvec)
        doc = corpus.get(rid, {})
        concepts = doc.get("analysis", {}).get("concepts", [])
        print(f"{score: .3f}  {rid}  concepts={concepts}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="Query text (usually target skeleton text)")
    ap.add_argument("--index-dir", default="index")
    ap.add_argument("--view", default="skeleton", choices=["skeleton", "question"])
    ap.add_argument("--corpus", default="data/enriched_schemae/2015-schemae.jsonl")
    ap.add_argument("--k", type=int, default=10, help="final results after MMR")
    ap.add_argument("--candidates", type=int, default=50, help="initial top-N from FAISS")
    ap.add_argument("--lambda", dest="lambda_", type=float, default=0.7, help="MMR tradeoff (0-1)")
    ap.add_argument("--emb-model", default="text-embedding-3-large")
    args = ap.parse_args()

    main(
        args.query,
        index_dir=args.index_dir,
        view=args.view,
        corpus_jsonl=args.corpus,
        k=args.k,
        candidates=args.candidates,
        lambda_=args.lambda_,
        embedding_model=args.emb_model,
    )
