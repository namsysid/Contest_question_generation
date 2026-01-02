#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import faiss
from tqdm import tqdm
from openai import OpenAI
import os

from dotenv import load_dotenv
load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set. Please set it in .env file or environment variable.")


client = OpenAI()

def embed_texts(texts, model="text-embedding-3-large"):
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main(in_jsonl="enriched.jsonl", out_dir="index", view="skeleton"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    docs = list(load_jsonl(Path(in_jsonl)))
    ids = [d["id"] for d in docs]

    def get_view(d):
        if view == "skeleton":
            # use your canonical skeleton text (you already set _skeleton_text)
            return d["analysis"].get("_skeleton_text") or ""
        elif view == "question":
            stem = d["problem"].get("stem", "")
            choices = d["problem"].get("choices", [])
            return stem + "\n" + "\n".join(choices)
        else:
            raise ValueError("view must be 'skeleton' or 'question'")

    texts = [get_view(d) for d in docs]

    # batch embed
    vecs = []
    bs = 128
    for i in tqdm(range(0, len(texts), bs), desc=f"Embedding view={view}"):
        vecs.extend(embed_texts(texts[i:i+bs]))

    X = np.array(vecs, dtype="float32")
    faiss.normalize_L2(X)  # cosine via inner product

    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    # save
    np.save(out / f"{view}_embeddings.npy", X)
    faiss.write_index(index, str(out / f"faiss_{view}.index"))
    (out / f"{view}_ids.txt").write_text("\n".join(ids), encoding="utf-8")

    print(f"Saved: {out}/faiss_{view}.index, {out}/{view}_embeddings.npy, {out}/{view}_ids.txt")

if __name__ == "__main__":
    # example: python src/04_embed_and_index.py enriched.jsonl index skeleton
    import sys
    in_jsonl = sys.argv[1] if len(sys.argv) > 1 else "enriched.jsonl"
    out_dir  = sys.argv[2] if len(sys.argv) > 2 else "index"
    view     = sys.argv[3] if len(sys.argv) > 3 else "skeleton"
    main(in_jsonl=in_jsonl, out_dir=out_dir, view=view)
