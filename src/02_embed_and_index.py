#!/usr/bin/env python3
"""
02_embed_and_index_diverse.py

Build embeddings from abstract solution skeletons (optypes + laws),
preserving abstraction while improving separability.
"""

import json
import argparse
from openai import OpenAI
import os
import dotenv
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Check that API key is set
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set. Please set it in .env file or environment variable.")

client = OpenAI()

def build_embedding_text(rec):
    optypes = rec.get("optypes", [])
    laws = rec.get("laws", [])
    target = rec.get("target", "unknown")

    return (
        "Abstract solution structure.\n"
        f"Operators: {', '.join(sorted(optypes))}\n"
        f"Laws: {', '.join(sorted(laws))}\n"
        f"Target quantity: {target}\n"
    )

def main(inp, outp, model):
    with open(inp) as f:
        records = [json.loads(l) for l in f]

    out = []
    for r in records:
        text = r.get("embedding_text") or build_embedding_text(r)
        emb = client.embeddings.create(
            model=model,
            input=text
        ).data[0].embedding
        r["embedding"] = emb
        out.append(r)

    with open(outp, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="embedded.jsonl")
    ap.add_argument("--model", default="text-embedding-3-large")
    args = ap.parse_args()
    main(args.input, args.output, args.model)
