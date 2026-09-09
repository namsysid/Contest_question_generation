from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from .model_client import embed_texts
from .common import graph_text, read_jsonl, write_jsonl


def normalized(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)


def choose_anchors(vectors: list[list[float]], fraction: float, k: int) -> list[int]:
    if not vectors:
        return []
    matrix = normalized(vectors)
    similarities = matrix @ matrix.T
    np.fill_diagonal(similarities, -1.0)
    neighbor_count = max(1, min(k, len(matrix) - 1)) if len(matrix) > 1 else 1
    density = similarities.max(axis=1) if len(matrix) == 1 else np.partition(similarities, -neighbor_count, axis=1)[:, -neighbor_count:].mean(axis=1)
    count = max(1, round(len(matrix) * fraction))
    return np.argsort(-density)[:count].tolist()


def build_indices(rows: list[dict[str, Any]], model: str, batch: int, anchor_fraction: float, density_k: int, provider: str = "ollama"):
    question_texts = [str(row.get("prompt") or "") + "\n" + "\n".join(str(p.get("prompt") or "") for p in row.get("parts") or []) for row in rows]
    structure_texts = [str(row.get("graph_text") or graph_text(row)) for row in rows]
    question_vectors = embed_texts(model, question_texts, provider=provider)
    structure_vectors = embed_texts(model, structure_texts, provider=provider)
    q_rows, s_rows = [], []
    for row, question, structure, qvec, svec in zip(rows, question_texts, structure_texts, question_vectors, structure_vectors):
        metadata = {key: row.get(key) for key in ("id", "response_type", "topics", "section", "points")}
        q_rows.append({**metadata, "question_text": question, "embedding": qvec})
        s_rows.append({**metadata, "graph_text": structure, "embedding": svec})
    anchor_indices: set[int] = set()
    formats = {str(row.get("response_type")) for row in rows}
    for response_type in formats:
        group = [i for i, row in enumerate(rows) if str(row.get("response_type")) == response_type]
        local = choose_anchors([structure_vectors[i] for i in group], anchor_fraction, density_k)
        anchor_indices.update(group[i] for i in local)
    anchors = [{"id": rows[i].get("id"), "response_type": rows[i].get("response_type"), "topics": rows[i].get("topics")} for i in sorted(anchor_indices)]
    return q_rows, s_rows, anchors


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dual Circuit Lab question/structure embedding indices")
    parser.add_argument("--input", required=True)
    parser.add_argument("--question-out", required=True)
    parser.add_argument("--structure-out", required=True)
    parser.add_argument("--anchors-out", required=True)
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--provider", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--anchor-fraction", type=float, default=0.15)
    parser.add_argument("--density-k", type=int, default=10)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    q_rows, s_rows, anchors = build_indices(rows, args.model, args.batch, args.anchor_fraction, args.density_k, args.provider)
    write_jsonl(args.question_out, q_rows)
    write_jsonl(args.structure_out, s_rows)
    write_jsonl(args.anchors_out, anchors)
    print(f"Embedded {len(rows)} items; selected {len(anchors)} anchors")


if __name__ == "__main__":
    main()
