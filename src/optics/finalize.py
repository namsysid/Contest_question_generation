from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.circuit_lab.model_client import embed_texts
from src.circuit_lab.validate import answers_agree

from .common import read_jsonl, validate_item, write_jsonl
from .pipeline import judge_item, normalized, source_calibrated_audit


SELECTED = [
    # 10 introductory items, one from every major content area.
    ("production", "optics-b-generated-001"),
    ("production", "optics-b-generated-002"),
    ("production", "optics-b-generated-003"),
    ("production", "optics-b-generated-004"),
    ("production", "optics-b-generated-008"),
    ("production", "optics-b-generated-010"),
    ("production", "optics-b-generated-012"),
    ("shard_b", "optics-b-generated-004"),
    ("shard_c", "optics-b-generated-002"),
    ("targeted", "optics-b-generated-101"),
    # 6 applied/intermediate items.
    ("production", "optics-b-generated-006"),
    ("shard_c", "optics-b-generated-003"),
    ("targeted", "optics-b-generated-102"),
    ("targeted", "optics-b-generated-105"),
    ("targeted", "optics-b-generated-106"),
    ("targeted", "optics-b-generated-110"),
    # 2 linked-reasoning challenge items.
    ("targeted", "optics-b-generated-107"),
    ("targeted", "optics-b-generated-111"),
]


def load_candidates(root: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for directory in ("production", "shard_b", "shard_c", "targeted"):
        path = root / directory / "generated" / "items.jsonl"
        if not path.exists():
            continue
        for item in read_jsonl(path):
            item = copy.deepcopy(item)
            original_id = item["id"]
            item["candidate_origin"] = directory
            item["candidate_local_id"] = original_id
            item["id"] = f"optics-b-candidate-{len(rows) + 1:03d}"
            rows.append(item)
            lookup[(directory, original_id)] = item
    return rows, lookup


def embedding_audit(items: list[dict[str, Any]], candidate_vectors: dict[str, list[float]],
                    source_rows: list[dict[str, Any]], source_vectors: list[list[float]]) -> dict[str, Any]:
    matrix = normalized([candidate_vectors[item["id"]] for item in items])
    source_matrix = normalized(source_vectors)
    near_pairs = []
    maximum_pair = {"similarity": 0.0, "ids": []}
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            score = float(matrix[left] @ matrix[right])
            if score > maximum_pair["similarity"]:
                maximum_pair = {"similarity": score, "ids": [items[left]["id"], items[right]["id"]]}
            if score >= 0.875:
                near_pairs.append({"ids": [items[left]["id"], items[right]["id"]], "similarity": score})
    source_hits = []
    for index, item in enumerate(items):
        scores = source_matrix @ matrix[index]
        nearest = int(np.argmax(scores))
        source_hits.append({"id": item["id"], "source_id": source_rows[nearest]["id"],
                            "similarity": float(scores[nearest])})
    return {
        "method": "cosine similarity of text-embedding-3-small question embeddings",
        "generated_near_duplicate_threshold": 0.875,
        "source_rewrite_threshold": 0.94,
        "maximum_generated_pair": maximum_pair,
        "generated_pairs_at_or_above_threshold": near_pairs,
        "maximum_source_similarity": max(hit["similarity"] for hit in source_hits),
        "source_hits": source_hits,
        "difficulty_counts": dict(sorted(Counter(item["difficulty"] for item in items).items())),
        "response_type_counts": dict(Counter(item["response_type"] for item in items)),
        "topic_counts": dict(Counter(topic for item in items for topic in item["topics"])),
        "passes": not near_pairs and all(hit["similarity"] < 0.94 for hit in source_hits),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-shard Optics curation and final validation")
    parser.add_argument("--root", default="science_olympiad/optics_b")
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    output = root / "final"
    corpus = read_jsonl(root / "production" / "corpus" / "items.jsonl")
    source_q_rows = read_jsonl(root / "production" / "enriched" / "question_embeddings.jsonl")
    source_by_id = {row["id"]: row for row in corpus}
    candidates, lookup = load_candidates(root)
    write_jsonl(output / "generated" / "candidate_pool.jsonl", candidates)
    candidate_id_by_key = {(item["candidate_origin"], item["candidate_local_id"]): item["id"]
                           for item in candidates}
    candidate_reports = []
    for directory in ("production", "shard_b", "shard_c", "targeted"):
        report_path = root / directory / "validation" / "item_reports.jsonl"
        if not report_path.exists():
            continue
        for report in read_jsonl(report_path):
            key = (directory, str(report.get("id")))
            if key in candidate_id_by_key:
                report = copy.deepcopy(report)
                report["candidate_origin"] = directory
                report["candidate_local_id"] = report["id"]
                report["id"] = candidate_id_by_key[key]
                candidate_reports.append(report)
    write_jsonl(output / "validation" / "candidate_item_reports.jsonl", candidate_reports)
    candidate_embeddings_path = output / "enriched" / "candidate_question_embeddings.jsonl"
    if candidate_embeddings_path.exists() and {
            row["id"] for row in read_jsonl(candidate_embeddings_path)} == {item["id"] for item in candidates}:
        embedding_rows = read_jsonl(candidate_embeddings_path)
    else:
        vectors = embed_texts(args.embedding_model, [item["prompt"] for item in candidates], provider=args.provider)
        embedding_rows = [{"id": item["id"], "embedding": vector} for item, vector in zip(candidates, vectors)]
        write_jsonl(candidate_embeddings_path, embedding_rows)
    vector_by_id = {row["id"]: row["embedding"] for row in embedding_rows}
    staging_path = output / "generated" / "items_polished_staging.jsonl"
    if staging_path.exists():
        staged = {(row["candidate_origin"], row["candidate_local_id"]): row for row in read_jsonl(staging_path)}
        items = [copy.deepcopy(staged.get(key) or lookup[key]) for key in SELECTED]
    else:
        items = [copy.deepcopy(lookup[key]) for key in SELECTED]
    # The microscope product is a direct one-step warm-up, not an intermediate item.
    for serial, item in enumerate(items, 1):
        item["id"] = f"optics-b-final-{serial:03d}"
        if item["candidate_origin"] == "targeted" and item["candidate_local_id"] == "optics-b-generated-101":
            item["difficulty"] = 1
            (item.get("generation") or {}).get("plan", {})["difficulty"] = 1
    # Re-embed the final student-facing wording. This is intentionally separate
    # from candidate-pool embeddings because the concise-polish stage may change
    # surface text while preserving the reasoning plan.
    final_vectors = embed_texts(args.embedding_model, [item["prompt"] for item in items], provider=args.provider)
    final_embedding_rows = [{"id": item["id"], "embedding": vector}
                            for item, vector in zip(items, final_vectors)]
    write_jsonl(output / "enriched" / "final_question_embeddings.jsonl", final_embedding_rows)
    selected_vectors = {row["id"]: row["embedding"] for row in final_embedding_rows}
    source_vector_by_id = {row["id"]: row["embedding"] for row in source_q_rows}
    aligned_source_rows = [row for row in corpus if row["id"] in source_vector_by_id]
    source_vectors = [source_vector_by_id[row["id"]] for row in aligned_source_rows]
    reproducible = embedding_audit(items, selected_vectors, aligned_source_rows, source_vectors)
    audit_path = output / "validation" / "embedding_audit_final.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(reproducible, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not reproducible["passes"]:
        raise RuntimeError("Final selection failed deterministic embedding audit; inspect embedding_audit_final.json")

    reports_path = output / "validation" / "item_reports_final.jsonl"
    if args.audit_only and reports_path.exists():
        reports = read_jsonl(reports_path)
    else:
        reports = []
        for index, item in enumerate(items):
            errors = validate_item(item)
            generation = item.get("generation") or {}
            plan = generation.get("plan") or {}
            anchor = source_by_id.get(generation.get("anchor_id")) or {}
            judge = None
            if not errors:
                judge = judge_item(item, plan, anchor, args.model, args.provider, 61000 + index)
                judge["answer_agrees"] = answers_agree(item, judge.get("independent_answer"))
            required = ("solvable", "science_correct", "division_b_appropriate", "optics_relevant",
                        "difficulty_match", "competition_faithful", "style_faithful", "novel", "answer_agrees")
            valid = not errors and judge is not None and all(judge.get(key) is True for key in required)
            reports.append({"id": item["id"], "valid": valid, "deterministic_errors": errors, "judge": judge})
            print(f"Revalidated {index + 1}/{len(items)}: {item['id']} valid={valid}", flush=True)
        write_jsonl(reports_path, reports)
    write_jsonl(output / "generated" / "items_final.jsonl", items)
    batch = source_calibrated_audit(items, corpus, args.model, args.provider, 69999)
    (output / "validation" / "batch_audit_final.json").write_text(
        json.dumps(batch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Final: pool={len(candidates)}, valid={sum(report.get('valid') is True for report in reports)}/{len(items)}, "
          f"embedding_audit={reproducible['passes']}, batch_audit={batch.get('overall_good')}")


if __name__ == "__main__":
    main()
