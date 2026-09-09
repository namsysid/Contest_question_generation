from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from src.circuit_lab.model_client import embed_texts

from .common import read_jsonl, validate_item
from .pipeline import normalized
from .polish import word_cap


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible Machines bank quality gate")
    parser.add_argument("--run-dir", default="science_olympiad/machines_b/final")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai")
    args = parser.parse_args()
    root = Path(args.run_dir)
    items = read_jsonl(root / "generated" / "items_final.jsonl")
    source = read_jsonl(root / "corpus" / "items.jsonl")
    reports = {row["id"]: row for row in read_jsonl(root / "validation" / "item_reports_final.jsonl")}

    item_vectors = normalized(embed_texts(
        args.embedding_model, [row["prompt"] for row in items], provider=args.provider))
    source_vectors = normalized(embed_texts(
        args.embedding_model, [row["prompt"] for row in source], provider=args.provider))
    pairwise = item_vectors @ item_vectors.T
    np.fill_diagonal(pairwise, -1)
    pair_position = np.unravel_index(int(np.argmax(pairwise)), pairwise.shape)
    source_scores = item_vectors @ source_vectors.T
    source_position = np.unravel_index(int(np.argmax(source_scores)), source_scores.shape)

    source_counts = Counter(int(row.get("difficulty") or 2) for row in source)
    item_counts = Counter(int(row["difficulty"]) for row in items)
    expected = {difficulty: len(items) * source_counts[difficulty] / len(source) for difficulty in range(1, 6)}
    difficulty_good = all(abs(item_counts[difficulty] - expected[difficulty]) <= 1 for difficulty in range(1, 6))
    topics = sorted({topic for row in items for topic in row.get("topics") or []})
    source_topics = sorted({topic for row in source for topic in row.get("topics") or []
                            if topic != "classical_mechanics"})
    formats = sorted({row["response_type"] for row in items})
    deterministic_errors = {row["id"]: validate_item(row) for row in items if validate_item(row)}
    length_failures = {row["id"]: len(re.findall(r"\w+", row["prompt"])) for row in items
                       if len(re.findall(r"\w+", row["prompt"])) > word_cap(int(row["difficulty"]))}
    target_skills = [str((((row.get("generation") or {}).get("plan") or {}).get("target_skill") or "")).strip().lower()
                     for row in items]
    exact_skill_duplicates = sorted({skill for skill, count in Counter(target_skills).items() if skill and count > 1})
    individual_good = all(reports.get(row["id"], {}).get("valid") is True for row in items)
    max_pairwise = float(pairwise[pair_position])
    max_source = float(source_scores[source_position])
    checks = {
        "all_items_independently_valid": individual_good,
        "deterministic_schema_and_scope": not deterministic_errors,
        "source_proportional_difficulty_within_one_item": difficulty_good,
        "covers_all_source_machine_topic_families": set(source_topics).issubset(topics),
        "includes_mcq_and_constructed_response": "multiple_choice" in formats and
                                                  bool({"numeric", "short_answer"}.intersection(formats)),
        "within_style_word_caps": not length_failures,
        "no_exact_target_skill_duplicates": not exact_skill_duplicates,
        "pairwise_embedding_similarity_below_0_875": max_pairwise < 0.875,
        "source_embedding_similarity_below_0_94": max_source < 0.94,
    }
    result = {
        "overall_good": all(checks.values()),
        "checks": checks,
        "item_count": len(items),
        "difficulty_counts": dict(sorted(item_counts.items())),
        "projected_source_counts": expected,
        "topics": topics,
        "formats": formats,
        "maximum_pairwise_similarity": {
            "score": max_pairwise, "ids": [items[pair_position[0]]["id"], items[pair_position[1]]["id"]]},
        "maximum_source_similarity": {
            "score": max_source, "generated_id": items[source_position[0]]["id"],
            "source_id": source[source_position[1]]["id"]},
        "deterministic_errors": deterministic_errors,
        "length_failures": length_failures,
        "exact_skill_duplicates": exact_skill_duplicates,
    }
    target = root / "validation" / "reproducible_bank_audit.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["overall_good"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
