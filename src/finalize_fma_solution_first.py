#!/usr/bin/env python3
"""Merge gatekeeper-PASS solution-first F=ma generations into one balanced bank."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


TOPICS = {
    "kinematics": "Kinematics",
    "forces": "Forces",
    "energy": "Energy",
    "momentum": "Momentum",
    "circular_gravity": "Circular & gravity",
    "rotation": "Rotation",
    "oscillations": "Oscillations",
    "fluids": "Fluids",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def valid_pass(row: dict) -> bool:
    gate = row.get("_gatekeeper") or {}
    answer = str(row.get("answer") or "").upper()
    verified = str((gate.get("answer_consistency") or {}).get("answer_verified") or "").upper()
    choices = row.get("choices")
    return (
        gate.get("verdict") == "PASS"
        and answer in "ABCDE" and len(answer) == 1
        and verified == answer
        and isinstance(choices, dict) and set(choices) == set("ABCDE")
        and bool(str(row.get("question") or "").strip())
        and bool(str(row.get("solution") or "").strip())
    )


def stem_only(question: str) -> str:
    """Remove a trailing rendered A-E list; choices live in their own schema field."""
    match = re.search(
        r"(?s)\n\s*\(A\)\s+.+?\s+\(B\)\s+.+?\s+\(C\)\s+.+?\s+\(D\)\s+.+?\s+\(E\)\s+.+?\s*$",
        question,
    )
    return question[: match.start()].rstrip() if match else question.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("fma_solution_first/topics"))
    parser.add_argument("--out", type=Path, default=Path("fma_solution_first/questions.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("fma_solution_first/validation_summary.json"))
    args = parser.parse_args()

    bank: list[dict] = []
    summary: dict[str, dict] = {}
    seen_questions: set[str] = set()
    for topic_key, topic in TOPICS.items():
        topic_dir = args.root / topic_key
        by_bundle: dict[str, dict] = {}
        source_by_bundle: dict[str, str] = {}
        for filename in ("generated_problems.jsonl", "retry_problems_1.jsonl", "retry_problems_2.jsonl", "retry_problems_3.jsonl"):
            for row in read_jsonl(topic_dir / filename):
                bundle_id = str((row.get("_meta") or {}).get("bundle_id") or "")
                if bundle_id and valid_pass(row):
                    by_bundle[bundle_id] = row
                    source_by_bundle[bundle_id] = filename
        if len(by_bundle) != 15:
            raise ValueError(f"{topic_key}: expected 15 verified PASS records, found {len(by_bundle)}")

        skeleton_by_bundle: dict[str, dict] = {}
        for filename in ("generated_skeletons.jsonl", "retry_skeletons_2.jsonl", "retry_skeletons_3.jsonl"):
            for row in read_jsonl(topic_dir / filename):
                skeleton_by_bundle[str(row.get("bundle_id") or "")] = row

        for number, bundle_id in enumerate(sorted(by_bundle), 1):
            row = by_bundle[bundle_id]
            question = stem_only(str(row["question"]))
            normalized = re.sub(r"\s+", " ", question).strip().casefold()
            digest = hashlib.sha256(normalized.encode()).hexdigest()
            if digest in seen_questions:
                raise ValueError(f"duplicate normalized question at {topic_key}/{bundle_id}")
            seen_questions.add(digest)
            skeleton = skeleton_by_bundle.get(bundle_id, {})
            meta = dict(row.get("_meta") or {})
            meta.update({
                "topic_key": topic_key,
                "topic": topic,
                "pipeline": "all_in_one_solution_first_rag",
                "generation_provider": "openai",
                "generation_model": "gpt-5.1",
                "verification_model": "gpt-5.1",
                "selected_from": source_by_bundle[bundle_id],
            })
            bank.append({
                **row,
                "id": f"fma-sf-{topic_key}-{number:02d}",
                "question": question,
                "difficulty": skeleton.get("difficulty"),
                "topic_key": topic_key,
                "topic": topic,
                "_meta": meta,
            })
        summary[topic_key] = {"topic": topic, "count": 15, "gatekeeper_pass": 15}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in bank), encoding="utf-8")
    report = {
        "schema_version": 1,
        "pipeline": "all_in_one.py solution-first RAG",
        "generation_provider": "openai",
        "total": len(bank),
        "all_answer_letters_reverified": True,
        "topics": summary,
    }
    args.summary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(bank)} verified questions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
