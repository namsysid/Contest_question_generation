#!/usr/bin/env python3
"""Prepare or upload topic-balanced F=ma questions as MongoDB documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


TOPIC_KEYS = {
    "kinematics": "Kinematics",
    "forces": "Forces",
    "energy": "Energy",
    "momentum": "Momentum",
    "circular_gravity": "Circular & gravity",
    "rotation": "Rotation",
    "oscillations": "Oscillations",
    "fluids": "Fluids",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            rows.append(value)
    return rows


def normalized_question(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def report_by_id(path: Path | None) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in read_jsonl(path)} if path else {}


def build_document(record: dict[str, Any], report: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = record.get("_meta") if isinstance(record.get("_meta"), dict) else {}
    topic_key = str(meta.get("topic_key") or record.get("topic_key") or "")
    topic = str(meta.get("topic") or record.get("topic") or TOPIC_KEYS.get(topic_key, ""))
    choices = record.get("choices")
    if not isinstance(choices, dict):
        choices = {}
    choices = {key: str(choices.get(key) or "").strip() for key in "ABCDE"}
    answer = str(record.get("answer") or "").strip().upper()
    question = str(record.get("question") or "").strip()
    solution = str(record.get("solution") or "").strip()
    missing = []
    if not question:
        missing.append("question")
    if any(not choices[key] for key in "ABCDE"):
        missing.append("choices A-E")
    if answer not in "ABCDE" or len(answer) != 1:
        missing.append("answer A-E")
    if not solution:
        missing.append("solution")
    if topic_key not in TOPIC_KEYS or topic != TOPIC_KEYS[topic_key]:
        missing.append("recognized topic metadata")
    if missing:
        raise ValueError(f"record {record.get('id')!r} is missing/invalid: {', '.join(missing)}")

    grading: dict[str, Any] = {}
    status = "generated"
    difficulty = record.get("difficulty")
    if report:
        solve = report.get("solve_attempt") or {}
        competition = report.get("competition_appropriateness") or {}
        topic_assessment = report.get("topic_assessment") or {}
        selected = str(solve.get("selected_answer") or "").upper()
        agrees = solve.get("status") == "solved" and selected == answer
        topic_matches = topic_assessment.get("matches_required")
        difficulty = (report.get("difficulty_assessment") or {}).get("score", difficulty)
        grading = {
            "independently_solved": solve.get("status") == "solved",
            "selected_answer": selected,
            "answer_agrees": agrees,
            "topic_matches": topic_matches,
            "primary_topic": topic_assessment.get("primary_topic"),
            "issue_notes": solve.get("issue_notes") or [],
            "competition_appropriateness": competition,
        }
        status = "validated" if agrees and topic_matches is not False else "review_required"
    elif isinstance(record.get("_gatekeeper"), dict):
        gate = record.get("_gatekeeper") if isinstance(record.get("_gatekeeper"), dict) else {}
        consistency = gate.get("answer_consistency") if isinstance(gate.get("answer_consistency"), dict) else {}
        verified = str(consistency.get("answer_verified") or "").upper()
        agrees = gate.get("verdict") == "PASS" and verified == answer
        grading = {
            "solution_first_gatekeeper": gate.get("verdict") == "PASS",
            "selected_answer": verified,
            "answer_agrees": agrees,
            "issue_notes": gate.get("issues") or [],
            "verifier_notes": consistency.get("notes"),
        }
        status = "validated" if agrees else "review_required"

    digest = hashlib.sha256(normalized_question(question).encode()).hexdigest()
    return {
        "_id": f"fma-generated-{record['id']}",
        "schema_version": 1,
        "Question": question,
        "Choices": choices,
        "Answer": answer,
        "AnswerText": choices[answer],
        "Solution": solution,
        "Topic": topic,
        "TopicKey": topic_key,
        "Difficulty": difficulty,
        "Domain": "physics",
        "Program": "F=ma",
        "Competition": "F=ma",
        "QuestionType": "mcq",
        "Status": status,
        "DedupeKey": f"F=ma|{topic_key}|{digest}",
        "source_id": record["id"],
        "validation": grading,
        "generation_meta": meta,
    }


def prepare_documents(
    records: list[dict[str, Any]],
    reports: dict[str, dict[str, Any]] | None = None,
    require_answer_agreement: bool = False,
) -> list[dict[str, Any]]:
    reports = reports or {}
    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        record_id = str(record.get("id") or "")
        report = reports.get(record_id)
        document = build_document(record, report)
        if require_answer_agreement and document["Status"] != "validated":
            raise ValueError(f"record {record_id!r} lacks an agreeing independent solve")
        if document["DedupeKey"] in seen:
            raise ValueError(f"duplicate question in input: {record_id!r}")
        seen.add(document["DedupeKey"])
        documents.append(document)
    return documents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--reports", type=Path)
    parser.add_argument("--ids", nargs="*", help="Process only these source record IDs")
    parser.add_argument("--out", type=Path, help="Write MongoDB-ready JSONL without uploading")
    parser.add_argument("--require-answer-agreement", action="store_true")
    parser.add_argument("--upload", action="store_true", help="Insert documents; never overwrites existing IDs")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()

    reports = report_by_id(args.reports)
    records = read_jsonl(args.input)
    if args.ids:
        selected = set(args.ids)
        records = [record for record in records if str(record.get("id")) in selected]
        missing_ids = selected - {str(record.get("id")) for record in records}
        if missing_ids:
            raise RuntimeError(f"requested ids not found: {sorted(missing_ids)}")
    documents = prepare_documents(
        records, reports, require_answer_agreement=args.require_answer_agreement
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            for document in documents:
                handle.write(json.dumps(document, ensure_ascii=False) + "\n")
        print(f"Prepared {len(documents)} MongoDB documents -> {args.out}")

    if args.upload:
        load_dotenv(args.env_file)
        required = ("MONGODB_URI", "DB_NAME", "COLLECTION_NAME")
        missing = [key for key in required if not os.getenv(key)]
        if missing:
            raise RuntimeError("missing environment settings: " + ", ".join(missing))
        try:
            import certifi
            from pymongo import MongoClient
        except ImportError as exc:
            raise RuntimeError("upload requires pymongo and certifi") from exc
        client = MongoClient(os.environ["MONGODB_URI"], tlsCAFile=certifi.where())
        try:
            collection = client[os.environ["DB_NAME"]][os.environ["COLLECTION_NAME"]]
            ids = [document["_id"] for document in documents]
            collisions = [row["_id"] for row in collection.find({"_id": {"$in": ids}}, {"_id": 1})]
            if collisions:
                raise RuntimeError(f"refusing to overwrite {len(collisions)} existing document(s)")
            result = collection.insert_many(documents, ordered=False)
            print(f"Uploaded {len(result.inserted_ids)} documents")
        finally:
            client.close()
    elif not args.out:
        raise ValueError("specify --out and/or --upload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
