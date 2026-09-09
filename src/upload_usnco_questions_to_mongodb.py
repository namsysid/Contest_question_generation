#!/usr/bin/env python3
"""Validate/package generated USNCO questions and optionally insert them in MongoDB."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from usnco_generation import USNCO_CHOICE_KEYS, USNCO_TOPIC_BY_KEY


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            rows.append(row)
    return rows


def _fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    return hashlib.sha256(normalized.encode()).hexdigest()


def build_document(record: dict[str, Any]) -> dict[str, Any]:
    meta = record.get("_meta") if isinstance(record.get("_meta"), dict) else {}
    topic_key = str(record.get("topic_key") or meta.get("topic_key") or "")
    expected_topic = USNCO_TOPIC_BY_KEY.get(topic_key, ("", ""))[0]
    topic = str(record.get("topic") or meta.get("topic") or "")
    question = str(record.get("question") or "").strip()
    raw_choices = record.get("choices")
    choices = (
        {key: str(raw_choices.get(key) or "").strip() for key in USNCO_CHOICE_KEYS}
        if isinstance(raw_choices, dict) else {}
    )
    answer = str(record.get("answer") or "").strip().upper()
    solution = str(record.get("solution") or "").strip()
    raw_steps = record.get("solution_steps")
    solution_steps = [str(step).strip() for step in raw_steps] if isinstance(raw_steps, list) else []
    solution_steps = [step for step in solution_steps if step]
    difficulty = record.get("difficulty")
    missing: list[str] = []
    if not question:
        missing.append("question")
    if set(choices) != set(USNCO_CHOICE_KEYS) or any(not value for value in choices.values()):
        missing.append("choices A-D")
    if answer not in USNCO_CHOICE_KEYS:
        missing.append("answer A-D")
    if not solution:
        missing.append("solution")
    if len(solution_steps) < 2:
        missing.append("at least two solution_steps")
    if not isinstance(difficulty, int) or isinstance(difficulty, bool) or not 1 <= difficulty <= 5:
        missing.append("difficulty 1-5")
    if not expected_topic or topic != expected_topic:
        missing.append("recognized topic_key/topic pair")
    if missing:
        raise ValueError(f"record {record.get('id')!r} is invalid: {', '.join(missing)}")

    source_id = str(record.get("id") or "").strip()
    if not source_id:
        raise ValueError("record is missing id")
    digest = _fingerprint(question)
    return {
        "_id": f"usnco-generated-{source_id}",
        "schema_version": 1,
        "Question": question,
        "Choices": choices,
        "Answer": answer,
        "AnswerText": choices[answer],
        "Solution": solution,
        "SolutionSteps": solution_steps,
        "Topic": topic,
        "TopicKey": topic_key,
        "Difficulty": difficulty,
        "Domain": "chemistry",
        "Program": "USNCO",
        "Competition": "USNCO",
        "QuestionType": "mcq",
        "Status": "generated",
        "DedupeKey": f"USNCO|{topic_key}|{digest}",
        "source_id": source_id,
        "generation_meta": meta,
    }


def prepare_documents(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_keys: set[str] = set()
    for record in records:
        document = build_document(record)
        if document["_id"] in seen_ids:
            raise ValueError(f"duplicate id in input: {document['_id']}")
        if document["DedupeKey"] in seen_keys:
            raise ValueError(f"duplicate normalized question in input: {document['source_id']}")
        seen_ids.add(document["_id"])
        seen_keys.add(document["DedupeKey"])
        documents.append(document)
    return documents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, help="write MongoDB-ready JSONL without uploading")
    parser.add_argument("--upload", action="store_true", help="insert without overwriting existing IDs/questions")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()

    documents = prepare_documents(read_jsonl(args.input))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            for document in documents:
                handle.write(json.dumps(document, ensure_ascii=False) + "\n")
        print(f"Prepared {len(documents)} MongoDB documents -> {args.out}")

    if args.upload:
        load_dotenv(args.env_file)
        db_name = os.getenv("CHEM_DB_NAME") or os.getenv("DB_NAME")
        collection_name = os.getenv("CHEM_COLLECTION_NAME") or os.getenv("COLLECTION_NAME")
        if not os.getenv("MONGODB_URI") or not db_name or not collection_name:
            raise RuntimeError(
                "set MONGODB_URI and CHEM_DB_NAME/CHEM_COLLECTION_NAME (or DB_NAME/COLLECTION_NAME)"
            )
        try:
            import certifi
            from pymongo import MongoClient
        except ImportError as exc:
            raise RuntimeError("upload requires pymongo and certifi") from exc
        client = MongoClient(os.environ["MONGODB_URI"], tlsCAFile=certifi.where())
        try:
            collection = client[db_name][collection_name]
            ids = [document["_id"] for document in documents]
            keys = [document["DedupeKey"] for document in documents]
            collisions = list(collection.find(
                {"$or": [{"_id": {"$in": ids}}, {"DedupeKey": {"$in": keys}}]}, {"_id": 1}
            ))
            if collisions:
                raise RuntimeError(f"refusing to overwrite/duplicate {len(collisions)} existing document(s)")
            result = collection.insert_many(documents, ordered=False)
            print(f"Uploaded {len(result.inserted_ids)} documents to {db_name}.{collection_name}")
        finally:
            client.close()
    elif not args.out:
        raise ValueError("specify --out and/or --upload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
