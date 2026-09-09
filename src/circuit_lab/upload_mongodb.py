from __future__ import annotations

import argparse
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .common import read_jsonl, validate_item


def database_document(item: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    response_type = item.get("response_type")
    question_type = "mcq" if response_type == "multiple_choice" else "frq"
    parts = item.get("parts") or []
    content: dict[str, Any] = {"prompt": item.get("prompt")}
    answer_key: dict[str, Any]
    if question_type == "mcq":
        content["choices"] = item.get("choices")
        answer_key = {"answer": item.get("answer"), "solution": item.get("solution")}
    else:
        content["parts"] = [
            {key: part.get(key) for key in ("label", "response_type", "prompt", "choices", "points", "depends_on")
             if part.get(key) is not None}
            for part in parts
        ]
        answer_key = {"parts": [
            {key: part.get(key) for key in ("label", "answer", "solution", "rubric", "points")
             if part.get(key) is not None}
            for part in parts
        ]}
    generation = item.get("generation") or {}
    judge = report.get("judge") or {}
    return {
        "_id": f"scioly-circuit-lab-b-{item.get('id')}",
        "schema_version": 1,
        "competition": "Science Olympiad",
        "event": "Circuit Lab",
        "division": "B",
        "question_type": question_type,
        "response_type": response_type,
        "content": content,
        "answer_key": answer_key,
        "points": item.get("points"),
        "difficulty": item.get("difficulty"),
        "level": item.get("level"),
        "topics": item.get("topics") or [],
        "status": "validated",
        "validation": {
            "valid": report.get("valid") is True,
            "deterministic_errors": report.get("deterministic_errors") or [],
            "independently_solved": bool(judge),
            "quality_checks": {key: judge.get(key) for key in (
                "solvable", "answer_agrees", "science_correct", "level_appropriate",
                "circuit_lab_relevant", "plan_faithful", "competition_faithful",
                "style_faithful", "difficulty_match", "novel")},
        },
        "provenance": {
            "kind": "generated",
            "pipeline": "circuit_lab_graph_rag",
            "bundle_id": generation.get("bundle_id"),
            "source_response_type": generation.get("source_response_type"),
        },
        "created_at": datetime.now(timezone.utc),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload validated Circuit Lab questions to MongoDB")
    parser.add_argument("--input", required=True)
    parser.add_argument("--reports", required=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--valid-only", action="store_true",
                        help="Skip items whose validation report is not valid instead of failing the upload")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip records already present by _id; never overwrites them")
    args = parser.parse_args()

    load_dotenv(args.env_file)
    uri = os.getenv("MONGODB_URI", "").strip()
    database_name = os.getenv("SCIOLY_DB_NAME", "").strip()
    collection_name = os.getenv("SCIOLY_COLLECTION_NAME", "").strip()
    if not all((uri, database_name, collection_name)):
        raise RuntimeError("MONGODB_URI, SCIOLY_DB_NAME, and SCIOLY_COLLECTION_NAME must be set")

    try:
        from pymongo import MongoClient
    except ModuleNotFoundError as exc:
        raise RuntimeError("pymongo is required: pip install pymongo") from exc

    reports = {row.get("id"): row for row in read_jsonl(args.reports)}
    documents = []
    for item in read_jsonl(args.input):
        errors = validate_item(item)
        report = reports.get(item.get("id")) or {}
        if errors or report.get("valid") is not True:
            if args.valid_only:
                continue
            raise RuntimeError(f"Refusing to upload unvalidated item {item.get('id')}: {errors or report}")
        documents.append(database_document(item, report))
    if not documents:
        raise RuntimeError("No validated documents to upload")

    seen_ids: set[str] = set()
    for document in documents:
        if document["_id"] in seen_ids:
            bundle = str(document.get("provenance", {}).get("bundle_id") or "duplicate")
            suffix = re.sub(r"[^a-zA-Z0-9_-]+", "-", bundle).strip("-")
            document["_id"] = f"{document['_id']}-{suffix}"
        if document["_id"] in seen_ids:
            raise RuntimeError(f"Could not resolve duplicate input id: {document['_id']}")
        seen_ids.add(document["_id"])

    client = MongoClient(uri, serverSelectionTimeoutMS=15000)
    try:
        collection = client[database_name][collection_name]
        ids = [document["_id"] for document in documents]
        collisions = [row["_id"] for row in collection.find({"_id": {"$in": ids}}, {"_id": 1})]
        if collisions:
            if args.skip_existing:
                collision_set = set(collisions)
                documents = [document for document in documents if document["_id"] not in collision_set]
            else:
                raise RuntimeError("Refusing to overwrite existing records: " + ", ".join(collisions))
        if not documents:
            print("Inserted 0 questions; every validated record already exists")
            return
        result = collection.insert_many(documents, ordered=True)
        print(f"Inserted {len(result.inserted_ids)} validated questions into the configured Science Olympiad collection")
    finally:
        client.close()


if __name__ == "__main__":
    main()
