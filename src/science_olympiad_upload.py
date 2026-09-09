from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from src.circuit_lab.common import read_jsonl
from src.food_science.common import validate_item as validate_food_science_item
from src.machines.common import validate_item as validate_machines_item
from src.optics.common import validate_item as validate_optics_item


EVENTS: dict[str, dict[str, Any]] = {
    "food_science_b": {
        "event": "Food Science",
        "division": "B",
        "season": 2022,
        "validator": validate_food_science_item,
    },
    "machines_b": {
        "event": "Machines",
        "division": "B",
        "season": 2026,
        "validator": validate_machines_item,
    },
    "optics_b": {
        "event": "Optics",
        "division": "B",
        "season": 2025,
        "validator": validate_optics_item,
    },
}


def database_document(
    item: dict[str, Any], report: dict[str, Any], event_key: str
) -> dict[str, Any]:
    config = EVENTS[event_key]
    response_type = item.get("response_type")
    question_type = "mcq" if response_type == "multiple_choice" else "frq"
    content: dict[str, Any] = {"prompt": item.get("prompt")}
    if response_type == "multiple_choice":
        content["choices"] = item.get("choices")
    if item.get("parts"):
        content["parts"] = [
            {
                key: part.get(key)
                for key in (
                    "label",
                    "response_type",
                    "prompt",
                    "choices",
                    "points",
                    "depends_on",
                )
                if part.get(key) is not None
            }
            for part in item["parts"]
        ]

    answer_key: dict[str, Any] = {
        "answer": item.get("answer"),
        "solution": item.get("solution"),
    }
    if item.get("parts"):
        answer_key["parts"] = [
            {
                key: part.get(key)
                for key in ("label", "answer", "solution", "rubric", "points")
                if part.get(key) is not None
            }
            for part in item["parts"]
        ]

    generation = item.get("generation") or {}
    judge = report.get("judge") or {}
    quality_checks = {
        key: value
        for key, value in judge.items()
        if isinstance(value, bool)
    }
    return {
        "_id": f"scioly-{item['id']}",
        "schema_version": 1,
        "competition": "Science Olympiad",
        "event": config["event"],
        "division": config["division"],
        "season": item.get("season") or config["season"],
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
            "valid": True,
            "deterministic_errors": report.get("deterministic_errors") or [],
            "independently_solved": bool(judge),
            "quality_checks": quality_checks,
            "source_embedding_similarity": generation.get("source_embedding_similarity"),
        },
        "provenance": {
            "kind": "generated",
            "pipeline": generation.get("pipeline") or f"{event_key}_graph_rag",
            "anchor_id": generation.get("anchor_id"),
            "candidate_origin": item.get("candidate_origin"),
        },
        "created_at": datetime.now(timezone.utc),
    }


def prepare_documents(
    input_path: str | Path,
    reports_paths: list[str | Path],
    event_key: str,
) -> list[dict[str, Any]]:
    validator: Callable[[dict[str, Any]], list[str]] = EVENTS[event_key]["validator"]
    reports: dict[str, dict[str, Any]] = {}
    for reports_path in reports_paths:
        reports.update({row.get("id"): row for row in read_jsonl(reports_path)})
    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in read_jsonl(input_path):
        report = reports.get(item.get("id")) or {}
        errors = validator(item)
        if errors or report.get("valid") is not True:
            raise RuntimeError(
                f"Refusing to upload unvalidated item {item.get('id')}: "
                f"{errors or report}"
            )
        document = database_document(item, report, event_key)
        if document["_id"] in seen:
            raise RuntimeError(f"Duplicate input id: {document['_id']}")
        seen.add(document["_id"])
        documents.append(document)
    if not documents:
        raise RuntimeError("No validated documents to upload")
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload one validated Science Olympiad event bank to MongoDB"
    )
    parser.add_argument("--event", required=True, choices=sorted(EVENTS))
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--reports",
        required=True,
        action="append",
        help="Validation report JSONL; repeat for banks assembled from multiple runs",
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip matching _id values; never overwrite existing records",
    )
    args = parser.parse_args()

    documents = prepare_documents(args.input, args.reports, args.event)
    load_dotenv(args.env_file)
    uri = os.getenv("MONGODB_URI", "").strip()
    database_name = os.getenv("SCIOLY_DB_NAME", "").strip()
    collection_name = os.getenv("SCIOLY_COLLECTION_NAME", "").strip()
    if not all((uri, database_name, collection_name)):
        raise RuntimeError(
            "MONGODB_URI, SCIOLY_DB_NAME, and SCIOLY_COLLECTION_NAME must be set"
        )

    try:
        from pymongo import MongoClient
    except ModuleNotFoundError as exc:
        raise RuntimeError("pymongo is required: pip install pymongo") from exc

    client = MongoClient(uri, serverSelectionTimeoutMS=15000)
    try:
        collection = client[database_name][collection_name]
        ids = [document["_id"] for document in documents]
        existing = {
            row["_id"]
            for row in collection.find({"_id": {"$in": ids}}, {"_id": 1})
        }
        if existing and not args.skip_existing:
            raise RuntimeError(
                "Refusing to overwrite existing records: " + ", ".join(sorted(existing))
            )
        documents = [document for document in documents if document["_id"] not in existing]
        if documents:
            collection.insert_many(documents, ordered=True)

        found = collection.count_documents({"_id": {"$in": ids}})
        if found != len(ids):
            raise RuntimeError(
                f"Post-upload verification failed: found {found} of {len(ids)} records"
            )
        print(
            f"Inserted {len(documents)}; verified {found}/{len(ids)} "
            f"{EVENTS[args.event]['event']} B records in the configured collection"
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
