from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from src.circuit_lab.common import read_jsonl
from src.disease_detectives.common import validate_item as validate_disease_detectives_item
from src.dynamic_planet.model_pipeline import validate_item as validate_dynamic_planet_item
from src.food_science.common import validate_item as validate_food_science_item
from src.heredity.common import validate_item as validate_heredity_item
from src.machines.common import validate_item as validate_machines_item
from src.optics.common import validate_item as validate_optics_item


EVENTS: dict[str, dict[str, Any]] = {
    "disease_detectives_b": {
        "event": "Disease Detectives",
        "division": "B",
        "season": 2027,
        "validator": validate_disease_detectives_item,
    },
    "dynamic_planet_b": {
        "event": "Dynamic Planet",
        "division": "B",
        "season": 2027,
        "validator": lambda item: validate_dynamic_planet_item(
            item,
            str((item.get("topics") or [""])[0]),
            str(item.get("response_type") or ""),
        ),
    },
    "food_science_b": {
        "event": "Food Science",
        "division": "B",
        "season": 2022,
        "validator": validate_food_science_item,
    },
    "heredity_b": {
        "event": "Heredity",
        "division": "B",
        "season": 2027,
        "validator": validate_heredity_item,
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
    # Standalone constructed-response items keep their grading contract at the
    # item level.  Do not discard it merely because there is no ``parts`` list.
    if item.get("rubric") is not None:
        answer_key["rubric"] = item.get("rubric")
    if item.get("solution_steps") is not None:
        answer_key["solution_steps"] = item.get("solution_steps")
    if item.get("parts"):
        answer_key["parts"] = [
            {
                key: part.get(key)
                for key in (
                    "label", "answer", "solution", "solution_steps", "rubric", "points"
                )
                if part.get(key) is not None
            }
            for part in item["parts"]
        ]

    generation = item.get("generation") or {}
    judge = report.get("judge") or report.get("blind_judge") or {}
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
            "model_generated": generation.get("model_generated", True),
            "generation_model": generation.get("generation_model") or generation.get("model"),
            "embedding_model": generation.get("embedding_model"),
            "provider": generation.get("provider") or item.get("provider"),
            "blind_model_judged": generation.get("blind_model_judged") or bool(judge),
        },
        "created_at": datetime.now(timezone.utc),
    }


def prepare_documents(
    input_path: str | Path,
    reports_paths: list[str | Path],
    event_key: str,
    *,
    require_exact_model: str | None = None,
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
        if require_exact_model:
            errors.extend(exact_model_evidence_errors(item, report, require_exact_model))
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


def exact_model_evidence_errors(
    item: dict[str, Any], report: dict[str, Any], required_model: str = "gpt-5-mini"
) -> list[str]:
    """Return fail-closed provenance and blind-validation errors for replacements.

    Existing insertion flows remain backwards compatible.  Stable-ID repair uses
    this stricter contract so absent evidence is never interpreted as success.
    """
    errors: list[str] = []
    generation = item.get("generation") if isinstance(item.get("generation"), dict) else {}
    judge = report.get("judge")
    if not isinstance(judge, dict):
        judge = report.get("blind_judge")
    if not isinstance(judge, dict):
        judge = {}

    if generation.get("model_generated") is not True:
        errors.append("generation.model_generated must be explicitly true")
    if generation.get("generation_model") != required_model:
        errors.append(f"generation model must be exactly {required_model}")
    if report.get("valid") is not True:
        errors.append("validation report must be valid")
    if report.get("deterministic_errors"):
        errors.append("deterministic validation errors are present")

    for key in (
        "solvable",
        "answer_agrees",
        "science_correct",
        "division_b_appropriate",
        "event_relevant",
        "difficulty_match",
        "competition_faithful",
        "novel",
    ):
        if judge.get(key) is not True:
            errors.append(f"blind judge must explicitly pass {key}")
    if not str(judge.get("independent_answer") or "").strip():
        errors.append("blind judge must provide an independent answer")
    if judge.get("issues"):
        errors.append("blind judge reported issues")

    judge_provenance = judge.get("provenance") if isinstance(judge.get("provenance"), dict) else {}
    judge_model = (
        report.get("judge_model")
        or report.get("blind_judge_model")
        or judge_provenance.get("generation_model")
        or generation.get("judge_model")
    )
    if judge_model != required_model:
        errors.append(f"blind judge model must be exactly {required_model}")
    return errors


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
