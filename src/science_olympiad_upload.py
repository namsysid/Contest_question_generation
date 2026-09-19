from __future__ import annotations

import argparse
import hashlib
import json
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


def validate_club_pilot_item(item: dict[str, Any]) -> list[str]:
    """Fail-closed structural contract for the cross-event club pilot."""
    errors: list[str] = []
    response_type = item.get("response_type")
    if response_type not in {"multiple_choice", "short_answer", "numeric"}:
        errors.append("unsupported club-pilot response_type")
    if not str(item.get("id") or "").strip():
        errors.append("missing item id")
    for field in ("prompt", "answer", "solution"):
        if not str(item.get(field) or "").strip():
            errors.append(f"missing {field}")
    difficulty = item.get("difficulty")
    if isinstance(difficulty, bool) or not isinstance(difficulty, int) or difficulty not in {1, 2, 3}:
        errors.append("club-pilot difficulty must be an integer from 1 through 3")
    if isinstance(item.get("points"), bool) or not isinstance(item.get("points"), int) or item["points"] < 1:
        errors.append("points must be a positive integer")
    if not item.get("topics"):
        errors.append("at least one topic is required")
    if response_type == "multiple_choice":
        choices = item.get("choices") or {}
        if set(choices) != set("ABCD"):
            errors.append("multiple choice item must contain exactly A-D")
        if str(item.get("answer") or "").strip().upper() not in set("ABCD"):
            errors.append("multiple choice answer must be A-D")
    elif not item.get("rubric"):
        errors.append("constructed response requires a rubric")
    return errors


EVENTS: dict[str, dict[str, Any]] = {
    "anatomy_physiology_b": {
        "event": "Anatomy & Physiology",
        "division": "B",
        "season": 2027,
        "validator": validate_club_pilot_item,
        "id_prefix": "apb-2027-pilot-",
    },
    "crime_busters_b": {
        "event": "Crime Busters",
        "division": "B",
        "season": 2027,
        "validator": validate_club_pilot_item,
        "id_prefix": "crime-busters-b-2027-pilot-",
    },
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
    "meteorology_b": {
        "event": "Meteorology",
        "division": "B",
        "season": 2027,
        "validator": validate_club_pilot_item,
        "id_prefix": "meteorology-b-2027-spec-",
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
        expected_prefix = EVENTS[event_key].get("id_prefix")
        if expected_prefix and not str(item.get("id") or "").startswith(expected_prefix):
            errors.append(f"item id is not bound to event {event_key}")
        if report.get("deterministic_errors"):
            errors.append("validation report contains deterministic validation errors")
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


def _stable_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stable_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_stable_value(row) for row in value]
    if isinstance(value, datetime):
        normalized = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        normalized = normalized.astimezone(timezone.utc)
        normalized = normalized.replace(microsecond=(normalized.microsecond // 1000) * 1000)
        return normalized.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return value


def document_hash(document: dict[str, Any]) -> str:
    payload = json.dumps(
        _stable_value(document), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def append_bundle_transactionally(collection: Any, client: Any, documents: list[dict[str, Any]]) -> int:
    """Insert a bundle atomically and verify both new payloads and the baseline."""
    ids = [str(row.get("_id") or "") for row in documents]
    if not ids or "" in ids or len(ids) != len(set(ids)):
        raise RuntimeError("bundle must contain unique nonempty IDs")
    expected = {row["_id"]: document_hash(row) for row in documents}
    with client.start_session() as session:
        with session.start_transaction():
            baseline = list(collection.find({}, session=session))
            baseline_hashes = {row["_id"]: document_hash(row) for row in baseline}
            collisions = sorted(set(ids) & set(baseline_hashes))
            if collisions:
                raise RuntimeError("Refusing to overwrite existing records: " + ", ".join(collisions))
            collection.insert_many(documents, ordered=True, session=session)
            inserted = list(collection.find({"_id": {"$in": ids}}, session=session))
            actual = {row["_id"]: document_hash(row) for row in inserted}
            if actual != expected:
                raise RuntimeError("Inserted payload verification failed")
            current_baseline = list(
                collection.find({"_id": {"$in": list(baseline_hashes)}}, session=session)
            ) if baseline_hashes else []
            if {row["_id"]: document_hash(row) for row in current_baseline} != baseline_hashes:
                raise RuntimeError("Preexisting Science Olympiad records changed")
            if collection.count_documents({}, session=session) != len(baseline) + len(documents):
                raise RuntimeError("Collection count verification failed inside transaction")
        final_rows = list(collection.find({"_id": {"$in": ids}}))
        if {row["_id"]: document_hash(row) for row in final_rows} != expected:
            raise RuntimeError("Post-commit payload verification failed")
        if collection.count_documents({}) != len(baseline) + len(documents):
            raise RuntimeError("Post-commit collection count verification failed")
    return len(baseline) + len(documents)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload one validated Science Olympiad event bank to MongoDB"
    )
    parser.add_argument("--event", choices=sorted(EVENTS))
    parser.add_argument("--input")
    parser.add_argument(
        "--reports",
        action="append",
        help="Validation report JSONL; repeat for banks assembled from multiple runs",
    )
    parser.add_argument(
        "--bundle", nargs=3, action="append", metavar=("EVENT", "INPUT", "REPORTS"),
        help="Atomic bundle entry; repeat once per event",
    )
    parser.add_argument(
        "--bundle-profile", choices=("difficult", "club-pilot"), default="difficult",
        help="Validation contract for an atomic multi-event bundle",
    )
    parser.add_argument("--model", help="Require exact generation and judge model")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip matching _id values; never overwrite existing records",
    )
    args = parser.parse_args()

    if args.bundle:
        if args.event or args.input or args.reports or args.skip_existing:
            raise RuntimeError("--bundle cannot be combined with legacy single-event arguments")
        documents = []
        event_counts: dict[str, int] = {}
        for event_key, input_path, reports_path in args.bundle:
            if event_key not in EVENTS:
                raise RuntimeError(f"unknown event: {event_key}")
            prepared = prepare_documents(
                input_path, [reports_path], event_key, require_exact_model=args.model,
            )
            documents.extend(prepared)
            event_counts[event_key] = event_counts.get(event_key, 0) + len(prepared)
        ids = [row["_id"] for row in documents]
        if len(ids) != len(set(ids)):
            raise RuntimeError("duplicate IDs across bundle inputs")
        if args.bundle_profile == "difficult":
            if len(documents) != 15 or event_counts != {
                "disease_detectives_b": 5, "dynamic_planet_b": 5, "heredity_b": 5,
            }:
                raise RuntimeError("difficult tranche bundle must contain five items from each event")
            difficulty_counts = {
                level: sum(row.get("difficulty") == level for row in documents) for level in (3, 4)
            }
            if difficulty_counts != {3: 9, 4: 6}:
                raise RuntimeError("difficult tranche bundle must contain nine D3 and six D4 items")
        else:
            if len(documents) != 15 or event_counts != {
                "anatomy_physiology_b": 5, "crime_busters_b": 5, "meteorology_b": 5,
            }:
                raise RuntimeError("club pilot bundle must contain five items from each event")
            difficulty_counts = {
                level: sum(row.get("difficulty") == level for row in documents) for level in (1, 2, 3)
            }
            if (
                any(row.get("difficulty") not in {1, 2, 3} for row in documents)
                or difficulty_counts[1] < 3
                or difficulty_counts[2] < 6
            ):
                raise RuntimeError(
                    "club pilot bundle must stay within D1-D3 and contain at least three D1 and six D2 items"
                )
    else:
        if not args.event or not args.input or not args.reports:
            parser.error("legacy mode requires --event, --input, and --reports")
        documents = prepare_documents(
            args.input, args.reports, args.event, require_exact_model=args.model,
        )
    load_dotenv(args.env_file)
    uri = os.getenv("MONGODB_URI", "").strip()
    database_name = os.getenv("SCIOLY_DB_NAME", "").strip()
    collection_name = os.getenv("SCIOLY_COLLECTION_NAME", "").strip()
    if not all((uri, database_name, collection_name)):
        raise RuntimeError(
            "MONGODB_URI, SCIOLY_DB_NAME, and SCIOLY_COLLECTION_NAME must be set"
        )

    try:
        import certifi
        from pymongo import MongoClient
    except ModuleNotFoundError as exc:
        raise RuntimeError("pymongo and certifi are required") from exc

    client = MongoClient(uri, serverSelectionTimeoutMS=15000, tlsCAFile=certifi.where())
    try:
        collection = client[database_name][collection_name]
        if args.bundle:
            final_count = append_bundle_transactionally(collection, client, documents)
            print(f"Inserted and verified {len(documents)} records atomically; collection count={final_count}")
            return
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
