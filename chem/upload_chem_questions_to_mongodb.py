#!/usr/bin/env python3
"""Upload generated USNCO chemistry questions from JSONL/JSON into MongoDB."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable


REQUIRED_ENV_VARS = ("MONGODB_URI", "CHEM_DB_NAME", "CHEM_COLLECTION_NAME")


def normalize_for_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def dedupe_key(document: dict[str, Any]) -> str:
    return "|".join(
        (
            "USNCO",
            normalize_for_key(document.get("QuestionRange")),
            normalize_for_key(document.get("Question")),
        )
    )


def load_dotenv_file(path: Path) -> None:
    """Small .env loader so the script does not require python-dotenv."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"Line {line_number} is not a JSON object")
                records.append(value)
        return records

    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        if not all(isinstance(item, dict) for item in value):
            raise ValueError("JSON array must contain only objects")
        return value
    if isinstance(value, dict):
        return [value]
    raise ValueError("Input must be a JSON object, JSON array, or JSONL file")


def first_present(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def normalize_choices(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        out = {str(k).strip().upper(): str(v).strip() for k, v in value.items() if str(k).strip() and str(v).strip()}
        if set(out) >= {"A", "B", "C", "D"}:
            return {letter: out[letter] for letter in sorted(out)}
        raise ValueError("choices must include at least A, B, C, and D")

    if isinstance(value, list):
        out: dict[str, str] = {}
        for item in value:
            text = str(item).strip()
            match = re.match(r"^\(?\s*([A-E])\s*\)?[.)]?\s*(.*)$", text)
            if match:
                out[match.group(1).upper()] = match.group(2).strip()
        if set(out) >= {"A", "B", "C", "D"}:
            return {letter: out[letter] for letter in sorted(out)}

    raise ValueError("choices must be a dict or a list containing A-D choices")


def infer_range(record: dict[str, Any], input_path: Path) -> str | None:
    meta = record.get("_meta") if isinstance(record.get("_meta"), dict) else {}
    chem = meta.get("chem") if isinstance(meta.get("chem"), dict) else {}
    if chem.get("range"):
        return str(chem["range"])

    source_target_id = str(meta.get("source_target_id") or "")
    match = re.search(r"usnco_(\d+)_(\d+)_", source_target_id)
    if match:
        return f"{match.group(1)}-{match.group(2)}"

    file_match = re.search(r"usnco_(\d+)-(\d+)_", input_path.name)
    if file_match:
        return f"{file_match.group(1)}-{file_match.group(2)}"
    return None


def infer_generated_question_number(record: dict[str, Any], input_path: Path) -> int | None:
    meta = record.get("_meta") if isinstance(record.get("_meta"), dict) else {}
    chem = meta.get("chem") if isinstance(meta.get("chem"), dict) else {}
    if chem.get("generated_question_number") not in (None, ""):
        return int(chem["generated_question_number"])

    question_range = infer_range(record, input_path)
    source_target_id = str(meta.get("source_target_id") or "")
    match = re.search(r"_(\d+)$", source_target_id)
    if question_range and match:
        start = int(question_range.split("-", 1)[0])
        slot = int(match.group(1))
        return start + slot - 1
    return None


def nested_first_present(values: Iterable[Any]) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def answer_text(choices: dict[str, str], answer: Any) -> str | None:
    if answer in (None, ""):
        return None
    letter = str(answer).strip().upper()
    return choices.get(letter)


def build_document(
    record: dict[str, Any],
    *,
    input_path: Path,
    default_topic: str | None,
    default_difficulty: int | None,
    include_solution: bool,
    include_skeleton: bool,
) -> dict[str, Any]:
    meta = record.get("_meta") if isinstance(record.get("_meta"), dict) else {}
    chem = meta.get("chem") if isinstance(meta.get("chem"), dict) else {}
    question = first_present(record, ("Question", "question", "question_text", "prompt"))
    choices = normalize_choices(first_present(record, ("Choices", "choices", "options")))
    answer = first_present(record, ("Answer", "answer", "answer_key"))
    topic = first_present(record, ("Topic", "topic")) or meta.get("topic") or default_topic
    difficulty = first_present(record, ("Difficulty", "difficulty")) or meta.get("difficulty") or default_difficulty
    question_range = infer_range(record, input_path)
    generated_question_number = infer_generated_question_number(record, input_path)
    seed_question_number = nested_first_present(
        (
            chem.get("seed_question_number"),
            meta.get("seed_question_number"),
            record.get("seed_question_number"),
            record.get("question_number"),
            record.get("QuestionNumber"),
        )
    )
    year = nested_first_present((chem.get("year"), meta.get("year"), record.get("year"), record.get("Year")))
    exam_type = nested_first_present((chem.get("exam_type"), meta.get("exam_type"), record.get("exam_type"), record.get("ExamType")))
    exam_code = nested_first_present((chem.get("exam_code"), meta.get("exam_code"), record.get("exam_code"), record.get("ExamCode")))
    range_title = nested_first_present((chem.get("range_title"), meta.get("range_title"), record.get("range_title"), record.get("RangeTitle")))

    document: dict[str, Any] = {
        "Question": question,
        "Choices": choices,
        "Answer": str(answer).strip().upper() if answer not in (None, "") else None,
        "AnswerText": answer_text(choices, answer),
        "Topic": topic,
        "Difficulty": difficulty,
        "Domain": "chem",
        "Program": "USNCO",
        "QuestionRange": question_range,
        "QuestionNumber": generated_question_number,
        "RangeTitle": range_title,
        "Year": year,
        "ExamType": exam_type,
        "ExamCode": exam_code,
        "SeedQuestionNumber": seed_question_number,
    }
    document["DedupeKey"] = dedupe_key(document)

    if include_solution:
        document["Solution"] = first_present(record, ("Solution", "solution", "full_solution"))
    if include_skeleton:
        document["SkeletonText"] = first_present(record, ("skeleton_text", "graph_text"))

    source_id = first_present(record, ("id", "_id", "source_id"))
    if source_id is not None:
        document["source_id"] = source_id
    if meta:
        document["generation_meta"] = meta
    if isinstance(record.get("anti_copy_report"), dict):
        document["anti_copy_report"] = record["anti_copy_report"]

    missing = [key for key in ("Question", "Choices", "Answer", "Topic") if document.get(key) in (None, "", {})]
    if missing:
        record_id = source_id or "<unknown>"
        raise ValueError(f"Record {record_id} is missing required field(s): {', '.join(missing)}")

    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload generated USNCO chemistry questions to MongoDB using .env settings."
    )
    parser.add_argument("input", type=Path, help="Path to a .jsonl or .json file of generated chemistry questions")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Path to .env file")
    parser.add_argument("--default-topic", default="USNCO chemistry", help="Topic to use when records do not include one")
    parser.add_argument("--default-difficulty", type=int, default=None, help="Difficulty to use when records do not include one")
    parser.add_argument("--omit-solution", action="store_true", help="Do not upload solution text")
    parser.add_argument("--omit-skeleton", action="store_true", help="Do not upload skeleton/graph text")
    parser.add_argument("--allow-duplicates", action="store_true", help="Upload records without checking for existing overlap")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print a preview without writing to MongoDB")
    return parser.parse_args()


def filter_existing_documents(collection: Any, documents: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Remove records that already exist in MongoDB or repeat within this batch."""
    seen_keys: set[str] = set()
    seen_source_ids: set[str] = set()
    unique_batch: list[dict[str, Any]] = []
    skipped = 0

    for document in documents:
        key = str(document.get("DedupeKey") or "")
        source_id = str(document.get("source_id") or "")
        if (key and key in seen_keys) or (source_id and source_id in seen_source_ids):
            skipped += 1
            continue
        if key:
            seen_keys.add(key)
        if source_id:
            seen_source_ids.add(source_id)
        unique_batch.append(document)

    keys = [str(doc["DedupeKey"]) for doc in unique_batch if doc.get("DedupeKey")]
    source_ids = [str(doc["source_id"]) for doc in unique_batch if doc.get("source_id")]

    existing_keys: set[str] = set()
    existing_source_ids: set[str] = set()

    query_parts = []
    if keys:
        query_parts.append({"DedupeKey": {"$in": keys}})
    if source_ids:
        query_parts.append({"source_id": {"$in": source_ids}})

    if query_parts:
        query = query_parts[0] if len(query_parts) == 1 else {"$or": query_parts}
        for existing in collection.find(query, {"DedupeKey": 1, "source_id": 1}):
            if existing.get("DedupeKey"):
                existing_keys.add(str(existing["DedupeKey"]))
            if existing.get("source_id"):
                existing_source_ids.add(str(existing["source_id"]))

    filtered: list[dict[str, Any]] = []
    for document in unique_batch:
        key = str(document.get("DedupeKey") or "")
        source_id = str(document.get("source_id") or "")
        if (key and key in existing_keys) or (source_id and source_id in existing_source_ids):
            skipped += 1
            continue
        filtered.append(document)

    return filtered, skipped


def main() -> int:
    args = parse_args()
    load_dotenv_file(args.env_file)

    missing_env = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing_env:
        print(f"Missing required environment variable(s): {', '.join(missing_env)}", file=sys.stderr)
        return 2

    try:
        raw_records = load_records(args.input)
        documents = [
            build_document(
                record,
                input_path=args.input,
                default_topic=args.default_topic,
                default_difficulty=args.default_difficulty,
                include_solution=not args.omit_solution,
                include_skeleton=not args.omit_skeleton,
            )
            for record in raw_records
        ]
    except (OSError, ValueError) as exc:
        print(f"Input validation failed: {exc}", file=sys.stderr)
        return 1

    if not documents:
        print("No records found to upload.")
        return 0

    if args.dry_run:
        print(f"Validated {len(documents)} document(s). Preview:")
        print(json.dumps(documents[0], indent=2, ensure_ascii=False))
        return 0

    try:
        from pymongo import MongoClient
    except ImportError:
        print("pymongo is not installed. Install it with: python3 -m pip install pymongo", file=sys.stderr)
        return 2

    try:
        import certifi
    except ImportError:
        print("certifi is not installed. Install it with: python3 -m pip install certifi", file=sys.stderr)
        return 2

    db_name = os.environ["CHEM_DB_NAME"]
    collection_name = os.environ["CHEM_COLLECTION_NAME"]
    client = MongoClient(os.environ["MONGODB_URI"], tlsCAFile=certifi.where())
    collection = client[db_name][collection_name]

    skipped = 0
    if not args.allow_duplicates:
        documents, skipped = filter_existing_documents(collection, documents)
        if not documents:
            print(f"No new documents to upload. Skipped {skipped} duplicate/overlapping document(s).")
            return 0

    try:
        result = collection.insert_many(documents, ordered=False)
    except Exception as exc:
        print(f"MongoDB upload failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Uploaded {len(result.inserted_ids)} document(s) to "
        f"{db_name}.{collection_name}. Skipped {skipped} duplicate/overlapping document(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
