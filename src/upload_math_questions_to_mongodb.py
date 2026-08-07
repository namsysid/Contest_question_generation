#!/usr/bin/env python3
"""Upload generated math questions from JSONL/JSON into MongoDB."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable


REQUIRED_ENV_VARS = ("MONGODB_URI", "DB_NAME", "COLLECTION_NAME")
DIFFICULTY_LEVELS = ("1", "2", "3")


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


def normalize_difficulty(value: Any) -> str | None:
    if value in (None, ""):
        return None

    if isinstance(value, (int, float)):
        if value in (1, 2, 3):
            return str(int(value))
        raise ValueError(f"Unsupported difficulty '{value}'. Use one of: 1, 2, 3")

    text = str(value).strip()
    if text in DIFFICULTY_LEVELS:
        return text
    raise ValueError(f"Unsupported difficulty '{value}'. Use one of: 1, 2, 3")


def build_document(
    record: dict[str, Any],
    *,
    default_topic: str | None,
    default_difficulty: str | None,
    include_question: bool,
) -> dict[str, Any]:
    topic = first_present(record, ("Topic", "topic"))
    difficulty = first_present(record, ("Difficulty", "difficulty"))
    answer = first_present(record, ("Answer", "answer"))
    question = first_present(record, ("Question", "question", "question_text", "prompt"))

    if topic is None:
        topic = default_topic
    if difficulty is None:
        difficulty = default_difficulty
    difficulty = normalize_difficulty(difficulty)

    document = {
        "Topic": topic,
        "Difficulty": difficulty,
        "Answer": answer,
    }
    if include_question:
        document = {"Question": question, **document}

    missing = [key for key, value in document.items() if value in (None, "")]
    if missing:
        record_id = first_present(record, ("id", "_id", "question_number")) or "<unknown>"
        raise ValueError(f"Record {record_id} is missing required field(s): {', '.join(missing)}")

    source_id = first_present(record, ("id", "_id", "question_number"))
    if source_id is not None:
        document["source_id"] = source_id

    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload generated math questions to MongoDB using .env settings."
    )
    parser.add_argument("input", type=Path, help="Path to a .jsonl or .json file of questions")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Path to .env file")
    parser.add_argument(
        "--default-topic",
        help="Topic to use when records do not include Topic/topic",
    )
    parser.add_argument(
        "--default-difficulty",
        help="Difficulty to use when records do not include Difficulty/difficulty: 1, 2, or 3",
    )
    parser.add_argument(
        "--omit-question",
        action="store_true",
        help="Upload only Topic, Difficulty, and Answer fields",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print a preview without writing to MongoDB",
    )
    return parser.parse_args()


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
                default_topic=args.default_topic,
                default_difficulty=args.default_difficulty,
                include_question=not args.omit_question,
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

    client = MongoClient(os.environ["MONGODB_URI"])
    collection = client[os.environ["DB_NAME"]][os.environ["COLLECTION_NAME"]]
    result = collection.insert_many(documents, ordered=False)
    print(
        f"Uploaded {len(result.inserted_ids)} document(s) to "
        f"{os.environ['DB_NAME']}.{os.environ['COLLECTION_NAME']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
