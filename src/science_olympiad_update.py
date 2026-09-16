"""Safely replace validated Science Olympiad documents while retaining stable IDs.

This is deliberately separate from the insert-only uploader.  It never deletes
or upserts, requires every requested ID to exist, uses optimistic revision
filters, and verifies the complete normalized payload after replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from dotenv import load_dotenv

from src.science_olympiad_upload import EVENTS, prepare_documents


UPDATE_PIPELINE = "science_olympiad_stable_id_repair_v1"


def _json_value(value: Any) -> Any:
    """Convert Mongo/Python values into a stable, JSON-comparable structure."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, datetime):
        # BSON datetimes are UTC milliseconds. PyMongo may return them naive
        # even when the input was timezone-aware, so hash their stored meaning
        # instead of Python-only timezone/microsecond decoration.
        utc_value = (
            value.replace(tzinfo=timezone.utc)
            if value.tzinfo is None
            else value.astimezone(timezone.utc)
        )
        utc_value = utc_value.replace(microsecond=(utc_value.microsecond // 1000) * 1000)
        return utc_value.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    return value


def normalized_payload(document: Mapping[str, Any]) -> str:
    return json.dumps(
        _json_value(document), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def payload_hash(document: Mapping[str, Any]) -> str:
    return hashlib.sha256(normalized_payload(document).encode("utf-8")).hexdigest()


def _revision(document: Mapping[str, Any]) -> int:
    value = document.get("revision", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"invalid revision for {document.get('_id')}: {value!r}")
    return value


def prepare_stable_replacements(
    candidates: Iterable[Mapping[str, Any]],
    existing: Iterable[Mapping[str, Any]],
    *,
    updated_at: datetime | None = None,
) -> list[dict[str, Any]]:
    """Bind replacement payloads to an exact preflight snapshot.

    Candidate and existing ID sets must be identical.  The original creation
    timestamp is retained and each payload records the prior revision and hash.
    """
    candidate_rows = list(candidates)
    existing_rows = list(existing)
    candidate_by_id = {
        str(row.get("_id") or ""): deepcopy(dict(row)) for row in candidate_rows
    }
    existing_by_id = {
        str(row.get("_id") or ""): deepcopy(dict(row)) for row in existing_rows
    }
    if "" in candidate_by_id or "" in existing_by_id:
        raise RuntimeError("every replacement and existing document must have an _id")
    if len(candidate_by_id) != len(candidate_rows):
        raise RuntimeError("duplicate candidate IDs")
    if len(existing_by_id) != len(existing_rows):
        raise RuntimeError("duplicate existing IDs")
    candidate_ids, existing_ids = set(candidate_by_id), set(existing_by_id)
    if candidate_ids != existing_ids:
        missing = sorted(candidate_ids - existing_ids)
        unexpected = sorted(existing_ids - candidate_ids)
        raise RuntimeError(
            f"preflight ID mismatch; missing existing={missing}, unexpected existing={unexpected}"
        )
    if not candidate_ids:
        raise RuntimeError("no replacement documents supplied")

    timestamp = updated_at or datetime.now(timezone.utc)
    replacements: list[dict[str, Any]] = []
    for document_id in sorted(candidate_ids):
        candidate = candidate_by_id[document_id]
        old = existing_by_id[document_id]
        prior_revision = _revision(old)
        if "created_at" not in old:
            raise RuntimeError(f"existing document {document_id} has no created_at to preserve")
        candidate["created_at"] = old["created_at"]
        candidate["updated_at"] = timestamp
        candidate["revision"] = prior_revision + 1
        provenance = candidate.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}
        candidate["provenance"] = {
            **provenance,
            "replacement": {
                "pipeline": UPDATE_PIPELINE,
                "previous_revision": prior_revision,
                "previous_payload_sha256": payload_hash(old),
            },
        }
        replacements.append(candidate)
    return replacements


def optimistic_filter(
    existing: Mapping[str, Any], replacement: Mapping[str, Any]
) -> dict[str, Any]:
    """Create a compare-and-swap filter without permitting insertion."""
    old_revision = _revision(existing)
    if replacement.get("revision") != old_revision + 1:
        raise RuntimeError(f"replacement revision is not consecutive for {existing.get('_id')}")
    query: dict[str, Any] = {"_id": existing["_id"]}
    if "revision" in existing:
        query["revision"] = old_revision
    else:
        query["revision"] = {"$exists": False}
    if "created_at" in existing:
        query["created_at"] = existing["created_at"]
    return query


def verify_post_read(
    expected: Iterable[Mapping[str, Any]], actual: Iterable[Mapping[str, Any]]
) -> dict[str, str]:
    expected_rows = list(expected)
    actual_rows = list(actual)
    expected_by_id = {str(row.get("_id") or ""): row for row in expected_rows}
    actual_by_id = {str(row.get("_id") or ""): row for row in actual_rows}
    if len(expected_by_id) != len(expected_rows) or len(actual_by_id) != len(actual_rows):
        raise RuntimeError("duplicate IDs in post-update verification")
    if set(expected_by_id) != set(actual_by_id):
        raise RuntimeError("post-update verification returned a different ID set")
    hashes: dict[str, str] = {}
    for document_id, expected_document in expected_by_id.items():
        expected_hash = payload_hash(expected_document)
        actual_hash = payload_hash(actual_by_id[document_id])
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"post-update payload mismatch for {document_id}: "
                f"expected {expected_hash}, found {actual_hash}"
            )
        hashes[document_id] = actual_hash
    return hashes


def replace_existing_documents(
    collection: Any,
    replacements: list[dict[str, Any]],
    existing: list[dict[str, Any]],
    *,
    session: Any = None,
) -> dict[str, str]:
    """Perform ordered, optimistic replacements and verify every complete payload."""
    existing_by_id = {str(row["_id"]): row for row in existing}
    if set(existing_by_id) != {str(row["_id"]) for row in replacements}:
        raise RuntimeError("replacement execution did not receive the exact preflight ID set")
    for replacement in replacements:
        document_id = str(replacement["_id"])
        kwargs = {"upsert": False}
        if session is not None:
            kwargs["session"] = session
        result = collection.replace_one(
            optimistic_filter(existing_by_id[document_id], replacement),
            replacement,
            **kwargs,
        )
        if result.matched_count != 1:
            raise RuntimeError(f"concurrent modification or missing document: {document_id}")
    ids = [row["_id"] for row in replacements]
    find_kwargs = {"session": session} if session is not None else {}
    actual = list(collection.find({"_id": {"$in": ids}}, **find_kwargs))
    return verify_post_read(replacements, actual)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", choices=sorted(EVENTS))
    parser.add_argument("--input")
    parser.add_argument("--reports", action="append")
    parser.add_argument(
        "--bundle", action="append", nargs=3, metavar=("EVENT", "INPUT", "REPORTS"),
        help="repeat for an all-or-nothing multi-event transaction",
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--model", default="gpt-5-mini", choices=["gpt-5-mini"])
    args = parser.parse_args()

    if args.bundle:
        if args.event or args.input or args.reports:
            parser.error("--bundle cannot be combined with --event/--input/--reports")
        candidates = []
        for event_key, input_path, reports_path in args.bundle:
            if event_key not in EVENTS:
                parser.error(f"unknown bundle event: {event_key}")
            candidates.extend(prepare_documents(
                input_path, [reports_path], event_key, require_exact_model=args.model,
            ))
        ids = [row["_id"] for row in candidates]
        if len(ids) != len(set(ids)):
            raise RuntimeError("duplicate IDs across update bundles")
    else:
        if not args.event or not args.input or not args.reports:
            parser.error("provide --event, --input, and --reports, or repeat --bundle")
        candidates = prepare_documents(
            args.input,
            args.reports,
            args.event,
            require_exact_model=args.model,
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
        from pymongo import MongoClient
    except ModuleNotFoundError as exc:
        raise RuntimeError("pymongo is required: pip install pymongo") from exc

    client = MongoClient(uri, serverSelectionTimeoutMS=15000)
    try:
        collection = client[database_name][collection_name]
        ids = [row["_id"] for row in candidates]
        # Preflight, writes, and post-read comparison share one snapshot. Any
        # mismatch, concurrent write, or verification error aborts the batch.
        with client.start_session() as session:
            with session.start_transaction():
                existing = list(
                    collection.find({"_id": {"$in": ids}}, session=session)
                )
                replacements = prepare_stable_replacements(candidates, existing)
                hashes = replace_existing_documents(
                    collection, replacements, existing, session=session
                )
        print(f"Replaced and verified {len(hashes)} stable-ID documents")
    finally:
        client.close()


if __name__ == "__main__":
    main()
