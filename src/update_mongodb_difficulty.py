#!/usr/bin/env python3
"""Apply independently audited difficulty ratings to generated contest questions."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


CONFIG = {
    "fma": {
        "competition": "F=ma",
        "db_env": "DB_NAME",
        "collection_env": "COLLECTION_NAME",
        "identity": {
            "Program": "F=ma",
            "generation_meta.pipeline": "all_in_one_solution_first_rag",
        },
    },
    "usnco": {
        "competition": "USNCO",
        "db_env": "CHEM_DB_NAME",
        "collection_env": "CHEM_COLLECTION_NAME",
        "identity": {
            "Program": "USNCO",
            "generation_meta.generator": "07_direct_generate_openai_usnco_validated",
        },
    },
}


def read_audit(path: Path, competition: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids: set[str] = set()
    for index, row in enumerate(rows, 1):
        item_id = str(row.get("id") or "")
        score = row.get("difficulty")
        if not item_id or item_id in ids:
            raise ValueError(f"line {index}: missing or duplicate id {item_id!r}")
        if not isinstance(score, int) or isinstance(score, bool) or score not in range(1, 6):
            raise ValueError(f"line {index}: difficulty must be an integer from 1 through 5")
        if row.get("competition") != competition:
            raise ValueError(f"line {index}: expected competition {competition!r}")
        ids.add(item_id)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", choices=sorted(CONFIG))
    parser.add_argument("audit", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--apply", action="store_true", help="Write updates; otherwise perform a dry run")
    args = parser.parse_args()

    config = CONFIG[args.bank]
    rows = read_audit(args.audit, config["competition"])
    load_dotenv(args.env_file)
    required = ("MONGODB_URI", config["db_env"], config["collection_env"])
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError("missing environment settings: " + ", ".join(missing))

    try:
        import certifi
        from pymongo import MongoClient, UpdateOne
    except ImportError as exc:
        raise RuntimeError("MongoDB update requires pymongo and certifi") from exc

    client = MongoClient(os.environ["MONGODB_URI"], tlsCAFile=certifi.where())
    try:
        collection = client[os.environ[config["db_env"]]][os.environ[config["collection_env"]]]
        audit_by_id = {str(row["id"]): row for row in rows}
        query = {**config["identity"], "source_id": {"$in": list(audit_by_id)}}
        documents = list(collection.find(query, {"source_id": 1, "Difficulty": 1, "TopicKey": 1}))
        found_ids = [str(document.get("source_id") or "") for document in documents]
        found = set(found_ids)
        missing_ids = set(audit_by_id) - found
        duplicate_ids = [item_id for item_id, count in Counter(found_ids).items() if count != 1]
        if missing_ids or duplicate_ids or len(documents) != len(rows):
            raise RuntimeError(
                f"identity check failed: audit={len(rows)}, matched={len(documents)}, "
                f"missing={sorted(missing_ids)}, duplicate={sorted(duplicate_ids)}"
            )

        print("Current difficulty:", dict(sorted(Counter(doc.get("Difficulty") for doc in documents).items())))
        print("Audited difficulty:", dict(sorted(Counter(row["difficulty"] for row in rows).items())))
        if not args.apply:
            print(f"Dry run matched exactly {len(documents)} documents; no changes made")
            return 0

        audited_at = datetime.now(timezone.utc)
        operations = []
        for document in documents:
            item_id = str(document["source_id"])
            audit = audit_by_id[item_id]
            operations.append(UpdateOne(
                {"_id": document["_id"], **config["identity"], "source_id": item_id},
                {"$set": {
                    "Difficulty": audit["difficulty"],
                    "DifficultyStatus": "independently_audited",
                    "difficulty_audit": {
                        "status": "complete",
                        "rubric_version": audit.get("rubric_version", "finished_item_v1"),
                        "review_method": audit.get("review_method"),
                        "competition": config["competition"],
                        "model": audit.get("model"),
                        "score": audit["difficulty"],
                        "competition_level": bool(audit.get("competition_level")),
                        "reasoning_steps": audit.get("reasoning_steps"),
                        "notes": audit.get("notes"),
                        "previous_difficulty": document.get("Difficulty"),
                        "audited_at": audited_at,
                    },
                }},
            ))
        result = collection.bulk_write(operations, ordered=True)
        if result.matched_count != len(rows):
            raise RuntimeError(f"expected {len(rows)} matched updates, got {result.matched_count}")

        verified = list(collection.find(query, {
            "source_id": 1, "Difficulty": 1, "DifficultyStatus": 1,
            "difficulty_audit.status": 1, "Status": 1,
        }))
        mismatches = [
            doc["source_id"] for doc in verified
            if doc.get("Difficulty") != audit_by_id[str(doc["source_id"])]["difficulty"]
            or doc.get("DifficultyStatus") != "independently_audited"
            or (doc.get("difficulty_audit") or {}).get("status") != "complete"
        ]
        if mismatches:
            raise RuntimeError(f"read-back verification failed for {mismatches}")
        print(
            f"Updated and verified {len(verified)} documents in "
            f"{os.environ[config['db_env']]}.{os.environ[config['collection_env']]}"
        )
        print("Verified difficulty:", dict(sorted(Counter(doc["Difficulty"] for doc in verified).items())))
        print("Preserved Status:", dict(sorted(Counter(doc.get("Status") for doc in verified).items())))
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
