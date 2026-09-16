from copy import deepcopy
from datetime import datetime, timezone

import pytest

from src.science_olympiad_update import (
    optimistic_filter,
    payload_hash,
    prepare_stable_replacements,
    replace_existing_documents,
    verify_post_read,
)
from src.science_olympiad_upload import database_document, exact_model_evidence_errors


def candidate(document_id="scioly-example-001"):
    return {
        "_id": document_id,
        "content": {"prompt": "new prompt"},
        "answer_key": {"answer": "B", "solution": "new solution"},
        "status": "validated",
        "provenance": {
            "model_generated": True,
            "generation_model": "gpt-5-mini",
        },
        "created_at": datetime(2026, 9, 15, tzinfo=timezone.utc),
    }


def existing(document_id="scioly-example-001", revision=None):
    row = {
        "_id": document_id,
        "content": {"prompt": "old prompt"},
        "created_at": datetime(2026, 9, 1, tzinfo=timezone.utc),
    }
    if revision is not None:
        row["revision"] = revision
    return row


def test_database_document_preserves_standalone_rubric_and_solution_steps():
    item = {
        "id": "heredity-b-generated-999",
        "response_type": "short_answer",
        "prompt": "Explain the observed ratio.",
        "answer": "3:1",
        "solution": "A monohybrid cross gives 3:1.",
        "solution_steps": ["Write Aa x Aa.", "Count phenotypes."],
        "rubric": ["1 point for cross", "1 point for ratio"],
        "points": 2,
        "difficulty": 2,
        "topics": ["inheritance_patterns"],
        "generation": {},
    }
    document = database_document(item, {"valid": True}, "heredity_b")
    assert document["answer_key"]["rubric"] == item["rubric"]
    assert document["answer_key"]["solution_steps"] == item["solution_steps"]


def test_exact_model_evidence_is_fail_closed():
    item = {
        "generation": {
            "model_generated": True,
            "generation_model": "gpt-5-mini",
            "judge_model": "gpt-5-mini",
        }
    }
    report = {
        "valid": True,
        "deterministic_errors": [],
        "judge": {
            "solvable": True,
            "answer_agrees": True,
            "science_correct": True,
            "division_b_appropriate": True,
            "event_relevant": True,
            "difficulty_match": True,
            "competition_faithful": True,
            "novel": True,
            "independent_answer": "B",
            "issues": [],
        },
    }
    assert exact_model_evidence_errors(item, report) == []
    broken = deepcopy(item)
    broken["generation"]["generation_model"] = "gpt-5.1"
    errors = exact_model_evidence_errors(broken, report)
    assert "generation model must be exactly gpt-5-mini" in errors
    del report["judge"]["science_correct"]
    assert "blind judge must explicitly pass science_correct" in exact_model_evidence_errors(
        item, report
    )


def test_prepare_replacement_preserves_created_at_and_records_revision_hash():
    old = existing(revision=4)
    when = datetime(2026, 9, 16, tzinfo=timezone.utc)
    replacement = prepare_stable_replacements([candidate()], [old], updated_at=when)[0]
    assert replacement["created_at"] == old["created_at"]
    assert replacement["updated_at"] == when
    assert replacement["revision"] == 5
    audit = replacement["provenance"]["replacement"]
    assert audit["previous_revision"] == 4
    assert audit["previous_payload_sha256"] == payload_hash(old)
    assert optimistic_filter(old, replacement) == {
        "_id": old["_id"], "revision": 4, "created_at": old["created_at"]
    }


def test_legacy_document_uses_absent_revision_compare_and_swap():
    old = existing()
    replacement = prepare_stable_replacements([candidate()], [old])[0]
    assert replacement["revision"] == 1
    assert optimistic_filter(old, replacement)["revision"] == {"$exists": False}


def test_preflight_requires_exact_existing_id_set():
    with pytest.raises(RuntimeError, match="preflight ID mismatch"):
        prepare_stable_replacements([candidate()], [existing("scioly-other")])


def test_post_read_compares_complete_normalized_payload():
    expected = prepare_stable_replacements([candidate()], [existing()])
    assert verify_post_read(expected, deepcopy(expected))[expected[0]["_id"]] == payload_hash(
        expected[0]
    )
    actual = deepcopy(expected)
    actual[0]["content"]["prompt"] = "tampered"
    with pytest.raises(RuntimeError, match="payload mismatch"):
        verify_post_read(expected, actual)


def test_payload_hash_uses_bson_datetime_semantics():
    aware = {"when": datetime(2026, 9, 16, 12, 30, 45, 123999, tzinfo=timezone.utc)}
    mongo_round_trip = {"when": datetime(2026, 9, 16, 12, 30, 45, 123000)}
    assert payload_hash(aware) == payload_hash(mongo_round_trip)


class Result:
    matched_count = 1


class FakeCollection:
    def __init__(self, rows):
        self.rows = {row["_id"]: deepcopy(row) for row in rows}
        self.calls = []

    def replace_one(self, query, replacement, upsert):
        assert upsert is False
        self.calls.append((deepcopy(query), deepcopy(replacement), upsert))
        self.rows[replacement["_id"]] = deepcopy(replacement)
        return Result()

    def find(self, query):
        ids = query["_id"]["$in"]
        return [deepcopy(self.rows[document_id]) for document_id in ids]


def test_execution_never_upserts_and_post_verifies():
    old = existing(revision=2)
    replacements = prepare_stable_replacements([candidate()], [old])
    collection = FakeCollection([old])
    hashes = replace_existing_documents(collection, replacements, [old])
    assert len(collection.calls) == 1
    assert collection.calls[0][2] is False
    assert hashes[old["_id"]] == payload_hash(replacements[0])
