from __future__ import annotations

import copy
import json
from collections import Counter

import pytest

from src import science_olympiad_club_pilot as pilot
from src import science_olympiad_upload as uploader


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _question_specs(prefix: str) -> list[dict]:
    difficulties = [1, 2, 2, 3, 3]
    return [
        {
            "id": f"{prefix}-spec-{index:02d}",
            "difficulty": difficulty,
            "topic": f"topic_{index}",
            "response_type": "multiple_choice" if index < 3 else "short_answer",
            "points": 2 if difficulty == 1 else 3,
            "generation_guidance": {"task": f"Generate item {index}."},
            "authoritative_urls": ["https://example.test/reference"],
            "source_question_urls": ["https://example.test/questions"],
        }
        for index, difficulty in enumerate(difficulties, 1)
    ]


def _write_specs(spec_dir, *, mutate=None):
    for event_key in sorted(pilot.EXPECTED_EVENTS):
        spec = {
            "event": {
                "id": event_key,
                "label": event_key.replace("_", " ").title(),
                "season": 2027,
            },
            "source_manifest": [
                {"id": f"{event_key}-source-{index}", "url": row["url"]}
                for index, row in enumerate(pilot.SOURCE_FILES[event_key], 1)
            ],
            "question_specs": _question_specs(event_key),
        }
        if mutate is not None:
            mutate(event_key, spec)
        (spec_dir / f"{event_key}.json").write_text(
            json.dumps(spec), encoding="utf-8"
        )


def _valid_item(
    *,
    item_id: str = "meteorology-b-2027-spec-04",
    difficulty: int = 3,
    response_type: str = "short_answer",
) -> dict:
    item = {
        "id": item_id,
        "response_type": response_type,
        "prompt": "Use the supplied observations to identify the stronger storm signal.",
        "choices": None,
        "answer": "Cell A, based on the paired observations.",
        "solution_steps": ["Compare the two observations.", "Apply the supplied rule."],
        "solution": "Cell A has both required observations, while Cell B does not.",
        "rubric": ["Identifies Cell A.", "Cites both observations."],
        "points": 3,
        "difficulty": difficulty,
        "topics": ["doppler_radar_tornado_evidence"],
        "season": 2027,
        "level": "Division B",
        "generation": {
            "pipeline": "science_olympiad_club_pilot_v1_retrieval_generation",
            "model_generated": True,
            "generation_model": "gpt-5-mini",
            "embedding_model": "text-embedding-3-small",
            "provider": "openai",
        },
    }
    if response_type == "multiple_choice":
        item.update(
            choices={"A": "First", "B": "Second", "C": "Third", "D": "Fourth"},
            answer="A",
        )
    return item


def _valid_report(item_id: str) -> dict:
    return {
        "id": item_id,
        "valid": True,
        "deterministic_errors": [],
        "judge_model": "gpt-5-mini",
        "judge": {
            "solvable": True,
            "answer_agrees": True,
            "science_correct": True,
            "division_b_appropriate": True,
            "event_relevant": True,
            "difficulty_match": True,
            "competition_faithful": True,
            "novel": True,
            "independent_answer": "Cell A",
            "issues": [],
            "provenance": {"generation_model": "gpt-5-mini"},
        },
    }


def test_load_specs_accepts_only_exact_pilot_shape(tmp_path, monkeypatch):
    _write_specs(tmp_path)
    monkeypatch.setattr(pilot, "SPEC_DIR", tmp_path)

    specs = pilot.load_specs()

    assert set(specs) == pilot.EXPECTED_EVENTS
    questions = [row for spec in specs.values() for row in spec["question_specs"]]
    assert len(questions) == 15
    assert Counter(row["difficulty"] for row in questions) == {1: 3, 2: 6, 3: 6}
    assert max(row["difficulty"] for row in questions) == 3


def test_load_specs_rejects_any_d4_manifest_item(tmp_path, monkeypatch):
    def mutate(event_key, spec):
        if event_key == "meteorology_b":
            spec["question_specs"][-1]["difficulty"] = 4

    _write_specs(tmp_path, mutate=mutate)
    monkeypatch.setattr(pilot, "SPEC_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="D1/D2/D3 = 1/2/2"):
        pilot.load_specs()


def test_load_specs_rejects_duplicate_ids_across_events(tmp_path, monkeypatch):
    duplicate_id = "shared-pilot-id"

    def mutate(_event_key, spec):
        spec["question_specs"][0]["id"] = duplicate_id

    _write_specs(tmp_path, mutate=mutate)
    monkeypatch.setattr(pilot, "SPEC_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="fifteen unique nonempty item IDs"):
        pilot.load_specs()


def test_load_specs_rejects_missing_source_manifest(tmp_path, monkeypatch):
    def mutate(event_key, spec):
        if event_key == "crime_busters_b":
            spec["source_manifest"] = []

    _write_specs(tmp_path, mutate=mutate)
    monkeypatch.setattr(pilot, "SPEC_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="crime_busters_b has no source manifest"):
        pilot.load_specs()


@pytest.mark.parametrize(
    ("source_type", "normalized"),
    [
        ("numeric_with_interpretation", "short_answer"),
        ("numeric_response", "short_answer"),
        ("multipart_short_answer", "short_answer"),
        ("multiple_choice", "multiple_choice"),
    ],
)
def test_response_type_normalization(source_type, normalized):
    assert pilot.normalized_response_type(source_type) == normalized


def test_deterministic_errors_rejects_d4_without_forcing_blueprint_label():
    item = _valid_item(difficulty=4)
    question_spec = {
        "id": item["id"],
        "response_type": "short_answer",
        "difficulty": 3,
        "points": 3,
    }

    errors = pilot.deterministic_errors(item, question_spec)

    assert "difficulty exceeds D3 ceiling" in errors


def test_audit_item_rejects_blind_d4_calibration(monkeypatch):
    item = _valid_item()
    question_spec = {
        "id": item["id"],
        "response_type": "short_answer",
        "difficulty": 3,
        "points": 3,
    }
    blind = {
        "independent_answer": "Cell A",
        "reasoning": "Two observations agree.",
        "solvable": True,
        "science_correct": True,
        "self_contained": True,
        "event_faithful": True,
        "division_b_appropriate": True,
        "unique_answer": True,
        "calibrated_difficulty": 4,
        "issues": [],
    }
    editorial = {
        "answer_agrees": True,
        "answer_correct": True,
        "rubric_complete": True,
        "student_ready": True,
        "competition_faithful": True,
        "difficulty_match": True,
        "no_overclaiming": True,
        "source_independent": True,
        "issues": [],
    }
    replies = iter((blind, editorial))
    monkeypatch.setattr(pilot, "generate_json", lambda *args, **kwargs: next(replies))

    report = pilot.audit_item(
        item,
        {"event": {"label": "Meteorology"}},
        question_spec,
        [{"id": "source-1", "excerpt": "calibration text"}],
        "openai",
    )

    assert report["valid"] is False
    assert "blind-calibrated difficulty exceeds D3 ceiling or is invalid: 4" in report["acceptance_errors"]


def _final_items() -> list[dict]:
    difficulties = [1, 2, 2, 3, 3] * 3
    prefixes = ["anatomy"] * 5 + ["crime"] * 5 + ["meteorology"] * 5
    return [
        {
            "id": f"{prefix}-{index:02d}",
            "response_type": "short_answer",
            "prompt": f"Distinct prompt number {index} for {prefix}.",
            "choices": None,
            "answer": f"Answer {index}",
            "difficulty": difficulty,
            "topics": [f"topic_{index}"],
        }
        for index, (prefix, difficulty) in enumerate(zip(prefixes, difficulties), 1)
    ]


def _patch_offline_final_audit(monkeypatch):
    def embeddings(_model, texts, **_kwargs):
        size = len(texts)
        return [[1.0 if left == right else 0.0 for right in range(size)] for left in range(size)]

    model_audit = {
        "overall_good": True,
        "coverage_good": True,
        "difficulty_ceiling_respected": True,
        "reasoning_diversity_good": True,
        "competition_faithful": True,
        "middle_school_club_ready": True,
        "weak_item_ids": [],
        "issues": [],
    }
    monkeypatch.setattr(pilot, "embed_texts", embeddings)
    monkeypatch.setattr(pilot, "structural_similarity", lambda _left, _right: 0.0)
    monkeypatch.setattr(pilot, "generate_json", lambda *args, **kwargs: copy.deepcopy(model_audit))


def test_final_audit_accepts_exact_offline_difficulty_distribution(monkeypatch):
    _patch_offline_final_audit(monkeypatch)

    audit = pilot.final_audit(_final_items(), "openai")

    assert audit["valid"] is True
    assert audit["difficulty_counts"] == {"1": 3, "2": 6, "3": 6}


def test_final_audit_accepts_honest_d1_d2_pilot_with_no_d3(monkeypatch):
    _patch_offline_final_audit(monkeypatch)
    items = _final_items()
    for item in items:
        if item["difficulty"] == 3:
            item["difficulty"] = 2

    audit = pilot.final_audit(items, "openai")

    assert audit["valid"] is True
    assert audit["difficulty_counts"] == {"1": 3, "2": 12}


def test_final_audit_fails_closed_when_one_item_is_d4(monkeypatch):
    _patch_offline_final_audit(monkeypatch)
    items = _final_items()
    items[-1]["difficulty"] = 4

    audit = pilot.final_audit(items, "openai")

    assert audit["valid"] is False
    assert audit["difficulty_counts"] == {"1": 3, "2": 6, "3": 5, "4": 1}


@pytest.mark.parametrize("difficulty", [0, 4, "3", 3.0, None])
def test_uploader_validator_rejects_non_integer_or_out_of_range_difficulty(difficulty):
    item = _valid_item()
    item["difficulty"] = difficulty

    errors = uploader.validate_club_pilot_item(item)

    assert "club-pilot difficulty must be an integer from 1 through 3" in errors


def test_prepare_documents_rejects_d4_even_with_valid_report(tmp_path):
    item = _valid_item(difficulty=4)
    input_path = tmp_path / "items.jsonl"
    reports_path = tmp_path / "reports.jsonl"
    _write_jsonl(input_path, [item])
    _write_jsonl(reports_path, [_valid_report(item["id"])])

    with pytest.raises(RuntimeError, match="difficulty must be an integer from 1 through 3"):
        uploader.prepare_documents(
            input_path, [reports_path], "meteorology_b", require_exact_model="gpt-5-mini"
        )


def test_prepare_documents_preserves_constructed_response_grading_contract(tmp_path):
    item = _valid_item()
    input_path = tmp_path / "items.jsonl"
    reports_path = tmp_path / "reports.jsonl"
    _write_jsonl(input_path, [item])
    _write_jsonl(reports_path, [_valid_report(item["id"])])

    documents = uploader.prepare_documents(
        input_path, [reports_path], "meteorology_b", require_exact_model="gpt-5-mini"
    )

    assert len(documents) == 1
    assert documents[0]["difficulty"] == 3
    assert documents[0]["answer_key"]["rubric"] == item["rubric"]
    assert documents[0]["answer_key"]["solution_steps"] == item["solution_steps"]
    assert documents[0]["validation"]["valid"] is True


def test_prepare_documents_exact_model_contract_is_fail_closed(tmp_path):
    item = _valid_item()
    report = _valid_report(item["id"])
    report["judge"].pop("difficulty_match")
    input_path = tmp_path / "items.jsonl"
    reports_path = tmp_path / "reports.jsonl"
    _write_jsonl(input_path, [item])
    _write_jsonl(reports_path, [report])

    with pytest.raises(RuntimeError, match="blind judge must explicitly pass difficulty_match"):
        uploader.prepare_documents(
            input_path, [reports_path], "meteorology_b", require_exact_model="gpt-5-mini"
        )


def test_uploader_rejects_boolean_difficulty():
    item = _valid_item()
    item["difficulty"] = True

    assert "club-pilot difficulty must be an integer from 1 through 3" in (
        uploader.validate_club_pilot_item(item)
    )


def test_prepare_documents_always_rejects_report_deterministic_errors(tmp_path):
    item = _valid_item()
    report = _valid_report(item["id"])
    report["deterministic_errors"] = ["stored difficulty differs from manifest"]
    input_path = tmp_path / "items.jsonl"
    reports_path = tmp_path / "reports.jsonl"
    _write_jsonl(input_path, [item])
    _write_jsonl(reports_path, [report])

    with pytest.raises(RuntimeError, match="deterministic validation errors"):
        uploader.prepare_documents(input_path, [reports_path], "meteorology_b")


def test_prepare_documents_rejects_cross_event_item_mislabeling(tmp_path):
    item = _valid_item(item_id="meteorology-b-2027-spec-01")
    input_path = tmp_path / "items.jsonl"
    reports_path = tmp_path / "reports.jsonl"
    _write_jsonl(input_path, [item])
    _write_jsonl(reports_path, [_valid_report(item["id"])])

    with pytest.raises(RuntimeError, match="event"):
        uploader.prepare_documents(
            input_path, [reports_path], "anatomy_physiology_b", require_exact_model="gpt-5-mini"
        )
