import json
from pathlib import Path

from src.dynamic_planet.curated_pilot import SPECS
from src.dynamic_planet.model_pipeline import (REQUIRED_GENERATION_MODEL, normalize_mcq_answer,
                                               select_for_enrichment, validate_item, validate_plan)
from src.dynamic_planet.pipeline import deterministic_embedding, deterministic_errors, lexical_similarity, split_numbered, topics_for


def test_split_numbered_and_topics():
    rows = split_numbered("1. A stream loses water to an aquifer.\n2. A lake turns over in fall.")
    assert set(rows) == {1, 2}
    assert "groundwater" in topics_for(rows[1])
    assert "lakes" in topics_for(rows[2])


def test_lexical_similarity_is_bounded():
    assert lexical_similarity("river channel gradient", "river channel gradient") == 1
    assert 0 <= lexical_similarity("aquifer recharge", "lake turnover") <= 1


def test_deterministic_mcq_validation_and_novelty():
    item = {"id": "x", "response_type": "multiple_choice", "prompt": "A new watershed scenario asks for runoff.",
            "choices": {"A": "1", "B": "2", "C": "3", "D": "4"}, "answer": "A", "solution": "Because...",
            "points": 1, "difficulty": 2, "topics": ["surface_water"]}
    errors, score, _ = deterministic_errors(item, [{"id": "s", "prompt": "Identify a confined aquifer."}])
    assert errors == []
    assert score < 0.55


def test_curated_pilot_has_ten_unique_plan_faithful_items():
    assert len(SPECS) == 10
    prompts = [spec["item"]["prompt"] for spec in SPECS]
    assert len(set(prompts)) == 10
    for spec in SPECS:
        assert spec["item"]["response_type"] == spec["plan"]["response_type"]
        assert spec["item"]["difficulty"] == spec["plan"]["difficulty"]
        assert set(spec["item"]["topics"]) & set(spec["plan"]["topics"])
        assert len(deterministic_embedding(spec["item"]["prompt"])) == 256


def test_model_bank_is_complete_model_generated_and_topic_complete():
    root = Path("science_olympiad/dynamic_planet_b")
    items = [json.loads(line) for line in (root / "generated/items.jsonl").read_text().splitlines()]
    reports = [json.loads(line) for line in (root / "validation/item_reports.jsonl").read_text().splitlines()]
    summary = json.loads((root / "validation/run_summary.json").read_text())
    expected_topics = {"surface_water", "stream_dynamics", "groundwater", "lakes", "water_cycle",
                       "water_quality", "maps_data", "karst_glacial"}
    assert len(items) == len(reports) == 10
    assert expected_topics <= {topic for item in items for topic in item["topics"]}
    assert all(item["generation"]["model_generated"] is True for item in items)
    assert all(report["valid"] and report["blind_judge"]["answer_agrees"] for report in reports)
    assert summary["success"] is True


def test_model_helpers_enforce_plan_and_item_contracts():
    corpus = [{"id": "a", "topics": ["lakes"]}, {"id": "b", "topics": ["groundwater"]}]
    assert {row["id"] for row in select_for_enrichment(corpus, 8)} == {"a", "b"}
    plan = {"topic": "lakes", "response_type": "multiple_choice", "fixed_givens": ["x"],
            "verification": {"expected_answer": "A"},
            "reasoning_graph": {"nodes": [{}, {}, {}], "edges": [{}, {}]}}
    assert validate_plan(plan, "lakes", "multiple_choice") == []
    item = {"id": "x", "prompt": "A lake cools.", "answer": "A", "solution": "It mixes.", "points": 1,
            "difficulty": 2, "topics": ["lakes"], "response_type": "multiple_choice",
            "choices": {"A": "one", "B": "two", "C": "three", "D": "four"}}
    assert validate_item(item, "lakes", "multiple_choice") == []


def test_production_model_and_mcq_answer_normalization():
    assert REQUIRED_GENERATION_MODEL == "gpt-5-mini"
    item = {"answer": "Choice b: infiltration", "choices": {"A": "runoff", "B": "infiltration",
                                                               "C": "discharge", "D": "evaporation"}}
    normalize_mcq_answer(item)
    assert item["answer"] == "B"
    item["answer"] = "infiltration"
    normalize_mcq_answer(item)
    assert item["answer"] == "B"
