from copy import deepcopy

import pytest

from src.science_olympiad_remediation import (
    remediation_errors,
    require_remediation_batch,
    validate_remediation_batch,
)


def valid_artifacts():
    original = {
        "id": "dynamic-planet-b-model-002",
        "event": "dynamic_planet",
        "division": "B",
        "season": 2027,
    }
    candidate = {
        **original,
        "response_type": "multiple_choice",
        "prompt": "A complete data table is supplied. Which interpretation follows?",
        "choices": {"A": "One", "B": "Two", "C": "Three", "D": "Four"},
        "answer": "B",
        "solution": "The supplied observations uniquely support choice B.",
        "difficulty": 3,
        "generation": {
            "model_generated": True,
            "generation_model": "gpt-5-mini",
            "planning_model": "gpt-5-mini",
            "embedding_model": "text-embedding-3-small",
        },
    }
    report = {
        "id": original["id"],
        "deterministic_valid": True,
        "deterministic_errors": [],
        "blind_solves": [
            {
                "model": "gpt-5-mini", "answer_hidden": True, "solution_hidden": True,
                "verdict": "PASS", "issues": [], "solvable": True,
                "science_correct": True, "well_posed": True, "unique_answer": True,
                "independent_answer": "B",
            },
            {
                "model": "gpt-5-mini", "answer_hidden": True, "solution_hidden": True,
                "verdict": "PASS", "issues": [], "solvable": True,
                "science_correct": True, "well_posed": True, "unique_answer": True,
                "independent_answer": "B",
            },
        ],
        "adversarial_audit": {
            "model": "gpt-5-mini", "verdict": "PASS", "issues": [],
            "no_ambiguity": True, "no_missing_information": True,
            "key_uniquely_supported": True, "competition_faithful": True,
            "division_appropriate": True, "independent_answers_agree_with_key": True,
        },
        "difficulty_audit": {
            "model": "gpt-5-mini", "verdict": "PASS", "issues": [],
            "actual_difficulty": 3, "reasoning_steps": 3,
            "non_obvious_decisions": ["select the relevant observations", "rule out a confounder"],
            "routine_template": False, "formula_recall_sufficient": False,
            "competition_level": True,
        },
        "novelty": {
            "passed": True, "nearest_source_id": "official-014",
            "max_source_similarity": 0.51, "max_bank_similarity": 0.42,
        },
        "rubric_alignment": {
            "passed": True, "uncovered_demands": [], "extraneous_criteria": [],
        },
    }
    return original, candidate, report


def test_complete_candidate_passes_fail_closed_gate():
    original, candidate, report = valid_artifacts()
    assert remediation_errors(original, candidate, report) == []
    assert validate_remediation_batch([original], [candidate], [report]) == []
    require_remediation_batch([original], [candidate], [report])


@pytest.mark.parametrize("field", ["id", "event", "division", "season"])
def test_identity_and_scope_are_immutable(field):
    original, candidate, report = valid_artifacts()
    candidate[field] = "changed"
    assert remediation_errors(original, candidate, report)


def test_batch_requires_exact_stable_id_sets_and_no_duplicates():
    original, candidate, report = valid_artifacts()
    extra = deepcopy(candidate)
    extra["id"] = "unexpected-id"
    errors = validate_remediation_batch([original], [candidate, extra, candidate], [report])
    assert any("candidate id set" in error for error in errors)
    assert any("duplicate id" in error for error in errors)


@pytest.mark.parametrize("model", ["gpt-5.1", "gpt-5", "GPT-5-mini", "gpt-5-mini-2025-01-01"])
def test_generation_model_must_be_literal_gpt5mini(model):
    original, candidate, report = valid_artifacts()
    candidate["generation"]["generation_model"] = model
    assert any("generation_model" in error for error in remediation_errors(original, candidate, report))


def test_any_declared_non_embedding_model_must_be_gpt5mini():
    original, candidate, report = valid_artifacts()
    candidate["generation"]["planning_model"] = "gpt-5.1"
    assert any("planning_model" in error for error in remediation_errors(original, candidate, report))


def test_requires_two_answer_and_solution_blind_agreeing_solves():
    original, candidate, report = valid_artifacts()
    report["blind_solves"][1]["independent_answer"] = "C"
    report["blind_solves"][0]["answer_hidden"] = False
    errors = remediation_errors(original, candidate, report)
    assert any("not answer-and-solution blind" in error for error in errors)
    assert any("disagrees with the candidate key" in error for error in errors)
    assert any("two blind solves disagree" in error for error in errors)


def test_constructed_blind_solves_may_use_equivalent_numeric_wording():
    original, candidate, report = valid_artifacts()
    candidate.update({
        "response_type": "numeric",
        "answer": "4",
        "rubric": "2 points for a value of 4 with correct supporting work.",
    })
    candidate.pop("choices")
    report["blind_solves"][0]["independent_answer"] = "4"
    report["blind_solves"][1]["independent_answer"] = "4.0 units"
    assert remediation_errors(original, candidate, report) == []


def test_editor_may_adjudicate_semantically_equivalent_short_answers():
    original, candidate, report = valid_artifacts()
    candidate.update({
        "response_type": "short_answer",
        "answer": "Remove the communal food source.",
        "rubric": "2 points for stopping exposure to the implicated communal food.",
    })
    candidate.pop("choices")
    report["blind_solves"][0]["independent_answer"] = "Stop self-service immediately."
    report["blind_solves"][1]["independent_answer"] = "Discard the shared platter."
    assert remediation_errors(original, candidate, report) == []


def test_deterministic_and_adversarial_checks_are_fail_closed():
    original, candidate, report = valid_artifacts()
    report["deterministic_valid"] = False
    report["adversarial_audit"]["no_missing_information"] = False
    errors = remediation_errors(original, candidate, report)
    assert any("deterministic validation" in error for error in errors)
    assert any("no_missing_information" in error for error in errors)


def test_difficulty_uses_calibrated_shortest_solve_not_model_label():
    original, candidate, report = valid_artifacts()
    report["difficulty_audit"].update({
        "actual_difficulty": 4,
        "reasoning_steps": 1,
        "non_obvious_decisions": [],
        "routine_template": True,
    })
    errors = remediation_errors(original, candidate, report)
    assert any("calibrated difficulty 1" in error for error in errors)


@pytest.mark.parametrize(
    "field,value",
    [("max_source_similarity", 0.82), ("max_bank_similarity", 0.88), ("max_source_similarity", None)],
)
def test_novelty_requires_measured_scores_below_strict_thresholds(field, value):
    original, candidate, report = valid_artifacts()
    report["novelty"][field] = value
    assert any(field in error for error in remediation_errors(original, candidate, report))


def test_constructed_response_requires_rubric_and_prompt_alignment():
    original, candidate, report = valid_artifacts()
    candidate["response_type"] = "short_answer"
    candidate.pop("choices")
    candidate["answer"] = "a concise result"
    for solve in report["blind_solves"]:
        solve["independent_answer"] = "a concise result"
    report["rubric_alignment"]["uncovered_demands"] = ["justify the trend"]
    errors = remediation_errors(original, candidate, report)
    assert any("does not cover every prompt demand" in error for error in errors)
    assert any("lacks a rubric" in error for error in errors)


def test_constructed_response_with_complete_rubric_passes():
    original, candidate, report = valid_artifacts()
    candidate.update({
        "response_type": "short_answer",
        "answer": "a concise result",
        "rubric": "1 point for the result; 1 point for evidence from the supplied data.",
    })
    candidate.pop("choices")
    for solve in report["blind_solves"]:
        solve["independent_answer"] = "a concise result"
    assert remediation_errors(original, candidate, report) == []


def test_answer_only_contract_rejects_split_work_rubric():
    from src.science_olympiad_remediation import spec_contract_errors

    item = {
        "response_type": "numeric",
        "rubric": [
            "1 point: compute the intermediate genotype probability.",
            "1 point: give the correct final probability.",
        ],
    }
    assert spec_contract_errors(item, {"answer_only_rubric": True}) == [
        "answer-only response requires exactly one rubric criterion"
    ]


def test_answer_only_contract_accepts_single_final_answer_rubric():
    from src.science_olympiad_remediation import spec_contract_errors

    item = {"response_type": "numeric", "rubric": ["2 points: Correct final probability 0.375."]}
    assert spec_contract_errors(item, {"answer_only_rubric": True}) == []


def test_require_gate_raises_with_actionable_errors():
    original, candidate, report = valid_artifacts()
    report["novelty"]["passed"] = False
    with pytest.raises(ValueError, match="novelty audit did not pass"):
        require_remediation_batch([original], [candidate], [report])
