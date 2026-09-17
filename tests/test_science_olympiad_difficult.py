from copy import deepcopy

from src.science_olympiad_difficult import (
    DEFAULT_MANIFEST,
    _manifest,
    hard_contract_errors,
    spec_sha256,
)


def valid_hard_artifacts():
    spec = {
        "id": "example-hard-001",
        "topic": "groundwater",
        "response_type": "multiple_choice",
        "points": 6,
        "target_difficulty": 4,
    }
    architecture = {
        "selected_index": 0,
        "mechanisms": [
            {
                "core_chain": ["deduction 1", "deduction 2", "deduction 3", "deduction 4"],
                "indispensable_decisions": ["choose head representation", "reject depth shortcut"],
            },
            {"core_chain": [], "indispensable_decisions": []},
            {"core_chain": [], "indispensable_decisions": []},
        ],
    }
    item = {
        "id": spec["id"],
        "topics": [spec["topic"]],
        "response_type": spec["response_type"],
        "points": spec["points"],
        "difficulty": 4,
        "generation": {
            "difficulty_architecture": architecture,
            "manifest_spec_sha256": spec_sha256(spec),
        },
    }
    report = {
        "difficulty_audit": {
            "reasoning_steps": 4,
            "non_obvious_decisions": ["choose head representation", "reject depth shortcut"],
            "routine_template": False,
            "formula_recall_sufficient": False,
        }
    }
    return spec, item, report


def test_manifest_has_fifteen_new_exact_difficulty_items():
    manifest = _manifest(DEFAULT_MANIFEST)
    specs = [row for event in manifest["events"].values() for row in event["items"]]
    assert len(specs) == 15
    assert sum(row["target_difficulty"] == 3 for row in specs) == 9
    assert sum(row["target_difficulty"] == 4 for row in specs) == 6


def test_hard_contract_accepts_indispensable_exact_depth():
    spec, item, report = valid_hard_artifacts()
    assert hard_contract_errors(item, report, spec) == []


def test_hard_contract_rejects_inflated_routine_label():
    spec, item, report = valid_hard_artifacts()
    report = deepcopy(report)
    report["difficulty_audit"].update({
        "reasoning_steps": 2,
        "non_obvious_decisions": ["choose a named formula"],
        "routine_template": True,
        "formula_recall_sufficient": True,
    })
    errors = hard_contract_errors(item, report, spec)
    assert any("atomic deductions" in error for error in errors)
    assert any("non-obvious" in error for error in errors)
    assert any("routine" in error for error in errors)
    assert any("formula recall" in error for error in errors)


def test_hard_contract_requires_exact_not_minimum_difficulty():
    spec, item, report = valid_hard_artifacts()
    item["difficulty"] = 5
    assert "finished difficulty must be exactly 4" in hard_contract_errors(item, report, spec)


def test_disease_012_does_not_give_away_standardization_choice():
    spec, item, report = valid_hard_artifacts()
    spec["id"] = "disease-detectives-b-generated-012"
    item["id"] = spec["id"]
    item["prompt"] = "Use direct standardization with the supplied population."
    item["generation"]["manifest_spec_sha256"] = spec_sha256(spec)
    assert any("rate representation" in error for error in hard_contract_errors(item, report, spec))


def test_disease_014_requires_unified_model_and_decisive_risk_ratios():
    spec, item, report = valid_hard_artifacts()
    spec["id"] = "disease-detectives-b-generated-014"
    item["id"] = spec["id"]
    item["prompt"] = (
        "Compare Model 1, a Day 5 fair source with spread; Model 2, a cafeteria source without "
        "spread; or Model 3, two unrelated point sources."
    )
    item["answer"] = "Model 1; fair RR 9.0, cafeteria RR 1.15, contact RR 26.0."
    item["generation"]["manifest_spec_sha256"] = spec_sha256(spec)
    assert hard_contract_errors(item, report, spec) == []

    item["answer"] = "Model 1 is best supported."
    assert any("three decisive risk ratios" in error for error in hard_contract_errors(item, report, spec))
