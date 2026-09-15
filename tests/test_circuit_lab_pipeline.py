import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.circuit_lab.common import classify_topics, graph_text, similarity_to_exemplars, text_answers_equivalent, validate_against_plan, validate_item
from src.circuit_lab import embed as circuit_embed
from src.circuit_lab.generate import planned_answer_error, retrieve, solution_key_error
from src.circuit_lab.plan import evaluate_numeric_expression, synchronize_expected_value, verification_error
from src.circuit_lab.retrieve import build_bundles
from src.circuit_lab.validate import answers_agree, strip_solutions
from src.circuit_lab import validate as circuit_validate


def test_classifies_circuit_topics():
    topics = classify_topics("Two resistors are connected in parallel to a battery. Find the current.")
    assert "circuit_analysis" in topics
    assert "quantities_ohms_law" in topics


def test_led_tag_does_not_match_controlled_or_closed():
    assert "led" not in classify_topics("A relay controlled a lamp when the switch closed.")


def test_valid_four_choice_item():
    item = {
        "response_type": "multiple_choice",
        "prompt": "A 6 V source is connected across a 3 ohm resistor. What current flows?",
        "choices": {"A": "0.5 A", "B": "2 A", "C": "3 A", "D": "18 A"},
        "answer": "B", "solution": "I=V/R=2 A.", "points": 1,
        "level": "regional", "allow_state_topics": False,
    }
    assert validate_item(item) == []


def test_rejects_unresolved_solution_repair_narration():
    item = {
        "response_type": "multiple_choice", "prompt": "A battery powers a resistor. Find current.",
        "choices": {"A": "1 A", "B": "2 A", "C": "3 A", "D": "4 A"}, "answer": "A",
        "solution": "The result is 5 A, which is not among the choices. Let's update the choices.", "points": 1,
    }
    errors = validate_item(item)
    assert any("draft-repair artifact" in error for error in errors)


def test_short_answer_requires_answer():
    item = {"response_type": "short_answer", "prompt": "State one purpose of a fuse.", "answer": None, "points": 1}
    assert "constructed-response answer is required" in validate_item(item)


def test_multipart_requires_ordered_dependencies_and_exact_point_total():
    item = {"response_type": "multipart", "prompt": "A 6 V battery powers a circuit.", "points": 3,
            "parts": [
                {"label": "a", "response_type": "numeric", "prompt": "Find the current.",
                 "answer": {"value": 2, "unit": "A"}, "solution": "Use Ohm's law.", "points": 1,
                 "depends_on": []},
                {"label": "b", "response_type": "short_answer", "prompt": "Explain the power change.",
                 "answer": "It increases.", "solution": "Power depends on current.", "points": 2,
                 "depends_on": ["a"]}]}
    assert validate_item(item) == []
    item["parts"][0]["depends_on"] = ["b"]
    item["points"] = 4
    errors = validate_item(item)
    assert any("reference earlier parts" in error for error in errors)
    assert any("part total" in error for error in errors)


def test_rejects_out_of_scope_content():
    item = {"response_type": "short_answer", "prompt": "Use a phasor to calculate impedance.", "answer": "x", "points": 1}
    assert any("out-of-scope" in error for error in validate_item(item))


def test_retrieval_respects_format_and_topic():
    rows = [
        {"id": "a", "response_type": "multiple_choice", "prompt": "Who was Ohm?", "topics": ["history"]},
        {"id": "b", "response_type": "multiple_choice", "prompt": "Resistors in a circuit", "topics": ["circuit_analysis"]},
        {"id": "c", "response_type": "multipart", "prompt": "Circuit calculations", "topics": ["circuit_analysis"]},
    ]
    selected = retrieve(rows, "circuit_analysis", "multiple_choice", 2)
    assert [row["id"] for row in selected] == ["b", "a"]
    selected = retrieve(rows, "circuit_analysis", "short_answer", 2)
    assert [row["id"] for row in selected] == ["c"]


def test_graph_text_contains_structure_and_format():
    item = {"response_type": "multiple_choice", "topics": ["circuit_analysis"], "points": 2,
            "analysis": {"skills": ["equivalent resistance"], "reasoning_graph": {
                "nodes": [{"id": "g1", "type": "Given", "label": "two branches"}], "edges": []}}}
    text = graph_text(item)
    assert "FORMAT: multiple_choice" in text
    assert "g1:Given:two branches" in text


def test_dual_embeddings_and_format_stratified_anchors(monkeypatch):
    monkeypatch.setattr(circuit_embed, "embed_texts", lambda model, texts, **kwargs: [[float(i + 1), 1.0] for i, _ in enumerate(texts)])
    rows = [
        {"id": "m1", "response_type": "multiple_choice", "prompt": "one", "topics": ["history"], "points": 1},
        {"id": "m2", "response_type": "multiple_choice", "prompt": "two", "topics": ["history"], "points": 1},
        {"id": "s1", "response_type": "multipart", "prompt": "three", "topics": ["circuit_analysis"], "points": 3},
    ]
    qrows, srows, anchors = circuit_embed.build_indices(rows, "fake", 64, 0.1, 2)
    assert len(qrows) == len(srows) == 3
    assert {a["response_type"] for a in anchors} == {"multiple_choice", "multipart"}


def test_embedding_retrieval_bundle_uses_matching_format_and_topic():
    enriched = [
        {"id": "a", "response_type": "multiple_choice", "prompt": "a", "topics": ["history"]},
        {"id": "b", "response_type": "multiple_choice", "prompt": "b", "topics": ["history"]},
        {"id": "c", "response_type": "multipart", "prompt": "c", "topics": ["circuit_analysis"]},
    ]
    qrows = [{"id": r["id"], "embedding": [1.0, i + 0.1]} for i, r in enumerate(enriched)]
    srows = [{"id": r["id"], "embedding": [i + 0.1, 1.0], "graph_text": r["id"]} for i, r in enumerate(enriched)]
    bundles = build_bundles(enriched, qrows, srows, [{"id": "a"}], 1, 2, 2, 0.7, 1)
    assert bundles[0]["anchor_id"] == "a"
    assert [x["id"] for x in bundles[0]["question_exemplars"]] == ["b"]


def test_blind_answer_comparison():
    assert answers_agree({"response_type": "multiple_choice", "answer": "B"}, "B")
    assert answers_agree({"response_type": "multiple_choice", "answer": "B"}, "The result corresponds to choice B.")
    assert answers_agree({"response_type": "multiple_choice", "answer": "A"}, "A. The GFCI trips first.")
    assert answers_agree({"response_type": "numeric", "answer": {"value": 2.0, "tolerance": 0.1}}, "2.05")
    assert answers_agree({"response_type": "numeric", "answer": {"value": 1358.2, "tolerance": 15}},
                         "1.36 × 10³ g/L")
    assert answers_agree({"response_type": "numeric", "answer": {"value": 1358, "tolerance": 1}},
                         "1,358 g/L")


def test_multipart_blinding_and_answer_comparison():
    item = {"response_type": "multipart", "prompt": "Analyze the circuit.", "parts": [
        {"label": "a", "response_type": "numeric", "prompt": "Find current.",
         "answer": {"value": 2, "unit": "A", "tolerance": 0.1}, "solution": "hidden", "points": 1},
        {"label": "b", "response_type": "short_answer", "prompt": "State the effect.",
         "answer": "lamp gets brighter", "solution": "hidden", "rubric": ["hidden"], "points": 1}]}
    blind = strip_solutions(item)
    assert "answer" not in blind["parts"][0] and "solution" not in blind["parts"][1]
    assert answers_agree(item, {"a": "2.05", "b": "The lamp gets brighter."})


def test_rejects_decorative_scientist_date_arithmetic():
    item = {"response_type": "numeric", "prompt": "Ampere was born in 1775. How many years ago was that?",
            "answer": {"value": 249, "unit": "years", "tolerance": 0}, "solution": "2024-1775=249", "points": 1}
    assert any("biographical/date arithmetic" in error for error in validate_item(item))


def test_rejects_format_drift_from_mcq_anchor():
    item = {"response_type": "numeric", "prompt": "Find current through a resistor.", "solution": "Use Ohm's law."}
    bundle = {"response_type": "multiple_choice", "topics": ["circuit_analysis"]}
    plan = {"response_type": "multiple_choice", "topics": ["circuit_analysis"], "target_skill": "apply Ohm's law"}
    assert any("response format drift" in error for error in validate_against_plan(item, bundle, plan))


def test_detects_close_rewrite_of_real_question():
    generated = "Kirchhoff's Voltage Law says voltage changes around a closed loop sum to zero. Which conservation principle does this express?"
    exemplars = [{"id": "real-18", "prompt": "Kirchhoff's Voltage Law is fundamentally a statement of which conservation law?"}]
    score, source_id = similarity_to_exemplars(generated, exemplars)
    assert score >= 0.50
    assert source_id == "real-18"


def test_detects_solution_and_answer_key_disagreement():
    item = {"response_type": "multiple_choice", "answer": "B", "solution": "The correct answer is C."}
    assert solution_key_error(item) == "solution states choice C but answer field is B"


def test_generated_choice_must_match_verified_plan_answer():
    item = {"response_type": "multiple_choice", "choices": {"A": "24 W", "B": "62.4 W"}, "answer": "A"}
    plan = {"verification": {"expected_answer": "62.4 W"}}
    assert planned_answer_error(item, plan)
    item["answer"] = "B"
    assert planned_answer_error(item, plan) is None


def test_machine_checks_plan_arithmetic():
    assert abs(evaluate_numeric_expression("24**2/(6+1/(1/12+1/8))") - 53.333333333333336) < 1e-12
    assert verification_error({"expression": "24**2/(6+1/(1/12+1/8))", "expected_value": 48})
    assert verification_error({"expression": "24**2/(6+1/(1/12+1/8))", "expected_value": 53.3333333333}) is None


def test_expression_synchronizes_model_arithmetic():
    verification = {"expression": "30-18", "expected_value": 24, "expected_answer": "24 A entering"}
    synchronize_expected_value(verification)
    assert verification["expected_value"] == 12
    assert verification["expected_answer"] == "12 A entering"


def test_regional_led_item_is_allowed_when_present_in_source_ruleset():
    item = {"response_type": "multiple_choice", "prompt": "Which LED is lit?", "choices": {k: k for k in "ABCD"},
            "answer": "A", "solution": "The LED is forward biased.", "points": 1, "level": "regional"}
    assert not any("LED operation is reserved" in error for error in validate_item(item))


def test_equivalent_short_answers_allow_minor_paraphrase():
    assert text_answers_equivalent(
        "The lamp is on if S1 is closed or S2 is closed.",
        "The lamp will be ON when either switch S1 or switch S2 is closed.",
    )


def test_validation_rejects_duplicate_item_ids(tmp_path, monkeypatch):
    item = {
        "id": "duplicate-id", "response_type": "multiple_choice",
        "prompt": "A 6 V source is connected across a 3 ohm resistor. What current flows?",
        "choices": {"A": "0.5 A", "B": "2 A", "C": "3 A", "D": "18 A"},
        "answer": "B", "solution": "I=V/R=2 A.", "points": 1,
    }
    input_path = tmp_path / "items.jsonl"
    output_path = tmp_path / "reports.jsonl"
    input_path.write_text("\n".join(json.dumps(item) for _ in range(2)) + "\n")
    monkeypatch.setattr(sys, "argv", ["validate", "--input", str(input_path), "--out", str(output_path)])
    circuit_validate.main()
    reports = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert all("duplicate item id: duplicate-id" in report["deterministic_errors"] for report in reports)
