from src.disease_detectives.common import classify_topics, evaluate_expression, validate_item, verification_error
from src.disease_detectives.pipeline import (DEFAULT_GENERATION_MODEL, deterministic_bank_audit,
                                              blind_answer_agrees, load_or_embed, schedule_anchors)


def test_production_model_is_exact_gpt_5_mini():
    assert DEFAULT_GENERATION_MODEL == "gpt-5-mini"


def test_existing_embeddings_are_aligned_to_filtered_corpus_ids(tmp_path):
    from src.disease_detectives.common import write_jsonl
    corpus = [{"id":"b"}, {"id":"a"}]
    q_path, g_path = tmp_path/"q.jsonl", tmp_path/"g.jsonl"
    write_jsonl(q_path, [{"id":"stale","embedding":[9]}, {"id":"a","embedding":[1]}, {"id":"b","embedding":[2]}])
    write_jsonl(g_path, [{"id":"a","embedding":[3]}, {"id":"stale","embedding":[9]}, {"id":"b","embedding":[4]}])
    q_rows, g_rows = load_or_embed(corpus, q_path, g_path, "unused", "openai")
    assert [row["id"] for row in q_rows] == ["b", "a"]
    assert [row["id"] for row in g_rows] == ["b", "a"]


def test_blind_short_answer_accepts_clear_paraphrase():
    item = {"response_type":"short_answer",
            "answer":"An association supports the food as a likely source but does not prove causation without confirmation."}
    independent = "The food association suggests a likely source, but it cannot prove causation without confirmation."
    assert blind_answer_agrees(item, independent)


def test_topics_and_numeric_verification():
    assert "study_design" in classify_topics("A case-control study estimates an odds ratio.")
    assert evaluate_expression("(20*30)/(10*15)") == 4
    assert verification_error({"verification":{"expression":"(20*30)/(10*15)","expected_value":5}})


def test_valid_mcq_and_external_figure_rejection():
    item = {"id":"x","response_type":"multiple_choice","prompt":"In surveillance, what is a case definition?",
            "choices":{"A":"A consistent set of criteria","B":"A treatment","C":"A pathogen","D":"A graph"},
            "answer":"A","solution":"A case definition consistently classifies cases for surveillance.",
            "points":1,"difficulty":1,"topics":["surveillance"]}
    assert validate_item(item) == []
    item["prompt"] = "Using the following graph, identify the outbreak pattern."
    assert any("unavailable external" in error for error in validate_item(item))


def test_constructed_response_rejects_embedded_choices():
    item = {"response_type":"short_answer","prompt":"Pick one.\nA. First\nB. Second\nC. Third\nD. Fourth",
            "answer":"A","solution":"Surveillance makes this a relevant answer.","points":1,"difficulty":1}
    assert any("must not contain A-D" in error for error in validate_item(item))


def test_schedule_and_duplicate_audit():
    corpus = [{"id":f"x{i}","difficulty":1+i%3,"topics":["surveillance"],"prompt":"surveillance",
               "response_type":"multiple_choice"} for i in range(12)]
    assert len(schedule_anchors(corpus, 10)) == 10
    item = {"id":"same","response_type":"short_answer","prompt":"Define public health surveillance.",
            "answer":"Ongoing systematic collection and analysis of health data.",
            "solution":"Surveillance is ongoing systematic collection and analysis.","points":1,"difficulty":1,
            "topics":["surveillance"]}
    assert deterministic_bank_audit([item, dict(item)], corpus)["valid"] is False
