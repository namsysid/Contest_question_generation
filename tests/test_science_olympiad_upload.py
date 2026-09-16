from src.science_olympiad_upload import database_document, prepare_documents


def test_database_document_preserves_standalone_numeric_answer():
    item = {
        "id": "machines-b-generated-test",
        "response_type": "numeric",
        "prompt": "A lever has an effort arm of 2 m and load arm of 1 m. Find its IMA.",
        "answer": "2",
        "solution": "IMA = 2/1 = 2.",
        "points": 1,
        "difficulty": 1,
        "topics": ["mechanical_advantage", "levers_torque"],
        "generation": {
            "pipeline": "machines_b_graph_rag",
            "anchor_id": "source-1",
            "source_embedding_similarity": 0.4,
        },
    }
    report = {"valid": True, "judge": {"solvable": True, "answer_agrees": True}}
    document = database_document(item, report, "machines_b")
    assert document["_id"] == "scioly-machines-b-generated-test"
    assert document["question_type"] == "frq"
    assert document["content"] == {"prompt": item["prompt"]}
    assert document["answer_key"]["answer"] == "2"
    assert document["event"] == "Machines"
    assert document["season"] == 2026


def test_database_document_preserves_mcq_choices_and_optics_season():
    item = {
        "id": "optics-b-final-test",
        "response_type": "multiple_choice",
        "prompt": "Which phenomenon bends light at a boundary?",
        "choices": {"A": "Reflection", "B": "Refraction", "C": "Diffraction", "D": "Scattering"},
        "answer": "B",
        "solution": "Refraction bends light at a boundary.",
        "points": 1,
        "difficulty": 1,
        "topics": ["refraction_tir"],
        "generation": {"pipeline": "optics_b_graph_rag"},
    }
    report = {"valid": True, "judge": {"solvable": True}}
    document = database_document(item, report, "optics_b")
    assert document["question_type"] == "mcq"
    assert document["content"]["choices"]["B"] == "Refraction"
    assert document["event"] == "Optics"
    assert document["season"] == 2025


def test_database_document_preserves_gpt5mini_provenance_and_blind_judge():
    item = {
        "id": "dynamic-planet-b-model-test",
        "response_type": "multiple_choice",
        "prompt": "Which process moves water from leaves to the atmosphere?",
        "choices": {"A": "Infiltration", "B": "Transpiration", "C": "Runoff", "D": "Condensation"},
        "answer": "B",
        "solution": "Transpiration releases water vapor from leaves.",
        "points": 1,
        "difficulty": 1,
        "topics": ["surface_water"],
        "generation": {
            "pipeline": "dynamic_planet_b_model_graph_rag_v1",
            "model_generated": True,
            "generation_model": "gpt-5-mini",
            "embedding_model": "text-embedding-3-small",
            "provider": "openai",
        },
    }
    report = {"valid": True, "blind_judge": {"solvable": True, "answer_agrees": True}}
    document = database_document(item, report, "dynamic_planet_b")
    assert document["event"] == "Dynamic Planet"
    assert document["season"] == 2027
    assert document["validation"]["independently_solved"] is True
    assert document["validation"]["quality_checks"]["answer_agrees"] is True
    assert document["provenance"]["model_generated"] is True
    assert document["provenance"]["generation_model"] == "gpt-5-mini"
    assert document["provenance"]["embedding_model"] == "text-embedding-3-small"
