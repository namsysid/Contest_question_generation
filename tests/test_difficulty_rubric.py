from src.difficulty_rubric import calibrated_difficulty


def test_routine_formula_template_is_basic_even_if_model_calls_it_challenging():
    evidence = {
        "difficulty": 3,
        "reasoning_steps": 3,
        "non_obvious_decisions": [],
        "routine_template": True,
        "formula_recall_sufficient": True,
    }
    assert calibrated_difficulty(evidence) == 1


def test_competition_standard_requires_linked_steps_and_real_decision():
    assert calibrated_difficulty({
        "difficulty": 2, "reasoning_steps": 2,
        "non_obvious_decisions": ["infer an unstated constraint that changes the governing equation"],
        "routine_template": False, "formula_recall_sufficient": False,
    }) == 2
    assert calibrated_difficulty({
        "difficulty": 2, "reasoning_steps": 4, "non_obvious_decisions": [],
        "routine_template": False, "formula_recall_sufficient": False,
    }) == 1


def test_challenging_requires_two_non_obvious_decisions():
    assert calibrated_difficulty({
        "difficulty": 3, "reasoning_steps": 3,
        "non_obvious_decisions": ["choose frame"],
    }) == 2


def test_challenging_allows_one_decision_with_confirmed_hidden_constraint():
    assert calibrated_difficulty({
        "difficulty": 3,
        "reasoning_steps": 4,
        "non_obvious_decisions": ["choose the lagged exposure representation"],
        "hidden_constraint_or_shortcut": True,
    }) == 3
