"""Shared finished-item difficulty contract for contest generation and auditing."""
from __future__ import annotations


DIFFICULTY_RUBRIC = """DIFFICULTY RUBRIC (judge the shortest finished student-facing solve, not stem length,
topic names, algebra volume, blueprint node count, solution verbosity, or the source item's label):
1 — Basic/textbook: recall, a familiar named template, or one/two obvious formula substitutions with no
    genuine model selection. Adding rotational inertia, friction, vectors, or ugly arithmetic does not
    raise this level when the governing equation is standard and the stem supplies every needed condition.
2 — Competition standard: at least two causally linked physics deductions and at least one genuine,
    unstated modeling decision. The student must decide how to represent or connect the situation; merely
    recognizing an Atwood machine, collision formula, banked turn, or power formula is not enough.
3 — Challenging competition: a linked multi-step chain with at least two non-obvious decisions, a hidden
    constraint/regime, or a tempting shortcut that must be rejected. Representative of hard F=ma work.
4 — Difficult: several coupled ideas and a non-obvious model/representation; substantial critical thinking
    with limited scaffolding. Representative of the hardest quartile of the named competition.
5 — Exceptional challenge: very hard even for a strong contestant; requires a deep insight plus multiple
    linked steps and careful case/constraint handling. Never award difficulty for exotic nouns, tedious
    calculation, excessive prose, obscure trivia, or missing information.
"""


def calibrated_difficulty(evidence: dict) -> int:
    """Apply fail-closed evidence floors so a model cannot inflate a routine solve."""
    raw = evidence.get("difficulty", evidence.get("actual_difficulty", 1))
    try:
        score = max(1, min(5, int(raw)))
        steps = max(0, int(evidence.get("reasoning_steps", 0)))
    except (TypeError, ValueError):
        return 1
    decisions_value = evidence.get("non_obvious_decisions", evidence.get("critical_decisions", 0))
    try:
        decisions = len(decisions_value) if isinstance(decisions_value, list) else int(decisions_value or 0)
    except (TypeError, ValueError):
        decisions = 0
    routine = evidence.get("routine_template") is True or evidence.get("familiar_template") is True
    formula_only = evidence.get("formula_recall_sufficient") is True
    if evidence.get("competition_level") is False:
        return 1
    if (routine or formula_only) and decisions == 0:
        return 1
    for level, minimum_steps, minimum_decisions in ((5, 5, 3), (4, 4, 2)):
        if score >= level and steps >= minimum_steps and decisions >= minimum_decisions:
            return level
    hidden_constraint = evidence.get("hidden_constraint_or_shortcut") is True
    if score >= 3 and steps >= 3 and (
        decisions >= 2 or (decisions >= 1 and hidden_constraint)
    ):
        return 3
    if score >= 2 and steps >= 2 and decisions >= 1:
        return 2
    return 1


def difficulty_instruction(target: int) -> str:
    if target not in range(1, 6):
        raise ValueError("target difficulty must be an integer from 1 through 5")
    return (
        DIFFICULTY_RUBRIC
        + f"\nREQUIRED FINISHED-ITEM DIFFICULTY: exactly {target}/5. "
        + "Design the reasoning structure to hit that level and self-reject a draft that does not."
    )
