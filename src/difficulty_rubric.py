"""Shared finished-item difficulty contract for contest generation and auditing."""
from __future__ import annotations


DIFFICULTY_RUBRIC = """DIFFICULTY RUBRIC (judge the finished student-facing solve, not stem length,
algebra volume, blueprint node count, or solution verbosity):
1 — Easy/foundational: recall or one direct substitution/inference; little method selection.
2 — Accessible: one or two standard applications with obvious setup; easier than normal competition level.
3 — Normal competition: genuinely multi-step and reasoning-heavy; the solver must select and connect
    principles, interpret constraints, or reject a tempting approach. Not routine plug-and-chug.
4 — Hard competition: several coupled ideas or a non-obvious model/representation; substantial critical
    thinking, with limited scaffolding. Representative of the harder quartile of the named competition.
5 — Challenge: very hard even for a strong contestant; requires a deep insight plus multiple linked steps,
    careful case/constraint handling, or an elegant non-obvious synthesis. Do not fake difficulty with
    ugly arithmetic, excessive prose, obscure trivia, or missing information.
"""


def difficulty_instruction(target: int) -> str:
    if target not in range(1, 6):
        raise ValueError("target difficulty must be an integer from 1 through 5")
    return (
        DIFFICULTY_RUBRIC
        + f"\nREQUIRED FINISHED-ITEM DIFFICULTY: exactly {target}/5. "
        + "Design the reasoning structure to hit that level and self-reject a draft that does not."
    )
