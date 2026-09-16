from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


TOPICS = {
    "epidemiology_foundations": ("epidemiology", "determinant", "distribution", "agent", "host", "environment"),
    "surveillance": ("surveillance", "reporting", "baseline", "notifiable", "case definition", "line list"),
    "outbreak_investigation": (
        "outbreak", "hypothesis", "epidemic curve", "epi curve", "cluster", "investigation",
        "incubation", "exposure window", "symptom onset",
    ),
    "study_design": ("cohort", "case-control", "case control", "cross-sectional", "experimental", "bias"),
    "measures_of_disease": ("attack rate", "incidence", "prevalence", "mortality", "relative risk", "odds ratio"),
    "data_interpretation": ("table", "graph", "curve", "calculate", "rate", "ratio", "risk"),
    "transmission_and_control": ("transmission", "reservoir", "vector", "vehicle", "isolation", "quarantine"),
    "prevention": ("prevention", "screening", "immunization", "primary", "secondary", "tertiary", "quaternary"),
}
DOMAIN_TERMS = tuple(sorted({term for values in TOPICS.values() for term in values}))
RESPONSE_TYPES = {"multiple_choice", "numeric", "short_answer"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def classify_topics(text: str) -> list[str]:
    lowered = text.lower()
    found = [topic for topic, terms in TOPICS.items() if any(term in lowered for term in terms)]
    return found or ["epidemiology_foundations"]


def graph_text(item: dict[str, Any]) -> str:
    graph = (item.get("analysis") or {}).get("reasoning_graph") or {}
    nodes = " | ".join(f"{n.get('type')}:{n.get('label')}" for n in graph.get("nodes") or [])
    edges = " | ".join(f"{e.get('src')}-{e.get('type')}->{e.get('dst')}" for e in graph.get("edges") or [])
    return f"FORMAT:{item.get('response_type')} TOPICS:{','.join(item.get('topics') or [])} NODES:{nodes} EDGES:{edges}"


def evaluate_expression(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    allowed = (ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
               ast.Div, ast.Pow, ast.USub, ast.UAdd)
    if any(not isinstance(node, allowed) or isinstance(node, ast.Constant)
           and not isinstance(node.value, (int, float)) for node in ast.walk(tree)):
        raise ValueError("expression must contain only numeric arithmetic")
    value = float(eval(compile(tree, "<verification>", "eval"), {"__builtins__": {}}, {}))
    if not math.isfinite(value):
        raise ValueError("expression is not finite")
    return value


def verification_error(plan: dict[str, Any]) -> str | None:
    verification = plan.get("verification") or {}
    expression, expected = verification.get("expression"), verification.get("expected_value")
    if expression is None and expected is None:
        return None
    if expression is None or not isinstance(expected, (int, float)):
        return "numeric verification requires expression and expected_value"
    try:
        actual = evaluate_expression(str(expression))
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError) as exc:
        return str(exc)
    if abs(actual - float(expected)) > max(1e-7, abs(float(expected)) * 1e-5):
        return f"expression evaluates to {actual}, not {expected}"
    return None


def validate_item(item: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    response_type = item.get("response_type")
    prompt, solution = str(item.get("prompt") or "").strip(), str(item.get("solution") or "").strip()
    if response_type not in RESPONSE_TYPES:
        errors.append(f"invalid response_type: {response_type}")
    if not prompt:
        errors.append("prompt is required")
    if any(phrase in prompt.lower() for phrase in ("shown below", "figure above", "following graph", "table above")):
        errors.append("item depends on unavailable external material")
    if not any(term in (prompt + " " + solution).lower() for term in DOMAIN_TERMS):
        errors.append("item does not substantively test Disease Detectives")
    if not isinstance(item.get("difficulty"), int) or not 1 <= item["difficulty"] <= 5:
        errors.append("difficulty must be 1-5")
    if not isinstance(item.get("points"), (int, float)) or item["points"] <= 0:
        errors.append("points must be positive")
    if response_type == "multiple_choice":
        choices = item.get("choices") or {}
        if set(choices) != set("ABCD"):
            errors.append("MCQ requires exactly choices A-D")
        if item.get("answer") not in choices:
            errors.append("answer must name one choice")
        if len({" ".join(str(v).lower().split()) for v in choices.values()}) != 4:
            errors.append("choices must be distinct")
    elif item.get("answer") in (None, ""):
        errors.append("constructed response requires an answer")
    if response_type in {"short_answer", "numeric"} and re.search(
            r"(?mi)^\s*A[.)]\s+.*\n\s*B[.)]\s+", prompt):
        errors.append("constructed-response prompt must not contain A-D choices")
    if not solution:
        errors.append("worked solution is required")
    return errors
