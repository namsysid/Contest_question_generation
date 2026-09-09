from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


TOPICS = {
    "mechanical_advantage": ("mechanical advantage", "ima", "ama", "effort force", "load force"),
    "efficiency_work_power": ("efficiency", "work", "power", "energy", "input", "output"),
    "levers_torque": ("lever", "fulcrum", "torque", "moment", "equilibrium", "effort arm", "load arm"),
    "pulleys": ("pulley", "rope", "supporting strand", "block and tackle", "sheave"),
    "inclined_planes_wedges": ("inclined plane", "ramp", "wedge", "slope", "angle"),
    "wheel_axle_screws": ("wheel and axle", "wheel", "axle", "screw", "pitch", "thread"),
    "gears_compound": ("gear", "teeth", "rpm", "compound machine", "gear ratio", "sprocket"),
    "classical_mechanics": ("newton", "friction", "velocity", "acceleration", "momentum", "centripetal"),
}
MACHINE_TERMS = tuple(sorted({term for terms in TOPICS.values() for term in terms}))
OUT_OF_SCOPE = ("calculus", "integral", "derivative", "relativity", "quantum", "fluid pressure",
                "electric circuit", "molar", "acid-base", "projectile motion")
RESPONSE_TYPES = {"multiple_choice", "short_answer", "numeric"}


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
    topics = [topic for topic, terms in TOPICS.items() if any(term in lowered for term in terms)]
    return topics or ["mechanical_advantage"]


def graph_text(item: dict[str, Any]) -> str:
    analysis = item.get("analysis") or {}
    graph = analysis.get("reasoning_graph") or {}
    nodes = " | ".join(f"{node.get('type')}:{node.get('label')}" for node in graph.get("nodes") or [])
    edges = " | ".join(f"{edge.get('src')}-{edge.get('type')}->{edge.get('dst')}" for edge in graph.get("edges") or [])
    return f"FORMAT:{item.get('response_type')} TOPICS:{','.join(item.get('topics') or [])} NODES:{nodes} EDGES:{edges}"


def validate_item(item: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    response_type = item.get("response_type")
    prompt = str(item.get("prompt") or "").strip()
    solution = str(item.get("solution") or "").strip()
    if response_type not in RESPONSE_TYPES:
        errors.append(f"invalid response_type: {response_type}")
    if not prompt:
        errors.append("prompt is required")
    if any(token in prompt.lower() for token in ("shown below", "pictured", "diagram above", "figure below")):
        errors.append("question depends on an unavailable diagram")
    if not any(term in (prompt + " " + solution).lower() for term in MACHINE_TERMS):
        errors.append("question does not substantively test Machines")
    for phrase in OUT_OF_SCOPE:
        if phrase in (prompt + " " + solution).lower():
            errors.append(f"out-of-scope content: {phrase}")
    points = item.get("points")
    if not isinstance(points, (int, float)) or points <= 0:
        errors.append("points must be positive")
    difficulty = item.get("difficulty")
    if not isinstance(difficulty, int) or not 1 <= difficulty <= 5:
        errors.append("difficulty must be 1-5")
    if response_type == "multiple_choice":
        choices = item.get("choices") or {}
        if set(choices) != {"A", "B", "C", "D"}:
            errors.append("MCQ requires exactly choices A-D")
        if item.get("answer") not in choices:
            errors.append("answer must name one choice")
        if len({" ".join(str(value).lower().split()) for value in choices.values()}) != 4:
            errors.append("choices must be distinct")
    elif item.get("answer") in (None, ""):
        errors.append("constructed response requires an answer")
    if response_type in {"short_answer", "numeric"} and re.search(
            r"(?m)^\s*A[.)]\s+.*\n\s*B[.)]\s+", prompt):
        errors.append("constructed-response prompt must not contain A-D answer choices")
    if not solution:
        errors.append("worked solution is required")
    percentages = [float(value) for value in re.findall(r"(-?\d+(?:\.\d+)?)\s*%", prompt + " " + solution)]
    if any(value < 0 or value > 100 for value in percentages):
        errors.append("machine efficiency must be between 0% and 100%")
    return errors


def evaluate_expression(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    allowed = (ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
               ast.Div, ast.Pow, ast.USub, ast.UAdd)
    if any(not isinstance(node, allowed) or isinstance(node, ast.Constant) and not isinstance(node.value, (int, float))
           for node in ast.walk(tree)):
        raise ValueError("expression must contain numbers and arithmetic operators only")
    value = float(eval(compile(tree, "<verification>", "eval"), {"__builtins__": {}}, {}))
    if not math.isfinite(value):
        raise ValueError("expression is not finite")
    return value


def verification_error(plan: dict[str, Any]) -> str | None:
    verification = plan.get("verification") or {}
    expression = verification.get("expression")
    expected = verification.get("expected_value")
    if expression is None and expected is None:
        return None
    if expression is None or not isinstance(expected, (int, float)):
        return "numeric verification requires expression and expected_value"
    try:
        calculated = evaluate_expression(str(expression))
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError) as exc:
        return str(exc)
    if abs(calculated - float(expected)) > max(1e-7, abs(float(expected)) * 1e-5):
        return f"expression evaluates to {calculated}, not {expected}"
    return None
