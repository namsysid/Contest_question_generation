from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


TOPICS = {
    "reflection_mirrors": ("reflection", "mirror", "specular", "diffuse", "image distance"),
    "refraction_tir": ("refraction", "snell", "refractive index", "critical angle", "total internal", "fiber"),
    "lenses_images": ("lens", "focal length", "focal point", "magnification", "diopter", "thin lens"),
    "human_eye_vision": ("eye", "retina", "cornea", "iris", "ciliary", "myopia", "hyperopia", "astigmatism"),
    "color_spectra": ("color", "colour", "wavelength", "spectrum", "absorbance", "filter", "additive", "subtractive"),
    "physical_optics": ("diffraction", "interference", "polarization", "polarisation", "brewster", "wave"),
    "optical_instruments": ("microscope", "telescope", "camera", "periscope", "sextant", "spectrometer"),
    "light_photons": ("photon", "frequency", "speed of light", "energy", "emission", "rydberg", "doppler"),
}
OPTICS_TERMS = tuple(sorted({term for terms in TOPICS.values() for term in terms}))
OUT_OF_SCOPE = (
    "electric circuit", "ohm's law", "simple machine", "mechanical advantage", "molarity",
    "stoichiometry", "projectile motion", "f=ma", "force equals mass",
)
RESPONSE_TYPES = {"multiple_choice", "short_answer", "numeric"}
MISSING_VISUAL_PHRASES = (
    "shown below", "pictured", "diagram above", "figure below", "figure 1", "figure 2",
    "following diagram", "graph below", "spectrum shown", "image sheet",
)


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
    return found or ["reflection_mirrors"]


def graph_text(item: dict[str, Any]) -> str:
    graph = (item.get("analysis") or {}).get("reasoning_graph") or {}
    nodes = " | ".join(f"{node.get('type')}:{node.get('label')}" for node in graph.get("nodes") or [])
    edges = " | ".join(f"{edge.get('src')}-{edge.get('type')}->{edge.get('dst')}"
                       for edge in graph.get("edges") or [])
    return f"FORMAT:{item.get('response_type')} TOPICS:{','.join(item.get('topics') or [])} NODES:{nodes} EDGES:{edges}"


def validate_item(item: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    response_type = item.get("response_type")
    prompt = str(item.get("prompt") or "").strip()
    solution = str(item.get("solution") or "").strip()
    combined = (prompt + " " + solution).lower()
    if response_type not in RESPONSE_TYPES:
        errors.append(f"invalid response_type: {response_type}")
    if not prompt:
        errors.append("prompt is required")
    if any(phrase in prompt.lower() for phrase in MISSING_VISUAL_PHRASES):
        errors.append("question depends on an unavailable visual")
    if not any(term in combined for term in OPTICS_TERMS):
        errors.append("question does not substantively test Optics")
    for phrase in OUT_OF_SCOPE:
        if phrase in combined:
            errors.append(f"out-of-scope content: {phrase}")
    if not isinstance(item.get("points"), (int, float)) or item.get("points", 0) <= 0:
        errors.append("points must be positive")
    if not isinstance(item.get("difficulty"), int) or not 1 <= item.get("difficulty", 0) <= 5:
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
        errors.append("constructed-response prompt must not contain A-D choices")
    if not solution:
        errors.append("worked solution is required")
    return errors


def evaluate_expression(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    allowed = (ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
               ast.Div, ast.Pow, ast.USub, ast.UAdd)
    if any(not isinstance(node, allowed) or isinstance(node, ast.Constant) and
           not isinstance(node.value, (int, float)) for node in ast.walk(tree)):
        raise ValueError("expression must contain numbers and arithmetic operators only")
    result = float(eval(compile(tree, "<optics-verification>", "eval"), {"__builtins__": {}}, {}))
    if not math.isfinite(result):
        raise ValueError("expression is not finite")
    return result


def verification_error(plan: dict[str, Any]) -> str | None:
    verification = plan.get("verification") or {}
    expression, expected = verification.get("expression"), verification.get("expected_value")
    if expression is None and expected is None:
        return None
    if expression is None or not isinstance(expected, (int, float)):
        return "numeric verification requires both expression and expected_value"
    try:
        actual = evaluate_expression(str(expression))
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError) as exc:
        return str(exc)
    if abs(actual - float(expected)) > max(1e-7, abs(float(expected)) * 1e-5):
        return f"expression evaluates to {actual}, not {expected}"
    return None
