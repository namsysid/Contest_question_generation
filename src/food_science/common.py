from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


TOPICS = {
    "carbohydrates_sweeteners": (
        "carbohydrate", "monosaccharide", "disaccharide", "polysaccharide", "sugar",
        "glucose", "fructose", "sucrose", "lactose", "sweetener", "glycemic",
    ),
    "proteins_enzymes": (
        "protein", "amino acid", "peptide", "enzyme", "denaturation", "protease",
        "amylase", "lipase", "biuret", "casein", "rennet",
    ),
    "lipids_emulsions": (
        "lipid", "fat", "oil", "emulsion", "emulsifier", "lecithin", "ethanol emulsion",
        "saturated", "unsaturated", "triglyceride",
    ),
    "food_safety_preservation": (
        "pasteurization", "sterilization", "pathogen", "food safety", "preservation",
        "canning", "refrigeration", "dehydration", "botulinum", "salmonella", "water activity",
        "osmophile", "xerophile", "halophile",
    ),
    "food_chemistry_reactions": (
        "maillard", "caramelization", "oxidation", "hydrolysis", "browning", "reducing sugar",
        "ph", "acid", "base", "buffer",
    ),
    "nutrition_processing": (
        "vitamin", "mineral", "calorie", "nutrition", "dietary fiber", "digestion",
        "fortification", "processing",
    ),
    "analytical_tests": (
        "benedict", "biuret", "iodine test", "sudan", "spectrophotometer", "absorbance",
        "titration", "indicator", "chromatography", "reagent", "assay", "aliquot", "dilution",
    ),
    "experimental_analysis": (
        "independent variable", "dependent variable", "control group", "replicate", "uncertainty",
        "density", "accuracy", "precision", "experimental design",
    ),
}
FOOD_SCIENCE_TERMS = tuple(sorted({term for terms in TOPICS.values() for term in terms}))
OUT_OF_SCOPE = (
    "electric circuit", "ohm's law", "simple machine", "mechanical advantage", "projectile motion",
    "f=ma", "force equals mass", "ray diagram", "focal length", "planetary", "astronomy",
)
RESPONSE_TYPES = {"multiple_choice", "short_answer", "numeric", "multipart"}
MISSING_VISUAL_PHRASES = (
    "shown below", "pictured", "diagram above", "figure below", "figure 1", "figure 2",
    "following diagram", "graph below", "spectrum shown", "image sheet",
)
CONTEXT_DEPENDENT_PHRASES = (
    "previous question", "preceding question", "mentioned in the previous", "described above",
    "the organism mentioned", "the enzyme mentioned", "the potential carcinogen mentioned",
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
    def contains(term: str) -> bool:
        return re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", lowered) is not None

    found = [topic for topic, terms in TOPICS.items() if any(contains(term) for term in terms)]
    return found or ["food_chemistry_reactions"]


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
    combined = json.dumps(item, ensure_ascii=False).lower()
    if response_type not in RESPONSE_TYPES:
        errors.append(f"invalid response_type: {response_type}")
    if not prompt:
        errors.append("prompt is required")
    if any(phrase in prompt.lower() for phrase in MISSING_VISUAL_PHRASES):
        errors.append("question depends on an unavailable visual")
    if any(phrase in prompt.lower() for phrase in CONTEXT_DEPENDENT_PHRASES):
        errors.append("question depends on preceding test context")
    if not any(term in combined for term in FOOD_SCIENCE_TERMS):
        errors.append("question does not substantively test Food Science")
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
    elif response_type in {"short_answer", "numeric"} and item.get("answer") in (None, ""):
        errors.append("constructed response requires an answer")
    elif response_type == "multipart":
        parts = item.get("parts")
        if not isinstance(parts, list) or not 3 <= len(parts) <= 5:
            errors.append("multipart FRQ requires 3-5 parts")
        else:
            labels: list[str] = []
            for index, part in enumerate(parts, 1):
                label = str(part.get("label") or part.get("id") or "").strip().lower()
                if not label:
                    errors.append(f"part {index}: label is required")
                elif label in labels:
                    errors.append(f"part {index}: duplicate label {label}")
                dependencies = [str(value).strip().lower() for value in part.get("depends_on") or []]
                unknown = [value for value in dependencies if value not in labels]
                if unknown:
                    errors.append(f"part {index}: dependency must reference an earlier part: {', '.join(unknown)}")
                labels.append(label)
                errors.extend(f"part {index}: {error}" for error in validate_item(part))
            part_total = sum(part.get("points", 0) for part in parts
                             if isinstance(part.get("points"), (int, float)))
            if isinstance(item.get("points"), (int, float)) and item.get("points") != part_total:
                errors.append(f"multipart points must equal part total ({part_total})")
    if response_type in {"short_answer", "numeric"} and re.search(
            r"(?m)^\s*A[.)]\s+.*\n\s*B[.)]\s+", prompt):
        errors.append("constructed-response prompt must not contain A-D choices")
    if response_type != "multipart" and not solution:
        errors.append("worked solution is required")
    return errors


def evaluate_expression(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    allowed = (ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
               ast.Div, ast.Pow, ast.USub, ast.UAdd)
    if any(not isinstance(node, allowed) or isinstance(node, ast.Constant) and
           not isinstance(node.value, (int, float)) for node in ast.walk(tree)):
        raise ValueError("expression must contain numbers and arithmetic operators only")
    result = float(eval(compile(tree, "<food science-verification>", "eval"), {"__builtins__": {}}, {}))
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
