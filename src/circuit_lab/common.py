from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


TOPICS = {
    "history": ("scientist", "credited", "invented", "discovered", "historical", "named after"),
    "electrostatics": ("charge", "electric field", "coulomb", "static", "triboelectric", "capacit"),
    "dc": ("direct current", " dc ", "battery", "polarity"),
    "ac_household": ("alternating current", " ac ", "household", "hot wire", "rms", "frequency"),
    "quantities_ohms_law": ("current", "voltage", "resistance", "power", "energy", "ohm"),
    "magnetism": ("magnet", "solenoid", "transformer", "motor", "generator", "right-hand", "induc"),
    "controls_safety": ("switch", "relay", "fuse", "breaker", "gfci", "hazard", "electrocut"),
    "circuit_analysis": ("circuit", "resistor", "series", "parallel", "node", "loop", "kirchhoff"),
    "led": ("led", "light emitting diode", "band gap"),
}

OUT_OF_SCOPE = (
    "three-phase", "3 phase", "oscilloscope", "frequency analysis", "phasor",
    "inductive reactance", "capacitive reactance", "transistor", "operational amplifier",
)

UNRELATED_DOMAINS = (
    "kinematics", "projectile", "inclined plane", "newton's second law", "momentum collision",
    "initial acceleration", "m/s²", "m/s^2",
    "stoichiometry", "molarity", "chemical equilibrium", "organic chemistry", "periodic table",
    "buzzer", "toss-up question", "bonus question",
)

ELECTRICAL_TERMS = tuple(sorted({term.strip() for topic, terms in TOPICS.items() if topic != "history" for term in terms} | {
    "ampere", "volt", "watt", "joule", "coulomb", "electric", "electromagnetic", "conduct", "ground",
}))

RESPONSE_TYPES = {"multiple_choice", "short_answer", "numeric", "multipart"}
NOVELTY_THRESHOLD = 0.64

REPAIR_ARTIFACTS = (
    "not among the choices", "none of the choices match", "doesn't match the choices",
    "does not match the choices", "let's adjust", "let's update", "update the choices",
    "update answer", "corrected prompt:", "closest correct answer", "there is a mismatch",
)


def text_answers_equivalent(left: object, right: object) -> bool:
    def tokens(value: object) -> set[str]:
        stop = {"a", "an", "the", "is", "will", "be", "when", "if", "either", "or", "both",
                "they", "so", "through", "from", "to", "of", "in"}
        return set(re.findall(r"[a-z0-9]+", str(value).lower())) - stop
    a, b = tokens(left), tokens(right)
    return bool(a and b) and len(a & b) / max(1, min(len(a), len(b))) >= 0.8


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
    lowered = f" {text.lower()} "
    def contains(term: str) -> bool:
        stripped = term.strip().lower()
        if stripped.isalnum():
            return re.search(rf"\b{re.escape(stripped)}\b", lowered) is not None
        return term.lower() in lowered
    matches = [topic for topic, terms in TOPICS.items() if any(contains(term) for term in terms)]
    return matches or ["general_electricity_magnetism"]


def normalize_choices(value: Any) -> dict[str, str]:
    if isinstance(value, list):
        value = {chr(65 + i): body for i, body in enumerate(value)}
    if not isinstance(value, dict):
        return {}
    return {str(k).upper(): " ".join(str(v).split()) for k, v in value.items() if str(k).upper() in "ABCDE"}


def validate_item(item: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    response_type = item.get("response_type")
    if response_type not in RESPONSE_TYPES:
        errors.append(f"invalid response_type: {response_type}")
    if not str(item.get("prompt") or "").strip():
        errors.append("prompt is required")
    if response_type == "multiple_choice":
        choices = normalize_choices(item.get("choices"))
        if len(choices) not in {4, 5} or set(choices) not in ({"A", "B", "C", "D"}, {"A", "B", "C", "D", "E"}):
            errors.append("multiple-choice items require contiguous A-D or A-E choices")
        if str(item.get("answer") or "").upper() not in choices:
            errors.append("MCQ answer must name one available choice")
    elif response_type in {"short_answer", "numeric"}:
        answer = item.get("answer")
        if answer in (None, ""):
            errors.append("constructed-response answer is required")
        if response_type == "numeric" and not isinstance(answer, (int, float, dict)):
            errors.append("numeric answer must be a number or value/unit object")
    elif response_type == "multipart":
        parts = item.get("parts")
        if not isinstance(parts, list) or not parts:
            errors.append("multipart item requires parts")
        else:
            labels: list[str] = []
            for index, part in enumerate(parts, 1):
                label = str(part.get("label") or part.get("id") or "").strip().lower()
                if not label:
                    errors.append(f"part {index}: label is required")
                elif label in labels:
                    errors.append(f"part {index}: duplicate label {label}")
                dependencies = [str(value).strip().lower() for value in (part.get("depends_on") or [])]
                unknown = [value for value in dependencies if value not in labels]
                if unknown:
                    errors.append(f"part {index}: depends_on must reference earlier parts: {', '.join(unknown)}")
                labels.append(label)
                errors.extend(f"part {index}: {error}" for error in validate_item(part))
            part_total = sum(part.get("points", 0) for part in parts
                             if isinstance(part.get("points"), (int, float)))
            if isinstance(item.get("points"), (int, float)) and item.get("points") != part_total:
                errors.append(f"multipart points must equal part total ({part_total})")
    points = item.get("points", 0)
    if not isinstance(points, (int, float)) or points <= 0:
        errors.append("points must be positive")
    difficulty = item.get("difficulty")
    if difficulty is not None and (not isinstance(difficulty, int) or not 1 <= difficulty <= 5):
        errors.append("difficulty must be an integer from 1 to 5")
    combined = json.dumps(item, ensure_ascii=False).lower()
    for phrase in OUT_OF_SCOPE:
        if phrase in combined:
            errors.append(f"out-of-scope Circuit Lab topic: {phrase}")
    for phrase in UNRELATED_DOMAINS:
        if phrase in combined:
            errors.append(f"unrelated domain content: {phrase}")
    prompt = str(item.get("prompt") or "").lower()
    solution = str(item.get("solution") or "").lower()
    for phrase in REPAIR_ARTIFACTS:
        if phrase in solution:
            errors.append(f"solution contains an unresolved draft-repair artifact: {phrase}")
    if not any(term in prompt or term in solution for term in ELECTRICAL_TERMS):
        errors.append("item does not test electricity, magnetism, circuit safety, or required event history")
    if any(token in prompt for token in ("born", "age of", "how many years ago", "year was")):
        errors.append("biographical/date arithmetic is not a Circuit Lab skill")
    return errors


def validate_against_plan(item: dict[str, Any], bundle: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    source_format = bundle.get("response_type")
    expected = source_format if source_format in {"multiple_choice", "multipart"} else plan.get("response_type")
    if expected not in RESPONSE_TYPES:
        errors.append("reasoning plan has an invalid response type")
    elif item.get("response_type") != expected:
        errors.append(f"response format drift: expected {expected}, got {item.get('response_type')}")
    planned_topics = set(plan.get("topics") or bundle.get("topics") or [])
    actual_topics = set(item.get("topics") or []) | set(classify_topics(str(item.get("prompt", "")) + " " + str(item.get("solution", ""))))
    if planned_topics and not planned_topics.intersection(actual_topics):
        errors.append("generated item does not overlap the planned Circuit Lab topics")
    target_skill = str(plan.get("target_skill") or "").lower()
    if target_skill and not any(term in (str(item.get("prompt", "")) + " " + str(item.get("solution", ""))).lower() for term in ELECTRICAL_TERMS):
        errors.append(f"generated item does not realize target skill: {target_skill}")
    return errors


def similarity_to_exemplars(prompt: str, exemplars: list[dict[str, Any]]) -> tuple[float, str | None]:
    def normalize(value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", value.lower()))
    stopwords = {"a", "an", "and", "any", "does", "in", "is", "of", "the", "this", "to", "what", "which"}
    candidate = normalize(prompt)
    best_score, best_id = 0.0, None
    for exemplar in exemplars:
        source = normalize(str(exemplar.get("prompt") or ""))
        if not source:
            continue
        sequence = SequenceMatcher(None, candidate, source).ratio()
        candidate_tokens, source_tokens = set(candidate.split()), set(source.split())
        jaccard = len(candidate_tokens & source_tokens) / max(1, len(candidate_tokens | source_tokens))
        source_content = source_tokens - stopwords
        candidate_content = candidate_tokens - stopwords
        source_coverage = len(candidate_content & source_content) / max(1, len(source_content)) if len(source_content) >= 3 else 0.0
        score = max(sequence, jaccard, source_coverage)
        if score > best_score:
            best_score, best_id = score, str(exemplar.get("id") or "unknown")
    return best_score, best_id


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def graph_text(item: dict[str, Any]) -> str:
    """Stable structural representation used by the structure embedding index."""
    analysis = item.get("analysis") or {}
    graph = analysis.get("reasoning_graph") or item.get("reasoning_graph") or {}
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    parts = [
        "EVENT: CIRCUIT LAB B",
        "FORMAT: " + str(item.get("response_type") or "unknown"),
        "TOPICS: " + ", ".join(item.get("topics") or classify_topics(str(item))),
        "SKILLS: " + ", ".join(analysis.get("skills") or item.get("skills") or []),
    ]
    if nodes:
        parts.append("NODES: " + " | ".join(
            f"{node.get('id','')}:{node.get('type','State')}:{node.get('label','')}"
            for node in nodes if isinstance(node, dict)
        ))
    if edges:
        parts.append("EDGES: " + " | ".join(
            f"{edge.get('src','')}-{edge.get('type','supports')}->{edge.get('dst','')}"
            for edge in edges if isinstance(edge, dict)
        ))
    misconceptions = analysis.get("misconceptions") or []
    if misconceptions:
        parts.append("TRAPS: " + " | ".join(map(str, misconceptions)))
    parts.append("PART_COUNT: " + str(len(item.get("parts") or []) or 1))
    parts.append("POINTS: " + str(item.get("points") or 1))
    return "\n".join(parts)
