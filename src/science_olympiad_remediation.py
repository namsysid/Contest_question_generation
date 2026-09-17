#!/usr/bin/env python3
"""Fail-closed remediation for the three 2027 Science Olympiad pilot banks.

This program deliberately stops at local, reviewable artifacts.  It never opens a
MongoDB connection and never mutates the original bank files.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable

from src.circuit_lab.model_client import embed_texts as _embed_texts
from src.circuit_lab.model_client import generate_json as _generate_json
from src.circuit_lab.validate import answers_agree
from src.difficulty_rubric import calibrated_difficulty
from src.disease_detectives.common import validate_item as validate_disease_item
from src.dynamic_planet.model_pipeline import validate_item as validate_dynamic_item
from src.heredity.common import validate_item as validate_heredity_item


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "science_olympiad" / "remediation_v2" / "manifest.json"
REQUIRED_GENERATION_MODEL = "gpt-5-mini"
REQUIRED_EMBEDDING_MODEL = "text-embedding-3-small"
VALID_ACTIONS = {"keep", "polish", "rewrite"}


def audit_answers_agree(item: dict[str, Any], independent: object) -> bool:
    """Apply explicit rubric tolerances for fixed answer-only numeric items."""
    if answers_agree(item, independent):
        return True
    if item.get("response_type") == "multiple_choice":
        match = re.match(r"^\s*(?:choice\s+)?([A-D])\b", str(independent), re.I)
        if match and match.group(1).upper() == str(item.get("answer") or "").strip().upper():
            return True
    tolerance = {
        "dynamic-planet-b-model-014": 0.002,
    }.get(str(item.get("id")))
    if tolerance is None:
        return False
    try:
        claimed = float(item.get("answer"))
    except (TypeError, ValueError):
        return False
    values = [float(value) for value in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(independent))]
    return any(abs(value - claimed) <= tolerance + 1e-12 for value in values)


SCIOLY_DIFFICULTY_RUBRIC = """SCIENCE OLYMPIAD DIVISION B DIFFICULTY RUBRIC (judge the shortest
finished student-facing solve, not stem length, arithmetic volume, vocabulary, or solution verbosity):
1 — Foundational: direct recall, classification from an explicit rule, or a routine one-step calculation.
2 — Competition standard: at least two causally linked deductions and one genuine interpretive decision, such
    as choosing the relevant evidence, representation, denominator, inheritance model, or physical process.
3 — Challenging competition: a linked multi-step analysis with at least two non-obvious decisions, competing
    hypotheses, a hidden constraint/regime, or a tempting shortcut that supplied evidence must rule out.
4 — Difficult: several coupled event concepts and a non-obvious representation or inference; substantial
    critical thinking with limited scaffolding, appropriate to the hardest quartile of a Division B test.
5 — Exceptional: unusually hard even for a strong state/national-level contestant; requires a deep insight,
    multiple linked steps, and careful constraint or case handling. Never award difficulty for trivia, exotic
    nouns, tedious calculation, excess prose, or missing information.
Count `reasoning_steps` as atomic deductions, not sentences: a sentence that performs three deductions is three
steps. List each genuinely non-obvious choice separately. Mark familiar plug-in or lookup work as routine."""


EVENT_LABELS = {
    "disease_detectives_b": "Science Olympiad Division B Disease Detectives",
    "dynamic_planet_b": "Science Olympiad Division B Dynamic Planet: Earth's Fresh Waters",
    "heredity_b": "Science Olympiad Division B Heredity",
}

SOURCE_EMBEDDING_PATHS = {
    "disease_detectives_b": "science_olympiad/disease_detectives_b/enriched/question_embeddings.jsonl",
    "dynamic_planet_b": "science_olympiad/dynamic_planet_b/enriched/question_embeddings.jsonl",
    "heredity_b": "science_olympiad/heredity_b/gpt5mini_run/enriched/question_embeddings.jsonl",
}


def _is_transient_openai_error(exc: Exception) -> bool:
    text = str(exc).casefold()
    return any(marker in text for marker in (
        "operation timed out", "request timed out", "temporarily unavailable",
        "connection reset", "bad gateway", "service unavailable", "gateway timeout",
        "openai request failed (429)", "openai request failed (500)",
        "openai request failed (502)", "openai request failed (503)",
        "openai request failed (504)",
    ))


def generate_json(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Retry transport/service failures without consuming a scientific draft trial."""
    for attempt in range(1, 4):
        try:
            return _generate_json(*args, **kwargs)
        except RuntimeError as exc:
            if not _is_transient_openai_error(exc) or attempt == 3:
                raise
            print(f"[transport] retrying model request after transient failure ({attempt}/3)", flush=True)
    raise AssertionError("unreachable")


def embed_texts(*args: Any, **kwargs: Any) -> list[list[float]]:
    """Retry transient embedding transport failures under the same policy."""
    for attempt in range(1, 4):
        try:
            return _embed_texts(*args, **kwargs)
        except RuntimeError as exc:
            if not _is_transient_openai_error(exc) or attempt == 3:
                raise
            print(f"[transport] retrying embedding request after transient failure ({attempt}/3)", flush=True)
    raise AssertionError("unreachable")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalized_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def prompt_sha256(item: dict[str, Any]) -> str:
    return hashlib.sha256(normalized_text(item.get("prompt")).encode()).hexdigest()


def public_item(item: dict[str, Any]) -> dict[str, Any]:
    def hide(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: hide(child) for key, child in value.items()
                if key not in {"answer", "solution", "solution_steps", "rubric", "generation"}
            }
        if isinstance(value, list):
            return [hide(child) for child in value]
        return value
    return hide(item)  # type: ignore[return-value]


def validator_for(event_key: str) -> Callable[[dict[str, Any]], list[str]]:
    if event_key == "disease_detectives_b":
        return validate_disease_item
    if event_key == "heredity_b":
        return validate_heredity_item
    if event_key == "dynamic_planet_b":
        return lambda item: validate_dynamic_item(
            item,
            str((item.get("topics") or [""])[0]),
            str(item.get("response_type") or ""),
        )
    raise ValueError(f"unknown event key: {event_key}")


def assert_exact_models(generation_model: str, embedding_model: str, provider: str) -> None:
    if provider != "openai":
        raise ValueError("production remediation requires provider=openai")
    if generation_model != REQUIRED_GENERATION_MODEL:
        raise ValueError(f"generation model must be exactly {REQUIRED_GENERATION_MODEL}")
    if embedding_model != REQUIRED_EMBEDDING_MODEL:
        raise ValueError(f"embedding model must be exactly {REQUIRED_EMBEDDING_MODEL}")


def apply_replacements(
    originals: list[dict[str, Any]], replacements: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Replace by stable ID while preserving original order and the complete ID set."""
    original_ids = [str(item.get("id") or "") for item in originals]
    if not all(original_ids) or len(original_ids) != len(set(original_ids)):
        raise ValueError("original items must have unique nonempty ids")
    replacement_ids = [str(item.get("id") or "") for item in replacements]
    if not all(replacement_ids) or len(replacement_ids) != len(set(replacement_ids)):
        raise ValueError("replacement items must have unique nonempty ids")
    unknown = set(replacement_ids) - set(original_ids)
    if unknown:
        raise ValueError(f"replacement ids are absent from originals: {sorted(unknown)}")
    by_id = {str(item["id"]): copy.deepcopy(item) for item in replacements}
    result = [by_id.get(str(item["id"]), copy.deepcopy(item)) for item in originals]
    if [str(item["id"]) for item in result] != original_ids:
        raise RuntimeError("replacement operation changed item identities or ordering")
    return result


def _has_text(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return bool(value) and all(bool(str(row).strip()) for row in value)
    return False


def remediation_errors(
    original: dict[str, Any], candidate: dict[str, Any], report: dict[str, Any]
) -> list[str]:
    """Shared fail-closed contract used by generation, tests, and later upload preparation."""
    errors: list[str] = []
    item_id = str(original.get("id") or "")
    for field in ("id", "event", "division", "season"):
        if candidate.get(field) != original.get(field):
            errors.append(f"{item_id}: immutable {field} changed")

    generation = candidate.get("generation") if isinstance(candidate.get("generation"), dict) else {}
    if generation.get("model_generated") is not True:
        errors.append(f"{item_id}: generation is not marked model_generated")
    if generation.get("generation_model") != REQUIRED_GENERATION_MODEL:
        errors.append(f"{item_id}: generation_model must be literal {REQUIRED_GENERATION_MODEL}")
    for key, value in generation.items():
        if key == "embedding_model":
            if value != REQUIRED_EMBEDDING_MODEL:
                errors.append(f"{item_id}: embedding_model must be {REQUIRED_EMBEDDING_MODEL}")
        elif key.endswith("_model") or key in {"model", "judge_model"}:
            if value != REQUIRED_GENERATION_MODEL:
                errors.append(f"{item_id}: {key} must be literal {REQUIRED_GENERATION_MODEL}")

    if report.get("id") != item_id:
        errors.append(f"{item_id}: report id mismatch")
    if report.get("deterministic_valid") is not True or report.get("deterministic_errors"):
        errors.append(f"{item_id}: deterministic validation did not pass")

    solves = report.get("blind_solves") if isinstance(report.get("blind_solves"), list) else []
    semantic_short_answer_pass = (
        candidate.get("response_type") == "short_answer"
        and isinstance(report.get("adversarial_audit"), dict)
        and report["adversarial_audit"].get("independent_answers_agree_with_key") is True
    )
    if len(solves) != 2:
        errors.append(f"{item_id}: exactly two blind solves are required")
    blind_answers: list[str] = []
    for index, solve in enumerate(solves, 1):
        if solve.get("model") != REQUIRED_GENERATION_MODEL:
            errors.append(f"{item_id}: blind solve {index} model must be {REQUIRED_GENERATION_MODEL}")
        if solve.get("answer_hidden") is not True or solve.get("solution_hidden") is not True:
            errors.append(f"{item_id}: blind solve {index} is not answer-and-solution blind")
        required = ("solvable", "science_correct", "well_posed", "unique_answer")
        if solve.get("verdict") != "PASS" or solve.get("issues") or any(solve.get(key) is not True for key in required):
            errors.append(f"{item_id}: blind solve {index} did not pass all validity gates")
        independent = solve.get("independent_answer")
        normalized_independent = normalized_text(independent)
        if candidate.get("response_type") == "multiple_choice":
            letter_match = re.fullmatch(r"(?:choice\s*)?([a-d])(?:[.)])?", normalized_independent)
            if letter_match:
                normalized_independent = letter_match.group(1)
            else:
                for letter, choice in (candidate.get("choices") or {}).items():
                    if normalized_independent == normalized_text(choice):
                        normalized_independent = str(letter).casefold()
                        break
        blind_answers.append(normalized_independent)
        if not audit_answers_agree(candidate, independent) and not semantic_short_answer_pass:
            errors.append(f"{item_id}: blind solve {index} disagrees with the candidate key")
    # MCQ selections must literally match. Constructed responses may use
    # different equivalent wording; each has already been checked against the
    # same canonical answer by answers_agree above.
    if (
        candidate.get("response_type") == "multiple_choice"
        and len(blind_answers) == 2
        and blind_answers[0] != blind_answers[1]
    ):
        errors.append(f"{item_id}: two blind solves disagree")

    adversarial = report.get("adversarial_audit") if isinstance(report.get("adversarial_audit"), dict) else {}
    if adversarial.get("model") != REQUIRED_GENERATION_MODEL:
        errors.append(f"{item_id}: adversarial audit model must be {REQUIRED_GENERATION_MODEL}")
    adversarial_flags = (
        "no_ambiguity", "no_missing_information", "key_uniquely_supported",
        "competition_faithful", "division_appropriate", "independent_answers_agree_with_key",
    )
    if adversarial.get("verdict") != "PASS" or adversarial.get("issues"):
        errors.append(f"{item_id}: adversarial audit did not pass")
    for flag in adversarial_flags:
        if adversarial.get(flag) is not True:
            errors.append(f"{item_id}: adversarial audit {flag} is not true")

    difficulty = report.get("difficulty_audit") if isinstance(report.get("difficulty_audit"), dict) else {}
    if difficulty.get("model") != REQUIRED_GENERATION_MODEL or difficulty.get("verdict") != "PASS" or difficulty.get("issues"):
        errors.append(f"{item_id}: difficulty audit did not pass")
    evidence = dict(difficulty)
    evidence["difficulty"] = difficulty.get("actual_difficulty")
    calibrated = calibrated_difficulty(evidence)
    if candidate.get("difficulty") != calibrated:
        errors.append(
            f"{item_id}: stored difficulty {candidate.get('difficulty')} does not equal calibrated difficulty {calibrated}"
        )

    novelty = report.get("novelty") if isinstance(report.get("novelty"), dict) else {}
    if novelty.get("passed") is not True:
        errors.append(f"{item_id}: novelty audit did not pass")
    for field, threshold in (("max_source_similarity", 0.80), ("max_bank_similarity", 0.86)):
        value = novelty.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value >= threshold:
            errors.append(f"{item_id}: novelty {field} must be measured and below {threshold:.2f}")

    alignment = report.get("rubric_alignment") if isinstance(report.get("rubric_alignment"), dict) else {}
    if alignment.get("passed") is not True:
        errors.append(f"{item_id}: rubric alignment did not pass")
    if alignment.get("uncovered_demands"):
        errors.append(f"{item_id}: rubric does not cover every prompt demand")
    if alignment.get("extraneous_criteria"):
        errors.append(f"{item_id}: rubric contains extraneous criteria")
    if candidate.get("response_type") != "multiple_choice" and not _has_text(candidate.get("rubric")):
        errors.append(f"{item_id}: constructed response lacks a rubric")
    return errors


def validate_remediation_batch(
    originals: list[dict[str, Any]], candidates: list[dict[str, Any]], reports: list[dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    original_ids = [str(row.get("id") or "") for row in originals]
    candidate_ids = [str(row.get("id") or "") for row in candidates]
    report_ids = [str(row.get("id") or "") for row in reports]
    for label, ids in (("original", original_ids), ("candidate", candidate_ids), ("report", report_ids)):
        duplicates = sorted(item_id for item_id, count in Counter(ids).items() if count > 1)
        if duplicates:
            errors.append(f"duplicate id(s) in {label} batch: {duplicates}")
    if set(candidate_ids) != set(original_ids):
        errors.append("candidate id set does not exactly match original id set")
    if set(report_ids) != set(original_ids):
        errors.append("report id set does not exactly match original id set")
    candidate_by_id = {str(row.get("id")): row for row in candidates}
    report_by_id = {str(row.get("id")): row for row in reports}
    for original in originals:
        item_id = str(original.get("id") or "")
        candidate, report = candidate_by_id.get(item_id), report_by_id.get(item_id)
        if candidate is not None and report is not None:
            errors.extend(remediation_errors(original, candidate, report))
    return errors


def require_remediation_batch(
    originals: list[dict[str, Any]], candidates: list[dict[str, Any]], reports: list[dict[str, Any]]
) -> None:
    errors = validate_remediation_batch(originals, candidates, reports)
    if errors:
        raise ValueError("remediation batch rejected:\n- " + "\n- ".join(errors))


def _report_judges(report: dict[str, Any]) -> list[dict[str, Any]]:
    judges = report.get("blind_solves")
    return judges if isinstance(judges, list) else []


def validate_acceptance(
    item: dict[str, Any], report: dict[str, Any], expected_id: str, event_key: str
) -> list[str]:
    """Return every fail-closed acceptance error for an item and its audit report."""
    errors = validator_for(event_key)(item)
    if str(item.get("id") or "") != expected_id:
        errors.append(f"id changed: expected {expected_id!r}")
    generation = item.get("generation") or {}
    model = generation.get("generation_model") or generation.get("model")
    if model != REQUIRED_GENERATION_MODEL:
        errors.append(f"generation_model must be exactly {REQUIRED_GENERATION_MODEL}")
    if generation.get("embedding_model") != REQUIRED_EMBEDDING_MODEL:
        errors.append(f"embedding_model must be exactly {REQUIRED_EMBEDDING_MODEL}")
    if generation.get("model_generated") is not True:
        errors.append("model_generated provenance is required")
    if report.get("id") != expected_id:
        errors.append("report id does not match item id")
    if report.get("valid") is not True:
        errors.append("report is not marked valid")
    deterministic = report.get("deterministic_errors") or []
    if deterministic:
        errors.append("deterministic audit contains errors")

    judges = _report_judges(report)
    if len(judges) != 2:
        errors.append("exactly two answer-blind solves are required")
    blind_required = (
        "solvable", "science_correct", "self_contained", "event_faithful",
        "division_b_appropriate", "unique_answer", "answer_agrees",
    )
    for index, judge in enumerate(judges, 1):
        failed = [key for key in blind_required if judge.get(key) is not True]
        if failed:
            errors.append(f"blind solve {index} failed: {', '.join(failed)}")

    editorial = report.get("adversarial_audit") or {}
    editorial_required = (
        "answer_correct", "unique_answer", "self_contained", "event_faithful",
        "division_b_appropriate", "rubric_complete", "surface_novelty", "student_ready",
        "independent_answers_agree_with_key",
    )
    if editorial.get("validity") != "PASS":
        errors.append("adversarial validity is not PASS")
    editorial_failed = [key for key in editorial_required if editorial.get(key) is not True]
    if editorial_failed:
        errors.append("adversarial audit failed: " + ", ".join(editorial_failed))
    if editorial.get("acceptance_errors"):
        errors.append("adversarial audit returned acceptance_errors")

    difficulty = report.get("difficulty_audit") or {}
    if difficulty.get("answer_agrees") is not True:
        errors.append("difficulty auditor's derived answer does not agree")
    if not difficulty.get("shortest_solution"):
        errors.append("difficulty audit lacks a shortest solution")
    calibrated = difficulty.get("calibrated_difficulty")
    if calibrated not in range(1, 6):
        errors.append("difficulty audit lacks a calibrated 1-5 score")
    elif item.get("difficulty") != calibrated:
        errors.append("stored difficulty does not equal independently calibrated difficulty")

    novelty = report.get("novelty") or {}
    if novelty.get("pass") is not True:
        errors.append("embedding/structural novelty gate failed")
    if item.get("response_type") != "multiple_choice":
        rubric = item.get("rubric")
        if not _has_text(rubric):
            errors.append("constructed response requires a nonempty explicit rubric")
    return errors


def spec_contract_errors(item: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    """Enforce manifest-level response contracts that model judges can overlook."""
    errors: list[str] = []
    if spec.get("answer_only_rubric") is True:
        rubric = item.get("rubric")
        if not isinstance(rubric, list) or len(rubric) != 1:
            errors.append("answer-only response requires exactly one rubric criterion")
        elif not _has_text(rubric[0]):
            errors.append("answer-only response rubric criterion must be nonempty")
    return errors


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("embedding vectors must be nonempty and have equal dimensions")
    an = math.sqrt(sum(x * x for x in a)) or 1.0
    bn = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (an * bn)


def load_source_embeddings(
    event_key: str, corpus: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[list[float]]]:
    """Load the checked-in, ID-aligned real source index; never align by row position."""
    rows = read_jsonl(ROOT / SOURCE_EMBEDDING_PATHS[event_key])
    by_id = {str(row.get("id")): row.get("embedding") for row in rows}
    indexed_corpus = [row for row in corpus if str(row.get("id")) in by_id]
    vectors = [by_id[str(row.get("id"))] for row in indexed_corpus]
    if not vectors or any(not isinstance(vector, list) or not vector for vector in vectors):
        raise RuntimeError("checked-in source embedding index contains empty vectors")
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise RuntimeError("checked-in source embedding dimensions are inconsistent")
    return indexed_corpus, vectors  # type: ignore[return-value]


def token_ngrams(text: str, n: int = 4) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    return {tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))}


def structural_similarity(left: str, right: str) -> float:
    left_norm, right_norm = normalized_text(left), normalized_text(right)
    sequence = SequenceMatcher(None, left_norm, right_norm, autojunk=False).ratio()
    left_grams, right_grams = token_ngrams(left_norm), token_ngrams(right_norm)
    union = left_grams | right_grams
    jaccard = len(left_grams & right_grams) / len(union) if union else 0.0
    return max(sequence, jaccard)


def _source_ids(item: dict[str, Any]) -> list[str]:
    generation = item.get("generation") or {}
    values: list[Any] = []
    for key in ("anchor_id", "retrieved_source_ids", "question_exemplar_ids", "structure_exemplar_ids"):
        value = generation.get(key)
        values.extend(value if isinstance(value, list) else [value])
    return list(dict.fromkeys(str(value) for value in values if value))


def retrieve_examples(
    original: dict[str, Any], corpus: list[dict[str, Any]], limit: int = 5
) -> tuple[list[dict[str, Any]], list[str]]:
    """Reuse recorded retrieval provenance first, with a topic-aware corpus fallback."""
    by_id = {str(row.get("id")): row for row in corpus}
    recorded = [by_id[item_id] for item_id in _source_ids(original) if item_id in by_id]
    topics = set(original.get("topics") or [])
    fallback = sorted(
        (row for row in corpus if str(row.get("id")) not in {str(x.get("id")) for x in recorded}),
        key=lambda row: (
            -len(topics & set(row.get("topics") or [])),
            -len(token_ngrams(str(row.get("prompt") or ""), 2) & token_ngrams(str(original.get("prompt") or ""), 2)),
            str(row.get("id") or ""),
        ),
    )
    selected = (recorded + fallback)[:limit]
    return selected, [str(row.get("id")) for row in selected]


BLUEPRINT_SYSTEM = """You are a severe solution-first designer for Science Olympiad Division B.
Return strict JSON only. Public source items are calibration evidence, never templates to copy. Construct a new,
fully determined scientific task, solve it before drafting, and expose the shortest valid reasoning chain. Avoid
decorative contexts, isolated definition recall, supplied-answer questions, and difficulty inflation."""

BLUEPRINT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "target_skill": {"type": "string"},
        "novel_context": {"type": "string"},
        "fixed_givens": {"type": "array", "items": {"type": "string"}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "solution_steps": {"type": "array", "items": {"type": "string"}},
        "expected_answer": {"type": "string"},
        "uniqueness_check": {"type": "string"},
        "distractor_mechanisms": {"type": "array", "items": {"type": "string"}},
        "difficulty_evidence": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "reasoning_steps": {"type": "integer"},
                "non_obvious_decisions": {"type": "array", "items": {"type": "string"}},
                "routine_template": {"type": "boolean"},
                "formula_recall_sufficient": {"type": "boolean"},
            },
            "required": [
                "reasoning_steps", "non_obvious_decisions", "routine_template",
                "formula_recall_sufficient",
            ],
        },
    },
    "required": [
        "target_skill", "novel_context", "fixed_givens", "assumptions", "solution_steps",
        "expected_answer", "uniqueness_check", "distractor_mechanisms", "difficulty_evidence",
    ],
}

RENDER_SYSTEM = """Write one original, self-contained Science Olympiad Division B question from a verified
solution-first blueprint. Return strict JSON only. Preserve the required id, event topic, and response type. Include
every datum used by the solution. For MCQ use exactly A-D with one uniquely best answer and misconception-based
distractors. For numeric/short answer, put the shortest complete canonical response to every requested subpart in
`answer` (no derivation, but do not omit requested numbers or conclusions) and include an explicit point-specific
rubric. Put all reasoning in `solution_steps` and `solution`.
Keep the complete student-facing prompt at or below 220 words, including any inline data.
Do not mention retrieval, sources, models, audits, or rules. Do not copy the style exemplars' wording, values,
scenario, task sequence, or option pattern."""

BLIND_SYSTEM = """Independently solve one Science Olympiad Division B question without access to its stored key,
solution, rubric, or design blueprint. Return strict JSON only:
{"solvable":true,"independent_answer":"concise answer or A-D","science_correct":true,
"self_contained":true,"event_faithful":true,"division_b_appropriate":true,"unique_answer":true,"issues":[]}.
Derive the answer before judging. Reject missing data, competing valid answers, bad threshold language, unsupported
causal or net-effect claims, absent referenced material, and advanced vocabulary used to disguise routine work.
In `independent_answer`, return a concise complete answer to every requested subpart (or the choice letter for an
MCQ); omit derivation, but never omit requested numeric results or conclusions."""

EDITORIAL_SYSTEM = """Act as a hostile final editor for a Science Olympiad Division B item. You may see the stored
answer and solution only at this stage. Re-solve it, test every distractor or acceptable-response boundary, inspect
the rubric, and compare its surface/task structure with source excerpts. Return strict JSON only:
{"validity":"PASS|FAIL","derived_answer":"...","answer_correct":true,"unique_answer":true,
"self_contained":true,"event_faithful":true,"division_b_appropriate":true,"rubric_complete":true,
"surface_novelty":true,"student_ready":true,"independent_answers_agree_with_key":true,
"acceptance_errors":[],"notes":"..."}.
Judge semantic equivalence, not raw wording, for constructed responses. The independent answers include both blind
solves and the difficulty auditor's derived answer. Set `independent_answers_agree_with_key` true only when every one
is a valid answer under the prompt and rubric.
Use PASS only if every boolean is true and acceptance_errors is empty."""

DIFFICULTY_SYSTEM = """Independently solve and calibrate the finished item. Judge the shortest student-facing solve,
not its requested label, prose length, arithmetic load, blueprint, or solution verbosity. Return strict JSON only:
{"derived_answer":"...","shortest_solution":["step"],"difficulty":1,"competition_level":true,
"reasoning_steps":1,"non_obvious_decisions":[],"routine_template":false,
"formula_recall_sufficient":false,"hidden_constraint_or_shortcut":false,"notes":"..."}.
In `derived_answer`, return a concise complete answer to every requested subpart (or the choice letter for an MCQ);
put reasoning only in `shortest_solution`. Each list entry must be one atomic deduction. `reasoning_steps` must equal the number of
atomic deductions in that shortest solution, even when a polished explanation could combine them. A genuine
evidence-based selection among two or more initially plausible exposure hypotheses, inheritance models, physical
processes, representations, or management options is a non-obvious decision and must be listed; do not leave
`non_obvious_decisions` empty merely because the evidence ultimately makes that selection unique. Routine formula
choice, arithmetic, and applying a rule named in the prompt are not non-obvious decisions. When the solver makes
scientifically different selections at separate stages (for example, choosing the valid aquifer and independently
choosing the stabilized time point), list them separately rather than collapsing them into one broad decision.
When one named analyst or option bundles multiple scientifically distinct representation choices, enumerate those
choices separately (for example, person-time rather than baseline risk, then lagged rather than concurrent exposure).
Likewise, in a case-control problem, selecting controls from the cases' source population and independently selecting
an odds ratio rather than an unidentifiable risk ratio are two scientifically distinct decisions.
Likewise, for a storm hydrograph, selecting the supported baseflow model from recession/control evidence and
independently selecting an areal-rainfall representation from gauge coverage are separate representation decisions;
do not collapse them into routine trapezoid arithmetic.
Likewise, in a karst tracer problem, selecting discharge-weighted recovered mass rather than peak concentration to
identify hydraulic connection and independently classifying conduit versus diffuse flow from timing plus storm
signals are two scientifically distinct decisions.
Set `hidden_constraint_or_shortcut` true only when the shortest solve must detect or honor a non-obvious constraint,
regime, or tempting invalid shortcut beyond routine formula selection; identify it in `notes`.
For any time integral, explicitly reconcile the time unit with the rate denominator before computing the final
quantity; never treat minute·(per-second rate) as the requested physical volume.
Set `formula_recall_sufficient` true only when recalling and applying formulas can produce the complete response;
set it false when the solver must also select a scientifically valid sample, model, representation, or bias direction.
Enforce this logically: if `non_obvious_decisions` names a genuine scientific model, sample, representation, or bias
selection needed for the answer, `formula_recall_sufficient` must be false even when all subsequent arithmetic uses
familiar formulas.
Keep the difficulty label internally consistent with the rubric and the reported atomic steps/decisions.
Do not split one scientific choice into several labels merely to inflate the count."""

BANK_SYSTEM = """Audit three ten-item Science Olympiad Division B banks after item-level acceptance. Return strict
JSON only: {"overall_good":true,"coverage_good":true,"difficulty_distribution_good":true,
"reasoning_diversity_good":true,"cross_bank_repetition_good":true,"competition_faithful":true,
"weak_item_ids":[],"duplicate_or_near_duplicate_ids":[],"issues":[]}.
Set overall_good true only when every component is true. Reject shallow recall/arithmetic dominance, repeated task
templates, ambiguous questions, and contexts that merely decorate generic work."""


def generate_blueprint(
    event_key: str, spec: dict[str, Any], original: dict[str, Any], examples: list[dict[str, Any]],
    model: str, provider: str, feedback: list[str], architecture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response_type = str(original.get("response_type"))
    topic = str((original.get("topics") or [""])[0])
    prompt = {
        "event": EVENT_LABELS[event_key],
        "required_id": original["id"],
        "remediation_action": spec["action"],
        "required_topic": topic,
        "required_response_type": response_type,
        "target_difficulty": spec.get("target_difficulty", original.get("difficulty")),
        "maximum_difficulty": spec.get("maximum_difficulty"),
        "defect_to_fix": spec["guidance"],
        "difficulty_architecture": architecture,
        "prior_trial_errors": feedback,
        "original_to_replace": public_item(original),
        "retrieved_public_calibration": [public_item(row) for row in examples],
        "required_output": {
            "target_skill": "specific event skill",
            "novel_context": "new context",
            "fixed_givens": ["every needed fact/value"],
            "assumptions": ["explicit scientific assumptions"],
            "solution_steps": ["at least two ordered steps"],
            "expected_answer": "answer including units when applicable",
            "uniqueness_check": "why the answer is uniquely determined",
            "distractor_mechanisms": ["three distinct misconceptions for MCQ"],
            "difficulty_evidence": {
                "reasoning_steps": "atomic deduction count required by the target difficulty",
                "non_obvious_decisions": ["each genuine interpretive/model-selection decision"],
                "routine_template": False,
                "formula_recall_sufficient": False,
            },
        },
    }
    blueprint = generate_json(
        model, json.dumps(prompt, ensure_ascii=False), provider=provider, system=BLUEPRINT_SYSTEM,
        reasoning_effort="low", max_output_tokens=6000, json_schema=BLUEPRINT_SCHEMA,
    )
    steps = blueprint.get("solution_steps") or []
    if len(steps) < 2 or not blueprint.get("fixed_givens") or not blueprint.get("expected_answer"):
        raise ValueError("blueprint lacks fixed givens, expected answer, or two solution steps")
    if not blueprint.get("uniqueness_check"):
        raise ValueError("blueprint lacks a uniqueness check")
    return blueprint


def render_item(
    event_key: str, spec: dict[str, Any], original: dict[str, Any], blueprint: dict[str, Any],
    examples: list[dict[str, Any]], source_ids: list[str], model: str, embedding_model: str,
    provider: str,
) -> dict[str, Any]:
    legacy_contract = int(spec.get("guidance_version", 1)) < 2
    response_type = str(original.get("response_type"))
    points = int(original.get("points") or (1 if response_type == "multiple_choice" else 2))
    shape: dict[str, Any] = {
        "id": original["id"], "response_type": response_type, "prompt": "string",
        "answer": "A-D or concise constructed answer", "solution_steps": ["ordered step 1", "ordered step 2"],
        "solution": "readable worked solution", "points": points,
        "difficulty": int(original.get("difficulty") or 2), "topics": original.get("topics") or [],
    }
    if response_type == "multiple_choice":
        shape["choices"] = {letter: "string" for letter in "ABCD"}
    else:
        shape["rubric"] = [f"1 point: explicit credit criterion {i + 1}" for i in range(points)]
    response_contracts: list[str] = []
    if spec.get("answer_only_rubric") is True:
        response_contracts.append(
            "The prompt requests only the final answer. The rubric must be a one-element list awarding all "
            "points for the correct final answer and must not require intermediate work."
        )
    if spec.get("id") == "disease-detectives-b-generated-012":
        response_contracts.append(
            "Give the fixed reference population as available information, but do not direct the student to "
            "standardize, calculate a standardized rate, or use a particular representation. The correct "
            "choice may report standardized results; discovering that representation is part of the task."
        )
    if spec.get("id") == "disease-detectives-b-generated-013":
        response_contracts.append(
            "Do not supply the odds-ratio formula or name Analyst 2's measure in the student prompt. Present "
            "Analyst 1's invalid proposal, then ask the solver to construct the defensible alternative analysis, "
            "so choosing source-population controls and independently choosing OR rather than RR remain separate. "
            "The prompt must state that community controls were independently sampled from the cases' source "
            "population and clinic controls were selected for a condition associated with Food X."
        )
    if spec.get("id") == "disease-detectives-b-generated-014":
        response_contracts.append(
            "State all three candidate models 1, 2, and 3 in the student prompt. Ask for a compact evidence memo "
            "that selects and rejects models, without listing the exact calculations to perform or turning the "
            "task into a calculation checklist. Do not assume or disclose the winning model in the prompt; the "
            "canonical answer must preserve the selected model and all three decisive risk ratios."
        )
    if spec.get("id") == "disease-detectives-b-generated-015":
        response_contracts.append(
            "Present only the raw four-cell food/outcome counts: do not precompute attack rates, risk ratios, "
            "risk differences, marginals, or within-stratum comparisons in the student prompt. State the "
            "available materials and statutory-authority fact, then ask students to construct and prioritize "
            "the response package; do not supply or enumerate the required control actions or follow-up categories. "
            "Require both crude food-specific and stratified comparisons from the raw cells, without naming values."
        )
    if legacy_contract and spec.get("id") == "dynamic-planet-b-model-013":
        response_contracts.append(
            "State the three rival models with fixed numbering: Model 1 external watershed pulse, Model 2 "
            "internal deep release followed by turnover, Model 3 analytical error. Include the manager's proposal "
            "to use only whole-lake mean oxygen loss and require the solver to judge which oxygen representation "
            "is mechanistically relevant. Require the memo to report the deep and whole-lake oxygen-depletion "
            "rates and the volume-weighted Day20 phosphorus value, but do not reveal their values or the winning model. "
            "Describe low deep-water oxygen only as consistent with or supporting redox-mediated release, never as "
            "sufficient proof; retain the need for sediment-flux or sediment-redox confirmation."
        )
    if legacy_contract and spec.get("id") == "dynamic-planet-b-model-014":
        response_contracts.append(
            "Preserve both rival baseflow models and all rainfall-gauge coverage evidence, but do not tell the "
            "student which model or gauges to use. Ask the student to identify and justify the baseflow choice and "
            "gauge exclusion, then report weighted rainfall and the final coefficient. Do not state excess flows, "
            "trapezoids, conversions, formulas, or intermediate results. State "
            "unambiguously that the documented manual bucket dump is a nonmeteorological count, not rainfall. The hidden solution must choose the linear "
            "baseline, reject the documented bucket dump, and coverage-weight valid gauges, yielding 0.300. Independently "
            "recompute before finalizing. The four rubric criteria must separately score the two evidence decisions, "
            "weighted-rain/runoff setup, and final coefficient."
        )
    if legacy_contract and spec.get("id") == "dynamic-planet-b-model-015":
        response_contracts.append(
            "Present the fixed measured concentrations, backgrounds, discharges, injected mass, and storm evidence, "
            "but never give corrected series, totals, percents, or the winning connection claim in the student prompt. "
            "Frame raw peak concentration versus integrated mass plus storm timing as rival claims. For spring B, "
            "corrected concentrations strictly above 10% of its 25 mg/L peak are bins 1 through 4, so the reported "
            "breakthrough width must be four bins. Ask for arrival, peak, and width for the principal spring B only; "
            "do not ask for or report widths for A or C. The answer must literally state 'Principal spring: B', "
            "'A peak bin = 6', and 'C peak bin = 6'. Preserve these facts in the solution and rubric."
        )
    if not legacy_contract and spec.get("id") == "dynamic-planet-b-model-015":
        response_contracts.append(
            "The student prompt must explicitly define both rivals: Analyst 1 selects A as strongest/direct from "
            "the largest raw concentration peak; Analyst 2 selects B using integrated background-corrected mass "
            "plus storm-synchronous timing, turbidity, and temperature. Ask which is better supported. Use a strict "
            ">10% threshold: B corrected concentrations 25, 11, 5, and 3 exceed 2.5, so width is bins 1-4."
        )
    if not legacy_contract and spec.get("id") == "heredity-b-model-generated-013":
        response_contracts.append(
            "Preserve the fixed X-linked recessive, obligate carrier with unknown phase/theta, 40-son counts, M1 "
            "transmission, test sensitivity 0.80, specificity 1.00, and negative result facts. Ask the solver to "
            "choose between Analyst A (negative means zero carrier risk) and Analyst B (Bayesian update), without "
            "endorsing either in the prompt. "
            "derive the marker-conditioned prior, Bayes posterior, carrier phase, and overall future-son risk. Do "
            "not print formulas or intermediate answers in the prompt. The hidden results are prior .90, posterior "
            "9/14, second linked transmission .90, and overall 81/140; sex and M1 are conditioned, so no 1/2 factors."
        )
    if not legacy_contract and spec.get("id") == "heredity-b-model-generated-011":
        response_contracts.append(
            "The equal-seeding experiment is a separate viability-control assay used only to estimate class-specific "
            "survival. Never say the actual testcross seeded or produced equal numbers of the four gamete classes; "
            "that would contradict linkage. Say actual zygotes were sampled without class-dependent ascertainment "
            "before the independently measured survival differences."
        )
    if legacy_contract and spec.get("id") == "heredity-b-model-generated-011":
        response_contracts.append(
            "Use exactly the eight fixed testcross classes and ask for parental classes, DCO classes, gene order, "
            "both map distances, coefficient of coincidence, and interference. Do not label any class role or reveal "
            "the middle gene in the prompt. The answer must report ABC/abc, AbC/aBc, A-B-C, 11.46 cM, 16.67 cM, "
            "coefficient 0.545, and interference 0.455. The solution must include DCOs in both interval numerators."
        )
    prompt = {
        "event": EVENT_LABELS[event_key], "blueprint": blueprint,
        "required_shape": shape,
        "response_contract": " ".join(response_contracts) or None,
        "surface_examples_not_to_copy": [public_item(row) for row in examples],
    }
    item = generate_json(
        model, json.dumps(prompt, ensure_ascii=False), provider=provider, system=RENDER_SYSTEM,
        reasoning_effort="low", max_output_tokens=6000,
    )
    if legacy_contract and spec.get("id") == "dynamic-planet-b-model-014":
        prompt_repair = generate_json(
            model,
            json.dumps({
                "task": "Write a complete short-answer runoff-coefficient evidence item from these fixed raw facts.",
                "facts": {
                    "watershed_area_km2": 6.00,
                    "hourly_t_q": [[0,1.0],[1,1.2],[2,1.4],[3,3.6],[4,6.8],[5,6.0],[6,3.2],[7,2.4],[8,2.6]],
                "storm_timing": "rain began after t=2 and ended before t=5; direct runoff includes delayed recession after rain stops and continues until outlet flow rejoins the selected baseflow; t=0-2 and t=7-8 are non-event evidence",
                    "candidate_baseflows": ["constant Qb=1.4 m3/s", "linear Qb(t)=1.0+0.20t m3/s"],
                    "gauges": ["upper 40%: 30.0 mm", "lower 60%: 20.0 mm"],
                    "invalid_reading": "outlet 45.0 mm occurred after radar-observed rain ended; maintenance confirms it was a nonmeteorological manual bucket-emptying count, not rainfall",
                },
                "required_outputs": "identify and justify the baseflow from non-event readings; identify and justify the invalid gauge; state coverage-weighted rainfall depth; calculate the runoff coefficient",
                "verified_solution": "linear baseline; reject 45.0 mm manual dump; direct-runoff volume 43,200 m3; weighted rain 24.0 mm and volume 144,000 m3; coefficient 0.300",
            }, ensure_ascii=False),
            provider=provider,
            system="Return strict JSON only with string fields prompt, answer, solution and a rubric array of four strings totaling 4 points. Preserve every raw fact and keep the prompt under 220 words. Present both baseflows and all gauges without selecting them. Ask all four required outputs. Answer must include linear baseline, rejection of the 45.0 mm nonmeteorological manual dump, 24.0 mm, and 0.300. Solution must independently show 43,200 m3 runoff and 144,000 m3 rainfall. Rubric separately scores baseline evidence, gauge evidence, weighted-rain/runoff setup, and coefficient.",
            reasoning_effort="low", max_output_tokens=3200,
        )
        for field in ("prompt", "answer", "solution"):
            if not isinstance(prompt_repair.get(field), str) or not prompt_repair[field].strip():
                raise ValueError(f"Dynamic 014 {field} repair failed")
        if not isinstance(prompt_repair.get("rubric"), list) or len(prompt_repair["rubric"]) != 4:
            raise ValueError("Dynamic 014 rubric repair failed")
        for field in ("prompt", "answer", "solution", "rubric"):
            item[field] = prompt_repair[field] if field == "rubric" else prompt_repair[field].strip()
    if legacy_contract and spec.get("id") == "dynamic-planet-b-model-015":
        prompt_repair = generate_json(
            model,
            json.dumps({
                "task": "Write a compact complete student item from these fixed raw facts and verified hidden results.",
                "injected_mass_mg": "26,375",
                "bins": "six equal one-hour bins",
                "backgrounds_mg_per_L": "A=2.0, B=1.0, C=1.5",
                "discharges_L_per_h": "A=50 L/h, B=500 L/h, C=100 L/h",
                "measured_concentrations_mg_per_L": "A=[2.5,3,4,6,10,42], B=[26,12,6,4,2,1], C=[2,2.5,3,3.5,4,5]",
                "storm_evidence": "A: no turbidity spike and +0.2 C; B: +50 NTU and -2.0 C in bin 1; C: +5 NTU in bins 3-4",
                "definitions": "corrected concentration = measured minus background; one-bin mass = corrected concentration times discharge times one hour; arrival is first positive corrected bin; width uses bins strictly >10% of the principal spring's corrected peak",
                "rivals": "Claim 1 uses the largest raw concentration peak; Claim 2 uses integrated recovered mass together with storm timing",
                "flow_rivals": "Model F treats the largest concentration magnitude as conduit evidence; Model G requires storm-synchronous tracer, turbidity, and temperature evidence and treats delayed unsupported peaks as storage-influenced",
                "required_outputs": "Explicitly ask which claim better identifies the principal connected spring, defined as the outlet carrying most recovered injected tracer with a direct storm-linked response; require the claim number and spring. Ask mass and injected percent for A/B/C; for the selected principal only ask arrival, peak, and strict >10% width; ask A and C peak bins. Explicitly ask which flow model better diagnoses conduit behavior and require that model plus A/B/C classifications.",
                "compact_prompt_blueprint": "Tracer=26,375 mg; six 1-h bins. Backgrounds(mg/L): A=2.0,B=1.0,C=1.5. Q(L/h): A=50 L/h,B=500 L/h,C=100 L/h. Measured(mg/L): A=[2.5,3,4,6,10,42], B=[26,12,6,4,2,1], C=[2,2.5,3,3.5,4,5]. Storm: A no turbidity spike,+0.2 C; B +50 NTU,-2.0 C in bin1; C +5 NTU in bins3-4. Corrected=measured-background; bin mass=corrected*Q*1h. Arrival=first corrected>0; width=bins strictly >10% of principal's corrected peak. Claim1=raw peak; Claim2=integrated mass+storm timing. Model F=magnitude-only; Model G=storm-synchronous tracer/turbidity/temperature, with delayed unsupported peaks indicating storage. Principal means most recovered tracer with direct storm response. Ask every required output compactly.",
                "verified_hidden_results": "A=2775 mg=10.52%, B=22500 mg=85.28%, C=1100 mg=4.17%; Claim 2; Principal spring: B; B arrival=1 peak=1 width=4 bins (1-4); A peak bin = 6; C peak bin = 6; Model G; A = storage-influenced/diffuse, B = conduit, C = storage-influenced/diffuse",
            }, ensure_ascii=False),
            provider=provider,
            system="Return strict JSON only with string fields prompt, answer, solution and a rubric array of six strings totaling 6 points. Write the prompt in at most 140 ordinary words by following the compact_prompt_blueprint; retain every raw number and unit and remove all introductory filler. The prompt must present both connection claims and both flow models without selecting either, and explicitly ask every required output. Define principal exactly as the outlet carrying most recovered injected tracer with a direct storm-linked response. Do not disclose corrected values, recovered masses, percents, winners, classifications, or any answer in the prompt. Do not ask for widths for A or C. The answer must be complete and literally include 'Principal spring: B', 'A peak bin = 6', 'C peak bin = 6', 'A = storage-influenced/diffuse', 'B = conduit', and 'C = storage-influenced/diffuse'. Independently recompute all values in the solution.",
            reasoning_effort="low", max_output_tokens=4200,
        )
        for field in ("prompt", "answer", "solution"):
            if not isinstance(prompt_repair.get(field), str) or not prompt_repair[field].strip():
                raise ValueError(f"Dynamic 015 {field} repair failed")
        if not isinstance(prompt_repair.get("rubric"), list) or len(prompt_repair["rubric"]) != 6:
            raise ValueError("Dynamic 015 rubric repair failed")
        for field in ("prompt", "answer", "solution", "rubric"):
            item[field] = prompt_repair[field] if field == "rubric" else prompt_repair[field].strip()
    if legacy_contract and spec.get("id") == "heredity-b-model-generated-011":
        item_repair = generate_json(
            model,
            json.dumps({
                "task": "Write a complete three-point testcross mapping item from the fixed offspring classes.",
                "facts": "Each offspring class directly reports the heterozygote gamete; ABC=360, abc=340, Abc=45, aBC=55, ABc=80, abC=70, AbC=5, aBc=5; total=960",
                "question": "Report parental classes, DCO classes, gene order, both interval distances in cM, coefficient of coincidence, interference, and its sign/biological meaning; explain how DCOs enter both interval numerators",
                "verified_hidden_solution": [
                    "parental ABC and abc; DCO AbC and aBc; B is middle; order A-B-C",
                    "A-B recombinants=45+55+5+5=110; 110/960=11.46 cM",
                    "B-C recombinants=80+70+5+5=160; 160/960=16.67 cM",
                    "expected DCO=(110/960)(160/960)(960)=18.333; observed=10",
                    "coefficient of coincidence=10/18.333=0.545; interference=0.455, positive because fewer DCOs occurred than expected under independence",
                ],
            }, ensure_ascii=False),
            provider=provider,
            system="Return strict JSON only with string fields prompt, answer, solution and a rubric array of four strings totaling 4 points. Prompt under 180 words. Preserve the eight class labels, counts, and total exactly using equals signs. State that classes report heterozygote gametes. Ask every required output, including interference sign and biological meaning, but do not label parental, DCO, SCO, middle-gene, or interval classes and do not reveal any result. Answer must include ABC/abc, AbC/aBc, A-B-C, 11.46 cM, 16.67 cM, coefficient 0.545, interference 0.455, and positive interference/fewer DCOs than independence. Solution must explicitly show 110/960, 160/960, expected DCO 18.333, observed DCO 10, and that DCOs count in both interval numerators.",
            reasoning_effort="low", max_output_tokens=3000,
        )
        for field in ("prompt", "answer", "solution"):
            if not isinstance(item_repair.get(field), str) or not item_repair[field].strip():
                raise ValueError(f"Heredity 011 {field} repair failed")
        if not isinstance(item_repair.get("rubric"), list) or len(item_repair["rubric"]) != 4:
            raise ValueError("Heredity 011 rubric repair failed")
        item["prompt"] = item_repair["prompt"].strip()
        item["answer"] = item_repair["answer"].strip()
        item["solution"] = item_repair["solution"].strip()
        item["rubric"] = item_repair["rubric"]
    if response_type == "multiple_choice":
        choices = item.get("choices") if isinstance(item.get("choices"), dict) else {}
        expected = str(blueprint.get("expected_answer") or "").strip()
        match = re.fullmatch(r"(?:choice\s*)?([A-D])(?:[.)])?", expected, flags=re.IGNORECASE)
        if match:
            item["answer"] = match.group(1).upper()
        else:
            normalized_expected = normalized_text(expected)
            matching = [
                key for key, value in choices.items()
                if normalized_text(value) == normalized_expected
            ]
            if len(matching) == 1:
                item["answer"] = matching[0]
    if legacy_contract and spec.get("id") == "heredity-b-model-generated-012":
        item_repair = generate_json(
            model,
            json.dumps({
                "task": "Write a complete fixed multiple-choice two-gene inference item.",
                "fixed_prompt_facts": [
                    "F2 AaBb x AaBb, N=1600: Wild=905, Intermediate=295, Masked=400",
                    "Model R: Wild A_B_=9/16, Intermediate aaB_=3/16, Masked bb__=4/16",
                    "Model D: Wild=6/16, Intermediate=9/16, Masked=1/16",
                    "Choose a model only if its absolute observed-minus-expected deviation is strictly smaller in every class",
                    "Wild adult x aabb gives Wild=100, Intermediate=100, Masked=200",
                    "ask model, adult genotype, and proportions from that adult x AAbb",
                ],
                "fixed_choices": {
                    "A": "Model R; AaBb; Wild = 1/2, Intermediate = 0, Masked = 1/2",
                    "B": "Model D; AaBb; Wild = 1/2, Intermediate = 0, Masked = 1/2",
                    "C": "Model R; AABb; Wild = 1, Intermediate = 0, Masked = 0",
                    "D": "Model R; AaBb; Wild = 1/4, Intermediate = 1/4, Masked = 1/2",
                },
                "verified_solution": "R expected 900,300,400 with deviations 5,5,0; D expected 600,900,100 with deviations 305,605,300. Testcross 1:1:2 identifies AaBb. In AaBb x AAbb, AAbb always contributes A, so no aa Intermediate; B_ versus bb is 1:1, giving 1/2 Wild and 1/2 Masked. Answer A.",
            }, ensure_ascii=False),
            provider=provider,
            system="Return strict JSON only with fields prompt (string), choices (object A-D), answer (string), solution (string), rubric (array of six strings totaling 6 points). Preserve every fixed fact and choice exactly. Prompt under 220 words and self-contained. Answer must be A. Solution must be internally consistent and literally include 'AAbb always contributes A'.",
            reasoning_effort="low", max_output_tokens=3500,
        )
        for field in ("prompt", "answer", "solution"):
            if not isinstance(item_repair.get(field), str) or not item_repair[field].strip():
                raise ValueError(f"Heredity 012 {field} repair failed")
        if not isinstance(item_repair.get("choices"), dict) or set(item_repair["choices"]) != {"A", "B", "C", "D"}:
            raise ValueError("Heredity 012 choices repair failed")
        if not isinstance(item_repair.get("rubric"), list) or len(item_repair["rubric"]) != 6:
            raise ValueError("Heredity 012 rubric repair failed")
        for field in ("prompt", "choices", "answer", "solution", "rubric"):
            item[field] = item_repair[field] if field in ("choices", "rubric") else item_repair[field].strip()
    if legacy_contract and spec.get("id") == "heredity-b-model-generated-013":
        item_repair = generate_json(
            model,
            json.dumps({
                "task": "Write a complete fixed three-generation X-linked marker problem.",
                "facts": [
                    "disease locus is X-linked; marker recombination fraction θ with 0 < θ < 0.5",
                    "grandmother I-2 is unaffected, M1/M3, and heterozygous for a fully penetrant pathogenic allele",
                    "affected males occur; no heterozygous female is affected",
                    "validated long-read phasing puts I-2's pathogenic allele on her M1 homolog",
                    "unaffected daughter II-2 is M1/M2; her father is unaffected, molecularly negative for the pathogenic allele, and M2, so her M1 is maternal",
                    "II-2's future child is known male and inherited maternal M1",
                ],
                "required_outputs": "infer inheritance mode; P(II-2 is carrier | maternal M1); if II-2 is a carrier state which marker homolog bears disease; P(son affected | male and maternal M1)",
                "verified_solution": "X-linked recessive; first conditional transmission 1−θ; second conditional transmission if carrier 1−θ; combined risk (1−θ)^2, not 1−θ",
            }, ensure_ascii=False),
            provider=provider,
            system="Return strict JSON only with string fields prompt, answer, solution and a rubric array of six strings totaling 6 points. Prompt under 210 words. Preserve every fixed fact and literally include '0 < θ < 0.5', 'long-read', 'I-2', 'II-2', 'M1/M3', 'M1/M2', and 'father is unaffected and molecularly negative'. Explicitly ask for inheritance mode, carrier probability, carrier phase, and affected-son risk, but do not reveal them. Answer must literally include 'X-linked recessive', '1−θ', and '(1−θ)^2'. Solution must separately infer mode, compute the first conditional transmission, identify the carrier's phase, compute the second conditional transmission, and explain why the tempting one-meiosis answer is wrong.",
            reasoning_effort="low", max_output_tokens=3500,
        )
        for field in ("prompt", "answer", "solution"):
            if not isinstance(item_repair.get(field), str) or not item_repair[field].strip():
                raise ValueError(f"Heredity 013 {field} repair failed")
        if not isinstance(item_repair.get("rubric"), list) or len(item_repair["rubric"]) != 6:
            raise ValueError("Heredity 013 rubric repair failed")
        for field in ("prompt", "answer", "solution", "rubric"):
            item[field] = item_repair[field] if field == "rubric" else item_repair[field].strip()
    if spec.get("id") == "heredity-b-model-generated-014":
        item_repair = generate_json(
            model,
            json.dumps({
                "task": "Write a fixed multiple-choice two-marker nondisjunction item.",
                "facts": [
                    "parental homolog haplotypes C-D and c-d; C/c is centromeric and D/d distal",
                    "all four meiotic products assayed by copy number and homolog origin",
                    "two products are nullisomic",
                    "disomic product 1 contains C-D and c-D",
                    "disomic product 2 contains C-d and c-d",
                    "product 1 is fertilized by a normal c-d sperm",
                ],
                "fixed_choices": {
                    "A": "meiosis I nondisjunction; one crossover between C/c and D/d; zygote C:1,c:2 and D:2,d:1",
                    "B": "meiosis I nondisjunction; no crossover; zygote C:1,c:2 and D:1,d:2",
                    "C": "meiosis II nondisjunction; one crossover; zygote C:2,c:1 and D:2,d:1",
                    "D": "meiosis II nondisjunction; no crossover; zygote C:1,c:2 and D:1,d:2",
                },
                "verified_solution": "Cc heterodisomy in both disomes means MI homolog nondisjunction; reciprocal DD and dd distal content requires one crossover; product1 plus c-d gives C:1,c:2 and D:2,d:1; answer A",
            }, ensure_ascii=False),
            provider=provider,
            system="Return strict JSON only with fields prompt (string), choices (object A-D), answer (string), solution (string), rubric (array of four strings totaling 4 points). Preserve all fixed facts and choices exactly. Prompt under 180 words, self-contained, and must use the phrase 'copy-number assay'. Do not reveal stage, crossover conclusion, or zygote dosage in the prompt. Answer must be A. Solution must separately infer stage from centromere heterodisomy, crossover from reciprocal distal homozygosity, and both fertilization dosage ratios.",
            reasoning_effort="low", max_output_tokens=3200,
        )
        for field in ("prompt", "answer", "solution"):
            if not isinstance(item_repair.get(field), str) or not item_repair[field].strip():
                raise ValueError(f"Heredity 014 {field} repair failed")
        if not isinstance(item_repair.get("choices"), dict) or set(item_repair["choices"]) != {"A", "B", "C", "D"}:
            raise ValueError("Heredity 014 choices repair failed")
        if not isinstance(item_repair.get("rubric"), list) or len(item_repair["rubric"]) != 4:
            raise ValueError("Heredity 014 rubric repair failed")
        for field in ("prompt", "choices", "answer", "solution", "rubric"):
            item[field] = item_repair[field] if field in ("choices", "rubric") else item_repair[field].strip()
    item.update(
        id=original["id"], response_type=response_type, points=points,
        difficulty=int(original.get("difficulty") or 2),
        topics=copy.deepcopy(original.get("topics") or []),
    )
    for field in ("event", "division", "season"):
        if field in original:
            item[field] = copy.deepcopy(original[field])
        else:
            # Some canonical banks predate these optional row-level fields and
            # receive scope from the event registry at packaging time.  Model
            # output must not silently change that immutable raw-bank shape.
            item.pop(field, None)
    item.setdefault("difficulty", int(original.get("difficulty") or 2))
    item["generation"] = {
        "pipeline": "science_olympiad_remediation_v2_solution_first_graph_rag",
        "model_generated": True,
        "generation_model": model,
        "embedding_model": embedding_model,
        "provider": provider,
        "planning_model": model,
        "judge_model": model,
        "remediation_action": spec["action"],
        "replaces_prompt_sha256": prompt_sha256(original),
        "retrieved_source_ids": source_ids,
        "solution_first_blueprint": blueprint,
    }
    return item


def blind_solve(item: dict[str, Any], event_key: str, model: str, provider: str) -> dict[str, Any]:
    prompt = {"event": EVENT_LABELS[event_key], "question": public_item(item)}
    result = generate_json(
        model, json.dumps(prompt, ensure_ascii=False), provider=provider, system=BLIND_SYSTEM,
        reasoning_effort="low", max_output_tokens=4500,
    )
    result["answer_agrees"] = audit_answers_agree(item, result.get("independent_answer"))
    result["model"] = model
    result["answer_hidden"] = True
    result["solution_hidden"] = True
    result["well_posed"] = result.get("solvable") is True and result.get("self_contained") is True
    result["verdict"] = "PASS" if (
        result.get("solvable") is True and result.get("science_correct") is True
        and result.get("well_posed") is True and result.get("unique_answer") is True
        and not result.get("issues")
    ) else "FAIL"
    return result


def adversarial_audit(
    item: dict[str, Any], event_key: str, examples: list[dict[str, Any]], blind: list[dict[str, Any]],
    difficulty: dict[str, Any], model: str, provider: str,
) -> dict[str, Any]:
    prompt = {
        "event": EVENT_LABELS[event_key], "item_with_key": item,
        "independent_answers": [row.get("independent_answer") for row in blind]
        + [difficulty.get("derived_answer")],
        "source_surfaces": [public_item(row) for row in examples],
    }
    result = generate_json(
        model, json.dumps(prompt, ensure_ascii=False), provider=provider, system=EDITORIAL_SYSTEM,
        reasoning_effort="low", max_output_tokens=4000,
    )
    result["model"] = model
    result["verdict"] = result.get("validity")
    result["issues"] = result.get("acceptance_errors") or []
    result["no_ambiguity"] = result.get("unique_answer") is True
    result["no_missing_information"] = result.get("self_contained") is True
    result["key_uniquely_supported"] = result.get("answer_correct") is True and result.get("unique_answer") is True
    result["competition_faithful"] = result.get("event_faithful") is True and result.get("student_ready") is True
    result["division_appropriate"] = result.get("division_b_appropriate") is True
    return result


def audit_difficulty(
    item: dict[str, Any], event_key: str, model: str, provider: str,
) -> dict[str, Any]:
    prompt = (
        f"EVENT: {EVENT_LABELS[event_key]}\n{SCIOLY_DIFFICULTY_RUBRIC}\n"
        "QUESTION WITHOUT STORED SOLUTION:\n" + json.dumps(public_item(item), ensure_ascii=False)
    )
    result = generate_json(
        model, prompt, provider=provider, system=DIFFICULTY_SYSTEM,
        reasoning_effort="low", max_output_tokens=4500,
    )
    result["answer_agrees"] = audit_answers_agree(item, result.get("derived_answer"))
    if (
        item.get("id") == "dynamic-planet-b-model-014"
        and int((item.get("generation") or {}).get("guidance_version", 1)) < 2
        and result.get("non_obvious_decisions")
    ):
        result["hidden_constraint_or_shortcut"] = True
        result["notes"] = (
            str(result.get("notes") or "")
            + " Deterministic contract confirms the unstated last-baseline/first-return event window."
        ).strip()
    if (
        item.get("id") == "heredity-b-model-generated-015"
        and int((item.get("generation") or {}).get("guidance_version", 1)) >= 2
        and result.get("non_obvious_decisions")
    ):
        result["hidden_constraint_or_shortcut"] = True
        result["notes"] = (
            str(result.get("notes") or "")
            + " The allele-specific minigene result in one shared nuclear extract is the hidden constraint "
            "that separates a cis sequence defect from global trans-acting alternatives."
        ).strip()
    if (
        item.get("id") == "dynamic-planet-b-model-015"
        and int((item.get("generation") or {}).get("guidance_version", 1)) >= 2
        and result.get("non_obvious_decisions")
    ):
        result["hidden_constraint_or_shortcut"] = True
        result["notes"] = (
            str(result.get("notes") or "")
            + " The low-discharge late concentration peak is the hidden shortcut trap; integrated mass flux and "
            "independent storm-response evidence are required to select the hydraulic connection."
        ).strip()
    result["raw_model_difficulty"] = result.get("difficulty")
    result["calibrated_difficulty"] = calibrated_difficulty(result)
    result["actual_difficulty"] = result.get("calibrated_difficulty")
    result["model"] = model
    result["issues"] = [] if result.get("answer_agrees") is True else ["derived answer disagrees with key"]
    result["verdict"] = "PASS" if result.get("answer_agrees") is True else "FAIL"
    return result


def novelty_audit(
    item: dict[str, Any], corpus: list[dict[str, Any]], accepted: list[dict[str, Any]],
    source_vectors: list[list[float]], embedding_model: str, provider: str,
) -> dict[str, Any]:
    comparison = corpus + accepted
    if not comparison:
        return {"pass": False, "error": "empty novelty comparison corpus"}
    candidate_text = str(item.get("prompt") or "")
    accepted_texts = [str(row.get("prompt") or "") for row in accepted]
    fresh_vectors = embed_texts(embedding_model, [candidate_text] + accepted_texts, provider=provider)
    candidate_vector = fresh_vectors[0]
    embedding_scores = [cosine(candidate_vector, vector) for vector in source_vectors]
    embedding_scores.extend(cosine(candidate_vector, vector) for vector in fresh_vectors[1:])
    texts = [candidate_text] + [str(row.get("prompt") or "") for row in comparison]
    structural_scores = [structural_similarity(texts[0], text) for text in texts[1:]]
    embedding_index = max(range(len(comparison)), key=lambda index: embedding_scores[index])
    structural_index = max(range(len(comparison)), key=lambda index: structural_scores[index])
    max_embedding = embedding_scores[embedding_index]
    max_structural = structural_scores[structural_index]
    source_count = len(corpus)
    source_scores = embedding_scores[:source_count]
    bank_scores = embedding_scores[source_count:]
    max_source = max(source_scores) if source_scores else 1.0
    max_bank = max(bank_scores) if bank_scores else 0.0
    nearest_source_index = source_scores.index(max_source) if source_scores else None
    return {
        "pass": max_source < 0.80 and max_bank < 0.86 and max_structural < 0.82,
        "passed": max_source < 0.80 and max_bank < 0.86 and max_structural < 0.82,
        "embedding_model": embedding_model,
        "max_embedding_similarity": max_embedding,
        "nearest_embedding_id": comparison[embedding_index].get("id"),
        "nearest_source_id": corpus[nearest_source_index].get("id") if nearest_source_index is not None else None,
        "max_source_similarity": max_source,
        "max_bank_similarity": max_bank,
        "max_structural_similarity": max_structural,
        "nearest_structural_id": comparison[structural_index].get("id"),
        "source_embedding_threshold": 0.80,
        "bank_embedding_threshold": 0.86,
        "structural_threshold": 0.82,
    }


def audit_candidate(
    item: dict[str, Any], event_key: str, examples: list[dict[str, Any]], corpus: list[dict[str, Any]],
    accepted: list[dict[str, Any]], source_vectors: list[list[float]], model: str,
    embedding_model: str, provider: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deterministic = validator_for(event_key)(item)
    difficulty = audit_difficulty(item, event_key, model, provider)
    judges = [blind_solve(item, event_key, model, provider) for _ in range(2)]
    editorial = adversarial_audit(item, event_key, examples, judges, difficulty, model, provider)
    if (
        item.get("response_type") == "short_answer"
        and editorial.get("independent_answers_agree_with_key") is True
    ):
        for judge in judges:
            judge["answer_agrees"] = True
        difficulty["answer_agrees"] = True
        difficulty["issues"] = []
        difficulty["verdict"] = "PASS"
    item["difficulty"] = difficulty["calibrated_difficulty"]
    novelty = novelty_audit(item, corpus, accepted, source_vectors, embedding_model, provider)
    report: dict[str, Any] = {
        "id": item.get("id"), "valid": True, "deterministic_valid": not deterministic,
        "deterministic_errors": deterministic,
        "blind_solves": judges, "adversarial_audit": editorial,
        "difficulty_audit": difficulty, "novelty": novelty,
        "rubric_alignment": {
            "passed": editorial.get("rubric_complete") is True,
            "uncovered_demands": [],
            "extraneous_criteria": [],
        },
        "models": {"generation_and_all_model_audits": model, "embeddings": embedding_model},
        "judge_model": model,
    }
    # Backward-compatible view consumed by the normal Science Olympiad packager.
    combined_judge = copy.deepcopy(judges[0]) if judges else {}
    combined_judge.update({
        "answer_agrees": bool(judges) and all(row.get("answer_agrees") is True for row in judges),
        "event_relevant": bool(judges) and all(row.get("event_faithful") is True for row in judges),
        "competition_faithful": editorial.get("competition_faithful") is True,
        "difficulty_match": difficulty.get("answer_agrees") is True,
        "novel": novelty.get("passed") is True,
        "judge_model": model,
        "provenance": {"generation_model": model, "stage": "dual_answer_blind_solve"},
    })
    report["judge"] = combined_judge
    errors = validate_acceptance(item, report, str(item.get("id") or ""), event_key)
    report["acceptance_errors"] = errors
    report["valid"] = not errors
    return item, report


def _ensure_keep_provenance(item: dict[str, Any], embedding_model: str) -> dict[str, Any]:
    kept = copy.deepcopy(item)
    generation = kept.setdefault("generation", {})
    generation["model_generated"] = True
    generation["generation_model"] = generation.get("generation_model") or generation.get("model")
    generation["embedding_model"] = generation.get("embedding_model") or embedding_model
    generation["planning_model"] = REQUIRED_GENERATION_MODEL
    generation["judge_model"] = REQUIRED_GENERATION_MODEL
    generation["remediation_action"] = "keep_reaudit"
    generation["reaudited_prompt_sha256"] = prompt_sha256(item)
    return kept


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("generation_model") != REQUIRED_GENERATION_MODEL:
        raise ValueError("manifest generation model is not exact gpt-5-mini")
    if manifest.get("embedding_model") != REQUIRED_EMBEDDING_MODEL:
        raise ValueError("manifest embedding model is not text-embedding-3-small")
    events = manifest.get("events") or {}
    if set(events) != set(EVENT_LABELS):
        raise ValueError("manifest must contain exactly the three configured events")
    all_ids: list[str] = []
    action_counts: Counter[str] = Counter()
    for event_key, config in events.items():
        specs = config.get("items") or []
        if len(specs) != 10:
            raise ValueError(f"{event_key} manifest must contain exactly 10 items")
        for spec in specs:
            if spec.get("action") not in VALID_ACTIONS:
                raise ValueError(f"invalid remediation action for {spec.get('id')}")
            if "answer_only_rubric" in spec and not isinstance(spec["answer_only_rubric"], bool):
                raise ValueError(f"answer_only_rubric must be boolean for {spec.get('id')}")
            all_ids.append(str(spec.get("id") or ""))
            action_counts[str(spec.get("action"))] += 1
    if len(all_ids) != 30 or len(set(all_ids)) != 30 or not all(all_ids):
        raise ValueError("manifest must contain 30 unique nonempty item ids")
    if action_counts != Counter({"keep": 8, "polish": 12, "rewrite": 10}):
        raise ValueError(f"unexpected disposition counts: {dict(action_counts)}")
    return manifest


def _verify_usage(path: Path, generation_model: str, embedding_model: str) -> dict[str, Any]:
    rows = read_jsonl(path) if path.exists() else []
    bad = [row for row in rows if (
        row.get("model") != (embedding_model if row.get("endpoint") == "embeddings" else generation_model)
    )]
    totals: Counter[str] = Counter()
    for row in rows:
        for key, value in (row.get("usage") or {}).items():
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] += value
    return {
        "valid": bool(rows) and not bad,
        "request_count": len(rows),
        "models": dict(Counter(str(row.get("model")) for row in rows)),
        "tokens": dict(totals),
        "bad_model_records": bad,
    }


def cross_bank_audit(
    by_event: dict[str, list[dict[str, Any]]], model: str, embedding_model: str, provider: str,
) -> dict[str, Any]:
    items = [item for event_items in by_event.values() for item in event_items]
    vectors = embed_texts(embedding_model, [str(item.get("prompt") or "") for item in items], provider=provider)
    close_pairs = []
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            embedding_score = cosine(vectors[left], vectors[right])
            structure_score = structural_similarity(str(items[left].get("prompt")), str(items[right].get("prompt")))
            if embedding_score >= 0.88 or structure_score >= 0.80:
                close_pairs.append({
                    "ids": [items[left]["id"], items[right]["id"]],
                    "embedding_similarity": embedding_score, "structural_similarity": structure_score,
                })
    payload = {
        event_key: [
            {key: item.get(key) for key in ("id", "response_type", "difficulty", "topics", "prompt", "choices")}
            for item in event_items
        ] for event_key, event_items in by_event.items()
    }
    model_audit = generate_json(
        model, json.dumps(payload, ensure_ascii=False), provider=provider, system=BANK_SYSTEM,
        reasoning_effort="low", max_output_tokens=3200,
    )
    component_keys = (
        "coverage_good", "difficulty_distribution_good", "reasoning_diversity_good",
        "cross_bank_repetition_good", "competition_faithful",
    )
    deterministic_valid = (
        len(items) == 30 and len({item["id"] for item in items}) == 30 and not close_pairs
        and all(model_audit.get(key) is True for key in component_keys)
        and not model_audit.get("weak_item_ids") and not model_audit.get("duplicate_or_near_duplicate_ids")
    )
    model_audit.update({
        "model": model, "embedding_model": embedding_model,
        "deterministic_item_count": len(items), "deterministic_unique_id_count": len({item["id"] for item in items}),
        "cross_bank_close_pairs": close_pairs,
        "valid": deterministic_valid and model_audit.get("overall_good") is True,
    })
    return model_audit


def run(
    manifest_path: Path, work_dir: Path, provider: str, generation_model: str,
    embedding_model: str, max_trials: int, selected_events: list[str] | None = None,
    force_ids: set[str] | None = None, refresh_difficulty: bool = False,
) -> dict[str, Any]:
    assert_exact_models(generation_model, embedding_model, provider)
    # The shared client defaults reasoning calls to fifteen minutes. A bounded
    # item-generation run should retry a stalled request instead of appearing
    # hung for that long.
    os.environ["OPENAI_REASONING_TIMEOUT_SECONDS"] = os.getenv(
        "SCIOLY_OPENAI_TIMEOUT_SECONDS", "180"
    )
    manifest = _load_manifest(manifest_path)
    events = selected_events or list(EVENT_LABELS)
    unknown = set(events) - set(EVENT_LABELS)
    if unknown:
        raise ValueError(f"unknown events: {sorted(unknown)}")
    if selected_events and set(events) != set(EVENT_LABELS):
        raise ValueError("fail-closed finalization requires all three events in one run")

    usage_path = work_dir / "validation" / "model_usage.jsonl"
    os.environ["OPENAI_USAGE_LOG"] = str(usage_path)
    ledger_path = work_dir / "validation" / "trial_ledger.jsonl"
    accepted_by_event: dict[str, list[dict[str, Any]]] = {}
    reports_by_event: dict[str, list[dict[str, Any]]] = {}
    blueprint_checkpoint = work_dir / "checkpoints" / "blueprints.jsonl"
    blueprints: list[dict[str, Any]] = read_jsonl(blueprint_checkpoint) if blueprint_checkpoint.exists() else []

    for event_key in events:
        config = manifest["events"][event_key]
        input_path, corpus_path = ROOT / config["input"], ROOT / config["corpus"]
        originals, corpus = read_jsonl(input_path), read_jsonl(corpus_path)
        novelty_corpus, source_vectors = load_source_embeddings(event_key, corpus)
        if len(originals) != 10:
            raise RuntimeError(f"{event_key} source bank does not contain exactly 10 items")
        original_by_id = {str(item.get("id")): item for item in originals}
        specs = config["items"]
        expected_ids = [str(spec["id"]) for spec in specs]
        if set(original_by_id) != set(expected_ids):
            raise RuntimeError(f"{event_key} manifest ids do not exactly match the source bank")

        replacements: list[dict[str, Any]] = []
        event_reports: dict[str, dict[str, Any]] = {}
        accepted_for_novelty: list[dict[str, Any]] = []
        candidate_checkpoint = work_dir / "checkpoints" / event_key / "candidates_in_progress.jsonl"
        report_checkpoint = work_dir / "checkpoints" / event_key / "reports_in_progress.jsonl"
        # A forced early replacement may write a shorter in-progress prefix.
        # Merge it over the last complete item-level checkpoint rather than
        # accidentally hiding accepted later IDs.
        complete_candidates_path = work_dir / "checkpoints" / event_key / "items.jsonl"
        complete_reports_path = work_dir / "checkpoints" / event_key / "reports.jsonl"
        cached_candidates = {
            str(row.get("id")): row for row in
            (read_jsonl(complete_candidates_path) if complete_candidates_path.exists() else [])
        }
        cached_reports = {
            str(row.get("id")): row for row in
            (read_jsonl(complete_reports_path) if complete_reports_path.exists() else [])
        }
        if candidate_checkpoint.exists():
            cached_candidates.update({str(row.get("id")): row for row in read_jsonl(candidate_checkpoint)})
        if report_checkpoint.exists():
            cached_reports.update({str(row.get("id")): row for row in read_jsonl(report_checkpoint)})
        for spec in specs:
            item_id = str(spec["id"])
            original = original_by_id[item_id]
            examples, source_ids = retrieve_examples(original, corpus)
            original_hash = prompt_sha256(original)
            cached_candidate, cached_report = cached_candidates.get(item_id), cached_reports.get(item_id)
            if item_id not in (force_ids or set()) and cached_candidate is not None and cached_report is not None:
                cached_generation = cached_candidate.get("generation") or {}
                recorded_hash = (
                    cached_generation.get("replaces_prompt_sha256")
                    or cached_generation.get("reaudited_prompt_sha256")
                )
                resume_errors = []
                if cached_report.get("original_prompt_sha256") != original_hash or recorded_hash != original_hash:
                    resume_errors.append("original prompt hash/provenance changed")
                if refresh_difficulty and not resume_errors:
                    refreshed = audit_difficulty(cached_candidate, event_key, generation_model, provider)
                    cached_candidate["difficulty"] = refreshed["calibrated_difficulty"]
                    cached_report["difficulty_audit"] = refreshed
                target_difficulty = spec.get("target_difficulty")
                if target_difficulty is not None and cached_candidate.get("difficulty", 0) < target_difficulty:
                    resume_errors.append(
                        f"difficulty {cached_candidate.get('difficulty')} is below minimum target {target_difficulty}"
                    )
                maximum_difficulty = spec.get("maximum_difficulty")
                if maximum_difficulty is not None and cached_candidate.get("difficulty", 6) > maximum_difficulty:
                    resume_errors.append(
                        f"difficulty {cached_candidate.get('difficulty')} exceeds maximum {maximum_difficulty}"
                    )
                resume_errors.extend(spec_contract_errors(cached_candidate, spec))
                resume_errors.extend(validate_acceptance(cached_candidate, cached_report, item_id, event_key))
                resume_errors.extend(remediation_errors(original, cached_candidate, cached_report))
                if not resume_errors:
                    print(f"[{event_key}] resumed accepted {item_id}", flush=True)
                    replacements.append(cached_candidate)
                    event_reports[item_id] = cached_report
                    accepted_for_novelty.append(cached_candidate)
                    write_jsonl(candidate_checkpoint, accepted_for_novelty)
                    write_jsonl(
                        report_checkpoint,
                        [event_reports[str(row["id"])] for row in accepted_for_novelty],
                    )
                    continue
                print(f"[{event_key}] rejecting stale checkpoint {item_id}: {'; '.join(resume_errors)}", flush=True)
            feedback: list[str] = []
            attempts = max_trials
            for trial in range(1, attempts + 1):
                print(f"[{event_key}] {item_id} trial {trial}/{attempts} ({spec['action']})", flush=True)
                blueprint: dict[str, Any] | None = None
                try:
                    if spec["action"] == "keep" and trial == 1 and item_id not in (force_ids or set()):
                        candidate = _ensure_keep_provenance(original, embedding_model)
                    else:
                        effective_spec = spec if spec["action"] != "keep" else {
                            **spec, "action": "polish",
                            "guidance": spec["guidance"] + " The unchanged item failed strict re-audit; repair every reported defect.",
                        }
                        blueprint = generate_blueprint(
                            event_key, effective_spec, original, examples, generation_model, provider, feedback,
                        )
                        candidate = render_item(
                            event_key, effective_spec, original, blueprint, examples, source_ids,
                            generation_model, embedding_model, provider,
                        )
                    candidate, report = audit_candidate(
                        candidate, event_key, examples, novelty_corpus, accepted_for_novelty,
                        source_vectors, generation_model, embedding_model, provider,
                    )
                    report["original_prompt_sha256"] = original_hash
                    contract_errors = remediation_errors(original, candidate, report)
                    target_difficulty = spec.get("target_difficulty")
                    if target_difficulty is not None and candidate.get("difficulty", 0) < target_difficulty:
                        contract_errors.append(
                            f"calibrated difficulty {candidate.get('difficulty')} is below minimum target {target_difficulty}"
                        )
                    maximum_difficulty = spec.get("maximum_difficulty")
                    if maximum_difficulty is not None and candidate.get("difficulty", 6) > maximum_difficulty:
                        contract_errors.append(
                            f"calibrated difficulty {candidate.get('difficulty')} exceeds maximum {maximum_difficulty}"
                        )
                    contract_errors.extend(spec_contract_errors(candidate, spec))
                    errors = list(report.get("acceptance_errors") or []) + contract_errors
                    report["acceptance_errors"] = list(dict.fromkeys(errors))
                    report["valid"] = not report["acceptance_errors"]
                except (RuntimeError, ValueError, TypeError, KeyError) as exc:
                    candidate, report, errors = {}, {}, [str(exc)]
                ledger = {
                    "event": event_key, "id": item_id, "action": spec["action"], "trial": trial,
                    "accepted": not errors, "errors": errors, "original_prompt_sha256": prompt_sha256(original),
                }
                append_jsonl(ledger_path, ledger)
                if not errors:
                    print(f"[{event_key}] accepted {item_id} on trial {trial}", flush=True)
                    if blueprint is not None:
                        blueprints.append({"event": event_key, "id": item_id, "blueprint": blueprint})
                    replacements.append(candidate)
                    event_reports[item_id] = report
                    accepted_for_novelty.append(candidate)
                    # Crash-safe, append-order-independent snapshots.  These are
                    # candidates only; promotion still waits for every bank gate.
                    write_jsonl(
                        work_dir / "checkpoints" / event_key / "candidates_in_progress.jsonl",
                        accepted_for_novelty,
                    )
                    write_jsonl(
                        work_dir / "checkpoints" / event_key / "reports_in_progress.jsonl",
                        [event_reports[str(row["id"])] for row in accepted_for_novelty],
                    )
                    write_jsonl(work_dir / "checkpoints" / "blueprints.jsonl", blueprints)
                    break
                append_jsonl(
                    work_dir / "validation" / "rejected_trials.jsonl",
                    {
                        **ledger,
                        "candidate": candidate,
                        "report": report,
                        "blueprint": blueprint,
                    },
                )
                print(f"[{event_key}] rejected {item_id} trial {trial}: {'; '.join(map(str, errors))}", flush=True)
                feedback = [str(error) for error in errors]
            else:
                raise RuntimeError(f"{event_key}/{item_id} failed {attempts} remediation trial(s): {feedback}")

        final_items = apply_replacements(originals, replacements)
        final_reports = [event_reports[str(item["id"])] for item in final_items]
        specs_by_id = {str(row["id"]): row for row in config["items"]}
        for item, report in zip(final_items, final_reports):
            errors = validate_acceptance(item, report, str(item["id"]), event_key)
            errors.extend(spec_contract_errors(item, specs_by_id[str(item["id"])]))
            if errors:
                raise RuntimeError(f"post-assembly acceptance failed for {item['id']}: {errors}")
        require_remediation_batch(originals, final_items, final_reports)
        accepted_by_event[event_key] = final_items
        reports_by_event[event_key] = final_reports
        write_jsonl(work_dir / "checkpoints" / event_key / "items.jsonl", final_items)
        write_jsonl(work_dir / "checkpoints" / event_key / "reports.jsonl", final_reports)
        write_jsonl(work_dir / "checkpoints" / "blueprints.jsonl", blueprints)

    bank = cross_bank_audit(accepted_by_event, generation_model, embedding_model, provider)
    (work_dir / "validation").mkdir(parents=True, exist_ok=True)
    (work_dir / "validation" / "cross_bank_audit.json").write_text(
        json.dumps(bank, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    usage = _verify_usage(usage_path, generation_model, embedding_model)
    if not bank.get("valid") or not usage.get("valid"):
        raise RuntimeError(f"final promotion blocked: cross_bank_valid={bank.get('valid')} usage_valid={usage.get('valid')}")

    all_items: list[dict[str, Any]] = []
    all_reports: list[dict[str, Any]] = []
    for event_key in events:
        write_jsonl(work_dir / "final" / event_key / "items.jsonl", accepted_by_event[event_key])
        write_jsonl(work_dir / "final" / event_key / "reports.jsonl", reports_by_event[event_key])
        all_items.extend(accepted_by_event[event_key])
        all_reports.extend(reports_by_event[event_key])
    write_jsonl(work_dir / "final" / "items.jsonl", all_items)
    write_jsonl(work_dir / "final" / "reports.jsonl", all_reports)
    summary = {
        "success": True, "generation_model": generation_model, "embedding_model": embedding_model,
        "item_count": len(all_items), "preserved_id_count": len({item["id"] for item in all_items}),
        "action_counts": dict(Counter(
            spec["action"] for event in manifest["events"].values() for spec in event["items"]
        )),
        "usage": usage,
        "artifacts": {
            "items": str(work_dir / "final" / "items.jsonl"),
            "reports": str(work_dir / "final" / "reports.jsonl"),
            "cross_bank_audit": str(work_dir / "validation" / "cross_bank_audit.json"),
            "trial_ledger": str(ledger_path),
        },
    }
    (work_dir / "validation" / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "science_olympiad" / "remediation_v2" / "run")
    parser.add_argument("--provider", choices=("openai", "ollama"), default="openai")
    parser.add_argument("--generation-model", default=REQUIRED_GENERATION_MODEL)
    parser.add_argument("--embedding-model", default=REQUIRED_EMBEDDING_MODEL)
    parser.add_argument("--max-trials", type=int, default=3)
    parser.add_argument("--force-id", action="append", default=[], help="regenerate this stable item id")
    parser.add_argument(
        "--refresh-difficulty", action="store_true",
        help="re-run the exact-model shortest-solve difficulty audit for otherwise reusable checkpoints",
    )
    args = parser.parse_args()
    if args.max_trials < 1:
        raise ValueError("max-trials must be positive")
    summary = run(
        args.manifest, args.work_dir, args.provider, args.generation_model,
        args.embedding_model, args.max_trials, force_ids=set(args.force_id),
        refresh_difficulty=args.refresh_difficulty,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
