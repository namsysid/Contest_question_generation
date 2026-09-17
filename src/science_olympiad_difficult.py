#!/usr/bin/env python3
"""Generate an append-only, fail-closed difficult Science Olympiad tranche.

The architecture follows the F=ma/USNCO research pipelines: compete several
mechanisms before drafting, solve and close the construction, freeze it during
rendering, then independently cold-solve and calibrate the finished item.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from src.science_olympiad_remediation import (
    EVENT_LABELS,
    REQUIRED_EMBEDDING_MODEL,
    REQUIRED_GENERATION_MODEL,
    ROOT,
    _verify_usage,
    append_jsonl,
    assert_exact_models,
    audit_candidate,
    cosine,
    embed_texts,
    generate_blueprint,
    generate_json,
    load_source_embeddings,
    public_item,
    read_jsonl,
    remediation_errors,
    render_item,
    spec_contract_errors,
    structural_similarity,
    validate_acceptance,
    write_jsonl,
)


DEFAULT_MANIFEST = ROOT / "science_olympiad" / "difficult_v1" / "manifest.json"
DEFAULT_WORK_DIR = ROOT / "science_olympiad" / "difficult_v1" / "run"
EXISTING_BANKS = {
    event: ROOT / "science_olympiad" / "remediation_v2" / "run" / "final" / event / "items.jsonl"
    for event in EVENT_LABELS
}
STRUCTURE_EMBEDDING_PATHS = {
    "disease_detectives_b": ROOT / "science_olympiad" / "disease_detectives_b" / "enriched" / "structure_embeddings.jsonl",
    "dynamic_planet_b": ROOT / "science_olympiad" / "dynamic_planet_b" / "enriched" / "structure_embeddings.jsonl",
    "heredity_b": ROOT / "science_olympiad" / "heredity_b" / "gpt5mini_run" / "enriched" / "structure_embeddings.jsonl",
}


ARCHITECT_SYSTEM = """Act as the difficulty architect used before question writing. Return strict JSON only.
Design exactly three substantially different mechanisms for the requested Science Olympiad Division B skill.
Difficulty must come from indispensable evidence/model-selection decisions in the shortest solution, never trivia,
word count, arithmetic volume, exotic vocabulary, or missing facts. Select exactly one mechanism. Its solution must
be closed, self-contained, grade 6-9 appropriate, and have one compact student target. Reject familiar one-template
or plug-in paths. Applying a formula or supplied decision threshold, choosing a named formula, converting units,
and doing arithmetic are not non-obvious decisions. Do not put the decisive interpretation rule into the stem.
Each counted decision must choose between at least two initially plausible scientific representations or hypotheses,
and later evidence must rule one out. Public examples calibrate scope only and must not be copied."""

ARCHITECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "mechanisms": {
            "type": "array", "minItems": 3, "maxItems": 3,
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "core_chain": {"type": "array", "items": {"type": "string"}},
                    "indispensable_decisions": {"type": "array", "items": {"type": "string"}},
                    "shortcut_to_rule_out": {"type": "string"},
                },
                "required": ["name", "core_chain", "indispensable_decisions", "shortcut_to_rule_out"],
            },
        },
        "selected_index": {"type": "integer", "minimum": 0, "maximum": 2},
        "selection_reason": {"type": "string"},
        "mandatory_solution_structure": {"type": "array", "items": {"type": "string"}},
        "closure_requirements": {"type": "array", "items": {"type": "string"}},
        "forbidden_shortcuts": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "mechanisms", "selected_index", "selection_reason", "mandatory_solution_structure",
        "closure_requirements", "forbidden_shortcuts",
    ],
}

FINAL_AUDIT_SYSTEM = """Brutally audit a new fifteen-item difficult Science Olympiad Division B tranche.
Return strict JSON only: {"overall_good":true,"difficulty_honest":true,"coverage_good":true,
"reasoning_diversity_good":true,"competition_faithful":true,"weak_item_ids":[],
"duplicate_or_near_duplicate_ids":[],"issues":[]}.
Reject an item if its shortest solution is routine, its supposed decisions are arithmetic/formula selection, its
science is ambiguous, or it is difficult only because it is long. Require a genuine mix of challenging D3 and
hardest-quartile D4 work across all three events. Judge each item against its labeled difficulty: do not mark an
honest challenging D3 weak merely because it is not hardest-quartile D4; mark it weak only if it falls below D3,
is scientifically flawed, incomplete, or unfaithful to competition practice."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spec_sha256(spec: dict[str, Any]) -> str:
    encoded = json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("generation_model") != REQUIRED_GENERATION_MODEL:
        raise ValueError("difficult manifest must use exact gpt-5-mini")
    if data.get("embedding_model") != REQUIRED_EMBEDDING_MODEL:
        raise ValueError("difficult manifest must use text-embedding-3-small")
    if set(data.get("events") or {}) != set(EVENT_LABELS):
        raise ValueError("difficult manifest must configure all three events")
    all_ids: list[str] = []
    targets: list[int] = []
    for event, config in data["events"].items():
        specs = config.get("items") or []
        if len(specs) != data.get("items_per_event") or len(specs) != 5:
            raise ValueError(f"{event} must define exactly five difficult items")
        event_targets = Counter(int(row.get("target_difficulty", 0)) for row in specs)
        if event_targets != Counter({3: 3, 4: 2}):
            raise ValueError(f"{event} must contain three D3 and two D4 targets")
        for spec in specs:
            if spec.get("response_type") not in {"multiple_choice", "numeric", "short_answer"}:
                raise ValueError(f"invalid response type for {spec.get('id')}")
            all_ids.append(str(spec.get("id") or ""))
            targets.append(int(spec["target_difficulty"]))
    existing_ids = {
        str(row.get("id")) for bank in EXISTING_BANKS.values() for row in read_jsonl(bank)
    }
    if len(all_ids) != 15 or len(set(all_ids)) != 15 or not all(all_ids):
        raise ValueError("difficult manifest must have fifteen unique IDs")
    if set(all_ids) & existing_ids:
        raise ValueError("difficult IDs overlap existing bank IDs")
    if Counter(targets) != Counter({3: 9, 4: 6}):
        raise ValueError("difficult tranche must contain nine D3 and six D4 items")
    return data


def prototype(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": spec["id"],
        "response_type": spec["response_type"],
        "prompt": spec["guidance"],
        "answer": "MODEL_GENERATES",
        "solution": "MODEL_GENERATES",
        "points": spec["points"],
        "difficulty": spec["target_difficulty"],
        "topics": [spec["topic"]],
    }


def _vectors_by_id(path: Path) -> dict[str, list[float]]:
    rows = read_jsonl(path)
    result = {str(row.get("id")): row.get("embedding") for row in rows}
    if not result or any(not isinstance(vector, list) or not vector for vector in result.values()):
        raise RuntimeError(f"invalid embedding index: {path}")
    return result  # type: ignore[return-value]


def dual_retrieve(
    event: str, spec: dict[str, Any], corpus: list[dict[str, Any]], provider: str, limit: int = 5,
) -> tuple[list[dict[str, Any]], list[str], dict[str, str]]:
    query = f"{spec['topic']} {spec['guidance']}"
    query_vector = embed_texts(REQUIRED_EMBEDDING_MODEL, [query], provider=provider)[0]
    question_corpus, question_vectors = load_source_embeddings(event, corpus)
    question_by_id = {
        str(row["id"]): vector for row, vector in zip(question_corpus, question_vectors)
    }
    structure_by_id = _vectors_by_id(STRUCTURE_EMBEDDING_PATHS[event])
    by_id = {str(row.get("id")): row for row in corpus}
    common = set(by_id) & set(question_by_id) & set(structure_by_id)
    topic = str(spec["topic"])
    topic_ids = {item_id for item_id in common if topic in (by_id[item_id].get("topics") or [])}
    pool = topic_ids if len(topic_ids) >= limit else common
    ranked = sorted(
        pool,
        key=lambda item_id: (
            0.55 * cosine(query_vector, question_by_id[item_id])
            + 0.45 * cosine(query_vector, structure_by_id[item_id])
        ),
        reverse=True,
    )
    ids = ranked[:limit]
    if len(ids) < limit:
        raise RuntimeError(f"{event}/{topic} has insufficient dual-index retrieval coverage")
    digests = {
        "structure_index_sha256": file_sha256(STRUCTURE_EMBEDDING_PATHS[event]),
    }
    return [by_id[item_id] for item_id in ids], ids, digests


def generate_architecture(
    event: str, spec: dict[str, Any], examples: list[dict[str, Any]], feedback: list[str],
    provider: str,
) -> dict[str, Any]:
    target = int(spec["target_difficulty"])
    payload = {
        "event": EVENT_LABELS[event],
        "topic": spec["topic"],
        "response_type": spec["response_type"],
        "exact_target_difficulty": target,
        "required_atomic_deductions": target,
        "required_non_obvious_decisions": 2,
        "non_decisions": [
            "applying a supplied formula or threshold", "choosing a named formula",
            "unit conversion", "arithmetic", "restating a definition",
        ],
        "task_requirement": spec["guidance"],
        "prior_rejections": feedback,
        "public_scope_examples": [public_item(row) for row in examples],
    }
    architecture = generate_json(
        REQUIRED_GENERATION_MODEL, json.dumps(payload, ensure_ascii=False), provider=provider,
        system=ARCHITECT_SYSTEM, reasoning_effort="low", max_output_tokens=5000,
        json_schema=ARCHITECT_SCHEMA,
    )
    selected = architecture["mechanisms"][architecture["selected_index"]]
    if len(selected.get("core_chain") or []) < target:
        raise ValueError("selected hard mechanism lacks the required deduction chain")
    if len(selected.get("indispensable_decisions") or []) < 2:
        raise ValueError("selected hard mechanism lacks two indispensable decisions")
    if len(architecture.get("mandatory_solution_structure") or []) < target:
        raise ValueError("hard architecture lacks mandatory solution depth")
    return architecture


def hard_contract_errors(
    item: dict[str, Any], report: dict[str, Any], spec: dict[str, Any]
) -> list[str]:
    errors = spec_contract_errors(item, spec)
    legacy_contract = int(spec.get("guidance_version", 1)) < 2
    target = int(spec["target_difficulty"])
    if item.get("difficulty") != target:
        errors.append(f"finished difficulty must be exactly {target}")
    if item.get("response_type") != spec["response_type"]:
        errors.append("response type differs from difficult manifest")
    if item.get("topics") != [spec["topic"]]:
        errors.append("topic differs from difficult manifest")
    if item.get("points") != spec["points"]:
        errors.append("points differ from difficult manifest")
    if spec.get("id") == "disease-detectives-b-generated-012":
        prompt = str(item.get("prompt") or "")
        if __import__("re").search(r"\b(?:use|apply|calculate|compute)\b.{0,25}\bstandardiz", prompt, __import__("re").I):
            errors.append("Disease 012 prompt gives away the required rate representation")
        if "new-only" not in prompt.casefold() or not __import__("re").search(
            r"site c.{0,100}period 1|period 1.{0,100}site c", prompt, __import__("re").I | __import__("re").S
        ):
            errors.append("Disease 012 prompt omits Site C's two-period definition-change evidence")
    if spec.get("id") == "disease-detectives-b-generated-013":
        answer = str(item.get("answer") or "")
        if not all(value in answer.casefold() for value in ("community", "9.0", "1.8", "1.2")):
            errors.append("Disease 013 answer must preserve community OR 9.0, clinic OR 1.8, and invalid ratio 1.2")
        if __import__("re").search(r"\bor\s*=\s*(?:a|\(?\s*\d+\s*[×x*])", str(item.get("prompt") or ""), __import__("re").I):
            errors.append("Disease 013 prompt gives away the odds-ratio formula")
        if __import__("re").search(
            r"analyst\s*2.{0,80}(?:odds\s+ratio|\bOR\b)", str(item.get("prompt") or ""),
            __import__("re").I | __import__("re").S,
        ):
            errors.append("Disease 013 prompt gives away Analyst 2's association measure")
        prompt_folded = str(item.get("prompt") or "").casefold()
        if not (
            "source population" in prompt_folded
            and "clinic" in prompt_folded
            and "associated with" in prompt_folded
            and "food x" in prompt_folded
        ):
            errors.append("Disease 013 prompt omits the sampling mechanisms needed to choose controls")
    if spec.get("id") == "disease-detectives-b-generated-014":
        prompt = str(item.get("prompt") or "")
        if __import__("re").search(r"\b(?:preserve|select|choose)\s+model\s*1\b", prompt, __import__("re").I):
            errors.append("Disease 014 prompt reveals the winning model")
        answer = str(item.get("answer") or "").casefold()
        prompt_folded = prompt.casefold()
        if not (
            all(f"model {number}" in prompt_folded for number in (1, 2, 3))
            and "day5" in prompt_folded.replace(" ", "")
            and "cafeteria" in prompt_folded
            and any(term in prompt_folded for term in ("unrelated", "independent", "two point"))
        ):
            errors.append("Disease 014 prompt must present all three candidate outbreak models")
        if not (
            ("model 1" in answer or "day5 fair" in answer or "day 5 fair" in answer)
            and "9.0" in answer and "1.15" in answer
            and __import__("re").search(r"\b26(?:\.0)?\b", answer)
        ):
            errors.append(
                "Disease 014 answer must select Model 1 and preserve the three decisive risk ratios"
            )
    if spec.get("id") == "disease-detectives-b-generated-015":
        prompt = str(item.get("prompt") or "").casefold()
        answer = str(item.get("answer") or "").casefold()
        solution = str(item.get("solution") or "").casefold()
        if __import__("re").search(r"\b(?:rr|risk ratio|attack rate|marginal)\b\s*(?:=|is|:)", prompt):
            errors.append("Disease 015 prompt precomputes the food comparisons")
        if "must include" in prompt or "must list" in prompt:
            errors.append("Disease 015 prompt supplies a response checklist")
        if not all(term in solution for term in ("laboratory", "handler", "trace")):
            errors.append("Disease 015 solution must include distinct laboratory, handler, and trace-back actions")
        if not (
            all(term in answer for term in ("chicken", "6.0", "1.67", "60%", "20%", "undefined"))
            and __import__("re").search(r"\b3(?:\.0)?\b", answer)
        ):
            errors.append("Disease 015 answer must preserve crude and stratified food evidence")
        if not (
            ("20 percentage" in solution or "20-point" in solution)
            and ("heterogeneous" in solution or "sparse" in solution or "unstable" in solution)
        ):
            errors.append("Disease 015 solution must characterize the sparse second dessert stratum accurately")
    if legacy_contract and spec.get("id") == "dynamic-planet-b-model-013":
        prompt = str(item.get("prompt") or "").casefold()
        answer = str(item.get("answer") or "").casefold()
        if not all(term in prompt for term in ("model 1", "model 2", "model 3", "whole-lake")):
            errors.append("Dynamic 013 prompt must state all models and the whole-lake oxygen shortcut")
        if not all(value in answer for value in ("model 2", "0.35", "0.14", "0.122")):
            errors.append("Dynamic 013 answer must preserve the selected model and three decisive calculations")
        scientific_claims = " ".join(str(item.get(key) or "") for key in ("answer", "solution")).casefold()
        if __import__("re").search(r"(?:sufficient|prove[sd]?).{0,40}redox|redox.{0,40}(?:sufficient|prove[sd]?)", scientific_claims):
            errors.append("Dynamic 013 must treat low water-column oxygen as supporting evidence, not proof of sediment redox release")
    if legacy_contract and spec.get("id") == "dynamic-planet-b-model-011":
        prompt = str(item.get("prompt") or "").casefold()
        answer = str(item.get("answer") or "").casefold()
        if "total trapped" not in prompt or "tagged-grain mass" in prompt:
            errors.append("Dynamic 011 must measure total trapped mass, not mass of unspecified tagged grains")
        if not all(term in prompt for term in ("competence", "transport capacity")):
            errors.append("Dynamic 011 prompt must explicitly request the competence-versus-capacity distinction")
        if __import__("re").search(r"(?:capacity.{0,35}(?:32\s*mm|competence)|competence.{0,35}capacity)", answer):
            errors.append("Dynamic 011 answer conflates competence with mass transport capacity")
    if spec.get("id") == "dynamic-planet-b-model-012":
        prompt = str(item.get("prompt") or "")
        answer = str(item.get("answer") or "").casefold()
        if not all(value in prompt for value in ("101.0", "101.4", "100.6", "99.2")):
            errors.append("Dynamic 012 prompt must preserve the near-site head reversal")
        if not all(term in answer for term in ("upward", "downward", "flowmeter")):
            errors.append("Dynamic 012 answer must infer reversal and select the flowmeter")
    if legacy_contract and spec.get("id") == "dynamic-planet-b-model-014":
        prompt = str(item.get("prompt") or "")
        if __import__("re").search(
            r"(?:use|choose|select)\s+(?:the\s+)?(?:linear|analyst\s+B|two\s+in-basin)|trapezoid|excess discharge",
            prompt, __import__("re").I,
        ):
            errors.append("Dynamic 014 prompt gives away a required representation choice")
        if not any(term in prompt.casefold() for term in ("nonmeteorological", "not rainfall", "not precipitation")):
            errors.append("Dynamic 014 prompt must identify the maintenance count as non-rainfall")
        if not __import__("re").search(r"\b0\.300\b", str(item.get("answer") or "")):
            errors.append("Dynamic 014 answer must equal 0.300")
        answer = str(item.get("answer") or "").casefold()
        if not all(term in answer for term in ("linear", "45.0", "24.0")):
            errors.append("Dynamic 014 answer must preserve both evidence choices and weighted rainfall")
    if legacy_contract and spec.get("id") == "dynamic-planet-b-model-015":
        prompt = str(item.get("prompt") or "")
        compact_prompt = __import__("re").sub(r"\s+", "", prompt).casefold()
        answer = str(item.get("answer") or "").casefold()
        spring_b = __import__("re").search(r"(?:principal|connected)\s+spring.{0,50}?\bb\b", answer)
        four_bins = __import__("re").search(r"(?:bins?\s*1\s*[-–]\s*4|width.{0,100}?=\s*4\s+bins|four bins)", answer)
        if not (spring_b and four_bins):
            errors.append("Dynamic 015 answer must select Spring B and report its four-bin width")
        if not (
            __import__("re").search(r"\ba\b.{0,35}?peak.{0,20}?bin\s*=?\s*6", answer)
            and __import__("re").search(r"\bc\b.{0,35}?peak.{0,20}?bin\s*=?\s*6", answer)
        ):
            errors.append("Dynamic 015 answer must preserve the delayed A and C peak bins")
        if not all(value in answer.replace(",", "") for value in ("2775", "22500", "1100")):
            errors.append("Dynamic 015 answer must report all three recovered masses")
        if not all(value in answer for value in ("a = storage", "b = conduit", "c = storage")):
            errors.append("Dynamic 015 answer must classify all three spring flow regimes")
        required_raw = (
            "26,375", "a=2.0", "b=1.0", "c=1.5",
            "a=[2.5,3,4,6,10,42]", "b=[26,12,6,4,2,1]", "c=[2,2.5,3,3.5,4,5]",
        )
        discharge_ok = (
            all(value in compact_prompt for value in ("a=50l/h", "b=500l/h", "c=100l/h"))
            or "q(l/h):a=50,b=500,c=100" in compact_prompt
        )
        if not all(value in compact_prompt for value in required_raw) or not discharge_ok:
            errors.append("Dynamic 015 prompt must preserve the complete fixed raw dataset")
        if not all(value in compact_prompt for value in ("+50ntu", "-2.0c", "+0.2c", "+5ntu", "bins3-4")):
            errors.append("Dynamic 015 prompt must preserve the quantitative storm evidence")
        if any(value in compact_prompt for value in ("2775mg", "22,500mg", "22500mg", "1100mg", "85.3%")):
            errors.append("Dynamic 015 prompt must not provide recovered totals or percents")
        if ">10%" not in compact_prompt or "principal" not in compact_prompt:
            errors.append("Dynamic 015 prompt must request the principal spring's strict >10% width")
    if legacy_contract and spec.get("id") == "heredity-b-model-generated-011":
        prompt = str(item.get("prompt") or "")
        compact_prompt = __import__("re").sub(r"\s+", "", prompt)
        required_classes = ("ABC=360", "abc=340", "Abc=45", "aBC=55", "ABc=80", "abC=70", "AbC=5", "aBc=5")
        if not all(value in compact_prompt for value in required_classes) or "960" not in prompt:
            errors.append("Heredity 011 prompt must preserve all eight fixed testcross classes")
        if __import__("re").search(r"parental.{0,20}(?:ABC|abc)|DCO.{0,20}(?:AbC|aBc)|middle.{0,15}\bB\b", prompt, __import__("re").I):
            errors.append("Heredity 011 prompt reveals class roles or gene order")
        answer = str(item.get("answer") or "")
        if not all(value.casefold() in answer.casefold() for value in ("ABC", "abc", "AbC", "aBc", "A-B-C", "11.46", "16.67", "0.545", "0.455", "positive", "fewer")):
            errors.append("Heredity 011 answer must preserve the verified mapping results")
        solution = str(item.get("solution") or "")
        if not all(value in solution for value in ("110/960", "160/960", "18.333", "10")):
            errors.append("Heredity 011 solution must count DCOs in both intervals and compute expected DCO")
    if legacy_contract and spec.get("id") == "heredity-b-model-generated-012":
        prompt = str(item.get("prompt") or "")
        if not all(value in prompt for value in ("1600", "905", "295", "400", "100", "200", "9/16", "3/16", "4/16", "6/16", "1/16", "AAbb")):
            errors.append("Heredity 012 prompt must preserve the fixed model, F2, testcross, and prediction data")
        if str(item.get("answer") or "").strip().upper() != "A":
            errors.append("Heredity 012 correct option must be A")
        choices = item.get("choices") or {}
        if not all(term in str(choices.get("A") or "") for term in ("Model R", "AaBb", "1/2", "Intermediate = 0")):
            errors.append("Heredity 012 option A must preserve the unique combined conclusion")
        solution = str(item.get("solution") or "")
        if not all(term in solution for term in ("900", "300", "400", "600", "900", "100", "AaBb", "AAbb always contributes A")):
            errors.append("Heredity 012 solution must complete all three inference stages correctly")
    if legacy_contract and spec.get("id") == "heredity-b-model-generated-013":
        prompt = str(item.get("prompt") or "")
        if not all(term in prompt for term in ("0 < θ < 0.5", "I-2", "II-2", "M1/M3", "M1/M2", "M2", "long-read", "male", "father is unaffected and molecularly negative", "inheritance mode")):
            errors.append("Heredity 013 prompt must preserve the fixed three-generation marker evidence")
        answer = str(item.get("answer") or "").casefold()
        if not all(term in answer for term in ("x-linked recessive", "1−θ", "(1−θ)^2")):
            errors.append("Heredity 013 answer must preserve mode, carrier risk, and two-meiosis risk")
        solution = str(item.get("solution") or "")
        if solution.count("1−θ") < 2 or "two" not in solution.casefold():
            errors.append("Heredity 013 solution must explicitly track both conditional transmissions")
    if spec.get("id") == "heredity-b-model-generated-014":
        prompt = str(item.get("prompt") or "")
        if not all(term in prompt for term in ("C-D", "c-d", "C-D and c-D", "C-d and c-d", "nullisomic", "copy-number")):
            errors.append("Heredity 014 prompt must preserve both-marker product evidence")
        if str(item.get("answer") or "").strip().upper() != "A":
            errors.append("Heredity 014 correct option must be A")
        choice_a = str((item.get("choices") or {}).get("A") or "")
        if not all(term in choice_a for term in ("meiosis I", "one crossover", "C:1,c:2", "D:2,d:1")):
            errors.append("Heredity 014 option A must preserve stage, crossover, and both dosages")
    if (item.get("generation") or {}).get("manifest_spec_sha256") != spec_sha256(spec):
        errors.append("candidate was not generated from the current manifest specification")
    if item.get("response_type") == "multiple_choice":
        for letter, value in (item.get("choices") or {}).items():
            if __import__("re").match(r"^(?:correct|incorrect)\b", str(value).strip(), __import__("re").I):
                errors.append(f"choice {letter} telegraphs its correctness")
    if spec.get("answer_only_rubric") is True:
        prompt = str(item.get("prompt") or "")
        if __import__("re").search(r"\b(classify|identify|explain|justify|show|list|name)\b", prompt, __import__("re").I):
            errors.append("answer-only prompt asks for ungradable intermediate work")
        rubric = item.get("rubric") or []
        criterion = str(rubric[0]) if isinstance(rubric, list) and len(rubric) == 1 else ""
        if not __import__("re").search(rf"\b{int(spec['points'])}\s+points?\b", criterion, __import__("re").I):
            errors.append("answer-only rubric must award all item points in its single criterion")
        if __import__("re").search(r"\b(work|steps?|derive|derivation|model|explain|justify)\b", criterion, __import__("re").I):
            errors.append("answer-only rubric improperly requires intermediate work")
    difficulty = report.get("difficulty_audit") or {}
    if difficulty.get("reasoning_steps", 0) < target:
        errors.append(f"shortest solve has fewer than {target} atomic deductions")
    decisions = difficulty.get("non_obvious_decisions") or []
    required_decisions = 2
    decision_count = len(decisions) if isinstance(decisions, list) else 0
    hidden_d3 = target == 3 and decision_count >= 1 and difficulty.get("hidden_constraint_or_shortcut") is True
    if decision_count < required_decisions and not hidden_d3:
        errors.append(
            f"shortest solve lacks {required_decisions} non-obvious evidence/model decision(s)"
        )
    if difficulty.get("routine_template") is not False:
        errors.append("finished item is routine or routine status is absent")
    if difficulty.get("formula_recall_sufficient") is not False:
        errors.append("formula recall is sufficient or its status is absent")
    architecture = (item.get("generation") or {}).get("difficulty_architecture") or {}
    mechanisms = architecture.get("mechanisms") or []
    if len(mechanisms) != 3:
        errors.append("missing three-way difficulty architecture")
    else:
        selected_index = architecture.get("selected_index")
        if selected_index not in (0, 1, 2):
            errors.append("difficulty architecture lacks one selected mechanism")
        else:
            selected = mechanisms[selected_index]
            if len(selected.get("core_chain") or []) < target:
                errors.append("selected architecture chain is too shallow")
            if len(selected.get("indispensable_decisions") or []) < 2:
                errors.append("selected architecture decisions are too shallow")
    return errors


def final_audit(
    by_event: dict[str, list[dict[str, Any]]], existing: list[dict[str, Any]], provider: str,
) -> dict[str, Any]:
    new = [row for rows in by_event.values() for row in rows]
    combined = existing + new
    vectors = embed_texts(
        REQUIRED_EMBEDDING_MODEL, [str(row.get("prompt") or "") for row in combined], provider=provider
    )
    close_pairs: list[dict[str, Any]] = []
    old_count = len(existing)
    for left in range(len(combined)):
        for right in range(max(left + 1, old_count), len(combined)):
            if left == right:
                continue
            embedding_score = cosine(vectors[left], vectors[right])
            structure_score = structural_similarity(
                str(combined[left].get("prompt") or ""), str(combined[right].get("prompt") or "")
            )
            if embedding_score >= 0.86 or structure_score >= 0.82:
                close_pairs.append({
                    "ids": [combined[left]["id"], combined[right]["id"]],
                    "embedding_similarity": embedding_score,
                    "structural_similarity": structure_score,
                })
    payload = {
        event: [
            {key: row.get(key) for key in (
                "id", "topic", "topics", "response_type", "difficulty", "prompt", "choices"
            )}
            for row in rows
        ] for event, rows in by_event.items()
    }
    audit = generate_json(
        REQUIRED_GENERATION_MODEL, json.dumps(payload, ensure_ascii=False), provider=provider,
        system=FINAL_AUDIT_SYSTEM, reasoning_effort="low", max_output_tokens=3500,
    )
    expected_flags = (
        "overall_good", "difficulty_honest", "coverage_good",
        "reasoning_diversity_good", "competition_faithful",
    )
    audit.update({
        "model": REQUIRED_GENERATION_MODEL,
        "embedding_model": REQUIRED_EMBEDDING_MODEL,
        "new_item_count": len(new),
        "new_unique_id_count": len({row["id"] for row in new}),
        "difficulty_counts": dict(Counter(row.get("difficulty") for row in new)),
        "close_pairs_against_existing_or_new": close_pairs,
    })
    audit["valid"] = (
        len(new) == 15 and len({row["id"] for row in new}) == 15
        and Counter(row.get("difficulty") for row in new) == Counter({3: 9, 4: 6})
        and not close_pairs and all(audit.get(flag) is True for flag in expected_flags)
        and not audit.get("weak_item_ids") and not audit.get("duplicate_or_near_duplicate_ids")
    )
    return audit


def run(manifest_path: Path, work_dir: Path, provider: str, max_trials: int) -> dict[str, Any]:
    assert_exact_models(REQUIRED_GENERATION_MODEL, REQUIRED_EMBEDDING_MODEL, provider)
    os.environ["OPENAI_REASONING_TIMEOUT_SECONDS"] = os.getenv("SCIOLY_OPENAI_TIMEOUT_SECONDS", "180")
    os.environ["OPENAI_USAGE_LOG"] = str(work_dir / "validation" / "model_usage.jsonl")
    manifest = _manifest(manifest_path)
    before_hashes = {event: file_sha256(path) for event, path in EXISTING_BANKS.items()}
    existing = [row for path in EXISTING_BANKS.values() for row in read_jsonl(path)]
    accepted_by_event: dict[str, list[dict[str, Any]]] = {}
    reports_by_event: dict[str, list[dict[str, Any]]] = {}
    trial_ledger = work_dir / "validation" / "trial_ledger.jsonl"
    rejected_path = work_dir / "validation" / "rejected_trials.jsonl"
    rejected_rows = read_jsonl(rejected_path) if rejected_path.exists() else []

    for event, config in manifest["events"].items():
        corpus = read_jsonl(ROOT / config["corpus"])
        novelty_corpus, source_vectors = load_source_embeddings(event, corpus)
        accepted: list[dict[str, Any]] = []
        reports: list[dict[str, Any]] = []
        checkpoint_items = work_dir / "checkpoints" / event / "items.jsonl"
        checkpoint_reports = work_dir / "checkpoints" / event / "reports.jsonl"
        cached_items = {str(row.get("id")): row for row in read_jsonl(checkpoint_items)} if checkpoint_items.exists() else {}
        cached_reports = {str(row.get("id")): row for row in read_jsonl(checkpoint_reports)} if checkpoint_reports.exists() else {}
        for spec in config["items"]:
            item_id = str(spec["id"])
            original = prototype(spec)
            cached_item, cached_report = cached_items.get(item_id), cached_reports.get(item_id)
            if cached_item and cached_report:
                resume_errors = validate_acceptance(cached_item, cached_report, item_id, event)
                resume_errors.extend(remediation_errors(original, cached_item, cached_report))
                resume_errors.extend(hard_contract_errors(cached_item, cached_report, spec))
                if not resume_errors:
                    print(f"[{event}] resumed accepted {item_id}", flush=True)
                    accepted.append(cached_item); reports.append(cached_report)
                    continue
            for rejected in reversed(rejected_rows):
                if rejected.get("event") != event or rejected.get("id") != item_id:
                    continue
                recovered_item = rejected.get("candidate")
                recovered_report = rejected.get("report")
                if not isinstance(recovered_item, dict) or not isinstance(recovered_report, dict):
                    continue
                recovered_report = copy.deepcopy(recovered_report)
                recovered_report["acceptance_errors"] = []
                recovered_report["valid"] = True
                recovery_errors = validate_acceptance(recovered_item, recovered_report, item_id, event)
                recovery_errors.extend(remediation_errors(original, recovered_item, recovered_report))
                recovery_errors.extend(hard_contract_errors(recovered_item, recovered_report, spec))
                if recovery_errors:
                    continue
                accepted.append(recovered_item); reports.append(recovered_report)
                cached_items[item_id] = recovered_item
                cached_reports[item_id] = recovered_report
                ordered_ids = [str(row["id"]) for row in config["items"]]
                write_jsonl(checkpoint_items, [cached_items[row_id] for row_id in ordered_ids if row_id in cached_items])
                write_jsonl(checkpoint_reports, [cached_reports[row_id] for row_id in ordered_ids if row_id in cached_reports])
                print(f"[{event}] recovered accepted {item_id} from prior audited trial", flush=True)
                break
            else:
                recovered_item = None
            if recovered_item is not None:
                continue
            examples, source_ids, index_digests = dual_retrieve(event, spec, corpus, provider)
            feedback: list[str] = []
            for trial in range(1, max_trials + 1):
                print(f"[{event}] {item_id} hard trial {trial}/{max_trials}", flush=True)
                architecture: dict[str, Any] | None = None
                candidate: dict[str, Any] = {}
                report: dict[str, Any] = {}
                try:
                    architecture = generate_architecture(event, spec, examples, feedback, provider)
                    generation_spec = {**spec, "action": "rewrite"}
                    blueprint = generate_blueprint(
                        event, generation_spec, original, examples, REQUIRED_GENERATION_MODEL,
                        provider, feedback, architecture=architecture,
                    )
                    candidate = render_item(
                        event, generation_spec, original, blueprint, examples, source_ids,
                        REQUIRED_GENERATION_MODEL, REQUIRED_EMBEDDING_MODEL, provider,
                    )
                    generation = candidate["generation"]
                    generation.update({
                        "pipeline": "science_olympiad_difficult_v1_solution_first_dual_rag",
                        "guidance_version": int(spec.get("guidance_version", 1)),
                        "difficulty_architecture": architecture,
                        "retrieval_index_digests": index_digests,
                        "manifest_spec_sha256": spec_sha256(spec),
                    })
                    candidate, report = audit_candidate(
                        candidate, event, examples, novelty_corpus, existing + accepted,
                        source_vectors, REQUIRED_GENERATION_MODEL, REQUIRED_EMBEDDING_MODEL, provider,
                    )
                    report["original_prompt_sha256"] = hashlib.sha256(
                        str(original["prompt"]).encode()
                    ).hexdigest()
                    errors = validate_acceptance(candidate, report, item_id, event)
                    errors.extend(remediation_errors(original, candidate, report))
                    errors.extend(hard_contract_errors(candidate, report, spec))
                    errors = list(dict.fromkeys(errors))
                    report["acceptance_errors"] = errors
                    report["valid"] = not errors
                except (RuntimeError, ValueError, TypeError, KeyError) as exc:
                    errors = [str(exc)]
                ledger = {
                    "event": event, "id": item_id, "trial": trial,
                    "accepted": not errors, "errors": errors,
                }
                append_jsonl(trial_ledger, ledger)
                if not errors:
                    accepted.append(candidate); reports.append(report)
                    cached_items[item_id] = candidate
                    cached_reports[item_id] = report
                    ordered_ids = [str(row["id"]) for row in config["items"]]
                    write_jsonl(checkpoint_items, [cached_items[row_id] for row_id in ordered_ids if row_id in cached_items])
                    write_jsonl(checkpoint_reports, [cached_reports[row_id] for row_id in ordered_ids if row_id in cached_reports])
                    print(f"[{event}] accepted {item_id} on hard trial {trial}", flush=True)
                    break
                append_jsonl(work_dir / "validation" / "rejected_trials.jsonl", {
                    **ledger, "architecture": architecture, "candidate": candidate, "report": report,
                })
                feedback = [
                    "Discard the failed mechanism and scenario; do not add cosmetic complexity. " + str(error)
                    for error in errors
                ]
                print(f"[{event}] rejected {item_id}: {'; '.join(errors)}", flush=True)
            else:
                raise RuntimeError(f"{event}/{item_id} exhausted {max_trials} hard trials")
        write_jsonl(checkpoint_items, accepted)
        write_jsonl(checkpoint_reports, reports)
        accepted_by_event[event] = accepted
        reports_by_event[event] = reports

    after_hashes = {event: file_sha256(path) for event, path in EXISTING_BANKS.items()}
    if before_hashes != after_hashes:
        raise RuntimeError("existing thirty-item banks changed during append-only expansion")
    audit = final_audit(accepted_by_event, existing, provider)
    usage = _verify_usage(
        work_dir / "validation" / "model_usage.jsonl",
        REQUIRED_GENERATION_MODEL, REQUIRED_EMBEDDING_MODEL,
    )
    (work_dir / "validation").mkdir(parents=True, exist_ok=True)
    (work_dir / "validation" / "final_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not audit.get("valid") or not usage.get("valid"):
        raise RuntimeError(f"difficult finalization blocked: audit={audit.get('valid')} usage={usage.get('valid')}")
    all_items: list[dict[str, Any]] = []
    all_reports: list[dict[str, Any]] = []
    for event in EVENT_LABELS:
        write_jsonl(work_dir / "final" / event / "items.jsonl", accepted_by_event[event])
        write_jsonl(work_dir / "final" / event / "reports.jsonl", reports_by_event[event])
        all_items.extend(accepted_by_event[event]); all_reports.extend(reports_by_event[event])
    write_jsonl(work_dir / "final" / "items.jsonl", all_items)
    write_jsonl(work_dir / "final" / "reports.jsonl", all_reports)
    summary = {
        "success": True, "item_count": len(all_items),
        "difficulty_counts": dict(Counter(row["difficulty"] for row in all_items)),
        "generation_model": REQUIRED_GENERATION_MODEL,
        "embedding_model": REQUIRED_EMBEDDING_MODEL,
        "existing_bank_hashes_unchanged": before_hashes == after_hashes,
        "usage": usage,
    }
    (work_dir / "validation" / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--provider", choices=("openai",), default="openai")
    parser.add_argument("--max-trials", type=int, default=6)
    args = parser.parse_args()
    if args.max_trials < 1:
        raise ValueError("max-trials must be positive")
    print(json.dumps(run(args.manifest, args.work_dir, args.provider, args.max_trials), indent=2))


if __name__ == "__main__":
    main()
