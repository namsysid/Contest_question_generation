#!/usr/bin/env python3
"""Build a retrieval-grounded, GPT-5-mini Division B club pilot.

The pilot intentionally caps difficulty at D3.  Public tests calibrate event
style; authoritative references and manifest guardrails control the science.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

import fitz
from dotenv import load_dotenv

from src.science_olympiad_remediation import (
    SCIOLY_DIFFICULTY_RUBRIC,
    cosine,
    embed_texts,
    generate_json,
    structural_similarity,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL = "gpt-5-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
PILOT_ROOT = ROOT / "science_olympiad" / "club_pilot_v1"
SPEC_DIR = PILOT_ROOT / "specs"
RUN_DIR = PILOT_ROOT / "run"
EXPECTED_EVENTS = {"anatomy_physiology_b", "meteorology_b", "crime_busters_b"}
SOURCE_FILES: dict[str, list[dict[str, str]]] = {
    "anatomy_physiology_b": [
        {"path": "sources/anatomy_physiology_b/official_2027_sample.pdf", "kind": "questions",
         "url": "https://www.soinc.org/sites/default/files/uploaded_files/5-%2027-A%26P-SAMPLE-YR3.pdf"},
        {"path": "sources/anatomy_physiology_b/openstax_breathing.html", "kind": "authority",
         "url": "https://openstax.org/books/anatomy-and-physiology-2e/pages/22-3-the-process-of-breathing"},
        {"path": "sources/anatomy_physiology_b/openstax_immunity.html", "kind": "authority",
         "url": "https://openstax.org/books/anatomy-and-physiology-2e/pages/21-chapter-review"},
        {"path": "sources/anatomy_physiology_b/openstax_digestive.html", "kind": "authority",
         "url": "https://openstax.org/books/anatomy-and-physiology-2e/pages/23-chapter-review"},
    ],
    "meteorology_b": [
        {"path": "sources/meteorology_b/bullso_2026_test.pdf", "kind": "questions",
         "url": "https://drive.google.com/drive/folders/1O1duR0PqVUilIS5VwtgkbmHq8Oo0fqv4?usp=sharing"},
        {"path": "sources/meteorology_b/bullso_2026_key.pdf", "kind": "questions_key",
         "url": "https://drive.google.com/drive/folders/1O1duR0PqVUilIS5VwtgkbmHq8Oo0fqv4?usp=sharing"},
        {"path": "sources/meteorology_b/noaa_thunderstorm_ingredients.html", "kind": "authority",
         "url": "https://www.nesdis.noaa.gov/about/k-12-education/severe-weather/what-causes-thunderstorm"},
        {"path": "sources/meteorology_b/nws_radar.html", "kind": "authority",
         "url": "https://www.weather.gov/mkx/using-radar"},
        {"path": "sources/meteorology_b/noaa_hurricane_hazards.html", "kind": "authority",
         "url": "https://www.nhc.noaa.gov/prepare/hazards.php"},
    ],
    "crime_busters_b": [
        {"path": "sources/crime_busters_b/bullso_2026_test.pdf", "kind": "questions",
         "url": "https://drive.google.com/drive/folders/1O1duR0PqVUilIS5VwtgkbmHq8Oo0fqv4?usp=sharing"},
        {"path": "sources/crime_busters_b/bullso_2026_key.pdf", "kind": "questions_key",
         "url": "https://drive.google.com/drive/folders/1O1duR0PqVUilIS5VwtgkbmHq8Oo0fqv4?usp=sharing"},
        {"path": "sources/crime_busters_b/official_lab_expectations.pdf", "kind": "authority",
         "url": "https://www.soinc.org/sites/default/files/uploaded_files/Lab_Expections_B26.pdf"},
        {"path": "sources/crime_busters_b/nist_chromatography.html", "kind": "authority",
         "url": "https://www.nist.gov/publications/what-chromatography-all-about"},
        {"path": "sources/crime_busters_b/nist_density.pdf", "kind": "authority",
         "url": "https://www.nist.gov/system/files/documents/2017/05/09/2012Week-1sm.pdf"},
    ],
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event_key(spec: dict[str, Any]) -> str:
    return str(spec.get("event_key") or (spec.get("event") or {}).get("id") or "")


def event_label(spec: dict[str, Any]) -> str:
    return str(spec.get("event_label") or (spec.get("event") or {}).get("label") or event_key(spec))


def event_season(spec: dict[str, Any]) -> int:
    return int(spec.get("season") or (spec.get("event") or {}).get("season") or 2027)


def source_manifest(spec: dict[str, Any]) -> list[dict[str, Any]]:
    raw = spec.get("source_manifest") or []
    return list(raw.get("sources") or []) if isinstance(raw, dict) else list(raw)


def source_policy(spec: dict[str, Any]) -> str:
    raw = spec.get("source_manifest") or []
    return str(raw.get("policy") or "") if isinstance(raw, dict) else ""


class _VisibleHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() in {"script", "style", "svg", "noscript"}:
            self.hidden += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style", "svg", "noscript"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def source_texts(path: Path) -> list[tuple[int, str]]:
    if path.suffix.casefold() == ".pdf":
        with fitz.open(path) as document:
            return [(index + 1, normalize_text(page.get_text("text"))) for index, page in enumerate(document)]
    parser = _VisibleHTML()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    text = normalize_text(" ".join(parser.parts))
    return [(1, text)]


def load_specs() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(SPEC_DIR.glob("*.json")):
        spec = read_json(path)
        key = event_key(spec)
        if key in EXPECTED_EVENTS:
            try:
                spec["_path"] = str(path.relative_to(ROOT))
            except ValueError:
                spec["_path"] = str(path)
            result[key] = spec
    if set(result) != EXPECTED_EVENTS:
        raise RuntimeError(f"expected specs for {sorted(EXPECTED_EVENTS)}, found {sorted(result)}")
    all_ids: list[str] = []
    all_difficulties: list[int] = []
    for key, spec in result.items():
        questions = spec.get("question_specs") or []
        levels = Counter(int(row.get("difficulty", 0)) for row in questions)
        if len(questions) != 5 or levels != Counter({1: 1, 2: 2, 3: 2}):
            raise RuntimeError(f"{key} must define five items with D1/D2/D3 = 1/2/2")
        if not source_manifest(spec):
            raise RuntimeError(f"{key} has no source manifest")
        declared_urls = {str(row.get("url") or "") for row in source_manifest(spec)}
        undeclared = sorted({row["url"] for row in SOURCE_FILES[key]} - declared_urls)
        if undeclared:
            raise RuntimeError(f"{key} local retrieval sources missing from manifest: {undeclared}")
        all_ids.extend(str(row.get("id") or "") for row in questions)
        all_difficulties.extend(int(row.get("difficulty", 0)) for row in questions)
    if len(set(all_ids)) != 15 or not all(all_ids):
        raise RuntimeError("pilot requires fifteen unique nonempty item IDs")
    if max(all_difficulties) > 3:
        raise RuntimeError("club pilot difficulty ceiling is D3")
    return result


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_corpus(specs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(specs):
        for source in SOURCE_FILES[key]:
            path = PILOT_ROOT / source["path"]
            if not path.exists() or path.stat().st_size < 1000:
                raise RuntimeError(f"missing source PDF: {path}")
            for page_number, text in source_texts(path):
                if len(text) < 120:
                    continue
                for chunk_index, start in enumerate(range(0, len(text), 3200)):
                    chunk = text[start:start + 3800].strip()
                    if len(chunk) < 120:
                        continue
                    rows.append({
                        "id": f"{key}:{path.stem}:p{page_number}:c{chunk_index + 1}",
                        "event_key": key,
                        "source_file": str(path.relative_to(ROOT)),
                        "source_url": source["url"],
                        "source_kind": source["kind"],
                        "source_sha256": sha256(path),
                        "page": page_number,
                        "text": chunk,
                        "use": "authoritative grounding" if source["kind"] == "authority" else "question-style calibration; do not copy",
                    })
    write_jsonl(RUN_DIR / "corpus" / "items.jsonl", rows)
    return rows


def ensure_embeddings(corpus: list[dict[str, Any]], provider: str) -> list[dict[str, Any]]:
    path = RUN_DIR / "enriched" / "source_embeddings.jsonl"
    existing = read_jsonl(path)
    expected_ids = [row["id"] for row in corpus]
    if [row.get("id") for row in existing] == expected_ids and all(row.get("embedding") for row in existing):
        return existing
    vectors = embed_texts(EMBEDDING_MODEL, [row["text"] for row in corpus], provider=provider)
    rows = [
        {"id": source["id"], "event_key": source["event_key"], "embedding": vector}
        for source, vector in zip(corpus, vectors)
    ]
    write_jsonl(path, rows)
    return rows


def retrieve(
    key: str,
    question_spec: dict[str, Any],
    corpus: list[dict[str, Any]],
    embeddings: list[dict[str, Any]],
    provider: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    query = " ".join([
        str(question_spec.get("topic") or ""),
        str(question_spec.get("target_skill") or ""),
        json.dumps(question_spec.get("generation_guidance") or {}, ensure_ascii=False),
    ])
    vector = embed_texts(EMBEDDING_MODEL, [query], provider=provider)[0]
    by_id = {row["id"]: row for row in corpus if row["event_key"] == key}
    ranked = sorted(
        (
            (cosine(vector, row["embedding"]), by_id[row["id"]])
            for row in embeddings if row.get("event_key") == key and row.get("id") in by_id
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )
    selected = []
    for score, row in ranked[:limit]:
        selected.append({
            "id": row["id"], "page": row["page"], "source_file": row["source_file"],
            "similarity": round(score, 6), "excerpt": row["text"][:1800],
        })
    if len(selected) < 2:
        raise RuntimeError(f"insufficient source retrieval for {key}/{question_spec['id']}")
    return selected


ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string"},
        "response_type": {"type": "string", "enum": ["multiple_choice", "short_answer", "numeric"]},
        "prompt": {"type": "string"},
        "choices": {"anyOf": [
            {"type": "object", "additionalProperties": False,
             "properties": {letter: {"type": "string"} for letter in "ABCD"},
             "required": list("ABCD")},
            {"type": "null"},
        ]},
        "answer": {"type": "string"},
        "solution_steps": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "solution": {"type": "string"},
        "rubric": {"type": "array", "items": {"type": "string"}},
        "points": {"type": "integer", "minimum": 1, "maximum": 10},
        "difficulty": {"type": "integer", "minimum": 1, "maximum": 3},
        "topics": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2},
    },
    "required": [
        "id", "response_type", "prompt", "choices", "answer", "solution_steps", "solution",
        "rubric", "points", "difficulty", "topics",
    ],
}


GENERATION_SYSTEM = """You write original Science Olympiad Division B questions for a real middle-school club.
Return strict JSON matching the schema. The manifest is binding. Source excerpts calibrate authentic scope and
format only: never copy their wording, values, named scenarios, or diagrams. Solve the item before finalizing it.
It must be self-contained, scientifically correct, concise, grade 6-9 appropriate, and uniquely gradable.
Difficulty is capped at D3. D1 is direct recognition or one routine inference. D2 has two linked deductions and
one interpretation. D3 has a short multi-step chain and at most two non-obvious evidence decisions. Never create
D4 complexity by coupling too many concepts, excessive reading, exotic vocabulary, or missing facts. Use an
original plain-text table instead of an external image. For MCQ, choices must be A-D and the answer only a letter.
For constructed response, supply an explicit point-by-point rubric whose awarded points sum exactly to the item's
points field. Avoid causal or diagnostic overclaiming."""


BLIND_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "independent_answer": {"type": "string"},
        "reasoning": {"type": "string"},
        "solvable": {"type": "boolean"},
        "science_correct": {"type": "boolean"},
        "self_contained": {"type": "boolean"},
        "event_faithful": {"type": "boolean"},
        "division_b_appropriate": {"type": "boolean"},
        "unique_answer": {"type": "boolean"},
        "calibrated_difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "independent_answer", "reasoning", "solvable", "science_correct", "self_contained",
        "event_faithful", "division_b_appropriate", "unique_answer", "calibrated_difficulty", "issues",
    ],
}


EDITORIAL_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "answer_agrees": {"type": "boolean"},
        "answer_correct": {"type": "boolean"},
        "rubric_complete": {"type": "boolean"},
        "student_ready": {"type": "boolean"},
        "competition_faithful": {"type": "boolean"},
        "difficulty_match": {"type": "boolean"},
        "no_overclaiming": {"type": "boolean"},
        "source_independent": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "answer_agrees", "answer_correct", "rubric_complete", "student_ready",
        "competition_faithful", "difficulty_match", "no_overclaiming", "source_independent", "issues",
    ],
}


def normalized_response_type(value: str) -> str:
    if value in {"numeric_with_interpretation", "numeric_response", "multipart_short_answer"}:
        return "short_answer"
    return value


def public_item(item: dict[str, Any]) -> dict[str, Any]:
    return {key: item.get(key) for key in ("id", "response_type", "prompt", "choices", "points", "topics")}


def deterministic_errors(item: dict[str, Any], question_spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_type = normalized_response_type(str(question_spec.get("response_type") or ""))
    if item.get("id") != question_spec.get("id"):
        errors.append("ID differs from manifest")
    if item.get("response_type") != expected_type:
        errors.append(f"response_type must be {expected_type}")
    if isinstance(item.get("difficulty"), bool) or not isinstance(item.get("difficulty"), int) or not 1 <= item["difficulty"] <= 3:
        errors.append("difficulty exceeds D3 ceiling")
    if isinstance(item.get("points"), bool) or item.get("points") != int(question_spec["points"]):
        errors.append("points differ from manifest")
    for field in ("prompt", "solution"):
        if len(str(item.get(field) or "").strip()) < 2:
            errors.append(f"missing {field}")
    if not str(item.get("answer") or "").strip():
        errors.append("missing answer")
    if item.get("response_type") == "multiple_choice":
        if set((item.get("choices") or {}).keys()) != set("ABCD"):
            errors.append("MCQ must have exactly A-D choices")
        if str(item.get("answer") or "").strip().upper() not in set("ABCD"):
            errors.append("MCQ answer must be one letter A-D")
    elif not item.get("rubric"):
        errors.append("constructed response requires explicit rubric")
    if len(str(item.get("prompt") or "").split()) > 330:
        errors.append("prompt is too long for club pilot")
    return errors


def model_generate(
    key: str,
    spec: dict[str, Any],
    question_spec: dict[str, Any],
    retrieved: list[dict[str, Any]],
    feedback: list[str],
    provider: str,
) -> dict[str, Any]:
    expected_type = normalized_response_type(str(question_spec["response_type"]))
    payload = {
        "event": event_label(spec),
        "season": event_season(spec),
        "audience": "Division B middle-school club, grades 6-9",
        "required_contract": {
            "id": question_spec["id"],
            "response_type": expected_type,
            "points": question_spec["points"],
            "difficulty": question_spec["difficulty"],
            "topic": question_spec["topic"],
        },
        "target_skill": question_spec.get("target_skill"),
        "generation_guidance": question_spec.get("generation_guidance"),
        "authoritative_urls": question_spec.get("authoritative_urls") or [],
        "source_manifest_policy": source_policy(spec),
        "declared_sources": source_manifest(spec),
        "prior_trial_feedback": feedback,
        "retrieved_public_source_excerpts": retrieved,
    }
    item = generate_json(
        MODEL, json.dumps(payload, ensure_ascii=False), provider=provider,
        system=GENERATION_SYSTEM, reasoning_effort="low", max_output_tokens=5500,
        json_schema=ITEM_SCHEMA,
    )
    # Stable storage identity is pipeline metadata, not generated scientific content.
    item["id"] = question_spec["id"]
    item["season"] = event_season(spec)
    item["level"] = "Division B"
    item["generation"] = {
        "pipeline": "science_olympiad_club_pilot_v1_retrieval_generation",
        "event_key": key,
        "model_generated": True,
        "generation_model": MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "provider": provider,
        "retrieved_source_ids": [row["id"] for row in retrieved],
        "retrieval_scores": {row["id"]: row["similarity"] for row in retrieved},
        "manifest_spec": question_spec,
        "source_policy": "calibration and factual grounding; no copying",
    }
    return item


def audit_item(
    item: dict[str, Any],
    spec: dict[str, Any],
    question_spec: dict[str, Any],
    retrieved: list[dict[str, Any]],
    provider: str,
) -> dict[str, Any]:
    blind = generate_json(
        MODEL,
        json.dumps({"event": event_label(spec), "question": public_item(item)}, ensure_ascii=False),
        provider=provider,
        system=(
            "Cold-solve this Division B question without access to its key. Return strict JSON. Objectively flag "
            "missing data, scientific ambiguity, overclaiming, or an actual difficulty above D3. Do not inflate "
            "difficulty merely because this is an adversarial review. Difficulty must reflect the shortest valid "
            "solve, not length or vocabulary. The issues array is only for actionable defects; do not list a "
            "scientific or forensic limitation as an issue when the prompt and key already acknowledge it and "
            "the requested conclusion is appropriately qualified. Apply this rubric exactly:\n" + SCIOLY_DIFFICULTY_RUBRIC
        ),
        reasoning_effort="medium", max_output_tokens=3500, json_schema=BLIND_SCHEMA,
    )
    calibrated = blind.get("calibrated_difficulty")
    if isinstance(calibrated, int) and not isinstance(calibrated, bool) and calibrated in {1, 2, 3}:
        item["difficulty"] = calibrated
        item.setdefault("generation", {})["blueprint_difficulty"] = int(question_spec["difficulty"])
        item["generation"]["blind_calibrated_difficulty"] = calibrated
    editorial_payload = {
        "event": event_label(spec),
        "target_difficulty": item.get("difficulty"),
        "blueprint_difficulty": question_spec["difficulty"],
        "item_with_key": item,
        "blind_solve": blind,
        "retrieved_source_surfaces": [{"id": row["id"], "excerpt": row["excerpt"]} for row in retrieved],
    }
    editorial = generate_json(
        MODEL, json.dumps(editorial_payload, ensure_ascii=False), provider=provider,
        system=(
            "Brutally edit this middle-school Science Olympiad pilot item. Return strict JSON. Verify the key "
            "against the blind solve, scientific correctness, complete grading, authentic event scope, exact "
            "D1/D2/D3 target, no causal or diagnostic overclaiming, and no close copying of source surfaces. "
            "The target_difficulty is the objective answer-hidden calibration and is authoritative; the separate "
            "blueprint_difficulty is only the original design aim and may differ without being a defect. Do not "
            "require the student-facing item to mention D1, D2, D3, learning targets, metadata, or sources. "
            "student_ready is false for any substantive or wording defect."
        ),
        reasoning_effort="low", max_output_tokens=3500, json_schema=EDITORIAL_SCHEMA,
    )
    errors = deterministic_errors(item, question_spec)
    for flag in (
        "solvable", "science_correct", "self_contained", "event_faithful",
        "division_b_appropriate", "unique_answer",
    ):
        if blind.get(flag) is not True:
            errors.append(f"blind audit failed {flag}")
    if not isinstance(calibrated, int) or isinstance(calibrated, bool) or calibrated not in {1, 2, 3}:
        errors.append(f"blind-calibrated difficulty exceeds D3 ceiling or is invalid: {calibrated}")
    for flag in (
        "answer_agrees", "answer_correct", "rubric_complete", "student_ready",
        "competition_faithful", "difficulty_match", "no_overclaiming", "source_independent",
    ):
        if editorial.get(flag) is not True:
            errors.append(f"editorial audit failed {flag}")
    def substantive_issues(values: Any) -> list[str]:
        result = []
        for value in values or []:
            text = str(value).strip()
            normalized = re.sub(r"[^a-z]+", " ", text.casefold()).strip()
            if normalized not in {"", "none", "none significant", "no issue", "no issues", "n a"}:
                result.append(text)
        return result

    errors.extend(substantive_issues(blind.get("issues")))
    errors.extend(substantive_issues(editorial.get("issues")))
    return {
        "id": item.get("id"),
        "valid": not errors,
        "deterministic_errors": deterministic_errors(item, question_spec),
        "acceptance_errors": errors,
        "blind_solve": blind,
        "adversarial_audit": editorial,
        "judge_model": MODEL,
        "models": {"generation_and_audits": MODEL, "embeddings": EMBEDDING_MODEL},
        "judge": {
            "solvable": blind.get("solvable") is True,
            "answer_agrees": editorial.get("answer_agrees") is True,
            "science_correct": blind.get("science_correct") is True and editorial.get("answer_correct") is True,
            "division_b_appropriate": blind.get("division_b_appropriate") is True,
            "event_relevant": blind.get("event_faithful") is True,
            "difficulty_match": editorial.get("difficulty_match") is True,
            "competition_faithful": editorial.get("competition_faithful") is True,
            "novel": editorial.get("source_independent") is True,
            "independent_answer": blind.get("independent_answer"),
            "issues": errors,
            "provenance": {"generation_model": MODEL, "stage": "answer_hidden_solve_plus_editorial"},
        },
    }


FINAL_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "overall_good": {"type": "boolean"},
        "coverage_good": {"type": "boolean"},
        "difficulty_ceiling_respected": {"type": "boolean"},
        "reasoning_diversity_good": {"type": "boolean"},
        "competition_faithful": {"type": "boolean"},
        "middle_school_club_ready": {"type": "boolean"},
        "weak_item_ids": {"type": "array", "items": {"type": "string"}},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "overall_good", "coverage_good", "difficulty_ceiling_respected", "reasoning_diversity_good",
        "competition_faithful", "middle_school_club_ready", "weak_item_ids", "issues",
    ],
}


def final_audit(items: list[dict[str, Any]], provider: str) -> dict[str, Any]:
    vectors = embed_texts(EMBEDDING_MODEL, [str(row.get("prompt") or "") for row in items], provider=provider)
    close_pairs = []
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            semantic = cosine(vectors[left], vectors[right])
            structural = structural_similarity(items[left]["prompt"], items[right]["prompt"])
            if semantic >= 0.88 or structural >= 0.82:
                close_pairs.append({
                    "ids": [items[left]["id"], items[right]["id"]],
                    "semantic": semantic, "structural": structural,
                })
    audit = generate_json(
        MODEL,
        json.dumps([{key: row.get(key) for key in (
            "id", "response_type", "prompt", "choices", "answer", "difficulty", "topics"
        )} for row in items], ensure_ascii=False),
        provider=provider,
        system=(
            "Brutally audit this 15-item Science Olympiad Division B club pilot. Return strict JSON. It must "
            "contain five Anatomy & Physiology, five Meteorology, and five Crime Busters questions; an honest "
            "honest club-level calibration with at least three D1 items and at least six D2 items; D3 is optional, "
            "and no item may exceed D3; authentic breadth; unambiguous keys; concise "
            "student-facing wording; and no unsafe, clinical, causal, or forensic overclaiming. Set overall_good "
            "false if any item needs substantive or wording repair."
        ),
        reasoning_effort="low", max_output_tokens=4000, json_schema=FINAL_SCHEMA,
    )
    counts = Counter(int(row["difficulty"]) for row in items)
    audit.update({
        "model": MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "item_count": len(items),
        "difficulty_counts": {str(key): value for key, value in sorted(counts.items())},
        "close_pairs": close_pairs,
    })
    audit["valid"] = (
        len(items) == 15 and len({row["id"] for row in items}) == 15
        and set(counts) <= {1, 2, 3} and counts[1] >= 3 and counts[2] >= 6 and not close_pairs
        and all(audit.get(flag) is True for flag in (
            "overall_good", "coverage_good", "difficulty_ceiling_respected",
            "reasoning_diversity_good", "competition_faithful", "middle_school_club_ready",
        ))
        and not audit.get("weak_item_ids") and not audit.get("issues")
    )
    return audit


def run(provider: str, max_trials: int) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")
    os.environ["OPENAI_USAGE_LOG"] = str(RUN_DIR / "validation" / "model_usage.jsonl")
    os.environ["OPENAI_REASONING_TIMEOUT_SECONDS"] = os.getenv("SCIOLY_OPENAI_TIMEOUT_SECONDS", "90")
    specs = load_specs()
    corpus = build_corpus(specs)
    embeddings = ensure_embeddings(corpus, provider)
    # Per-event files are durable checkpoints if a later run is interrupted while
    # the combined progress file is being rewritten.
    item_checkpoints: list[dict[str, Any]] = []
    report_checkpoints: list[dict[str, Any]] = []
    for key in specs:
        item_checkpoints.extend(read_jsonl(RUN_DIR / "final" / key / "items.jsonl"))
        report_checkpoints.extend(read_jsonl(RUN_DIR / "final" / key / "reports.jsonl"))
    # The combined file contains the newest accepted rows from an interrupted run.
    item_checkpoints.extend(read_jsonl(RUN_DIR / "final" / "items.jsonl"))
    report_checkpoints.extend(read_jsonl(RUN_DIR / "final" / "reports.jsonl"))
    existing_items = {row["id"]: row for row in item_checkpoints}
    existing_reports = {row["id"]: row for row in report_checkpoints}
    accepted: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    ledger_path = RUN_DIR / "validation" / "trial_ledger.jsonl"

    for key, spec in specs.items():
        for question_spec in spec["question_specs"]:
            item_id = question_spec["id"]
            cached_item = existing_items.get(item_id) or {}
            cached_report = existing_reports.get(item_id) or {}
            cache_current = (
                cached_report.get("valid") is True
                and not deterministic_errors(cached_item, question_spec)
                and (cached_item.get("generation") or {}).get("manifest_spec") == question_spec
                and (cached_item.get("generation") or {}).get("generation_model") == MODEL
                and cached_report.get("judge_model") == MODEL
            )
            if cache_current:
                cached_item.setdefault("generation", {})["event_key"] = key
                accepted.append(cached_item)
                reports.append(cached_report)
                continue
            retrieved = retrieve(key, question_spec, corpus, embeddings, provider)
            feedback: list[str] = []
            for trial in range(1, max_trials + 1):
                item = model_generate(key, spec, question_spec, retrieved, feedback, provider)
                report = audit_item(item, spec, question_spec, retrieved, provider)
                append_jsonl(ledger_path, {
                    "id": item_id, "event_key": key, "trial": trial,
                    "accepted": report["valid"], "errors": report["acceptance_errors"],
                    "model": MODEL, "embedding_model": EMBEDDING_MODEL,
                })
                if report["valid"]:
                    accepted.append(item)
                    reports.append(report)
                    break
                feedback = report["acceptance_errors"][-8:]
            else:
                raise RuntimeError(f"failed to produce accepted item {item_id}: {feedback}")
            write_jsonl(RUN_DIR / "final" / "items.jsonl", accepted)
            write_jsonl(RUN_DIR / "final" / "reports.jsonl", reports)

    accepted.sort(key=lambda row: row["id"])
    reports.sort(key=lambda row: row["id"])
    audit = final_audit(accepted, provider)
    write_jsonl(RUN_DIR / "final" / "items.jsonl", accepted)
    write_jsonl(RUN_DIR / "final" / "reports.jsonl", reports)
    item_event = {(row.get("generation") or {}).get("event_key"): [] for row in accepted}
    report_by_id = {row["id"]: row for row in reports}
    for row in accepted:
        item_event.setdefault((row.get("generation") or {}).get("event_key"), []).append(row)
    for key in specs:
        event_items = item_event.get(key, [])
        write_jsonl(RUN_DIR / "final" / key / "items.jsonl", event_items)
        write_jsonl(
            RUN_DIR / "final" / key / "reports.jsonl",
            [report_by_id[row["id"]] for row in event_items],
        )
    write_json(RUN_DIR / "validation" / "final_audit.json", audit)
    summary = {
        "valid": audit["valid"], "item_count": len(accepted),
        "events": {key: 5 for key in sorted(specs)},
        "difficulty_counts": audit["difficulty_counts"],
        "generation_model": MODEL, "embedding_model": EMBEDDING_MODEL,
        "source_files": {key: SOURCE_FILES[key] for key in sorted(SOURCE_FILES)},
    }
    write_json(RUN_DIR / "validation" / "run_summary.json", summary)
    if not audit["valid"]:
        raise RuntimeError(f"final pilot audit failed: {audit}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openai",), default="openai")
    parser.add_argument("--max-trials", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.provider, args.max_trials), indent=2))


if __name__ == "__main__":
    main()
