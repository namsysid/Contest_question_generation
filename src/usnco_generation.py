"""USNCO-specific prompting, exemplar normalization, and output validation."""
from __future__ import annotations

import re
from typing import Any, Iterable

from difficulty_rubric import difficulty_instruction


USNCO_TOPIC_SPECS = (
    ("molStoichiometry", "Stoichiometry", "moles, formulas, solutions, and chemical reactions"),
    ("eAtomicStructure", "Atomic structure", "electrons, periodicity, and atomic spectra"),
    ("sigmaBonding", "Bonding", "Lewis structures, VSEPR, orbitals, and intermolecular forces"),
    ("deltaHThermochemistry", "Thermochemistry", "heat, entropy, enthalpy, and Gibbs free energy"),
    ("KEquilibrium", "Equilibrium", "equilibrium constants, reaction quotients, and Le Chatelier's principle"),
    ("pHAcidsBases", "Acids & bases", "pH, acid-base equilibria, buffers, and titrations"),
    ("kKinetics", "Kinetics", "rate laws, mechanisms, and activation energy"),
    ("E0Electrochemistry", "Electrochemistry", "redox cells, voltage, electrolysis, and Faraday's law"),
)
USNCO_TOPIC_BY_KEY = {key: (label, description) for key, label, description in USNCO_TOPIC_SPECS}
USNCO_CHOICE_KEYS = ("A", "B", "C", "D")
USNCO_DIVERSITY_GUIDANCE = {
    "molStoichiometry": "Rotate among limiting reagents, empirical formulas, solution preparation, gas stoichiometry, yields, and particulate reasoning.",
    "eAtomicStructure": "Rotate among spectroscopy, quantum numbers, periodic trends, photoelectron data, configurations, and successive ionization energies.",
    "sigmaBonding": "Rotate among geometry, symmetry, polarity, orbital overlap, bond order, lattice energy, and intermolecular-force evidence.",
    "deltaHThermochemistry": "Rotate among Hess cycles, calorimetry, phase changes, entropy, spontaneity, formation data, and temperature dependence.",
    "KEquilibrium": "Rotate among K/Q comparisons, coupled equilibria, solubility, gas equilibria, perturbations, and thermodynamic links to K.",
    "pHAcidsBases": "Rotate among salt hydrolysis, indicators, polyprotic systems, amphiprotic species, percent ionization, buffer capacity, and titration-curve reasoning. Avoid another routine HA/A- buffer calculation unless structurally novel.",
    "kKinetics": "Rotate among initial-rate tables, integrated laws, half-lives, Arrhenius comparisons, mechanisms, isotope effects, and catalysis evidence.",
    "E0Electrochemistry": "Rotate among galvanic-cell direction, Nernst concentration effects, electrolysis stoichiometry, Faraday calculations, corrosion, and coupled half-reactions.",
}

USNCO_SYSTEM_PROMPT = """You generate original USNCO-style high-school chemistry multiple-choice questions.

Use the supplied official ACS USNCO questions only to calibrate scope, concision, and difficulty.
Never copy or closely paraphrase their wording, numbers, substances, or scenario structure.
Construct and solve the chemistry problem first, verify the result independently, then write the
question and distractors around that verified solution. Return strict JSON only.

Every problem must:
- primarily test the requested chemistry topic;
- have exactly four plausible choices A through D and exactly one correct answer;
- be self-contained and not depend on a missing diagram or external reference;
- include a concise full solution and two or more ordered solution_steps;
- include a difficulty integer from 1 (direct foundational application) through 5 (USNCO-level
  multi-step or non-obvious reasoning);
- use chemically correct formulas, signs, units, significant figures, and constants.
"""


def usnco_problem_json_schema(expected_topic_key: str, expected_topic: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "domain": {"type": "string", "enum": ["chemistry"]},
            "topic_key": {"type": "string", "enum": [expected_topic_key]},
            "topic": {"type": "string", "enum": [expected_topic]},
            "question": {"type": "string"},
            "choices": {
                "type": "object",
                "properties": {key: {"type": "string"} for key in USNCO_CHOICE_KEYS},
                "required": list(USNCO_CHOICE_KEYS),
                "additionalProperties": False,
            },
            "answer": {"type": "string", "enum": list(USNCO_CHOICE_KEYS)},
            "solution_steps": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
            },
            "solution": {"type": "string"},
            "difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": [
            "id", "domain", "topic_key", "topic", "question", "choices", "answer",
            "solution_steps", "solution", "difficulty",
        ],
        "additionalProperties": False,
    }


def usnco_batch_json_schema(
    expected_count: int, topic_key: str, topic: str, difficulty: int | None = None
) -> dict[str, Any]:
    problem_schema = usnco_problem_json_schema(topic_key, topic)
    if difficulty is not None:
        problem_schema["properties"]["difficulty"] = {"type": "integer", "enum": [difficulty]}
    return {
        "type": "object",
        "properties": {
            "problems": {
                "type": "array",
                "items": problem_schema,
                "minItems": expected_count,
                "maxItems": expected_count,
            }
        },
        "required": ["problems"],
        "additionalProperties": False,
    }


def _normalize_choices(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        choices = {str(key).strip().upper(): str(body).strip() for key, body in value.items()}
    elif isinstance(value, list) and len(value) == 4:
        choices = {key: str(body).strip() for key, body in zip(USNCO_CHOICE_KEYS, value)}
    else:
        return {}
    if set(choices) != set(USNCO_CHOICE_KEYS) or any(not choices[key] for key in USNCO_CHOICE_KEYS):
        return {}
    return {key: choices[key] for key in USNCO_CHOICE_KEYS}


def _source_fields(row: dict[str, Any]) -> tuple[str, dict[str, str], str]:
    content = row.get("content") if isinstance(row.get("content"), dict) else {}
    problem = row.get("problem") if isinstance(row.get("problem"), dict) else {}
    question = str(
        row.get("question") or row.get("question_text") or content.get("question_text")
        or problem.get("stem") or ""
    ).strip()
    choices = _normalize_choices(row.get("choices") or content.get("choices") or problem.get("choices"))
    answer_value: Any = row.get("answer") or row.get("answer_key") or problem.get("answer_key")
    if isinstance(answer_value, dict):
        answer_value = answer_value.get("letter")
    return question, choices, str(answer_value or "").strip().upper()


def usnco_exemplar_from_row(row: dict[str, Any]) -> str | None:
    question, choices, answer = _source_fields(row)
    if not question or not choices:
        return None
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    provenance = row.get("source_url") or source.get("url") or row.get("source_pdf") or "official USNCO corpus"
    rendered = " | ".join(f"({key}) {choices[key]}" for key in USNCO_CHOICE_KEYS)
    return (
        f"ID: {row.get('id') or 'unknown'}\nSOURCE: {provenance}\n"
        f"QUESTION:\n{question}\nCHOICES:\n{rendered}\n"
        f"ANSWER: {answer if answer in USNCO_CHOICE_KEYS else '(unknown)'}"
    )


def pack_usnco_exemplars(
    rows: Iterable[dict[str, Any]], max_prompt_chars: int, max_count: int | None = None
) -> list[str]:
    packed: list[str] = []
    used = 0
    for row in rows:
        if max_count is not None and len(packed) >= max_count:
            break
        exemplar = usnco_exemplar_from_row(row)
        if not exemplar:
            continue
        block = f"### OFFICIAL USNCO EXEMPLAR {len(packed) + 1}\n{exemplar}\n"
        if packed and used + len(block) > max_prompt_chars:
            break
        if not packed and len(block) > max_prompt_chars:
            continue
        packed.append(block)
        used += len(block)
    return packed


def build_usnco_prompt(
    exemplar_blocks: list[str],
    expected_count: int,
    topic_key: str,
    topic: str,
    topic_description: str,
    prior_questions: list[str],
    difficulty: int | None = None,
    retry_note: str = "",
) -> str:
    avoid = "\n".join(f"- {question[:240]}" for question in prior_questions[-30:]) or "(none yet)"
    target_difficulty = difficulty or 3
    retry = f"\nA prior draft was rejected: {retry_note}\nCorrect every issue.\n" if retry_note else ""
    return f"""Generate exactly {expected_count} original USNCO multiple-choice question(s).

Required topic_key: {topic_key}
Required topic: {topic}
Required content: {topic_description}
{difficulty_instruction(target_difficulty)}
Diversity requirement: {USNCO_DIVERSITY_GUIDANCE[topic_key]}

Use a solution-centric approach: construct a valid solution path first, calculate or reason to
the answer, verify it against all choices, and only then finalize the stem and distractors.
`solution_steps` must expose that ordered reasoning; `solution` must be a readable synthesis.
The topic fields must exactly match the values above. Do not copy an exemplar.
{retry}
Prior generated questions to avoid:
{avoid}

Return {{"problems": [...]}} with exactly {expected_count} items and no Markdown.

OFFICIAL ACS USNCO EXEMPLARS:
{chr(10).join(exemplar_blocks)}
"""


def _extract_candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(result.get("problems"), list):
        return [item for item in result["problems"] if isinstance(item, dict)]
    if all(key in result for key in ("question", "choices", "answer")):
        return [result]
    return []


def validate_usnco_batch(
    result: dict[str, Any], expected_count: int, topic_key: str, topic: str,
    expected_difficulty: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    candidates = _extract_candidates(result)
    errors: list[str] = []
    valid: list[dict[str, Any]] = []
    if len(candidates) != expected_count:
        errors.append(f"expected {expected_count} problems but received {len(candidates)}")
    for index, candidate in enumerate(candidates, 1):
        item_errors: list[str] = []
        question = str(candidate.get("question") or "").strip()
        choices = _normalize_choices(candidate.get("choices"))
        answer = str(candidate.get("answer") or "").strip().upper()
        solution = str(candidate.get("solution") or "").strip()
        raw_steps = candidate.get("solution_steps")
        steps = [str(step).strip() for step in raw_steps] if isinstance(raw_steps, list) else []
        steps = [step for step in steps if step]
        difficulty = candidate.get("difficulty")
        if not question:
            item_errors.append("question is missing")
        if not choices:
            item_errors.append("choices must contain exactly nonempty A-D entries")
        if answer not in USNCO_CHOICE_KEYS:
            item_errors.append("answer must be one letter A-D")
        if not solution:
            item_errors.append("solution is missing")
        if len(steps) < 2:
            item_errors.append("solution_steps must contain at least two nonempty steps")
        if not isinstance(difficulty, int) or isinstance(difficulty, bool) or not 1 <= difficulty <= 5:
            item_errors.append("difficulty must be an integer from 1 to 5")
        elif expected_difficulty is not None and difficulty != expected_difficulty:
            item_errors.append(f"difficulty must equal requested level {expected_difficulty}")
        if candidate.get("topic_key") != topic_key or candidate.get("topic") != topic:
            item_errors.append("topic metadata does not match the requested category")
        domain = str(candidate.get("domain") or "").strip().lower()
        if domain not in {"chem", "chemistry"}:
            item_errors.append("domain must be chemistry")
        if item_errors:
            errors.append(f"problem {index}: " + "; ".join(item_errors))
            continue
        normalized = dict(candidate)
        normalized.update({
            "domain": "chemistry", "topic_key": topic_key, "topic": topic,
            "question": question, "choices": choices, "answer": answer,
            "solution_steps": steps, "solution": solution, "difficulty": difficulty,
        })
        valid.append(normalized)
    return (valid if not errors else []), errors


def infer_usnco_topic(question_number: int | None, text: str = "") -> str:
    """Map official ACS buckets/keywords into the application's eight-topic taxonomy."""
    lowered = re.sub(r"\s+", " ", text).casefold()
    keyword_rules = (
        ("pHAcidsBases", r"\b(?:ph|pka|pkb|acid|base|buffer|titration|hydronium|hydroxide)\b"),
        ("E0Electrochemistry", r"\b(?:electrochem|electrolysis|galvanic|voltaic|cell potential|faraday|anode|cathode)\b"),
        ("kKinetics", r"\b(?:rate law|reaction rate|activation energy|arrhenius|half-life|mechanism)\b"),
        ("deltaHThermochemistry", r"\b(?:enthalpy|entropy|gibbs|calorim|heat capacity|hess|thermochem)\b"),
        ("KEquilibrium", r"\b(?:equilibrium|reaction quotient|le chatelier|solubility product|\bkp\b|\bkc\b)"),
        ("eAtomicStructure", r"\b(?:electron configuration|spectrum|spectra|periodic trend|ionization energy|quantum number)\b"),
        ("sigmaBonding", r"\b(?:lewis|vsepr|hybridization|bond angle|molecular geometry|intermolecular|orbital)\b"),
    )
    for key, pattern in keyword_rules:
        if re.search(pattern, lowered):
            return key
    if question_number:
        if 1 <= question_number <= 6:
            return "molStoichiometry"
        if 19 <= question_number <= 24:
            return "deltaHThermochemistry"
        if 25 <= question_number <= 30:
            return "kKinetics"
        if 31 <= question_number <= 36:
            return "KEquilibrium"
        if 37 <= question_number <= 42:
            return "E0Electrochemistry"
        if 43 <= question_number <= 48:
            return "eAtomicStructure"
        if 49 <= question_number <= 54:
            return "sigmaBonding"
    return "molStoichiometry"
