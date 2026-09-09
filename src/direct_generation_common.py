from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Tuple


CHOICE_KEYS = ("A", "B", "C", "D", "E")
CHOICE_RE = re.compile(
    r"\(\s*([A-Ea-e])\s*\)\s*(.+?)(?=(?:\n\s*\(\s*[A-Ea-e]\s*\)\s*)|\Z)",
    re.DOTALL,
)

# These patterns intentionally target unmistakable out-of-scope subject matter rather
# than isolated words such as "light" (which is common in "light string") or "current"
# (which can be ordinary prose). This is a guardrail in addition to the prompt; it is
# not intended to classify every possible physics question.
FORBIDDEN_FMA_PATTERNS = (
    ("electricity/circuits", re.compile(
        r"\b(?:electric\s+(?:field|potential|current)|magnetic\s+(?:field|flux)|"
        r"circuit|resistor|electrical\s+resistance|capacitor|capacitance|inductor|inductance|"
        r"battery|voltage|ohm(?:'s)?\s+law|coulomb(?:'s)?\s+law)\b",
        re.IGNORECASE,
    )),
    ("optics", re.compile(
        r"\b(?:light\s+ray|ray\s+of\s+light|beam\s+of\s+(?:monochromatic\s+)?light|"
        r"thin\s+(?:film|lens)|focal\s+length|image\s+distance|refractive\s+index|"
        r"angle\s+of\s+refraction|snell(?:'s)?\s+law|diffraction|interference)\b",
        re.IGNORECASE,
    )),
    ("thermodynamics", re.compile(
        r"\b(?:ideal\s+gas|adiabatic|isothermal|thermodynamic|entropy|heat\s+engine|"
        r"specific\s+heat|latent\s+heat)\b",
        re.IGNORECASE,
    )),
    ("waves", re.compile(
        r"\b(?:sound\s+wave|wave\s+speed|wavelength|standing\s+wave|"
        r"harmonic\s+mode|vibrating\s+string)\b",
        re.IGNORECASE,
    )),
    ("modern physics", re.compile(
        r"\b(?:photoelectric|photon|radioactive|half-life|quantum|nuclear\s+(?:decay|reaction)|"
        r"special\s+relativity|general\s+relativity)\b",
        re.IGNORECASE,
    )),
)


def forbidden_fma_topic(text: str) -> str | None:
    for topic, pattern in FORBIDDEN_FMA_PATTERNS:
        if pattern.search(text):
            return topic
    return None

SYSTEM_PROMPT = """You generate F=ma-style high-school physics contest multiple-choice problems.

Hard constraints:
- Every problem must be within the official F=ma mechanics scope: kinematics, statics,
  Newton's laws, momentum and energy, oscillations, orbital mechanics, rotational dynamics,
  fluids, dimensional analysis, or elementary data analysis.
- Do not generate electricity and magnetism, circuits, optics, waves, thermodynamics, modern
  physics, relativity, or a pure mathematics exercise.
- Produce novel scenarios and phrasing; do not paraphrase or lightly rewrite exemplars.
- Provide exactly five answer choices, keyed A through E, with plausible physics distractors.
- Exactly one choice must be correct, and `answer` must be its single letter.
- Keep every problem self-contained and solvable from stated information. Do not require a
  missing diagram, unstated geometry, or outside facts beyond standard contest physics.
- Verify numerical values, units, and the keyed answer before returning the problem.
- Return strict JSON only, without Markdown.
"""

PROBLEM_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "domain": {"type": "string", "enum": ["physics"]},
        "question": {"type": "string"},
        "choices": {
            "type": "object",
            "properties": {key: {"type": "string"} for key in CHOICE_KEYS},
            "required": list(CHOICE_KEYS),
            "additionalProperties": False,
        },
        "answer": {"type": "string", "enum": list(CHOICE_KEYS)},
        "solution": {"type": "string"},
    },
    "required": ["id", "domain", "question", "choices", "answer", "solution"],
    "additionalProperties": False,
}


def batch_json_schema(expected_count: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "problems": {
                "type": "array",
                "items": PROBLEM_JSON_SCHEMA,
                "minItems": expected_count,
                "maxItems": expected_count,
            }
        },
        "required": ["problems"],
        "additionalProperties": False,
    }


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_line(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def split_embedded_choices(text: str) -> Tuple[str, Dict[str, str]]:
    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    matches = list(CHOICE_RE.finditer(cleaned))
    if not matches:
        return cleaned, {}
    choices = {
        match.group(1).upper(): re.sub(r"\s+", " ", match.group(2)).strip()
        for match in matches
    }
    if set(choices) != set(CHOICE_KEYS):
        return cleaned, {}
    return cleaned[: matches[0].start()].strip(), choices


def normalize_choices(value: Any) -> Dict[str, str]:
    if isinstance(value, dict):
        normalized = {str(key).upper(): str(body).strip() for key, body in value.items()}
    elif isinstance(value, list) and len(value) == 5:
        normalized = {key: str(body).strip() for key, body in zip(CHOICE_KEYS, value)}
    else:
        return {}
    if set(normalized) != set(CHOICE_KEYS) or any(not normalized[key] for key in CHOICE_KEYS):
        return {}
    return {key: normalized[key] for key in CHOICE_KEYS}


def exemplar_from_row(row: Dict[str, Any]) -> str:
    problem = row.get("problem") if isinstance(row.get("problem"), dict) else None
    if problem is not None:
        question = str(problem.get("stem") or "").strip()
        choices = normalize_choices(problem.get("choices"))
        answer = str(problem.get("answer_key") or "").strip().upper()
        domain = str(row.get("domain") or "physics").strip()
    else:
        question = str(row.get("question") or row.get("question_text") or "").strip()
        choices = normalize_choices(row.get("choices"))
        answer = str(row.get("answer") or "").strip().upper()
        domain = str(row.get("domain") or "physics").strip()

    if not choices:
        question, embedded = split_embedded_choices(question)
        choices = embedded

    choice_text = " | ".join(f"({key}) {choices[key]}" for key in CHOICE_KEYS) if choices else "(none listed)"
    return (
        f"ID: {row.get('id') or 'unknown'}\n"
        f"DOMAIN: {domain}\n"
        f"QUESTION:\n{question}\n"
        f"CHOICES:\n{choice_text}\n"
        f"ANSWER: {answer if answer in CHOICE_KEYS else '(unknown)'}"
    )


def pack_exemplars(
    rows: Iterable[Dict[str, Any]], max_prompt_chars: int, max_count: int | None = None
) -> List[str]:
    packed: List[str] = []
    used = 0
    for row in rows:
        if max_count is not None and len(packed) >= max_count:
            break
        exemplar = exemplar_from_row(row)
        # Do not teach the model that an MCQ without choices is an acceptable exemplar.
        if "CHOICES:\n(none listed)" in exemplar:
            continue
        # The source corpus can contain general-physics contamination even when the
        # generation task is F=ma. Never expose those rows as demonstrations.
        if forbidden_fma_topic(exemplar):
            continue
        block = f"### PHYSICS EXEMPLAR {len(packed) + 1}\n{exemplar}\n"
        if packed and used + len(block) > max_prompt_chars:
            break
        if not packed and len(block) > max_prompt_chars:
            block = block[:max_prompt_chars]
        packed.append(block)
        used += len(block)
    return packed


def build_prompt(exemplar_blocks: List[str], target_count: int, retry_note: str = "") -> str:
    retry = f"\nA prior attempt was rejected: {retry_note}\nCorrect every listed issue.\n" if retry_note else ""
    return f"""Generate {target_count} new F=ma-style MECHANICS multiple-choice problems.

The exemplars below establish the intended mechanics domain and contest style. Stay within
F=ma mechanics even if an exemplar contains another subject due to corpus noise. Do not
generate E&M, circuits, optics, waves, thermodynamics, modern physics, calculus exercises,
pure algebra, combinatorics, or other mathematics questions. Mathematics may be used only as
part of solving a substantive mechanics problem.
{retry}
Return this exact JSON structure:
{{
  "problems": [
    {{
      "id": "<unique string>",
      "domain": "physics",
      "question": "<self-contained F=ma mechanics question stem>",
      "choices": {{"A":"...", "B":"...", "C":"...", "D":"...", "E":"..."}},
      "answer": "A|B|C|D|E",
      "solution": "<derive and verify the keyed answer>"
    }}
  ]
}}

The `problems` array must contain exactly {target_count} complete items. Before responding,
check that every item is physics, has five nonempty choices A-E, has exactly one defensible
answer, and that its solution reaches the keyed choice.

PHYSICS EXEMPLARS:
{chr(10).join(exemplar_blocks)}
"""


def extract_problems(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    value = result.get("problems")
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    # Smaller local models often omit the wrapper for a one-item request. Accept the
    # object only when it looks like the requested problem schema; validate_problem
    # performs the strict field checks immediately afterward.
    if all(key in result for key in ("question", "answer")) and (
        "choices" in result or "options" in result
    ) and (
        "solution" in result or "explanation" in result
    ):
        return [result]
    for key in ("generated_problems", "questions", "items", "results"):
        candidate = result.get(key)
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]
    raise ValueError(f"missing problem list; top-level keys={sorted(result)}")


def validate_problem(candidate: Dict[str, Any]) -> Tuple[Dict[str, Any] | None, List[str]]:
    errors: List[str] = []
    question = str(candidate.get("question") or "").strip()
    choices = normalize_choices(candidate.get("choices") or candidate.get("options"))
    raw_answer = str(candidate.get("answer") or "").strip()
    answer = raw_answer.upper()
    match = re.match(r"^(?:ANSWER\s*[:=-]?\s*)?(?:CHOICE|OPTION)?\s*\(?([A-E])\)?(?:\s|[.):,-]|$)", answer)
    if match:
        answer = match.group(1)
    elif choices:
        matching_keys = [key for key, value in choices.items() if raw_answer.casefold() == value.casefold()]
        if len(matching_keys) == 1:
            answer = matching_keys[0]
    solution = str(candidate.get("solution") or candidate.get("explanation") or "").strip()
    # Some local models omit the redundant domain label even under a mechanics-only
    # system prompt. The question is still subject to downstream blind F=ma grading.
    domain = str(candidate.get("domain") or "physics").strip().lower()

    topic_text = "\n".join([question, *choices.values(), solution])
    forbidden_topic = forbidden_fma_topic(topic_text)

    if domain != "physics":
        errors.append("domain must be `physics`")
    if not question:
        errors.append("question is missing")
    if not choices:
        errors.append("choices must contain exactly nonempty A-E entries")
    if answer not in CHOICE_KEYS:
        errors.append(f"answer must resolve to one letter A-E (received {raw_answer!r})")
    if not solution:
        errors.append("solution is missing")
    if forbidden_topic:
        errors.append(f"problem is outside F=ma mechanics scope ({forbidden_topic})")
    if errors:
        return None, errors

    normalized = dict(candidate)
    normalized.pop("explanation", None)
    normalized.pop("options", None)
    normalized.update(
        {"domain": "physics", "question": question, "choices": choices, "answer": answer, "solution": solution}
    )
    return normalized, []


def validate_batch(result: Dict[str, Any], expected_count: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    try:
        candidates = extract_problems(result)
    except ValueError as exc:
        return [], [str(exc)]
    errors: List[str] = []
    valid: List[Dict[str, Any]] = []
    if len(candidates) != expected_count:
        errors.append(f"expected {expected_count} problems but received {len(candidates)}")
    for index, candidate in enumerate(candidates, start=1):
        normalized, item_errors = validate_problem(candidate)
        if item_errors:
            errors.append(f"problem {index}: " + "; ".join(item_errors))
        elif normalized is not None:
            valid.append(normalized)
    return (valid if not errors else []), errors
