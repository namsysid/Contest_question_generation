#!/usr/bin/env python3
"""
05_generate_questions.py  (STAGE-2: QUESTION GENERATION CONDITIONED ON GENERATED GRAPHS)

Uses generated typed problem graphs as the authoritative structural target.
Backward compatibility: retains `skeleton_text` field as alias of `graph_text`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from circuit_lab.model_client import generate_json

load_dotenv()

SYSTEM_GEN = """You write contest-faithful STEM multiple-choice problems from an authoritative solved blueprint.

Hard constraints:
- Do NOT copy, paraphrase, or minimally edit any exemplar scenario, wording, variable names, or numbers.
- You MUST produce novel phrasing, but the blueprint's physical scenario is authoritative.
- Preserve every fixed given, derivation, correct result, and distractor mechanism in the SOLUTION_BLUEPRINT.
- Never change, replace, tune, or omit a numerical given. Do not introduce a new numerical given.
- Provide exactly the requested number of answer choices with plausible, confusable distractors.
- Keep the problem self-contained and solvable without outside references.
- Keep the solution concise and presentation-ready (normally 250-700 words). Never narrate
  abandoned approaches, contradictions, corrections, or drafting history.
Return strict JSON only (no markdown).
"""

SYSTEM_GATEKEEP = """You are a severe, fail-closed gatekeeper for contest-style multiple-choice STEM problems.
Given the target difficulty, authentic competition exemplars, solution graph, solved blueprint, and candidate:
- verify the requested choice format and unique answer
- self-contained and solvable
- solution logically supports the claimed answer
- exactly consistent with the solved blueprint and primarily within the required topic
- faithful to TARGET_GRAPH_TEXT at a high level
- judge the SHORTEST valid student solve, ignoring prose length and arithmetic volume
- reject routine recall, one-formula substitution, cosmetic complications, or a key insight stated in the stem
- reject if actual difficulty is below the exact requested difficulty
Return strict JSON only:
{
  "verdict": "PASS"|"FAIL",
  "issues": ["..."],
  "required_fixes": ["..."],
  "actual_difficulty": 1,
  "competition_faithful": true,
  "reasoning_depth": 1,
  "routine_shortcut_exists": false,
  "familiar_template": false,
  "answer_consistency": {"answer_claimed":"A","answer_verified":"A|UNKNOWN|DIFFERS","notes":"..."}
}
Independently redo the calculation. If no option exactly matches (apart from ordinary stated
rounding), the answer letter is stale, or the solution contains correction chatter, mark FAIL.
"""


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl_line(f, row: Dict[str, Any]) -> None:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    f.flush()


def llm_json(model: str, system: str, user: str, provider: str, temperature: float = 0.2,
             reasoning_effort: str | None = None) -> Dict[str, Any]:
    return generate_json(model, user, provider=provider, system=system, temperature=temperature,
                         max_output_tokens={"low": 12000, "medium": 16000, "high": 20000}.get(
                             reasoning_effort, 5000
                         ),
                         reasoning_effort=reasoning_effort)


def llm_json_with_retries(
    model: str,
    system: str,
    user: str,
    *,
    temperature: float,
    retries: int,
    debug: bool,
    debug_prefix: str,
    provider: str = "ollama",
    reasoning_effort: str | None = None,
) -> Dict[str, Any]:
    last_exc: Optional[Exception] = None
    max_attempts = max(1, retries + 1)
    for attempt in range(1, max_attempts + 1):
        try:
            return llm_json(
                model, system, user, provider, temperature=temperature,
                reasoning_effort=reasoning_effort,
            )
        except Exception as exc:
            last_exc = exc
            if debug:
                print(
                    f"{debug_prefix} JSON call failed attempt {attempt}/{max_attempts}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            if attempt < max_attempts:
                time.sleep(0.6 * attempt)
    assert last_exc is not None
    raise last_exc


def _truncate(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"


def build_prompt(bundle: Dict[str, Any], target_graph_text: str, solution_blueprint: Dict[str, Any],
                 required_topic: str, max_q_ex: int, max_paired: int,
                 target_difficulty: int = 3, competition: str = "F=ma",
                 num_choices: int = 5) -> str:
    q_ex = (bundle.get("question_exemplars") or [])[:max_q_ex]
    p_ex = (bundle.get("paired_exemplars") or [])[:max_paired]

    q_blocks = []
    for i, ex in enumerate(q_ex, start=1):
        q_blocks.append(f"QUESTION EXEMPLAR {i}:\n{_truncate(ex.get('question_text',''), 1600)}")
    q_section = "\n\n".join(q_blocks) if q_blocks else "(none)"

    p_blocks = []
    for i, ex in enumerate(p_ex, start=1):
        p_blocks.append(
            f"PAIRED EXEMPLAR {i} QUESTION:\n{_truncate(ex.get('question_text',''), 1200)}\n\n"
            f"PAIRED EXEMPLAR {i} GRAPH_TEXT:\n{_truncate(ex.get('graph_text') or ex.get('skeleton_text',''), 1200)}"
        )
    p_section = "\n\n".join(p_blocks) if p_blocks else "(none)"

    return f"""You are given:

REQUIRED TOPIC: {required_topic}

1) TARGET_GRAPH_TEXT (authoritative latent dependency plan):
{target_graph_text}

SOLUTION_BLUEPRINT (authoritative; solve-first stage output):
{json.dumps(solution_blueprint, ensure_ascii=False)}

2) QUESTION EXEMPLARS (question space; style/semantics priors ONLY; must not be copied):
{q_section}

3) PAIRED EXEMPLARS (mapping demonstrations between dependency graphs and problems; must not be copied):
{p_section}

TASK:
Generate ONE NEW {competition}-style multiple-choice problem faithful to the graph and solved blueprint.
- Render canonical_problem faithfully; only surface wording may be polished.
- Preserve all blueprint variables, values, units, conditions, and the exact requested target.
- Use canonical_choices as the answer-choice content; only reorder and label them.
- Exactly {num_choices} choices {"A-D" if num_choices == 4 else "A-E"} (confusable distractors).
- Include at least 2 distractors corresponding to traps or common mistakes implied by the graph.
- Do not reveal the full solution path in the statement.
- Preserve the high-level dependency backbone: what hidden states matter, which laws/constraints couple, and what sort of trap profile exists.
- The final wording should still feel natural, not like a graph dump.
- The solution must be clean and presentation-ready: no editing chatter, corrections, or uncertainty.
- Keep the solution under 700 words and graph_text under 1,200 characters.
- Recalculate once, ensure exactly one choice matches, and end the solution with the keyed choice.

OUTPUT strict JSON schema:
{{
  "id": "<string>",
  "question": "<string stem>",
  "choices": {{"A":"...", "B":"...", "C":"...", "D":"...", "E":"..."}},
  "answer": "A|B|C|D|E",
  "solution": "<clear solution; may include equations>",
  "graph_text": "<compressed graph trace you actually used>",
  "skeleton_text": "<same as graph_text or shorter alias>",
  "anti_copy_report": {{
     "novel_scenario_summary": "<1-2 sentences>",
     "differences_from_exemplars": ["...", "..."],
     "possible_overlap_risks": ["...", "..."]
  }}
}}
"""


def build_gatekeeper_prompt(bundle: Dict[str, Any], target_graph_text: str,
                            solution_blueprint: Dict[str, Any], required_topic: str,
                            target_difficulty: int, competition: str, num_choices: int,
                            candidate: Dict[str, Any]) -> str:
    exemplars = [ex.get("question_text", "") for ex in (bundle.get("question_exemplars") or [])[:4]]
    return f"""COMPETITION: {competition}
CHOICE_FORMAT: {num_choices} choices ({"A-D" if num_choices == 4 else "A-E"})

AUTHENTIC_COMPETITION_EXEMPLARS_FOR_CALIBRATION:
{json.dumps(exemplars, ensure_ascii=False)}

TARGET_GRAPH_TEXT:
{target_graph_text}

REQUIRED_TOPIC:
{required_topic}

SOLUTION_BLUEPRINT:
{json.dumps(solution_blueprint, ensure_ascii=False)}

CANDIDATE_JSON:
{json.dumps(candidate, ensure_ascii=False)}
"""


def build_repair_prompt(target_graph_text: str, solution_blueprint: Dict[str, Any], required_topic: str,
                        target_difficulty: int, competition: str, num_choices: int,
                        candidate: Dict[str, Any], gate: Dict[str, Any]) -> str:
    return f"""Your candidate FAILED gatekeeper verification.

TARGET_GRAPH_TEXT (must remain faithful):
{target_graph_text}

SOLUTION_BLUEPRINT (do not change its physics or result):
{json.dumps(solution_blueprint, ensure_ascii=False)}

REQUIRED_TOPIC: {required_topic}
COMPETITION: {competition}
CHOICE_FORMAT: {num_choices} choices ({"A-D" if num_choices == 4 else "A-E"})

REQUIRED_FIXES:
{json.dumps(gate.get('required_fixes', []), ensure_ascii=False)}

ISSUES:
{json.dumps(gate.get('issues', []), ensure_ascii=False)}

FAILED_CANDIDATE_JSON:
{json.dumps(candidate, ensure_ascii=False)}

TASK:
Revise the candidate to address ALL required fixes while preserving:
- Novel scenario (no copying)
- Faithfulness to TARGET_GRAPH_TEXT
- Exactly {num_choices} confusable choices {"A-D" if num_choices == 4 else "A-E"}

Return strict JSON in the SAME schema.
"""


def is_basic_schema_ok(obj: Dict[str, Any], num_choices: int = 5) -> bool:
    if not isinstance(obj, dict):
        return False
    if "question" not in obj or "choices" not in obj or "answer" not in obj or "solution" not in obj:
        return False
    ch = obj.get("choices")
    if not isinstance(ch, dict):
        return False
    letters = set("ABCD" if num_choices == 4 else "ABCDE")
    if set(ch.keys()) != letters:
        return False
    if obj.get("answer") not in letters:
        return False
    return True


_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?![A-Za-z0-9_.])")
_FORBIDDEN_SOLUTION_CHATTER = (
    "closest answer", "closest choice", "matching the expected answer",
    "per the blueprint", "for the answer choices", "after correcting",
)


def _numeric_leaves(value: Any) -> List[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return [float(value)]
    if isinstance(value, dict):
        return [number for child in value.values() for number in _numeric_leaves(child)]
    if isinstance(value, list):
        return [number for child in value for number in _numeric_leaves(child)]
    return []


def _numbers_in_text(text: Any) -> List[float]:
    out: List[float] = []
    for match in _NUMBER_RE.findall(str(text or "")):
        try:
            out.append(float(match))
        except ValueError:
            pass
    return out


def _single_numeric_value(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    numbers = _numbers_in_text(value)
    return numbers[0] if len(numbers) == 1 else None


def candidate_blueprint_issues(candidate: Dict[str, Any], blueprint: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    question_numbers = _numbers_in_text(candidate.get("question"))
    for required in _numeric_leaves(blueprint.get("fixed_givens")):
        if not any(math.isclose(required, seen, rel_tol=1e-9, abs_tol=1e-12) for seen in question_numbers):
            issues.append(f"Blueprint fixed given {required:g} is absent from the question stem")

    answer = str(candidate.get("answer") or "")
    choice = (candidate.get("choices") or {}).get(answer)
    expected = blueprint.get("correct_choice", blueprint.get("expected_answer"))
    expected_num = _single_numeric_value(expected)
    choice_num = _single_numeric_value(choice)
    if expected_num is not None:
        if choice_num is None or not math.isclose(expected_num, choice_num, rel_tol=1e-6, abs_tol=1e-9):
            issues.append("Keyed choice does not equal the blueprint's verified result")
    elif "".join(str(expected).split()).casefold() not in "".join(str(choice or "").split()).casefold():
        issues.append("Keyed choice does not preserve the blueprint's verified result")

    solution = str(candidate.get("solution") or "").casefold()
    for marker in _FORBIDDEN_SOLUTION_CHATTER:
        if marker in solution:
            issues.append(f"Solution contains prohibited drafting/approximation language: {marker!r}")
    canonical_choices = blueprint.get("canonical_choices")
    if isinstance(canonical_choices, list):
        expected_choices = sorted("".join(str(value).split()).casefold() for value in canonical_choices)
        rendered_choices = sorted(
            "".join(str(value).split()).casefold()
            for value in (candidate.get("choices") or {}).values()
        )
        if expected_choices != rendered_choices:
            issues.append("Rendered answer choices differ from canonical_choices")
    return issues


def gate_passes(gate: Any, target_difficulty: int, candidate_answer: str | None = None) -> bool:
    if not isinstance(gate, dict) or gate.get("verdict") != "PASS":
        return False
    actual = gate.get("actual_difficulty")
    depth = gate.get("reasoning_depth")
    consistency = gate.get("answer_consistency") or {}
    return (
        isinstance(actual, int) and not isinstance(actual, bool)
        and actual >= target_difficulty
        and isinstance(depth, int) and not isinstance(depth, bool)
        and depth >= target_difficulty
        and gate.get("competition_faithful") is True
        and gate.get("routine_shortcut_exists") is False
        and (target_difficulty < 4 or gate.get("familiar_template") is False)
        and consistency.get("answer_verified") == consistency.get("answer_claimed")
        and (candidate_answer is None or consistency.get("answer_claimed") == candidate_answer)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", required=True)
    ap.add_argument("--skeletons", required=True)
    ap.add_argument("--out", default="generated_problems.jsonl")
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--verify_model", default="")
    ap.add_argument("--provider", choices=["ollama", "openai"], default="ollama")
    ap.add_argument("--required-topic", default="F=ma mechanics")
    ap.add_argument("--competition", default="F=ma")
    ap.add_argument("--num-choices", type=int, choices=(4, 5), default=5)
    ap.add_argument("--max_q_exemplars", type=int, default=4)
    ap.add_argument("--max_paired_exemplars", type=int, default=3)
    ap.add_argument("--repair_max", type=int, default=1)
    ap.add_argument("--json_retries", type=int, default=3, help="Retry count for malformed/non-JSON model responses")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--debug", action="store_true", help="print per-row progress to stderr")
    args = ap.parse_args()

    verify_model = args.verify_model or args.model

    bundles = read_jsonl(args.bundles)
    skels = read_jsonl(args.skeletons)
    sk_by_bundle = {r.get("bundle_id"): r for r in skels}

    if args.limit and args.limit > 0:
        bundles = bundles[: args.limit]

    total = len(bundles)

    if args.debug:
        print(f"[write] open (truncate): {args.out}", file=sys.stderr, flush=True)
    with open(args.out, "w", encoding="utf-8") as out_f:
        failures = 0
        for i, b in enumerate(bundles, start=1):
            bid = b.get("bundle_id")
            sk = sk_by_bundle.get(bid, {})
            if not sk or not (sk.get("solution_blueprint") and (sk.get("graph_text") or sk.get("skeleton_text"))):
                failures += 1
                if args.debug:
                    print(f"[{i}/{total}] bundle_id={bid} skipped: no accepted blueprint", file=sys.stderr)
                continue
            target_graph_text = sk.get("graph_text") or sk.get("skeleton_text") or ""
            solution_blueprint = sk.get("solution_blueprint") or {}
            target_difficulty = int(sk.get("difficulty") or 3)

            try:
                if args.debug:
                    print(f"[{i}/{total}] bundle_id={bid} building prompt", file=sys.stderr, flush=True)

                prompt = build_prompt(b, target_graph_text, solution_blueprint, args.required_topic,
                                      args.max_q_exemplars, args.max_paired_exemplars,
                                      target_difficulty, args.competition, args.num_choices)

                if args.debug:
                    print(f"[{i}/{total}] bundle_id={bid} calling gen model={args.model}", file=sys.stderr, flush=True)
                    t0 = time.time()

                cand = llm_json_with_retries(
                    args.model,
                    SYSTEM_GEN,
                    prompt,
                    temperature=0.25,
                    retries=args.json_retries,
                    debug=args.debug,
                    debug_prefix=f"[{i}/{total}] bundle_id={bid}",
                    provider=args.provider,
                )
                if "graph_text" not in cand and target_graph_text:
                    cand["graph_text"] = target_graph_text
                if "skeleton_text" not in cand:
                    cand["skeleton_text"] = cand.get("graph_text", "")

                if args.debug:
                    dt = time.time() - t0
                    print(f"[{i}/{total}] bundle_id={bid} gen done in {dt:.2f}s", file=sys.stderr, flush=True)

                repairs: List[Dict[str, Any]] = []
                gate: Optional[Dict[str, Any]] = None

                for r_i in range(max(0, args.repair_max) + 1):
                    deterministic_issues = candidate_blueprint_issues(cand, solution_blueprint)
                    if not is_basic_schema_ok(cand, args.num_choices):
                        gate = {"verdict":"FAIL","issues":["Basic schema invalid"],"required_fixes":[f"Fix JSON schema to contain exactly {args.num_choices} choices"],"answer_consistency":{"answer_claimed":cand.get("answer",""),"answer_verified":"UNKNOWN","notes":"schema invalid"}}
                    elif deterministic_issues:
                        gate = {
                            "verdict": "FAIL", "issues": deterministic_issues,
                            "required_fixes": deterministic_issues,
                            "answer_consistency": {
                                "answer_claimed": cand.get("answer", ""),
                                "answer_verified": "UNKNOWN",
                                "notes": "deterministic blueprint-fidelity failure",
                            },
                        }
                    else:
                        if args.debug:
                            print(f"[{i}/{total}] bundle_id={bid} gatekeeper call {r_i+1} model={verify_model}", file=sys.stderr, flush=True)
                        gate = llm_json_with_retries(
                            verify_model,
                            SYSTEM_GATEKEEP,
                            build_gatekeeper_prompt(
                                b, target_graph_text, solution_blueprint, args.required_topic,
                                target_difficulty, args.competition, args.num_choices, cand,
                            ),
                            temperature=0.0,
                            retries=args.json_retries,
                            debug=args.debug,
                            debug_prefix=f"[{i}/{total}] bundle_id={bid}",
                            provider=args.provider,
                            reasoning_effort=(
                                "medium" if args.provider == "openai"
                                and verify_model.startswith(("gpt-5", "o1", "o3", "o4")) else None
                            ),
                        )

                    if gate_passes(gate, target_difficulty, str(cand.get("answer") or "")):
                        if args.debug:
                            print(f"[{i}/{total}] bundle_id={bid} gatekeeper PASS", file=sys.stderr, flush=True)
                        break

                    repairs.append({"gatekeeper": gate, "candidate": cand})
                    if args.debug:
                        print(f"[{i}/{total}] bundle_id={bid} repair {r_i+1} calling model={args.model}", file=sys.stderr, flush=True)
                    cand = llm_json_with_retries(
                        args.model,
                        SYSTEM_GEN,
                        build_repair_prompt(
                            target_graph_text, solution_blueprint, args.required_topic,
                            target_difficulty, args.competition, args.num_choices, cand, gate,
                        ),
                        temperature=0.2,
                        retries=args.json_retries,
                        debug=args.debug,
                        debug_prefix=f"[{i}/{total}] bundle_id={bid}",
                        provider=args.provider,
                    )
                    if "graph_text" not in cand and target_graph_text:
                        cand["graph_text"] = target_graph_text
                    if "skeleton_text" not in cand:
                        cand["skeleton_text"] = cand.get("graph_text", "")

                if (
                    not gate_passes(gate, target_difficulty, str(cand.get("answer") or ""))
                    or candidate_blueprint_issues(cand, solution_blueprint)
                ):
                    failures += 1
                    print(
                        f"[{i}/{total}] bundle_id={bid} REJECTED after difficulty gate",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue

                out_row = {
                    **cand,
                    "difficulty": target_difficulty,
                    "_meta": {
                        "bundle_id": bid,
                        "seed_id": b.get("seed_id"),
                        "mode": b.get("mode"),
                        "anchor_id": b.get("anchor_id"),
                        "anchor_distance": b.get("anchor_distance"),
                        "required_topic": args.required_topic,
                        "generation_provider": args.provider,
                        "competition": args.competition,
                        "research_pipeline": "stages_01_05_dual_space_solution_graph",
                        "target_difficulty": target_difficulty,
                    },
                    "_gatekeeper": gate,
                    "_repairs": repairs,
                }

                write_jsonl_line(out_f, out_row)
                if args.debug:
                    print(f"[{i}/{total}] bundle_id={bid} wrote", file=sys.stderr, flush=True)

                if args.sleep:
                    time.sleep(args.sleep)
            except Exception as exc:
                failures += 1
                print(f"[{i}/{total}] bundle_id={bid} FAILED: {exc}", file=sys.stderr, flush=True)
                continue

    if args.debug:
        print(f"[write] done: {args.out}", file=sys.stderr, flush=True)
    print(f"Wrote generated problems -> {args.out} (success={total - failures}, failed={failures})")

if __name__ == "__main__":
    main()
