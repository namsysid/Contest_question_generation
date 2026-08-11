#!/usr/bin/env python3
"""
Schema-free ablation pipeline:
- retrieves from raw question/solution text only (no graph/schema fields)
- generates new contest-style problems from retrieved raw exemplars
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv() -> None:
        return None

from ollama_client import embed_texts, generate_json

load_dotenv()

SYSTEM_GEN = """You generate contest-faithful STEM multiple-choice problems from retrieved raw solved examples.

Hard constraints:
- Do NOT copy, paraphrase, or minimally edit exemplar scenario wording or numbers.
- Produce one novel, self-contained problem with exactly 5 choices (A)-(E).
- Keep distractors plausible and confusable.
- Return strict JSON only (no markdown).
"""

SYSTEM_PLAIN_SOLUTION = """You write plain, concise solution text for a STEM multiple-choice question.

Hard constraints:
- Do not use markdown.
- Explain the key reasoning steps.
- End by stating the most likely answer option as a single letter A-E.
- Return strict JSON only.
"""

SYSTEM_SOLVE_FIRST = """You create a novel contest-style solved MCQ core from retrieved raw solved examples.

Hard constraints:
- Do NOT copy, paraphrase, or minimally edit exemplar scenario wording or numbers.
- Produce one coherent solved core with exactly 5 choices (A)-(E), one correct answer, and a concise complete solution.
- Return strict JSON only (no markdown).
"""

SYSTEM_QUESTION_FROM_SOLUTION = """You convert a solved MCQ core into the final contest-style problem statement.

Hard constraints:
- Keep the provided choices exactly as-is.
- Keep the provided answer exactly as-is.
- Keep the provided solution semantically consistent.
- Return strict JSON only (no markdown).
"""


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _choices_to_text(choices: Any) -> str:
    if isinstance(choices, dict):
        parts = []
        for key in ("A", "B", "C", "D", "E"):
            val = str(choices.get(key) or "").strip()
            if val:
                parts.append(f"({key}) {val}")
        return "\n".join(parts)
    if isinstance(choices, list):
        out = []
        for idx, choice in enumerate(choices):
            letter = chr(ord("A") + idx) if idx < 5 else str(idx + 1)
            txt = str(choice).strip()
            if txt:
                out.append(f"({letter}) {txt}")
        return "\n".join(out)
    return ""


def _extract_question_text(row: Dict[str, Any]) -> str:
    for key in ("question", "question_text", "prompt"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    problem = row.get("problem")
    if isinstance(problem, dict):
        stem = str(problem.get("stem") or "").strip()
        choices = _choices_to_text(problem.get("choices"))
        if stem and choices:
            return f"{stem}\n\n{choices}".strip()
        return stem
    return ""


def _extract_solution_text(row: Dict[str, Any]) -> str:
    for key in ("solution_text", "solution", "reasoning_summary", "explanation", "rationale"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    solve_attempt = row.get("solve_attempt")
    if isinstance(solve_attempt, dict):
        val = solve_attempt.get("reasoning_summary")
        if isinstance(val, str) and val.strip():
            return val.strip()

    analysis = row.get("analysis")
    if isinstance(analysis, dict):
        for key in ("solution", "solution_text"):
            val = analysis.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


def normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        q = _extract_question_text(row)
        if not q:
            continue
        s = _extract_solution_text(row)
        rid = str(row.get("id") or f"row_{i:06d}")
        out.append(
            {
                "id": rid,
                "question_text": q,
                "solution_text": s,
                "topic": row.get("topic"),
                "difficulty": row.get("difficulty"),
            }
        )
    return out


def build_plain_solution_prompt(question_text: str, max_q_chars: int) -> str:
    q = _truncate(question_text, max_q_chars)
    return f"""Question:
{q}

Task:
Write a plain-language solution with key steps and give the most likely answer letter.

Return strict JSON with this schema:
{{
  "solution_text": "<string>",
  "answer_guess": "A|B|C|D|E|UNKNOWN"
}}
"""


def plain_solution_ok(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    if not isinstance(obj.get("solution_text"), str) or not obj.get("solution_text", "").strip():
        return False
    if obj.get("answer_guess") not in {"A", "B", "C", "D", "E", "UNKNOWN"}:
        return False
    return True


def _coerce_answer_guess(value: Any) -> str:
    if isinstance(value, str):
        s = value.strip().upper()
        if s in {"A", "B", "C", "D", "E"}:
            return s
        if s in {"UNKNOWN", "UNK", "N/A", "NA", "NONE", ""}:
            return "UNKNOWN"
        if len(s) >= 1 and s[0] in {"A", "B", "C", "D", "E"}:
            return s[0]
    return "UNKNOWN"


def coerce_plain_solution_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Plain-solution payload is not a JSON object")

    # Accept common field aliases from model drift.
    raw_solution = payload.get("solution_text")
    if not isinstance(raw_solution, str) or not raw_solution.strip():
        for key in ("solution", "reasoning_summary", "explanation", "rationale"):
            v = payload.get(key)
            if isinstance(v, str) and v.strip():
                raw_solution = v
                break

    if not isinstance(raw_solution, str) or not raw_solution.strip():
        solve_attempt = payload.get("solve_attempt")
        if isinstance(solve_attempt, dict):
            v = solve_attempt.get("reasoning_summary")
            if isinstance(v, str) and v.strip():
                raw_solution = v

    if not isinstance(raw_solution, str) or not raw_solution.strip():
        raise ValueError("Plain-solution payload missing usable solution text")

    answer_guess = payload.get("answer_guess")
    if answer_guess is None and isinstance(payload.get("answer"), str):
        answer_guess = payload.get("answer")
    if answer_guess is None:
        solve_attempt = payload.get("solve_attempt")
        if isinstance(solve_attempt, dict):
            answer_guess = solve_attempt.get("selected_answer")

    return {
        "solution_text": raw_solution.strip(),
        "answer_guess": _coerce_answer_guess(answer_guess),
    }


def llm_plain_solution_with_retries(
    model: str,
    prompt: str,
    retries: int,
    temperature: float,
) -> Dict[str, Any]:
    last_exc: Optional[Exception] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            return generate_json(model, prompt, system=SYSTEM_PLAIN_SOLUTION, temperature=temperature)
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(0.6 * attempt)
    assert last_exc is not None
    raise last_exc


def fill_missing_plain_solutions(
    rows: List[Dict[str, Any]],
    *,
    model: str,
    retries: int,
    temperature: float,
    max_q_chars: int,
    debug: bool,
    checkpoint_path: str = "",
    checkpoint_every: int = 10,
) -> tuple[List[Dict[str, Any]], int]:
    updated: List[Dict[str, Any]] = []
    state_rows: List[Dict[str, Any]] = [dict(r) for r in rows]
    synthesized = 0
    for idx, row in enumerate(rows, start=1):
        out = dict(row)
        if str(out.get("solution_text") or "").strip():
            out["solution_text"] = str(out.get("solution_text") or "").strip()
            updated.append(out)
            state_rows[idx - 1] = out
            continue

        prompt = build_plain_solution_prompt(out["question_text"], max_q_chars)
        payload = llm_plain_solution_with_retries(model, prompt, retries=retries, temperature=temperature)
        coerced = coerce_plain_solution_payload(payload)
        if not plain_solution_ok(coerced):
            raise ValueError(f"Plain-solution schema failed for row={out.get('id')}")
        out["solution_text"] = str(coerced["solution_text"]).strip()
        out["answer_guess"] = coerced.get("answer_guess", "UNKNOWN")
        updated.append(out)
        state_rows[idx - 1] = out
        synthesized += 1
        if checkpoint_path and checkpoint_every > 0 and (synthesized % checkpoint_every == 0):
            write_jsonl(checkpoint_path, state_rows)
            if debug:
                print(
                    f"[plain-solutions] checkpoint rows={len(state_rows)} synthesized={synthesized} -> {checkpoint_path}",
                    flush=True,
                )
        if debug and (synthesized % 10 == 0 or idx == len(rows)):
            print(f"[plain-solutions] synthesized={synthesized}/{len(rows)}", flush=True)
    return state_rows, synthesized


def merge_solution_text_by_id(
    base_rows: List[Dict[str, Any]],
    resume_rows: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], int]:
    resume_by_id: Dict[str, Dict[str, Any]] = {}
    for r in resume_rows:
        rid = str(r.get("id") or "").strip()
        if rid:
            resume_by_id[rid] = r

    merged: List[Dict[str, Any]] = []
    reused = 0
    for row in base_rows:
        out = dict(row)
        rid = str(out.get("id") or "").strip()
        existing = str(out.get("solution_text") or "").strip()
        if (not existing) and rid in resume_by_id:
            cand = str((resume_by_id[rid] or {}).get("solution_text") or "").strip()
            if cand:
                out["solution_text"] = cand
                ans = (resume_by_id[rid] or {}).get("answer_guess")
                if ans is not None:
                    out["answer_guess"] = ans
                reused += 1
        merged.append(out)
    return merged, reused


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / denom


def mmr_select(query: np.ndarray, cand_idx: List[int], mat: np.ndarray, k: int, lambda_mult: float) -> List[int]:
    if not cand_idx or k <= 0:
        return []
    selected: List[int] = []
    cand_set = set(cand_idx)
    sims_q = {i: float(mat[i] @ query) for i in cand_idx}
    while len(selected) < k and cand_set:
        if not selected:
            best = max(cand_set, key=lambda i: sims_q[i])
            selected.append(best)
            cand_set.remove(best)
            continue

        def score(i: int) -> float:
            sim_q = sims_q[i]
            sim_sel = max(float(mat[i] @ mat[j]) for j in selected)
            return lambda_mult * sim_q - (1.0 - lambda_mult) * sim_sel

        best = max(cand_set, key=score)
        selected.append(best)
        cand_set.remove(best)
    return selected


def compute_anchor_indices(mat: np.ndarray, anchor_frac: float, k_density: int) -> List[int]:
    n = int(mat.shape[0])
    if n == 0:
        return []
    k = max(1, min(int(k_density), n - 1 if n > 1 else 1))
    sims = mat @ mat.T
    densities: List[tuple[float, int]] = []
    for i in range(n):
        row = sims[i]
        vals = np.partition(row, -k)[-k:] if n > 1 else row
        densities.append((float(np.mean(vals)), i))
    densities.sort(reverse=True)
    take = max(1, int(round(n * max(0.01, min(anchor_frac, 1.0)))))
    return [idx for _, idx in densities[:take]]


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "..."


def build_prompt(bundle: Dict[str, Any], max_q_chars: int, max_s_chars: int) -> str:
    seed = bundle.get("seed", {})
    q_examples = bundle.get("question_exemplars", [])
    s_examples = bundle.get("solution_exemplars", [])
    paired = bundle.get("paired_exemplars", [])

    q_blocks = []
    for i, ex in enumerate(q_examples, start=1):
        q_blocks.append(f"QUESTION EXEMPLAR {i}:\n{_truncate(ex.get('question_text', ''), max_q_chars)}")
    q_text = "\n\n".join(q_blocks) if q_blocks else "(none)"

    s_blocks = []
    for i, ex in enumerate(s_examples, start=1):
        s_blocks.append(f"SOLUTION EXEMPLAR {i}:\n{_truncate(ex.get('solution_text', ''), max_s_chars)}")
    s_text = "\n\n".join(s_blocks) if s_blocks else "(none)"

    p_blocks = []
    for i, ex in enumerate(paired, start=1):
        p_blocks.append(
            f"PAIRED EXEMPLAR {i} QUESTION:\n{_truncate(ex.get('question_text', ''), max_q_chars)}\n\n"
            f"PAIRED EXEMPLAR {i} SOLUTION:\n{_truncate(ex.get('solution_text', ''), max_s_chars)}"
        )
    p_text = "\n\n".join(p_blocks) if p_blocks else "(none)"

    return f"""You are given retrieved raw examples from solved problems.

SEED QUESTION:
{_truncate(seed.get('question_text', ''), max_q_chars)}

QUESTION EXEMPLARS (style/topic priors only; do not copy):
{q_text}

SOLUTION EXEMPLARS (reasoning pattern priors only; do not copy):
{s_text}

PAIRED QUESTION+SOLUTION EXEMPLARS (mapping priors only; do not copy):
{p_text}

TASK:
Generate ONE new hard contest-style multiple-choice problem with 5 choices (A)-(E), a correct answer, and a concise but complete solution.
Use the retrieved examples only as weak priors for style/reasoning profile.
Do not copy scenario details, wording, variable names, or numbers.

Return strict JSON with this exact schema:
{{
  "id": "<string>",
  "question": "<string>",
  "choices": {{"A":"...", "B":"...", "C":"...", "D":"...", "E":"..."}},
  "answer": "A|B|C|D|E",
  "solution": "<string>",
  "anti_copy_report": {{
    "novel_scenario_summary": "<1-2 sentences>",
    "differences_from_exemplars": ["...", "..."],
    "possible_overlap_risks": ["...", "..."]
  }}
}}
"""


def build_solve_first_prompt(bundle: Dict[str, Any], max_q_chars: int, max_s_chars: int) -> str:
    seed = bundle.get("seed", {})
    q_examples = bundle.get("question_exemplars", [])
    s_examples = bundle.get("solution_exemplars", [])
    paired = bundle.get("paired_exemplars", [])

    q_blocks = []
    for i, ex in enumerate(q_examples, start=1):
        q_blocks.append(f"QUESTION EXEMPLAR {i}:\n{_truncate(ex.get('question_text', ''), max_q_chars)}")
    q_text = "\n\n".join(q_blocks) if q_blocks else "(none)"

    s_blocks = []
    for i, ex in enumerate(s_examples, start=1):
        s_blocks.append(f"SOLUTION EXEMPLAR {i}:\n{_truncate(ex.get('solution_text', ''), max_s_chars)}")
    s_text = "\n\n".join(s_blocks) if s_blocks else "(none)"

    p_blocks = []
    for i, ex in enumerate(paired, start=1):
        p_blocks.append(
            f"PAIRED EXEMPLAR {i} QUESTION:\n{_truncate(ex.get('question_text', ''), max_q_chars)}\n\n"
            f"PAIRED EXEMPLAR {i} SOLUTION:\n{_truncate(ex.get('solution_text', ''), max_s_chars)}"
        )
    p_text = "\n\n".join(p_blocks) if p_blocks else "(none)"

    return f"""You are given retrieved raw examples from solved problems.

SEED QUESTION:
{_truncate(seed.get('question_text', ''), max_q_chars)}

QUESTION EXEMPLARS (style/topic priors only; do not copy):
{q_text}

SOLUTION EXEMPLARS (reasoning pattern priors only; do not copy):
{s_text}

PAIRED QUESTION+SOLUTION EXEMPLARS (mapping priors only; do not copy):
{p_text}

TASK:
Generate ONLY the solved core first (no final question wording yet).

Return strict JSON with this exact schema:
{{
  "id": "<string>",
  "choices": {{"A":"...", "B":"...", "C":"...", "D":"...", "E":"..."}},
  "answer": "A|B|C|D|E",
  "solution": "<string>",
  "anti_copy_report": {{
    "novel_scenario_summary": "<1-2 sentences>",
    "differences_from_exemplars": ["...", "..."],
    "possible_overlap_risks": ["...", "..."]
  }}
}}
"""


def build_question_from_solution_prompt(solved: Dict[str, Any]) -> str:
    return f"""You are given a solved MCQ core. Write the final question statement.

SOLVED CORE JSON:
{json.dumps(solved, ensure_ascii=False)}

TASK:
Write the final contest-style question that matches this solved core.
Do not change the choices or answer.

Return strict JSON with this exact schema:
{{
  "id": "{solved.get('id', '')}",
  "question": "<string>",
  "choices": {{"A":"...", "B":"...", "C":"...", "D":"...", "E":"..."}},
  "answer": "A|B|C|D|E",
  "solution": "<string>",
  "anti_copy_report": {{
    "novel_scenario_summary": "<1-2 sentences>",
    "differences_from_exemplars": ["...", "..."],
    "possible_overlap_risks": ["...", "..."]
  }}
}}
"""


def llm_json_with_retries(
    model: str,
    prompt: str,
    retries: int,
    temperature: float,
    *,
    system: str = SYSTEM_GEN,
) -> Dict[str, Any]:
    last_exc: Optional[Exception] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            return generate_json(model, prompt, system=system, temperature=temperature)
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(0.6 * attempt)
    assert last_exc is not None
    raise last_exc


def schema_ok(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    if not isinstance(obj.get("question"), str) or not obj.get("question", "").strip():
        return False
    if obj.get("answer") not in {"A", "B", "C", "D", "E"}:
        return False
    choices = obj.get("choices")
    if not isinstance(choices, dict) or set(choices.keys()) != {"A", "B", "C", "D", "E"}:
        return False
    if not isinstance(obj.get("solution"), str) or not obj.get("solution", "").strip():
        return False
    return True


def solved_schema_ok(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    if obj.get("answer") not in {"A", "B", "C", "D", "E"}:
        return False
    choices = obj.get("choices")
    if not isinstance(choices, dict) or set(choices.keys()) != {"A", "B", "C", "D", "E"}:
        return False
    if not isinstance(obj.get("solution"), str) or not obj.get("solution", "").strip():
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Schema-free all-in-one ablation: raw-text retrieval + generation.")
    ap.add_argument("--input", required=True, help="Raw solved-problem corpus JSONL (e.g., scored/*.jsonl)")
    ap.add_argument("--out", default="ablation_3/generated/no_schema_generated.jsonl")
    ap.add_argument("--bundles-out", default="ablation_3/retrieval_bundles.jsonl", help="Retrieval bundle audit output JSONL")
    ap.add_argument("--plain-solutions-out", default="ablation_3/plain_solutions.jsonl", help="Full corpus with plain solution text.")
    ap.add_argument("--only-plain-solutions", action="store_true", help="Stop after writing plain-solutions corpus.")
    ap.add_argument("--resume-solutions-from", default="", help="Optional JSONL with previously generated solution_text to merge by id before synthesis.")
    ap.add_argument("--question-embeddings-out", default="ablation_3/question_embedded.jsonl")
    ap.add_argument("--solution-embeddings-out", default="ablation_3/solution_embedded.jsonl")
    ap.add_argument("--embed-model", default="qwen3-embedding")
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--model-solve", default="", help="Optional override for solve-first stage.")
    ap.add_argument("--model-question", default="", help="Optional override for question-from-solution stage.")
    ap.add_argument("--solved-drafts-out", default="ablation_3/generated/solved_drafts.jsonl")
    ap.add_argument("--plain-solution-model", default="qwen2.5:7b-instruct")
    ap.add_argument("--skip-plain-solutions", action="store_true", help="Do not synthesize missing solution_text rows.")
    ap.add_argument("--generation-mode", choices=["direct", "solution_first"], default="solution_first")
    ap.add_argument("--num_bundles", type=int, default=25)
    ap.add_argument("--anchor_frac", type=float, default=0.18)
    ap.add_argument("--k_density", type=int, default=20)
    ap.add_argument("--annulus_min", type=float, default=0.08)
    ap.add_argument("--annulus_max", type=float, default=0.40)
    ap.add_argument("--tail_frac", type=float, default=0.35)
    ap.add_argument("--tail_min", type=float, default=0.30)
    ap.add_argument("--k_solution", type=int, default=4)
    ap.add_argument("--k_question", type=int, default=4)
    ap.add_argument("--k_paired", type=int, default=3)
    ap.add_argument("--mmr_lambda_solution", type=float, default=0.7)
    ap.add_argument("--mmr_lambda_question", type=float, default=0.7)
    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--temperature_solve", type=float, default=0.25)
    ap.add_argument("--temperature_question", type=float, default=0.2)
    ap.add_argument("--json_retries", type=int, default=3)
    ap.add_argument("--plain_json_retries", type=int, default=3)
    ap.add_argument("--plain-checkpoint-every", type=int, default=10, help="Write plain-solutions checkpoint every N synthesized rows (0 to disable).")
    ap.add_argument("--max_q_chars", type=int, default=1400)
    ap.add_argument("--max_s_chars", type=int, default=1400)
    ap.add_argument("--plain_temperature", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=0, help="If > 0, only process the first N usable rows before embedding.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    model_solve = args.model_solve or args.model
    model_question = args.model_question or args.model

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    rows = normalize_rows(read_jsonl(args.input))
    if not rows:
        raise ValueError("No usable rows found. Need rows with question text.")
    if args.limit > 0:
        rows = rows[: args.limit]
        if args.debug:
            print(f"[limit] using first {len(rows)} usable rows")

    if args.resume_solutions_from:
        resume_rows = normalize_rows(read_jsonl(args.resume_solutions_from))
        rows, reused = merge_solution_text_by_id(rows, resume_rows)
        if args.debug:
            print(f"[plain-solutions] reused existing solution_text for {reused} rows from {args.resume_solutions_from}")

    missing_before = sum(1 for r in rows if not str(r.get("solution_text") or "").strip())
    if args.skip_plain_solutions and missing_before > 0:
        raise ValueError(
            f"--skip-plain-solutions was set, but {missing_before} rows are missing solution_text. "
            "Remove --skip-plain-solutions (recommended) to synthesize missing solutions."
        )
    if not args.skip_plain_solutions and missing_before > 0:
        if args.debug:
            print(f"[plain-solutions] missing={missing_before}, synthesizing with model={args.plain_solution_model}")
        rows, synthesized = fill_missing_plain_solutions(
            rows,
            model=args.plain_solution_model,
            retries=args.plain_json_retries,
            temperature=args.plain_temperature,
            max_q_chars=args.max_q_chars,
            debug=args.debug,
            checkpoint_path=args.plain_solutions_out,
            checkpoint_every=args.plain_checkpoint_every,
        )
        if args.debug:
            print(f"[plain-solutions] done synthesized={synthesized}")
    missing_after = sum(1 for r in rows if not str(r.get("solution_text") or "").strip())
    if missing_after > 0:
        raise RuntimeError(
            f"solution_text still missing for {missing_after} rows after plain-solution stage. "
            "Ensure model endpoint is reachable and rerun without --skip-plain-solutions."
        )
    write_jsonl(args.plain_solutions_out, rows)
    if args.debug:
        print(f"[write] plain solutions corpus={len(rows)} -> {args.plain_solutions_out}")
    if args.only_plain_solutions:
        print(f"Wrote {len(rows)} rows with non-empty solution_text -> {args.plain_solutions_out}")
        return

    ids = [r["id"] for r in rows]
    q_texts = [r["question_text"] for r in rows]
    s_texts = [r["solution_text"] or r["question_text"] for r in rows]

    if args.debug:
        print(f"[load] usable rows={len(rows)}")
        print(f"[embed] model={args.embed_model} question+solution spaces")
    q_mat = l2_normalize(np.asarray(embed_texts(args.embed_model, q_texts), dtype=np.float32))
    s_mat = l2_normalize(np.asarray(embed_texts(args.embed_model, s_texts), dtype=np.float32))
    q_embed_rows = [
        {"id": rows[i]["id"], "question_text": rows[i]["question_text"], "embedding": q_mat[i].astype(float).tolist()}
        for i in range(len(rows))
    ]
    s_embed_rows = [
        {"id": rows[i]["id"], "solution_text": rows[i]["solution_text"], "embedding": s_mat[i].astype(float).tolist()}
        for i in range(len(rows))
    ]
    write_jsonl(args.question_embeddings_out, q_embed_rows)
    write_jsonl(args.solution_embeddings_out, s_embed_rows)
    if args.debug:
        print(f"[write] question embeddings={len(q_embed_rows)} -> {args.question_embeddings_out}")
        print(f"[write] solution embeddings={len(s_embed_rows)} -> {args.solution_embeddings_out}")

    anchor_idx = compute_anchor_indices(q_mat, args.anchor_frac, args.k_density)
    if not anchor_idx:
        raise ValueError("Failed to compute anchors.")

    bundles: List[Dict[str, Any]] = []
    used: set[str] = set()
    attempts = 0
    max_attempts = max(100, args.num_bundles * 50)

    while len(bundles) < args.num_bundles and attempts < max_attempts:
        attempts += 1
        a_idx = random.choice(anchor_idx)
        a_id = ids[a_idx]
        a_vec = q_mat[a_idx]
        sims = q_mat @ a_vec
        dists = 1.0 - sims
        mode = "tail" if random.random() < args.tail_frac else "annulus"
        mask = (dists >= args.tail_min) if mode == "tail" else ((dists >= args.annulus_min) & (dists <= args.annulus_max))
        cand_idx = np.where(mask)[0].tolist()
        cand_idx = [i for i in cand_idx if ids[i] != a_id]
        if not cand_idx:
            cand_idx = [i for i in np.argsort(dists).tolist() if ids[i] != a_id][:200]
        if not cand_idx:
            continue

        seed_idx = random.choice(cand_idx)
        seed_id = ids[seed_idx]
        if seed_id in used:
            continue
        used.add(seed_id)

        q_query = q_mat[seed_idx]
        q_rank = [i for i in np.argsort(-(q_mat @ q_query)).tolist() if ids[i] != seed_id][:240]
        q_pick = mmr_select(q_query, q_rank, q_mat, args.k_question, args.mmr_lambda_question)

        s_query = s_mat[seed_idx]
        s_rank = [i for i in np.argsort(-(s_mat @ s_query)).tolist() if ids[i] != seed_id][:240]
        s_pick = mmr_select(s_query, s_rank, s_mat, args.k_solution, args.mmr_lambda_solution)

        paired = []
        for i in q_pick[: args.k_paired]:
            paired.append(
                {
                    "id": ids[i],
                    "question_text": rows[i]["question_text"],
                    "solution_text": rows[i]["solution_text"],
                }
            )

        bundles.append(
            {
                "bundle_id": f"bundle_{len(bundles)+1:05d}",
                "mode": mode,
                "anchor_id": a_id,
                "seed_id": seed_id,
                "anchor_distance": float(dists[seed_idx]),
                "seed": {"id": seed_id, "question_text": rows[seed_idx]["question_text"], "solution_text": rows[seed_idx]["solution_text"]},
                "question_exemplars": [{"id": ids[i], "question_text": rows[i]["question_text"]} for i in q_pick],
                "solution_exemplars": [{"id": ids[i], "solution_text": rows[i]["solution_text"]} for i in s_pick],
                "paired_exemplars": paired,
            }
        )

    if not bundles:
        raise ValueError("Failed to build any bundles.")

    if args.bundles_out:
        write_jsonl(args.bundles_out, bundles)
        if args.debug:
            print(f"[write] bundles={len(bundles)} -> {args.bundles_out}")

    out_rows: List[Dict[str, Any]] = []
    solved_rows: List[Dict[str, Any]] = []
    failures = 0
    for i, bundle in enumerate(bundles, start=1):
        try:
            if args.generation_mode == "solution_first":
                solve_prompt = build_solve_first_prompt(bundle, args.max_q_chars, args.max_s_chars)
                solved = llm_json_with_retries(
                    model_solve,
                    solve_prompt,
                    retries=args.json_retries,
                    temperature=args.temperature_solve,
                    system=SYSTEM_SOLVE_FIRST,
                )
                if not solved_schema_ok(solved):
                    raise ValueError("Solve-first JSON failed schema checks.")
                solved["id"] = solved.get("id") or f"ablation_{i:05d}"
                solved_rows.append(
                    {
                        **solved,
                        "_meta": {
                            "phase": "solve_first",
                            "bundle_id": bundle["bundle_id"],
                            "seed_id": bundle["seed_id"],
                            "anchor_id": bundle["anchor_id"],
                            "mode": bundle["mode"],
                        },
                    }
                )

                question_prompt = build_question_from_solution_prompt(solved)
                cand = llm_json_with_retries(
                    model_question,
                    question_prompt,
                    retries=args.json_retries,
                    temperature=args.temperature_question,
                    system=SYSTEM_QUESTION_FROM_SOLUTION,
                )
                if not schema_ok(cand):
                    raise ValueError("Question-from-solution JSON failed schema checks.")
                out_rows.append(
                    {
                        **cand,
                        "id": cand.get("id") or solved["id"],
                        "_meta": {
                            "generator": "all_in_one_no_schema_solution_first",
                            "bundle_id": bundle["bundle_id"],
                            "seed_id": bundle["seed_id"],
                            "anchor_id": bundle["anchor_id"],
                            "mode": bundle["mode"],
                            "source_input": args.input,
                            "model_solve": model_solve,
                            "model_question": model_question,
                        },
                    }
                )
            else:
                prompt = build_prompt(bundle, args.max_q_chars, args.max_s_chars)
                cand = llm_json_with_retries(
                    args.model,
                    prompt,
                    retries=args.json_retries,
                    temperature=args.temperature,
                    system=SYSTEM_GEN,
                )
                if not schema_ok(cand):
                    raise ValueError("Generated JSON failed basic schema checks.")
                out_rows.append(
                    {
                        **cand,
                        "id": cand.get("id") or f"ablation_{i:05d}",
                        "_meta": {
                            "generator": "all_in_one_no_schema_raw_text",
                            "bundle_id": bundle["bundle_id"],
                            "seed_id": bundle["seed_id"],
                            "anchor_id": bundle["anchor_id"],
                            "mode": bundle["mode"],
                            "source_input": args.input,
                        },
                    }
                )
            if args.debug:
                print(f"[gen {i}/{len(bundles)}] PASS bundle={bundle['bundle_id']}")
        except Exception as exc:
            failures += 1
            if args.debug:
                print(f"[gen {i}/{len(bundles)}] FAIL bundle={bundle['bundle_id']} err={exc}")

    write_jsonl(args.solved_drafts_out, solved_rows)
    write_jsonl(args.out, out_rows)
    print(
        f"Wrote {len(out_rows)} generated rows -> {args.out} "
        f"(solved_drafts={len(solved_rows)} -> {args.solved_drafts_out}, failures={failures})"
    )


if __name__ == "__main__":
    main()
