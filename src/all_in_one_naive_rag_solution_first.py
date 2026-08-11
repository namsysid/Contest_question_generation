#!/usr/bin/env python3
"""
Naive all-in-one RAG over question text only, with solution-first generation.

Pipeline:
1) Read question corpus JSONL and extract question text.
2) Embed all question texts.
3) For each sampled seed question, retrieve top-k nearest question exemplars.
4) Generate a solved item first (solution + answer + choices).
5) Generate the final question conditioned on the solved item.
6) Write generated rows and intermediate solved drafts to JSONL.
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

SYSTEM_SOLVE_FIRST = """You create a novel contest-style solved multiple-choice item from retrieved question-only exemplars.

Hard constraints:
- Do NOT copy exemplar wording, numbers, variable names, or scenarios.
- First design the solved core: reasoning path, 5 options, and one correct option.
- Keep it coherent and self-contained.
- Return strict JSON only (no markdown).
"""

SYSTEM_QUESTION_FROM_SOLUTION = """You convert a solved core into a clear contest-style question.

Hard constraints:
- Keep the provided choices exactly as-is.
- Keep the provided answer exactly as-is.
- Keep the solution consistent with the final question.
- Do NOT return markdown.
- Return strict JSON only.
"""


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _choices_to_text(choices: Any) -> str:
    if isinstance(choices, dict):
        out: List[str] = []
        for key in ("A", "B", "C", "D", "E"):
            val = str(choices.get(key) or "").strip()
            if val:
                out.append(f"({key}) {val}")
        return "\n".join(out)
    if isinstance(choices, list):
        out: List[str] = []
        for i, c in enumerate(choices):
            label = chr(ord("A") + i) if i < 5 else str(i + 1)
            txt = str(c).strip()
            if txt:
                out.append(f"({label}) {txt}")
        return "\n".join(out)
    return ""


def extract_question_text(row: Dict[str, Any]) -> str:
    for key in ("question_text", "question", "prompt", "stem"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    problem = row.get("problem")
    if isinstance(problem, dict):
        stem = str(problem.get("stem") or "").strip()
        choices = _choices_to_text(problem.get("choices"))
        if stem and choices:
            return f"{stem}\n\n{choices}"
        return stem
    return ""


def normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for i, row in enumerate(rows, start=1):
        q = extract_question_text(row)
        if not q:
            continue
        out.append({"id": str(row.get("id") or f"row_{i:06d}"), "question_text": q})
    return out


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / denom


def top_k_neighbors(mat: np.ndarray, query_idx: int, k: int) -> List[int]:
    sims = mat @ mat[query_idx]
    order = np.argsort(-sims).tolist()
    out = [i for i in order if i != query_idx]
    return out[: max(0, k)]


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "..."


def build_solve_first_prompt(seed: Dict[str, str], exemplars: List[Dict[str, str]], max_chars: int) -> str:
    ex_text = "\n\n".join(
        f"EXEMPLAR {i}:\n{_truncate(ex['question_text'], max_chars)}"
        for i, ex in enumerate(exemplars, start=1)
    )
    if not ex_text:
        ex_text = "(none)"

    return f"""You are given a seed question and nearest retrieved question exemplars.

SEED QUESTION:
{_truncate(seed['question_text'], max_chars)}

RETRIEVED QUESTION EXEMPLARS (style priors only; do not copy):
{ex_text}

TASK:
Create one NEW solved contest-style MCQ core before writing the final question wording.
Design:
- a valid reasoning path,
- exactly five choices A-E,
- exactly one correct answer letter,
- a concise complete solution.

Return strict JSON with this exact schema:
{{
  "id": "<string>",
  "choices": {{"A":"...", "B":"...", "C":"...", "D":"...", "E":"..."}},
  "answer": "A|B|C|D|E",
  "solution": "<string>",
  "solution_outline": ["<step 1>", "<step 2>"]
}}
"""


def build_question_from_solution_prompt(solved: Dict[str, Any]) -> str:
    return f"""You are given a solved MCQ core. Write the final question statement that matches it.

SOLVED CORE JSON:
{json.dumps(solved, ensure_ascii=False)}

TASK:
Write the final multiple-choice question so the given choices, answer, and solution are all consistent.
Do not alter the choices or answer.

Return strict JSON with this exact schema:
{{
  "id": "{solved.get('id', '')}",
  "question": "<string>",
  "choices": {{"A":"...", "B":"...", "C":"...", "D":"...", "E":"..."}},
  "answer": "A|B|C|D|E",
  "solution": "<string>"
}}
"""


def solved_schema_ok(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    choices = obj.get("choices")
    if not isinstance(choices, dict) or set(choices.keys()) != {"A", "B", "C", "D", "E"}:
        return False
    if obj.get("answer") not in {"A", "B", "C", "D", "E"}:
        return False
    if not isinstance(obj.get("solution"), str) or not obj.get("solution", "").strip():
        return False
    return True


def final_schema_ok(obj: Dict[str, Any]) -> bool:
    if not solved_schema_ok(obj):
        return False
    if not isinstance(obj.get("question"), str) or not obj.get("question", "").strip():
        return False
    return True


def llm_json_with_retries(
    model: str,
    prompt: str,
    *,
    system: str,
    retries: int,
    temperature: float,
) -> Dict[str, Any]:
    last_exc: Optional[Exception] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            return generate_json(model, prompt, system=system, temperature=temperature)
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(0.7 * attempt)
    assert last_exc is not None
    raise last_exc


def main() -> None:
    ap = argparse.ArgumentParser(description="All-in-one naive RAG with solution-first generation.")
    ap.add_argument("--input", default="data/txts/all_questions.jsonl")
    ap.add_argument("--out", default="ablation_2/generated/naive_rag_solution_first_questions.jsonl")
    ap.add_argument("--bundles-out", default="ablation_2/retrieval_bundles.jsonl")
    ap.add_argument("--embeddings-out", default="ablation_2/question_embedded.jsonl")
    ap.add_argument("--solved-drafts-out", default="ablation_2/generated/solved_drafts.jsonl")
    ap.add_argument("--embed-model", default="qwen3-embedding")
    ap.add_argument("--model-solve", default="qwen2.5:7b-instruct")
    ap.add_argument("--model-question", default="qwen2.5:7b-instruct")
    ap.add_argument("--num", type=int, default=25)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temperature-solve", type=float, default=0.25)
    ap.add_argument("--temperature-question", type=float, default=0.2)
    ap.add_argument("--json-retries", type=int, default=3)
    ap.add_argument("--max-chars", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    rows = normalize_rows(read_jsonl(args.input))
    if not rows:
        raise ValueError("No usable question rows found in input.")

    texts = [r["question_text"] for r in rows]
    if args.debug:
        print(f"[load] usable rows={len(rows)}")
        print(f"[embed] model={args.embed_model}")
    mat = l2_normalize(np.asarray(embed_texts(args.embed_model, texts), dtype=np.float32))

    embedding_rows = [
        {
            "id": rows[i]["id"],
            "question_text": rows[i]["question_text"],
            "embedding": mat[i].astype(float).tolist(),
        }
        for i in range(len(rows))
    ]
    write_jsonl(args.embeddings_out, embedding_rows)
    if args.debug:
        print(f"[write] embeddings={len(embedding_rows)} -> {args.embeddings_out}")

    n = len(rows)
    target = min(max(1, args.num), n)
    seed_indices = random.sample(range(n), k=target) if target < n else list(range(n))

    bundles: List[Dict[str, Any]] = []
    for i, sidx in enumerate(seed_indices, start=1):
        nn = top_k_neighbors(mat, sidx, args.k)
        bundles.append(
            {
                "bundle_id": f"bundle_{i:05d}",
                "seed": rows[sidx],
                "retrieved_exemplars": [rows[j] for j in nn],
            }
        )

    write_jsonl(args.bundles_out, bundles)
    if args.debug:
        print(f"[write] bundles={len(bundles)} -> {args.bundles_out}")

    solved_drafts: List[Dict[str, Any]] = []
    out_rows: List[Dict[str, Any]] = []
    failures = 0

    for i, bundle in enumerate(bundles, start=1):
        try:
            solve_prompt = build_solve_first_prompt(bundle["seed"], bundle["retrieved_exemplars"], args.max_chars)
            solved = llm_json_with_retries(
                args.model_solve,
                solve_prompt,
                system=SYSTEM_SOLVE_FIRST,
                retries=args.json_retries,
                temperature=args.temperature_solve,
            )
            if not solved_schema_ok(solved):
                raise ValueError("Solve-first JSON failed schema checks.")

            solved["id"] = solved.get("id") or f"naive_rag_sf_{i:05d}"
            solved_drafts.append(
                {
                    **solved,
                    "_meta": {
                        "phase": "solve_first",
                        "bundle_id": bundle["bundle_id"],
                        "seed_id": bundle["seed"]["id"],
                        "retrieved_ids": [ex["id"] for ex in bundle["retrieved_exemplars"]],
                    },
                }
            )

            question_prompt = build_question_from_solution_prompt(solved)
            final_item = llm_json_with_retries(
                args.model_question,
                question_prompt,
                system=SYSTEM_QUESTION_FROM_SOLUTION,
                retries=args.json_retries,
                temperature=args.temperature_question,
            )
            if not final_schema_ok(final_item):
                raise ValueError("Question-from-solution JSON failed schema checks.")

            out_rows.append(
                {
                    **final_item,
                    "id": final_item.get("id") or solved["id"],
                    "_meta": {
                        "generator": "all_in_one_naive_rag_solution_first",
                        "bundle_id": bundle["bundle_id"],
                        "seed_id": bundle["seed"]["id"],
                        "retrieved_ids": [ex["id"] for ex in bundle["retrieved_exemplars"]],
                        "source_input": args.input,
                        "model_solve": args.model_solve,
                        "model_question": args.model_question,
                    },
                }
            )
            if args.debug:
                print(f"[gen {i}/{len(bundles)}] PASS bundle={bundle['bundle_id']}")
        except Exception as exc:
            failures += 1
            if args.debug:
                print(f"[gen {i}/{len(bundles)}] FAIL bundle={bundle['bundle_id']} err={exc}")

    write_jsonl(args.solved_drafts_out, solved_drafts)
    write_jsonl(args.out, out_rows)
    print(
        f"Wrote {len(out_rows)} generated rows -> {args.out} "
        f"(solved_drafts={len(solved_drafts)} -> {args.solved_drafts_out}, failures={failures})"
    )


if __name__ == "__main__":
    main()
