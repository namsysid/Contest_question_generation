#!/usr/bin/env python3
"""Naive question-only RAG baseline for validated F=ma mechanics MCQ generation."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from dotenv import load_dotenv

from direct_generation_common import PROBLEM_JSON_SCHEMA, SYSTEM_PROMPT, split_embedded_choices, validate_problem
from ollama_client import embed_texts, generate_json


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        text = str(row.get("question_text") or row.get("question") or "").strip()
        if not text and isinstance(row.get("problem"), dict):
            text = str(row["problem"].get("stem") or "").strip()
        if not text:
            continue
        question, choices = split_embedded_choices(text)
        rendered = question
        if choices:
            rendered += "\n" + "\n".join(f"({key}) {choices[key]}" for key in "ABCDE")
        normalized.append({"id": str(row.get("id") or f"row_{index:06d}"), "question_text": rendered})
    return normalized


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)


def nearest_indices(matrix: np.ndarray, query_index: int, k: int) -> List[int]:
    order = np.argsort(-(matrix @ matrix[query_index])).tolist()
    return [index for index in order if index != query_index][:k]


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def build_prompt(seed: Dict[str, str], exemplars: List[Dict[str, str]], max_chars: int, retry: str) -> str:
    context = "\n\n".join(
        f"RETRIEVED EXEMPLAR {index}:\n{truncate(row['question_text'], max_chars)}"
        for index, row in enumerate(exemplars, start=1)
    )
    retry_text = f"\nA prior output was rejected: {retry}\nCorrect it completely.\n" if retry else ""
    return f"""Generate ONE novel F=ma mechanics multiple-choice problem.

Allowed scope only: kinematics, statics, Newton's laws, momentum and energy, oscillations,
orbital mechanics, rotational dynamics, fluids, dimensional analysis, or elementary data
analysis. Exclude E&M, circuits, optics, waves, thermodynamics, modern physics, relativity,
and pure mathematics. Ignore any out-of-scope corpus noise.

Use the seed and retrieved questions only as topic/style priors. Do not copy their wording,
numbers, variable names, or scenarios. The result must be self-contained, require no missing
diagram, have exactly five plausible choices, exactly one correct answer, and a solution that
verifies the keyed choice.{retry_text}

SEED:
{truncate(seed['question_text'], max_chars)}

{context}

Return strict JSON:
{{
  "id": "<unique string>",
  "domain": "physics",
  "question": "<F=ma mechanics question>",
  "choices": {{"A":"...", "B":"...", "C":"...", "D":"...", "E":"..."}},
  "answer": "A|B|C|D|E",
  "solution": "<derive and verify answer>"
}}
"""


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Naive question-only RAG F=ma baseline")
    parser.add_argument("--input", default="data/txts/all_questions.jsonl")
    parser.add_argument("--out", default="ablation_1/generated/naive_rag_questions_v2.jsonl")
    parser.add_argument("--bundles-out", default="ablation_1/retrieval_bundles_v2.jsonl")
    parser.add_argument("--embeddings-out", default="ablation_1/question_embedded_v2.jsonl")
    parser.add_argument("--embed-model", default="qwen2.5:7b-instruct")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--ollama-base-url", default="http://192.168.50.130:11434")
    parser.add_argument("--num", type=int, default=25)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=1400)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if min(args.num, args.k, args.max_attempts, args.embed_batch_size) <= 0:
        parser.error("num, k, max-attempts, and embed-batch-size must be positive")

    rng = random.Random(args.seed)
    rows = normalize_rows(read_jsonl(args.input))
    if args.limit > 0:
        rows = rows[: args.limit]
    if len(rows) < args.num:
        raise ValueError(f"only {len(rows)} usable rows for requested num={args.num}")

    if args.debug:
        print(f"embedding {len(rows)} rows via {args.ollama_base_url}", file=sys.stderr)
    texts = [row["question_text"] for row in rows]
    vectors: List[List[float]] = []
    for start in range(0, len(texts), args.embed_batch_size):
        batch = texts[start : start + args.embed_batch_size]
        vectors.extend(
            embed_texts(
                args.embed_model,
                batch,
                base_url=args.ollama_base_url,
                timeout=args.timeout_seconds,
            )
        )
        if args.debug:
            print(f"embedded {min(start + len(batch), len(texts))}/{len(texts)}", file=sys.stderr)
    matrix = l2_normalize(np.asarray(vectors, dtype=np.float32))
    write_jsonl(
        args.embeddings_out,
        [{"id": row["id"], "question_text": row["question_text"], "embedding": matrix[i].tolist()} for i, row in enumerate(rows)],
    )

    bundles: List[Dict[str, Any]] = []
    for number, seed_index in enumerate(rng.sample(range(len(rows)), args.num), start=1):
        retrieved = nearest_indices(matrix, seed_index, args.k)
        bundles.append(
            {
                "bundle_id": f"bundle_{number:05d}",
                "seed": rows[seed_index],
                "retrieved_exemplars": [rows[index] for index in retrieved],
            }
        )
    write_jsonl(args.bundles_out, bundles)

    generated: List[Dict[str, Any]] = []
    for number, bundle in enumerate(bundles, start=1):
        retry = ""
        for attempt in range(1, args.max_attempts + 1):
            candidate = generate_json(
                args.model,
                build_prompt(bundle["seed"], bundle["retrieved_exemplars"], args.max_chars, retry),
                system=SYSTEM_PROMPT,
                temperature=args.temperature,
                json_schema=PROBLEM_JSON_SCHEMA,
                base_url=args.ollama_base_url,
                timeout=args.timeout_seconds,
            )
            normalized, errors = validate_problem(candidate)
            if normalized is not None:
                break
            retry = "; ".join(errors)
            if args.debug:
                print(f"[{number}/{len(bundles)}] rejected attempt {attempt}: {retry}", file=sys.stderr)
        else:
            raise RuntimeError(f"bundle {bundle['bundle_id']} failed after {args.max_attempts} attempts: {retry}")

        normalized["id"] = normalized.get("id") or f"naive_rag_{number:05d}"
        normalized["_meta"] = {
            "generator": "all_in_one_naive_rag_questions_v2",
            "seed_id": bundle["seed"]["id"],
            "retrieved_ids": [row["id"] for row in bundle["retrieved_exemplars"]],
            "source_input": args.input,
            "model": args.model,
            "ollama_base_url": args.ollama_base_url,
        }
        generated.append(normalized)
        write_jsonl(args.out, generated)
        if args.debug:
            print(f"[{number}/{len(bundles)}] generated", file=sys.stderr)

    print(f"Wrote {len(generated)} validated naive-RAG questions -> {args.out}")


if __name__ == "__main__":
    main()
