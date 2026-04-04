#!/usr/bin/env python3
"""
06_verify_and_score.py  (QUESTION-ONLY SOLVE + DIFFICULTY SCORING)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, List

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv() -> None:  # type: ignore[no-redef]
        return

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()

SYSTEM_TEMPLATE = """You are an expert evaluator of Olympiad-style STEM multiple-choice problems.

You will receive a single problem JSON with only the rendered question and answer choices.
You also receive exemplars of real __COMPETITION_LABEL__ problems for style calibration.

Your task:
- Ignore any hidden metadata, graph structure, or supplied solution.
- Try to solve the problem from the question text and answer choices alone.
- If the problem has formatting issues, ambiguity, or a missing piece, do not reject it immediately.
- Instead, solve the problem as written using the most reasonable interpretation, and note the issue.
- Estimate difficulty based on the actual reasoning required to solve the rendered problem.
- Be concise. Keep `reasoning_summary` and all `notes` brief.

Return ONLY strict JSON with keys:
{
  "solve_attempt": {
    "status": "solved"|"partial"|"unclear",
    "selected_answer": "A|B|C|D|E|UNKNOWN",
    "reasoning_summary": "<<=120 words>",
    "issue_notes": ["short note", "..."]
  },
  "difficulty_assessment": {
    "score": 1-5,
    "notes": "<<=60 words>"
  },
  "competition_appropriateness": {
    "depth_reasoning": 1-5,
    "conceptual_richness": 1-5,
    "clarity": 1-5,
    "olympiad_similarity": 1-5,
    "notes": "<<=60 words>"
  }
}
"""


def load_exemplars(paths: Iterable[str], limit: int, max_chars: int) -> List[str]:
    exemplars: List[str] = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if len(exemplars) >= limit:
                        return exemplars
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = (obj.get("question_text") or "").strip()
                    if not text:
                        continue
                    if len(text) > max_chars:
                        continue
                    if "(A)" not in text or "(B)" not in text or "(C)" not in text:
                        continue
                    exemplars.append(text)
        except FileNotFoundError:
            continue
    return exemplars


def build_system_with_exemplars(system_base: str, exemplars: List[str], competition_label: str) -> str:
    if not exemplars:
        return system_base
    chunks = [f"\n\nEXEMPLARS (real {competition_label} problems; use as style reference, not to copy):"]
    for i, ex in enumerate(exemplars, start=1):
        chunks.append(f"\n---\nEXEMPLAR {i}:\n{ex}")
    return system_base + "".join(chunks)


def normalize_choices(choices: Any) -> Dict[str, str]:
    if isinstance(choices, dict):
        out: Dict[str, str] = {}
        for key in ("A", "B", "C", "D", "E"):
            if key in choices:
                out[key] = str(choices[key])
        return out
    if isinstance(choices, list):
        out = {}
        for idx, value in enumerate(choices[:5]):
            out[chr(ord("A") + idx)] = str(value)
        return out
    return {}


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


INVALID_ESCAPE_RE = re.compile(r"\\(?![\"\\/bfnrtu])")


def _loads_with_escape_repair(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        repaired = INVALID_ESCAPE_RE.sub(r"\\\\", text)
        if repaired == text:
            raise exc
        return json.loads(repaired)


def parse_json_text(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Model returned empty content")
    try:
        return _loads_with_escape_repair(raw)
    except json.JSONDecodeError:
        i = raw.find("{")
        j = raw.rfind("}")
        if i < 0 or j < 0 or j <= i:
            raise ValueError(f"Model did not return JSON. Output was:\n{raw}")
        return _loads_with_escape_repair(raw[i : j + 1])


def llm_json(client: Any, model: str, system: str, user: str) -> Dict[str, Any]:
    if client is None:
        raise RuntimeError(
            "OpenAI client not available. Install the `openai` package and set OPENAI_API_KEY."
        )

    last_raw = ""
    prompts = [
        user,
        user
        + "\n\nIMPORTANT: The previous answer may have been truncated. Return the same JSON schema, but with very short field values. Do not show step-by-step derivations.",
    ]
    token_budgets = [1200, 2200]

    for prompt, max_output_tokens in zip(prompts, token_budgets):
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=max_output_tokens,
        )

        text_parts: List[str] = []
        for out in getattr(resp, "output", []) or []:
            if getattr(out, "type", None) != "message":
                continue
            for content in getattr(out, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    text_parts.append(content.text)

        last_raw = "".join(text_parts).strip()
        try:
            return parse_json_text(last_raw)
        except Exception:
            continue

    raise ValueError(f"Model did not return parseable JSON. Output was:\n{last_raw}")


def read_existing_ids(path: str) -> set[str]:
    seen: set[str] = set()
    if not os.path.exists(path):
        return seen
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = obj.get("id")
            if rid is not None:
                seen.add(str(rid))
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="./scored/scored.jsonl")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--subject", default="Physics")
    ap.add_argument("--competition", default="f=ma")
    ap.add_argument("--exemplars", nargs="*", default=["data/text/exam1-2015-1-8.jsonl"])
    ap.add_argument("--exemplar-limit", type=int, default=3)
    ap.add_argument("--exemplar-max-chars", type=int, default=1200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="append and skip ids already present in --out")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    total = len(rows)
    competition_label = " & ".join(p for p in [args.subject, args.competition] if p) or "Olympiad"
    system_base = SYSTEM_TEMPLATE.replace("__COMPETITION_LABEL__", competition_label)
    exemplars = load_exemplars(args.exemplars, args.exemplar_limit, args.exemplar_max_chars)
    system = build_system_with_exemplars(system_base, exemplars, competition_label)
    client = OpenAI() if OpenAI is not None else None
    done_ids = read_existing_ids(args.out) if args.resume else set()
    mode = "a" if args.resume else "w"

    if args.debug:
        action = "append/resume" if args.resume else "open (truncate)"
        print(f"[write] {action}: {args.out}", file=sys.stderr, flush=True)
        if args.resume:
            print(f"[resume] existing ids={len(done_ids)}", file=sys.stderr, flush=True)
    wrote = 0
    skipped = 0
    with open(args.out, mode, encoding="utf-8") as out_f:
        for i, r in enumerate(rows, start=1):
            rid = str(r.get("id", "?"))
            if args.resume and rid in done_ids:
                skipped += 1
                if args.debug:
                    print(f"[{i}/{total}] id={rid} skipped (already scored)", file=sys.stderr, flush=True)
                continue
            if args.debug:
                print(f"[{i}/{total}] id={rid} calling model={args.model}", file=sys.stderr, flush=True)
                t0 = time.time()

            user = "PROBLEM_JSON:\n" + json.dumps({
                "id": r.get("id"),
                "question": r.get("question"),
                "choices": normalize_choices(r.get("choices")),
            }, ensure_ascii=False)

            score = llm_json(client, args.model, system, user)

            if args.debug:
                dt = time.time() - t0
                print(f"[{i}/{total}] id={r.get('id','?')} model done in {dt:.2f}s", file=sys.stderr, flush=True)

            out_row = {
                "id": r.get("id"),
                **score,
                "_meta": r.get("_meta") or {},
            }
            write_jsonl_line(out_f, out_row)
            wrote += 1
            done_ids.add(rid)
            if args.debug:
                print(f"[{i}/{total}] id={rid} wrote", file=sys.stderr, flush=True)

    if args.debug:
        print(f"[write] done: {args.out}", file=sys.stderr, flush=True)
        if args.resume:
            print(f"[resume] skipped {skipped} existing rows", file=sys.stderr, flush=True)
    print(f"Wrote {wrote} scored rows -> {args.out}")

if __name__ == "__main__":
    main()
