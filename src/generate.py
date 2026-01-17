#!/usr/bin/env python3
"""
generate.py

Reads targets.jsonl produced by 03_retrieve.py and generates new contest-style problems.

Key idea (dual-RAG):
- Target provides the authoritative TARGET SKELETON (skeleton_text + optional structured skeleton)
- Target also provides EXEMPLARS from the question/choice manifold with their compressed solution traces
  to teach surface-form + mapping, while enforcing strong anti-copy constraints.

Output: JSONL with {"id","question","choices","answer","solution","skeleton_text", ...}
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI


SYSTEM = """You generate contest-faithful STEM multiple-choice problems (e.g., F=ma / USNCO style).
Hard constraints:
- Do NOT copy, paraphrase, or minimally edit any exemplar scenario, wording, variable names, or numbers.
- You MUST produce a novel scenario and novel phrasing.
- Stay faithful to the TARGET SKELETON_TEXT's reasoning structure (operators/laws/structure), but you may choose
  new symbols and a new context.
- Provide 5 answer choices (A)-(E) with plausible distractors that are in tension (close, confusable),
  not random far-apart numbers.
- Prefer symbolic quantities where natural for the domain; only instantiate numbers when needed.
- Keep the problem self-contained and solvable without outside references.
Return strict JSON only (no markdown).
"""

def _truncate(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"

def build_user_prompt(target: Dict[str, Any], k_max_exemplars: int = 4) -> str:
    skel = target.get("skeleton_text") or ""
    exemplars: List[Dict[str, Any]] = target.get("exemplars") or []
    exemplars = exemplars[:k_max_exemplars]

    ex_blocks = []
    for j, ex in enumerate(exemplars, start=1):
        qtxt = ex.get("question_text") or ex.get("question") or ""
        stxt = ex.get("skeleton_text") or ""
        ex_blocks.append(
            f"EXEMPLAR {j} QUESTION:\n{_truncate(qtxt, 1600)}\n\n"
            f"EXEMPLAR {j} COMPRESSED SOLUTION TRACE (SKELETON_TEXT):\n{_truncate(stxt, 1400)}\n"
        )

    exemplar_section = "\n\n".join(ex_blocks) if ex_blocks else "(none)"

    return f"""You are given:

1) TARGET SKELETON_TEXT (authoritative reasoning plan):
{skel}

2) EXEMPLARS (for style + mapping ONLY; must not be copied):
{exemplar_section}

TASK:
Generate ONE new multiple-choice problem that is faithful to the TARGET SKELETON_TEXT.
- Invent a NEW scenario and NEW wording. Do not reuse any exemplar scenario elements.
- Use different variable names than any exemplars.
- Ensure the answer choices are confusable (same order of magnitude / near in value / share common symbolic forms).
- Include at least 2 distractors that correspond to common mistakes implied by the skeleton (sign, component swap, missing factor, wrong conservation equation, etc.).
- Keep the problem statement concise.

OUTPUT JSON schema (strict):
{{
  "id": "<string>",
  "question": "<string problem stem>",
  "choices": {{"A":"...", "B":"...", "C":"...", "D":"...", "E":"..."}},
  "answer": "<one of A/B/C/D/E>",
  "solution": "<clear solution, may include equations>",
  "skeleton_text": "<compressed trace you actually used>",
  "anti_copy_report": {{
     "novel_scenario_summary": "<1-2 sentences describing why it's different>",
     "differences_from_closest_exemplar": "<1-3 bullet-ish sentences>",
     "possible_overlap_risks": ["<short phrase>", "..."]
  }}
}}
"""

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="targets.jsonl from 03_retrieve.py")
    ap.add_argument("--out", default="generated_problems.jsonl")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--timeout", type=float, default=90.0, help="seconds per request")
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--max_exemplars", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set (env or .env).")

    client = OpenAI(timeout=args.timeout, max_retries=args.max_retries)

    n_done = 0
    t0 = time.time()

    with open(args.targets, "r", encoding="utf-8") as f_in, open(args.out, "w", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in, start=1):
            if not line.strip():
                continue
            if args.limit and n_done >= args.limit:
                break

            target = json.loads(line)
            target_id = target.get("id") or f"gen_{i}"

            try:
                user_prompt = build_user_prompt(target, k_max_exemplars=args.max_exemplars)

                resp = client.chat.completions.create(
                    model=args.model,
                    temperature=args.temperature,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                )

                content = resp.choices[0].message.content
                obj = json.loads(content)

                obj.setdefault("id", target_id)
                obj["_meta"] = {
                    "source_target_id": target_id,
                    "anchor_id": target.get("anchor_id"),
                    "anchor_distance": target.get("anchor_distance"),
                    "mode": target.get("mode"),
                    "topic": target.get("topic"),
                    "difficulty": target.get("difficulty"),
                }

                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                f_out.flush()

                n_done += 1
                print(f"[{i}] OK id={obj.get('id')} mode={target.get('mode')} anchor={target.get('anchor_id')}", flush=True)

            except Exception as e:
                err = {"target_id": target_id, "error": repr(e)}
                f_out.write(json.dumps(err, ensure_ascii=False) + "\n")
                f_out.flush()
                print(f"[{i}] ERROR target_id={target_id}: {e!r}", flush=True)

    dt = time.time() - t0
    print(f"Done. Wrote {args.out}. Generated={n_done}. Elapsed={dt:.1f}s")

if __name__ == "__main__":
    main()
