#!/usr/bin/env python3
"""
generate.py (WITH VERIFY + OPTIONAL AUTO-REPAIR)

Reads targets.jsonl produced by 03_retrieve.py and generates new contest-style problems.

Dual-RAG conditioning:
- TARGET SKELETON_TEXT: authoritative reasoning plan
- EXEMPLARS (question/choices) + compressed solution traces: mapping & surface priors

NEW:
- Verification step: a separate verifier prompt checks correctness/format/style and returns PASS/FAIL + issues.
- Optional auto-repair loop: if FAIL, ask model to revise and re-verify up to N times.

Output JSONL with:
{
  "id","question","choices","answer","solution","skeleton_text","anti_copy_report",
  "_meta", "_verify": {...}, "_repairs": [...]
}
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


SYSTEM_GEN = """You generate contest-faithful STEM multiple-choice problems (e.g., F=ma / USNCO style).
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

SYSTEM_VERIFY = """You are a strict verifier for contest-style multiple-choice STEM problems.
You will be given:
- TARGET_SKELETON_TEXT (authoritative reasoning plan)
- EXEMPLARS (questions + compressed traces) that must NOT be copied
- A CANDIDATE generated problem JSON

Your job:
1) Check the candidate is self-contained, solvable, and internally consistent.
2) Check the provided solution logically leads to the claimed answer choice.
3) Check there are exactly 5 choices A-E and answer is one of them.
4) Check choices are reasonably confusable (same order of magnitude / similar symbolic forms).
5) Check anti-copy: no reuse of exemplar scenario/phrasing/variable names/numbers.
6) Check surface feel: avoid "too many random numbers" if the domain seems symbolic (e.g., F=ma).

Return strict JSON only with:
{
  "verdict": "PASS" or "FAIL",
  "issues": ["...", ...],                  // short, specific
  "required_fixes": ["...", ...],          // actionable edits if FAIL
  "answer_consistency": {
     "answer_claimed": "A",
     "answer_verified": "A" or "UNKNOWN" or "DIFFERS",
     "notes": "..."
  },
  "copy_risk": {
     "risk_level": "LOW"|"MEDIUM"|"HIGH",
     "notes": "..."
  }
}
Be conservative: if you cannot verify answer correctness from the candidate solution, mark FAIL.
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


def build_verify_prompt(target: Dict[str, Any], candidate: Dict[str, Any], k_max_exemplars: int = 4) -> str:
    skel = target.get("skeleton_text") or ""
    exemplars: List[Dict[str, Any]] = target.get("exemplars") or []
    exemplars = exemplars[:k_max_exemplars]

    ex_blocks = []
    for j, ex in enumerate(exemplars, start=1):
        qtxt = ex.get("question_text") or ""
        stxt = ex.get("skeleton_text") or ""
        ex_blocks.append(
            f"EXEMPLAR {j} QUESTION:\n{_truncate(qtxt, 1200)}\n\n"
            f"EXEMPLAR {j} TRACE:\n{_truncate(stxt, 900)}\n"
        )
    exemplar_section = "\n\n".join(ex_blocks) if ex_blocks else "(none)"

    return f"""TARGET_SKELETON_TEXT:
{skel}

EXEMPLARS (must not be copied):
{exemplar_section}

CANDIDATE_JSON:
{json.dumps(candidate, ensure_ascii=False)}
"""


def llm_json(client: OpenAI, model: str, system: str, user: str, temperature: float = 0.0) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def repair_prompt(target: Dict[str, Any], bad: Dict[str, Any], verify: Dict[str, Any], k_max_exemplars: int = 4) -> str:
    # Give the model the issues and require a revised candidate JSON, same schema.
    return f"""You generated a candidate that FAILED verification.

TARGET_SKELETON_TEXT (must remain faithful):
{target.get("skeleton_text") or ""}

VERIFIER_REQUIRED_FIXES:
{json.dumps(verify.get("required_fixes", []), ensure_ascii=False)}

VERIFIER_ISSUES:
{json.dumps(verify.get("issues", []), ensure_ascii=False)}

FAILED_CANDIDATE_JSON:
{json.dumps(bad, ensure_ascii=False)}

TASK:
Revise the candidate to address ALL required fixes while preserving:
- Novel scenario (do not copy exemplars)
- Faithfulness to TARGET_SKELETON_TEXT
- 5 confusable choices A-E

Return strict JSON in the SAME schema as before.
"""


def is_basic_schema_ok(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    if "question" not in obj or "choices" not in obj or "answer" not in obj or "solution" not in obj:
        return False
    ch = obj.get("choices")
    if not isinstance(ch, dict):
        return False
    keys = set(ch.keys())
    if keys != {"A", "B", "C", "D", "E"}:
        return False
    if obj.get("answer") not in ["A", "B", "C", "D", "E"]:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="targets.jsonl from 03_retrieve.py")
    ap.add_argument("--out", default="generated_problems.jsonl")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--verify_model", default="", help="optional separate verifier model (defaults to --model)")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--timeout", type=float, default=90.0, help="seconds per request")
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--max_exemplars", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--do_verify", action="store_true", help="enable verification gate")
    ap.add_argument("--repair_attempts", type=int, default=1, help="number of auto-repair attempts if FAIL")
    args = ap.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set (env or .env).")

    client = OpenAI(timeout=args.timeout, max_retries=args.max_retries)

    verify_model = args.verify_model.strip() or args.model

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

                candidate = llm_json(
                    client=client,
                    model=args.model,
                    system=SYSTEM_GEN,
                    user=user_prompt,
                    temperature=args.temperature,
                )
                candidate.setdefault("id", target_id)

                repairs: List[Dict[str, Any]] = []
                verify_obj: Optional[Dict[str, Any]] = None

                # Basic schema sanity before verifier
                if not is_basic_schema_ok(candidate):
                    verify_obj = {
                        "verdict": "FAIL",
                        "issues": ["Candidate JSON schema invalid (missing fields or not exactly A-E)."],
                        "required_fixes": ["Return exactly the required schema with choices A-E and answer in A-E."],
                        "answer_consistency": {"answer_claimed": str(candidate.get("answer")), "answer_verified": "UNKNOWN", "notes": "Schema invalid."},
                        "copy_risk": {"risk_level": "UNKNOWN", "notes": "Schema invalid."},
                    }

                if args.do_verify:
                    # Verify / repair loop
                    attempts_left = args.repair_attempts
                    while True:
                        if verify_obj is None:
                            vprompt = build_verify_prompt(target, candidate, k_max_exemplars=args.max_exemplars)
                            verify_obj = llm_json(
                                client=client,
                                model=verify_model,
                                system=SYSTEM_VERIFY,
                                user=vprompt,
                                temperature=0.0,
                            )

                        if verify_obj.get("verdict") == "PASS":
                            break

                        if attempts_left <= 0:
                            break

                        # Attempt repair
                        rprompt = repair_prompt(target, candidate, verify_obj, k_max_exemplars=args.max_exemplars)
                        repaired = llm_json(
                            client=client,
                            model=args.model,
                            system=SYSTEM_GEN,
                            user=rprompt,
                            temperature=max(0.3, args.temperature),
                        )
                        repaired.setdefault("id", target_id)

                        repairs.append(
                            {
                                "attempt": (args.repair_attempts - attempts_left + 1),
                                "previous_verify": verify_obj,
                            }
                        )

                        candidate = repaired
                        verify_obj = None  # re-verify from scratch
                        attempts_left -= 1

                candidate["_meta"] = {
                    "source_target_id": target_id,
                    "anchor_id": target.get("anchor_id"),
                    "anchor_distance": target.get("anchor_distance"),
                    "mode": target.get("mode"),
                    "topic": target.get("topic"),
                    "difficulty": target.get("difficulty"),
                }
                if args.do_verify:
                    candidate["_verify"] = verify_obj or {"verdict": "FAIL", "issues": ["Unknown verifier state."], "required_fixes": []}
                    if repairs:
                        candidate["_repairs"] = repairs

                f_out.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                f_out.flush()

                n_done += 1
                status = "PASS" if (candidate.get("_verify", {}).get("verdict") == "PASS") else ("NO-VERIFY" if not args.do_verify else "FAIL")
                print(f"[{i}] OK id={candidate.get('id')} verify={status}", flush=True)

            except Exception as e:
                err = {"target_id": target_id, "error": repr(e)}
                f_out.write(json.dumps(err, ensure_ascii=False) + "\n")
                f_out.flush()
                print(f"[{i}] ERROR target_id={target_id}: {e!r}", flush=True)

    dt = time.time() - t0
    print(f"Done. Wrote {args.out}. Generated={n_done}. Elapsed={dt:.1f}s")


if __name__ == "__main__":
    main()
