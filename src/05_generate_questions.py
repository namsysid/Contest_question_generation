#!/usr/bin/env python3
"""
05_generate_questions.py  (STAGE-2: QUESTION GENERATION CONDITIONED ON GENERATED SKELETONS)

Paper-faithful:
- Uses generated skeletons (stage 1) as authoritative TARGET_SKELETON_TEXT
- Uses TWO independent exemplar sets:
  * question_exemplars (question space)
  * paired_exemplars (question + skeleton) to demonstrate mapping

Optional: gatekeeper verification + auto-repair (kept lightweight here; rubric scoring is in 06).

Inputs:
  --bundles retrieval_bundles.jsonl   (from 03_retrieve_paper.py)
  --skeletons generated_skeletons.jsonl (from 04_generate_skeletons.py)

Output:
  generated_problems.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not set (env or .env).")

SYSTEM_GEN = """You generate contest-faithful STEM multiple-choice problems (e.g., F=ma / USNCO style).

Hard constraints:
- Do NOT copy, paraphrase, or minimally edit any exemplar scenario, wording, variable names, or numbers.
- You MUST produce a novel scenario and novel phrasing.
- Stay faithful to the TARGET_SKELETON_TEXT's reasoning structure (operators/laws/structure), but you may choose
  new symbols and a new context.
- Provide exactly 5 answer choices (A)-(E) with plausible, confusable distractors.
- Keep the problem self-contained and solvable without outside references.
Return strict JSON only (no markdown).
"""

SYSTEM_GATEKEEP = """You are a strict gatekeeper verifier for contest-style multiple-choice STEM problems.
Given TARGET_SKELETON_TEXT and a CANDIDATE_JSON, determine if the candidate is:
- properly formatted (5 choices A-E, answer in A-E)
- self-contained and solvable
- solution logically supports the claimed answer
- faithful to TARGET_SKELETON_TEXT at a high level
Return strict JSON only:
{
  "verdict": "PASS"|"FAIL",
  "issues": ["..."],
  "required_fixes": ["..."],
  "answer_consistency": {"answer_claimed":"A","answer_verified":"A|UNKNOWN|DIFFERS","notes":"..."}
}
If you cannot verify answer correctness from the solution, mark FAIL.
"""

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_jsonl_line(f, row: Dict[str, Any]) -> None:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    f.flush()

def llm_json(client: OpenAI, model: str, system: str, user: str, temperature: float = 0.2) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        response_format={"type":"json_object"},
    )
    return json.loads(resp.choices[0].message.content)

def _truncate(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"

def build_prompt(bundle: Dict[str, Any], target_skeleton_text: str, max_q_ex: int, max_paired: int) -> str:
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
            f"PAIRED EXEMPLAR {i} SKELETON_TEXT:\n{_truncate(ex.get('skeleton_text',''), 900)}"
        )
    p_section = "\n\n".join(p_blocks) if p_blocks else "(none)"

    return f"""You are given:

1) TARGET_SKELETON_TEXT (authoritative reasoning plan; must be assisted by):
{target_skeleton_text}

2) QUESTION EXEMPLARS (question space; style/semantics priors ONLY; must not be copied):
{q_section}

3) PAIRED EXEMPLARS (mapping demonstrations between solutions and problems; must not be copied):
{p_section}

TASK:
Generate ONE NEW hard f=ma multiple-choice problem that is largely faithful to TARGET_SKELETON_TEXT. The problems must be complex, difficult, and interesting.
- Invent a NEW, hard scenario and NEW wording (no copying).
- Use different variable names than any exemplars.
- Exactly 5 choices A-E (confusable distractors).
- Include at least 2 distractors corresponding to common mistakes implied by the skeleton (missing factor, sign, wrong component, etc.).
- When writing the question, assume the reader does not know anything of the solution path. Do not expose more than what is absolutely necessary to solve the problem. Keep concepts that can get inferred, even with some challenge, up to the reader.
- The question should follow the rough pipeline of the solution skeleton. You can deviate to make the problem more complex, interesting, and difficult

OUTPUT strict JSON schema:
{{
  "id": "<string>",
  "question": "<string stem>",
  "choices": {{"A":"...", "B":"...", "C":"...", "D":"...", "E":"..."}},
  "answer": "A|B|C|D|E",
  "solution": "<clear solution; may include equations>",
  "skeleton_text": "<compressed trace you actually used>",
  "anti_copy_report": {{
     "novel_scenario_summary": "<1-2 sentences>",
     "differences_from_exemplars": ["...", "..."],
     "possible_overlap_risks": ["...", "..."]
  }}
}}
"""

def build_gatekeeper_prompt(target_skeleton_text: str, candidate: Dict[str, Any]) -> str:
    return f"""TARGET_SKELETON_TEXT:
{target_skeleton_text}

CANDIDATE_JSON:
{json.dumps(candidate, ensure_ascii=False)}
"""

def build_repair_prompt(target_skeleton_text: str, candidate: Dict[str, Any], gate: Dict[str, Any]) -> str:
    return f"""Your candidate FAILED gatekeeper verification.

TARGET_SKELETON_TEXT (must remain faithful):
{target_skeleton_text}

REQUIRED_FIXES:
{json.dumps(gate.get("required_fixes", []), ensure_ascii=False)}

ISSUES:
{json.dumps(gate.get("issues", []), ensure_ascii=False)}

FAILED_CANDIDATE_JSON:
{json.dumps(candidate, ensure_ascii=False)}

TASK:
Revise the candidate to address ALL required fixes while preserving:
- Novel scenario (no copying)
- Faithfulness to TARGET_SKELETON_TEXT
- Exactly 5 confusable choices A-E

Return strict JSON in the SAME schema.
"""

def is_basic_schema_ok(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    if "question" not in obj or "choices" not in obj or "answer" not in obj or "solution" not in obj:
        return False
    ch = obj.get("choices")
    if not isinstance(ch, dict):
        return False
    if set(ch.keys()) != {"A","B","C","D","E"}:
        return False
    if obj.get("answer") not in {"A","B","C","D","E"}:
        return False
    return True

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", required=True, help="retrieval_bundles.jsonl from 03_retrieve_paper.py")
    ap.add_argument("--skeletons", required=True, help="generated_skeletons.jsonl from 04_generate_skeletons.py")
    ap.add_argument("--out", default="generated_problems.jsonl")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--verify_model", default="", help="gatekeeper model (defaults to --model)")
    ap.add_argument("--max_q_exemplars", type=int, default=4)
    ap.add_argument("--max_paired_exemplars", type=int, default=3)
    ap.add_argument("--repair_max", type=int, default=1)
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

    client = OpenAI()
    total = len(bundles)

    if args.debug:
        print(f"[write] open (truncate): {args.out}", file=sys.stderr, flush=True)
    with open(args.out, "w", encoding="utf-8") as out_f:
        for i, b in enumerate(bundles, start=1):
            bid = b.get("bundle_id")
            sk = sk_by_bundle.get(bid, {})
            target_skeleton_text = sk.get("skeleton_text") or ""

            if args.debug:
                print(f"[{i}/{total}] bundle_id={bid} building prompt", file=sys.stderr, flush=True)

            prompt = build_prompt(b, target_skeleton_text, args.max_q_exemplars, args.max_paired_exemplars)

            if args.debug:
                print(f"[{i}/{total}] bundle_id={bid} calling gen model={args.model}", file=sys.stderr, flush=True)
                t0 = time.time()

            cand = llm_json(client, args.model, SYSTEM_GEN, prompt, temperature=0.25)

            if args.debug:
                dt = time.time() - t0
                print(f"[{i}/{total}] bundle_id={bid} gen done in {dt:.2f}s", file=sys.stderr, flush=True)

            repairs: List[Dict[str, Any]] = []
            gate: Optional[Dict[str, Any]] = None

            # lightweight repair loop
            for r_i in range(max(0, args.repair_max) + 1):
                if not is_basic_schema_ok(cand):
                    gate = {"verdict":"FAIL","issues":["Basic schema invalid"],"required_fixes":["Fix JSON schema to match required keys and 5 choices A-E"],"answer_consistency":{"answer_claimed":cand.get("answer",""),"answer_verified":"UNKNOWN","notes":"schema invalid"}}
                else:
                    if args.debug:
                        print(f"[{i}/{total}] bundle_id={bid} gatekeeper call {r_i+1} model={verify_model}", file=sys.stderr, flush=True)
                    gate = llm_json(client, verify_model, SYSTEM_GATEKEEP, build_gatekeeper_prompt(target_skeleton_text, cand), temperature=0.0)

                if gate.get("verdict") == "PASS":
                    if args.debug:
                        print(f"[{i}/{total}] bundle_id={bid} gatekeeper PASS", file=sys.stderr, flush=True)
                    break

                repairs.append({"gatekeeper": gate, "candidate": cand})
                # repair
                if args.debug:
                    print(f"[{i}/{total}] bundle_id={bid} repair {r_i+1} calling model={args.model}", file=sys.stderr, flush=True)
                cand = llm_json(client, args.model, SYSTEM_GEN, build_repair_prompt(target_skeleton_text, cand, gate), temperature=0.2)

            out_row = {
                **cand,
                "_meta": {
                    "bundle_id": bid,
                    "seed_id": b.get("seed_id"),
                    "mode": b.get("mode"),
                    "anchor_id": b.get("anchor_id"),
                    "anchor_distance": b.get("anchor_distance"),
                },
                "_gatekeeper": gate,
                "_repairs": repairs,
            }

            write_jsonl_line(out_f, out_row)
            if args.debug:
                print(f"[{i}/{total}] bundle_id={bid} wrote", file=sys.stderr, flush=True)

            if args.sleep:
                time.sleep(args.sleep)

    if args.debug:
        print(f"[write] done: {args.out}", file=sys.stderr, flush=True)
    print(f"Wrote {total} generated problems -> {args.out}")

if __name__ == "__main__":
    main()
