#!/usr/bin/env python3
"""
04_generate_skeletons.py  (STAGE-1: SOLUTION SKELETON GENERATION)

Paper-faithful:
- Skeletons are generated FIRST, conditioned ONLY on solution_exemplars (skeleton space).
- We still provide the seed skeleton text as a "style prior", but instruct the model to produce
  a NEW skeleton (no copying of exemplars or seed).

Input:  retrieval_bundles.jsonl from 03_retrieve_paper.py
Output: generated_skeletons.jsonl with:
{
  "bundle_id": "...",
  "generated_skeleton": {givens,target,laws,steps,final_form},
  "skeleton_text": "...",
  "_meta": {...}
}

Operator ontology is limited (same as enrichment).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not set (env or .env).")

SYSTEM = """You generate STRUCTURED solution skeletons (schemas) for Olympiad-style STEM problems.

Hard constraints:
- Do NOT copy any exemplar skeleton verbatim.
- Produce a NEW skeleton that is plausible, contest-like, and structurally coherent.
- Use ONLY the allowed operator set for every step.op.
- Keep it abstract: no full arithmetic, no long derivations.
Return strict JSON only.
"""

USER_TMPL = """Domain: {domain}

SEED_SKELETON_TEXT (style/structure prior; do not copy verbatim):
{seed_skel}

SOLUTION EXEMPLARS (do not copy verbatim; for style and typical structure only):
{solution_exemplars_block}

ALLOWED_OPS:
{allowed_ops_block}

TASK:
Generate ONE NEW solution skeleton (schema). It should be:
- similar in *structural difficulty* to the exemplars,
- but clearly novel (different givens/target/laws combination).

Return strict JSON with keys:
- skeleton: object with keys:
  - givens: list of strings (0-8)
  - target: string
  - laws: list of strings (0-6) (very short)
  - steps: list of objects {{op: string, text: string}} (3-{max_steps} steps)
  - final_form: string (abstract final computation plan)
- topic: short string or null
- difficulty: integer 1-10 or null
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

def safe_skeleton_text(sk: Any) -> str:
    if isinstance(sk, str):
        return sk.strip()
    if isinstance(sk, dict):
        laws = sk.get("laws", []) or []
        steps = sk.get("steps", []) or []
        parts = []
        if laws:
            parts.append("LAWS: " + ", ".join(map(str, laws)))
        if steps:
            parts.append("STEPS: " + " | ".join(f'{s.get("op")}:{s.get("text")}' for s in steps if isinstance(s, dict)))
        return "\n".join(parts).strip()
    return str(sk).strip()

def llm_json(client: OpenAI, model: str, system: str, user: str, temperature: float = 0.2) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        response_format={"type":"json_object"},
    )
    return json.loads(resp.choices[0].message.content)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", required=True, help="retrieval_bundles.jsonl from 03_retrieve_paper.py")
    ap.add_argument("--out", default="generated_skeletons.jsonl")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--domain", default="fma", help="default domain label when bundles lack it")
    ap.add_argument("--max_steps", type=int, default=8)
    ap.add_argument("--allowed_ops", default="IDENTIFY_GIVENS,IDENTIFY_RELATION,APPLY_RELATION,SAVE_RESULT,CHECK")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    allowed_ops = [s.strip() for s in args.allowed_ops.split(",") if s.strip()]
    if not allowed_ops:
        raise ValueError("allowed_ops must be non-empty")

    rows = read_jsonl(args.bundles)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    client = OpenAI()
    out_rows: List[Dict[str, Any]] = []

    for b in rows:
        domain = b.get("domain") or args.domain
        seed_skel = b.get("seed_skeleton_text") or ""

        sol_ex = b.get("solution_exemplars") or []
        ex_blocks = []
        for i, ex in enumerate(sol_ex[:8], start=1):
            ex_blocks.append(f"EX {i}:\n{ex.get('skeleton_text','')}")
        solution_exemplars_block = "\n\n".join(ex_blocks) if ex_blocks else "(none)"

        user = USER_TMPL.format(
            domain=domain,
            seed_skel=seed_skel,
            solution_exemplars_block=solution_exemplars_block,
            allowed_ops_block="\n".join(f"- {o}" for o in allowed_ops),
            max_steps=args.max_steps,
        )

        js = llm_json(client, args.model, SYSTEM, user, temperature=0.25)
        sk = (js.get("skeleton") or {})
        sk_text = safe_skeleton_text(sk)

        out_rows.append({
            "bundle_id": b.get("bundle_id"),
            "generated_skeleton": sk,
            "skeleton_text": sk_text,
            "topic": js.get("topic"),
            "difficulty": js.get("difficulty"),
            "_meta": {
                "seed_id": b.get("seed_id"),
                "mode": b.get("mode"),
                "anchor_id": b.get("anchor_id"),
                "anchor_distance": b.get("anchor_distance"),
            }
        })

    write_jsonl(args.out, out_rows)
    print(f"Wrote {len(out_rows)} generated skeletons -> {args.out}")

if __name__ == "__main__":
    main()
