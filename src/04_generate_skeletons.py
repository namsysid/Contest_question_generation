#!/usr/bin/env python3
"""
04_generate_skeletons_v3.py

Stage-1 skeleton generation with enforced insight operator usage.
Requires:
- laws >= 2
- steps >= 6
- at least one INSIGHT op in steps

INSIGHT_OPS default:
  INTRODUCE_AUX, CONSTRAINT_COUPLING, INVARIANT_SYMMETRY, CASEWORK_REGIME, CHECK
"""

from __future__ import annotations
import argparse, json, os, re, sys, time
from typing import Any, Dict, List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not set (env or .env).")

SYSTEM = """You generate STRUCTURED solution skeletons (schemas) for Olympiad-style STEM problems.

Hard constraints:
- Do NOT copy any exemplar skeleton verbatim.
- Use ONLY the allowed operator set for every step.op.
- Keep it abstract: no full arithmetic, no long derivations.
- The skeleton MUST be Olympiad-like (not one-equation).

Difficulty constraints (must satisfy):
- laws list length >= 2.
- Include at least ONE INSIGHT operator in the steps.
- Include >= 6 steps.
Return strict JSON only.
"""

USER_TMPL = """Domain: {domain}

SEED_SKELETON_TEXT (style prior; do not copy; be at least as complex):
{seed_skel}

SOLUTION EXEMPLARS (do not copy):
{solution_exemplars_block}

ALLOWED_OPS:
{allowed_ops_block}

INSIGHT_OPS (at least one step.op MUST be one of these):
{insight_ops_block}

TASK:
Generate ONE NEW solution skeleton (schema) that is contest-faithful and nontrivial.

Return strict JSON with keys:
- skeleton: object with keys:
  - givens: list of strings (0-8)
  - target: string
  - laws: list of strings (2-6) (short names)
  - steps: list of objects {{op: string, text: string}} (6-{max_steps} steps)
  - final_form: string
- topic: short string or null
- difficulty: integer 1-10 or null
"""

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                out.append(json.loads(line))
    return out

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path,"w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

def write_jsonl_line(f, row: Dict[str, Any]) -> None:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    f.flush()

def safe_skeleton_text(sk: Any) -> str:
    if isinstance(sk,str): return sk.strip()
    if isinstance(sk,dict):
        laws=sk.get("laws") or []
        steps=sk.get("steps") or []
        parts=[]
        if laws:
            parts.append("LAWS: " + ", ".join(map(str,laws)))
        if steps:
            parts.append("STEPS: " + " | ".join(f'{s.get("op")}:{s.get("text")}' for s in steps if isinstance(s,dict)))
        return "\n".join(parts).strip()
    return str(sk).strip()

def llm_json(client: OpenAI, model: str, system: str, user: str, temperature: float=0.25) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        response_format={"type":"json_object"},
    )
    return json.loads(resp.choices[0].message.content)

def parse_ops_from_skeleton_text(st: str) -> List[str]:
    m=re.search(r"STEPS:\s*(.+)", st or "", re.DOTALL)
    if not m:
        return []
    ops=[]
    for p in m.group(1).split("|"):
        p=p.strip()
        if ":" in p:
            ops.append(p.split(":",1)[0].strip())
    return ops

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bundles", required=True)
    ap.add_argument("--out", default="generated_skeletons.jsonl")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--domain", default="fma")
    ap.add_argument("--max_steps", type=int, default=10)
    ap.add_argument("--allowed_ops", default="IDENTIFY_GIVENS,IDENTIFY_RELATION,APPLY_RELATION,INTRODUCE_AUX,CONSTRAINT_COUPLING,INVARIANT_SYMMETRY,CASEWORK_REGIME,CHECK,SAVE_RESULT")
    ap.add_argument("--insight_ops", default="INTRODUCE_AUX,CONSTRAINT_COUPLING,INVARIANT_SYMMETRY,CASEWORK_REGIME,CHECK")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--debug", action="store_true", help="print per-row progress to stderr")
    args=ap.parse_args()

    allowed_ops=[s.strip() for s in args.allowed_ops.split(",") if s.strip()]
    insight_ops=set(s.strip() for s in args.insight_ops.split(",") if s.strip())

    rows=read_jsonl(args.bundles)
    if args.limit and args.limit>0:
        rows=rows[:args.limit]

    client=OpenAI()
    total=len(rows)

    if args.debug:
        print(f"[write] open (truncate): {args.out}", file=sys.stderr, flush=True)
    with open(args.out, "w", encoding="utf-8") as out_f:
        for i, b in enumerate(rows, start=1):
            domain=b.get("domain") or args.domain
            seed_skel=b.get("seed_skeleton_text") or ""
            sol_ex=b.get("solution_exemplars") or []
            ex_blocks=[]
            for j,ex in enumerate(sol_ex[:8], start=1):
                ex_blocks.append(f"EX {j}:\n{ex.get('skeleton_text','')}")
            sol_block="\n\n".join(ex_blocks) if ex_blocks else "(none)"

            user=USER_TMPL.format(
                domain=domain,
                seed_skel=seed_skel,
                solution_exemplars_block=sol_block,
                allowed_ops_block="\n".join(f"- {o}" for o in allowed_ops),
                insight_ops_block="\n".join(f"- {o}" for o in sorted(insight_ops)),
                max_steps=args.max_steps
            )

            if args.debug:
                bid=b.get("bundle_id","?")
                print(f"[{i}/{total}] bundle_id={bid} calling model={args.model}", file=sys.stderr, flush=True)
                t0=time.time()

            best=None
            for k in range(3):
                js=llm_json(client, args.model, SYSTEM, user, temperature=0.25)
                sk=(js.get("skeleton") or {})
                st=safe_skeleton_text(sk)
                ops=set(parse_ops_from_skeleton_text(st))
                laws=sk.get("laws") or []
                ok = isinstance(laws,list) and len(laws) >= 2 and any(op in insight_ops for op in ops) and isinstance(sk.get("steps"), list) and len(sk.get("steps")) >= 6
                best=(js, sk, st, ok)
                if ok:
                    break
                if args.debug:
                    print(f"[{i}/{total}] bundle_id={b.get('bundle_id','?')} retry {k+1} constraints_satisfied={bool(ok)}", file=sys.stderr, flush=True)

            js, sk, st, ok = best

            if args.debug:
                dt=time.time()-t0
                print(f"[{i}/{total}] bundle_id={b.get('bundle_id','?')} model done in {dt:.2f}s", file=sys.stderr, flush=True)

            out_row={
                "bundle_id": b.get("bundle_id"),
                "generated_skeleton": sk,
                "skeleton_text": st,
                "topic": js.get("topic"),
                "difficulty": js.get("difficulty"),
                "_meta": {"seed_id": b.get("seed_id"), "mode": b.get("mode"), "anchor_id": b.get("anchor_id"), "anchor_distance": b.get("anchor_distance")},
                "_diagnostics": {"constraints_satisfied": bool(ok)}
            }
            write_jsonl_line(out_f, out_row)
            if args.debug:
                print(f"[{i}/{total}] bundle_id={b.get('bundle_id','?')} wrote", file=sys.stderr, flush=True)

    if args.debug:
        print(f"[write] done: {args.out}", file=sys.stderr, flush=True)
    print(f"Wrote {total} generated skeletons -> {args.out}")

if __name__=="__main__":
    main()
