#!/usr/bin/env python3
import argparse, json, os, time
from dotenv import load_dotenv
from openai import OpenAI

SYSTEM = """You generate contest-faithful STEM problems.
You must not copy or paraphrase any training/exemplar text.
You must produce a novel problem with a clean, correct solution.
Stay faithful to the TARGET SKELETON (operators/laws/structure)."""

def bucket_distance(d):
    if d is None: return None
    if d < 0.15: return "near"
    if d < 0.30: return "mid"
    return "far"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", default="generated_problems.jsonl")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds per request")
    ap.add_argument("--max_retries", type=int, default=2)
    args = ap.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set (env or .env).")

    # IMPORTANT: timeout prevents “stall forever”
    client = OpenAI(timeout=args.timeout, max_retries=args.max_retries)

    n_done = 0
    t0 = time.time()

    with open(args.targets, "r", encoding="utf-8") as f_in, open(args.out, "w", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in, start=1):
            if not line.strip():
                continue
            if args.limit and n_done >= args.limit:
                break

            t = json.loads(line)
            dist_bucket = bucket_distance(t.get("anchor_distance"))

            user_prompt = f"""
TARGET SKELETON (authoritative):
{json.dumps(t.get("skeleton"), ensure_ascii=False, indent=2)}

CONTROL METADATA (do not mention in the problem text):
- mode: {t.get("mode")}
- anchor_distance_bucket: {dist_bucket}
- anchor_distance: {t.get("anchor_distance")}

CONTEST CONSTRAINTS:
- Difficulty: F=ma-style multiple choice or short answer (choose one consistent style)
- Must be solvable in ~5–10 minutes by a trained student
- Use clean numbers; avoid messy arithmetic unless essential
- Provide: (1) Problem statement, (2) Answer, (3) Full solution

REQUIREMENTS:
- Follow the skeleton's operators/laws/structure exactly (same type of method).
- Make the surface story NEW (different objects/context).
- Do NOT reuse any specific phrasing you have seen elsewhere.
"""

            print(f"[{i}] generating target_id={t.get('id')} model={args.model}", flush=True)

            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                text = (resp.choices[0].message.content or "").strip()

                out = {
                    "target_id": t.get("id"),
                    "mode": t.get("mode"),
                    "anchor_id": t.get("anchor_id"),
                    "anchor_distance": t.get("anchor_distance"),
                    "generated": text,
                }
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
                f_out.flush()
                n_done += 1

            except Exception as e:
                # Don’t stall the whole run; log and continue
                err = {
                    "target_id": t.get("id"),
                    "error": repr(e),
                }
                f_out.write(json.dumps(err, ensure_ascii=False) + "\n")
                f_out.flush()
                print(f"[{i}] ERROR target_id={t.get('id')}: {e!r}", flush=True)

    dt = time.time() - t0
    print(f"Done. Wrote {args.out}. Generated={n_done}. Elapsed={dt:.1f}s")

if __name__ == "__main__":
    main()
