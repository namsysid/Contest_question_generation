#!/usr/bin/env python3
"""
generate_diverse.py

- Prints the solution skeleton for each anchor
- Uses a solution-centric, solution-backwards prompt
- Generates one problem per anchor (no averaging collapse)
"""

import json
import argparse
from openai import OpenAI
import os

import dotenv
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Check that API key is set
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set. Please set it in .env file or environment variable.")


client = OpenAI()

def format_skeleton(anchor):
    core = {
        "id": anchor.get("id"),
        "optypes": anchor.get("optypes", []),
        "laws": anchor.get("laws", []),
        "target": anchor.get("target"),
    }

    optional_keys = [
        "concepts", "variables", "constraints",
        "steps", "solution_skeleton", "notes"
    ]
    for k in optional_keys:
        if k in anchor and anchor[k]:
            core[k] = anchor[k]

    return json.dumps(core, indent=2)

def build_prompt(anchor, prior_titles):
    optypes = anchor.get("optypes", [])
    laws = anchor.get("laws", [])
    target = anchor.get("target", "unknown")

    avoid = "\n".join(f"- {t}" for t in prior_titles[-10:]) if prior_titles else "(none)"

    return f"""
You are generating ORIGINAL competition-style STEM problems.

You MUST use a SOLUTION-CENTRIC, SOLUTION-BACKWARDS approach:

A) First write a SOLUTION OUTLINE that follows the abstract skeleton exactly.
   - Explicitly name the Operators and Laws.
   - Include equations and logical steps.
   - Do NOT plug in numbers yet.

B) Then design a PROBLEM STATEMENT whose givens force that solution path.

C) Finally provide the FULL WORKED SOLUTION and FINAL ANSWER.

Abstract solution skeleton:
- Operators: {optypes}
- Laws: {laws}
- Target quantity: {target}

Avoid problems similar to:
{avoid}

Output format:
PROBLEM:
...
SOLUTION OUTLINE:
...
FULL SOLUTION:
...
FINAL ANSWER:
...
"""

def main(inp, n, model):
    with open(inp) as f:
        anchors = [json.loads(l) for l in f]

    anchors = anchors[:n]
    prior_titles = []

    for i, a in enumerate(anchors, 1):
        print(f"=== ANCHOR {i}/{n}: SOLUTION SKELETON ===")
        print(format_skeleton(a))
        print()

        prompt = build_prompt(a, prior_titles)
        resp = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=900
        )
        out = resp.output_text.strip()

        first_line = out.splitlines()[0] if out else ""
        if first_line:
            prior_titles.append(first_line[:140])

        print(f"=== GENERATED PROBLEM {i}/{n} ===")
        print(out)
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieved", required=True)
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--model", default="gpt-4.1")
    args = ap.parse_args()
    main(args.retrieved, args.num, args.model)
