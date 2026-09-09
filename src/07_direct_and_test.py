#!/usr/bin/env python3
"""Run direct generation (07) and evaluation (06) in one command."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent


def _run(cmd: List[str], env: Dict[str, str], dry_run: bool = False) -> None:
    pretty = " ".join(shlex.quote(c) for c in cmd)
    print(f"[run] {pretty}")
    shown = ", ".join(f"{k}={v}" for k, v in sorted(env.items()))
    print(f"[env] {shown}")
    if dry_run:
        return
    child_env = os.environ.copy()
    child_env.update(env)
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=child_env)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run 07 direct generation + 06 scoring.")
    ap.add_argument("--input", required=True, help="JSONL prompt file for stage 07")
    ap.add_argument("--generated-out", default=str(ROOT / "direct_generated.jsonl"))
    ap.add_argument("--scored-out", default=str(ROOT / "direct_scored.jsonl"))
    ap.add_argument("--chat-model", default="qwen2.5:7b-instruct")
    ap.add_argument("--verify-model", default="")
    ap.add_argument("--api-base-url", default="http://192.168.50.186:11434/v1")
    ap.add_argument("--api-key", default="ollama")
    ap.add_argument("--subject", default="Chem")
    ap.add_argument("--competition", default="USNCO")
    ap.add_argument("--exemplars", nargs="*", default=["chem_data/txts/all_questions.jsonl"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    env = {
        "OPENAI_BASE_URL": args.api_base_url,
        "OPENAI_API_KEY": args.api_key,
    }

    generated_out = Path(args.generated_out)
    scored_out = Path(args.scored_out)
    generated_out.parent.mkdir(parents=True, exist_ok=True)
    scored_out.parent.mkdir(parents=True, exist_ok=True)

    cmd_07 = [
        sys.executable,
        str(SRC / "07_direct_generate.py"),
        "--input", args.input,
        "--out", str(generated_out),
        "--model", args.chat_model,
        "--allow-direct-ablation",
    ]
    if args.limit:
        cmd_07 += ["--limit", str(args.limit)]
    if args.debug:
        cmd_07 += ["--debug"]

    verify_model = args.verify_model or args.chat_model
    cmd_06 = [
        sys.executable,
        str(SRC / "06_verify_and_score.py"),
        "--input", str(generated_out),
        "--out", str(scored_out),
        "--model", verify_model,
        "--subject", args.subject,
        "--competition", args.competition,
        "--exemplars",
    ] + list(args.exemplars)
    if args.limit:
        cmd_06 += ["--limit", str(args.limit)]
    if args.debug:
        cmd_06 += ["--debug"]

    _run(cmd_07, env, args.dry_run)
    _run(cmd_06, env, args.dry_run)

    print("\nDone.")
    print(f"- Generated: {generated_out}")
    print(f"- Scored: {scored_out}")


if __name__ == "__main__":
    main()
