#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
VERIFY_SCRIPT = SRC_DIR / "06_verify_and_score.py"


def shell_join(parts: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def count_jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def infer_subject_competition(domain: str) -> Tuple[str, str]:
    key = domain.strip().lower()
    if key in {"chem", "chemistry"}:
        return ("Chemistry", "USNCO")
    return ("Physics", "f=ma")


def default_bucketed_runs_root(domain: str) -> Path:
    key = domain.strip().lower()
    if key in {"chem", "chemistry"}:
        base = ROOT / "chem_data"
    else:
        base = ROOT / "data"
    return base / "bucketed_runs"


def latest_run_root(domain: str) -> Path:
    runs_root = default_bucketed_runs_root(domain)
    if not runs_root.is_dir():
        raise ValueError(f"No bucketed runs directory found at: {runs_root}")
    candidates = sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not candidates:
        raise ValueError(f"No run folders found under: {runs_root}")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Verify and score generated problems for every bucket in a bucketed run."
    )
    ap.add_argument("--run-root", default="", help="Path to a bucketed run root. If omitted, uses latest for --domain.")
    ap.add_argument("--domain", default="phys", help="Used only when --run-root is omitted: phys|chem.")
    ap.add_argument("--input-name", default="generated_problems.jsonl", help="Per-bucket generated file name.")
    ap.add_argument("--out-dir", default=str(ROOT / "scored"), help="Directory for scored output files.")

    ap.add_argument("--score-model", default="gpt-4.1-mini")
    ap.add_argument("--score-subject", default="", help="Override stage 06 --subject.")
    ap.add_argument("--score-competition", default="", help="Override stage 06 --competition.")
    ap.add_argument("--score-exemplars", nargs="*", default=[], help="Optional exemplar JSONL paths passed to stage 06.")
    ap.add_argument("--score-exemplar-limit", type=int, default=3)
    ap.add_argument("--score-exemplar-max-chars", type=int, default=1200)
    ap.add_argument("--score-resume", action="store_true", help="Pass --resume to stage 06.")

    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    ap.add_argument("--debug", action="store_true", help="Pass --debug to stage 06.")
    ap.add_argument("--continue-on-error", action="store_true", help="Keep processing remaining buckets if one fails.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not VERIFY_SCRIPT.is_file():
        raise ValueError(f"Missing verifier script: {VERIFY_SCRIPT}")
    run_root = Path(args.run_root) if args.run_root else latest_run_root(args.domain)
    buckets_root = run_root / "buckets"
    if not buckets_root.is_dir():
        raise ValueError(f"No bucket directory found at: {buckets_root}")

    bucket_dirs = sorted([p for p in buckets_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not bucket_dirs:
        raise ValueError(f"No bucket folders found under: {buckets_root}")

    default_subject, default_competition = infer_subject_competition(args.domain)
    subject = args.score_subject or default_subject
    competition = args.score_competition or default_competition
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[verify-buckets] run_root={run_root}", flush=True)
    scored = 0
    skipped = 0
    failed = 0
    for bucket_dir in bucket_dirs:
        bucket_name = bucket_dir.name
        input_path = bucket_dir / args.input_name
        rows = count_jsonl_rows(input_path)
        if rows <= 0:
            skipped += 1
            print(f"[verify-buckets] {bucket_name}: skip (missing/empty {args.input_name})", flush=True)
            continue

        out_path = out_dir / f"{run_root.name}__{bucket_name}.jsonl"
        ensure_parent(out_path)
        cmd = [
            sys.executable,
            str(VERIFY_SCRIPT),
            "--input",
            str(input_path),
            "--out",
            str(out_path),
            "--model",
            args.score_model,
            "--subject",
            subject,
            "--competition",
            competition,
        ]
        if args.score_exemplars:
            cmd.extend(["--exemplars", *args.score_exemplars])
            cmd.extend(["--exemplar-limit", str(args.score_exemplar_limit)])
            cmd.extend(["--exemplar-max-chars", str(args.score_exemplar_max_chars)])
        if args.score_resume:
            cmd.append("--resume")
        if args.debug:
            cmd.append("--debug")

        print(f"[verify-buckets] {bucket_name}: {shell_join(cmd)}", flush=True)
        if not args.dry_run:
            try:
                subprocess.run(cmd, check=True, cwd=str(ROOT))
            except subprocess.CalledProcessError as exc:
                failed += 1
                print(
                    f"[verify-buckets] {bucket_name}: FAILED (exit={exc.returncode})",
                    file=sys.stderr,
                    flush=True,
                )
                if not args.continue_on_error:
                    raise
                continue
        scored += 1

    print(
        f"[verify-buckets] done: scored={scored}, skipped={skipped}, failed={failed}, out_dir={out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
