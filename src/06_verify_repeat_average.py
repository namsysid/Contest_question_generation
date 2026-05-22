#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
VERIFY_SCRIPT = SRC_DIR / "06_verify_and_score.py"


def shell_join(parts: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def mean(values: Iterable[float]) -> Optional[float]:
    seq = list(values)
    if not seq:
        return None
    return sum(seq) / len(seq)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run verifier multiple times on one file and average score statistics."
    )
    ap.add_argument("--input", required=True, help="Input generated_problems JSONL file.")
    ap.add_argument("--repeats", type=int, default=10, help="How many repeated verify runs.")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--subject", default="Physics")
    ap.add_argument("--competition", default="f=ma")
    ap.add_argument("--exemplars", nargs="*", default=[], help="Optional exemplar JSONL paths.")
    ap.add_argument("--exemplar-limit", type=int, default=3)
    ap.add_argument("--exemplar-max-chars", type=int, default=1200)
    ap.add_argument("--limit", type=int, default=0, help="Optional row limit forwarded to verifier.")

    ap.add_argument("--out-summary", default="", help="Path to summary JSON output.")
    ap.add_argument("--runs-dir", default="", help="Directory to store per-run scored JSONL files.")
    ap.add_argument(
        "--runs-folder-name",
        default="",
        help="Folder name under scored/repeat_runs (ignored when --runs-dir is set).",
    )
    ap.add_argument(
        "--runs-layout",
        choices=["per_input", "timestamp"],
        default="per_input",
        help="Default run folder strategy when --runs-dir is not provided.",
    )
    ap.add_argument("--keep-runs", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()


def aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    diff_scores: List[float] = []
    depth_scores: List[float] = []
    rich_scores: List[float] = []
    clarity_scores: List[float] = []
    olympiad_scores: List[float] = []
    solved_count = 0
    partial_count = 0
    unclear_count = 0
    unknown_answer_count = 0

    for row in rows:
        solve = row.get("solve_attempt") or {}
        status = str(solve.get("status", "")).lower()
        if status == "solved":
            solved_count += 1
        elif status == "partial":
            partial_count += 1
        elif status == "unclear":
            unclear_count += 1

        selected_answer = str(solve.get("selected_answer", "")).upper()
        if selected_answer == "UNKNOWN":
            unknown_answer_count += 1

        diff = to_float(((row.get("difficulty_assessment") or {}).get("score")))
        if diff is not None:
            diff_scores.append(diff)

        comp = row.get("competition_appropriateness") or {}
        depth = to_float(comp.get("depth_reasoning"))
        rich = to_float(comp.get("conceptual_richness"))
        clarity = to_float(comp.get("clarity"))
        olympiad = to_float(comp.get("olympiad_similarity"))
        if depth is not None:
            depth_scores.append(depth)
        if rich is not None:
            rich_scores.append(rich)
        if clarity is not None:
            clarity_scores.append(clarity)
        if olympiad is not None:
            olympiad_scores.append(olympiad)

    n = len(rows)
    return {
        "row_count": n,
        "means": {
            "difficulty_score": mean(diff_scores),
            "depth_reasoning": mean(depth_scores),
            "conceptual_richness": mean(rich_scores),
            "clarity": mean(clarity_scores),
            "olympiad_similarity": mean(olympiad_scores),
        },
        "rates": {
            "solved_rate": (solved_count / n) if n else None,
            "partial_rate": (partial_count / n) if n else None,
            "unclear_rate": (unclear_count / n) if n else None,
            "unknown_answer_rate": (unknown_answer_count / n) if n else None,
        },
    }


def main() -> int:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")
    if not VERIFY_SCRIPT.is_file():
        raise ValueError(f"Missing verifier script: {VERIFY_SCRIPT}")

    input_path = Path(args.input)
    if not input_path.is_file():
        raise ValueError(f"Input file not found: {input_path}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.runs_dir:
        runs_dir = Path(args.runs_dir)
    elif args.runs_folder_name:
        runs_dir = ROOT / "scored" / "repeat_runs" / args.runs_folder_name
    else:
        safe_name = input_path.stem.replace(" ", "_")
        if args.runs_layout == "timestamp":
            runs_dir = ROOT / "scored" / "repeat_runs" / f"{safe_name}_{stamp}"
        else:
            runs_dir = ROOT / "scored" / "repeat_runs" / safe_name
    runs_dir.mkdir(parents=True, exist_ok=True)

    out_summary = (
        Path(args.out_summary)
        if args.out_summary
        else (ROOT / "scored" / f"{input_path.stem}__repeat_avg_{stamp}.json")
    )
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[repeat-verify] input={input_path} repeats={args.repeats} runs_dir={runs_dir}",
        flush=True,
    )

    aggregate_input_rows: List[Dict[str, Any]] = []
    per_run_stats: List[Dict[str, Any]] = []

    for i in range(1, args.repeats + 1):
        run_out = runs_dir / f"{i}.jsonl"
        cmd = [
            sys.executable,
            str(VERIFY_SCRIPT),
            "--input",
            str(input_path),
            "--out",
            str(run_out),
            "--model",
            args.model,
            "--subject",
            args.subject,
            "--competition",
            args.competition,
        ]
        if args.exemplars:
            cmd.extend(["--exemplars", *args.exemplars])
            cmd.extend(["--exemplar-limit", str(args.exemplar_limit)])
            cmd.extend(["--exemplar-max-chars", str(args.exemplar_max_chars)])
        if args.limit > 0:
            cmd.extend(["--limit", str(args.limit)])
        if args.debug:
            cmd.append("--debug")

        print(f"[repeat-verify] run {i}/{args.repeats}: {shell_join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=str(ROOT))

        run_rows = read_jsonl(run_out)
        aggregate_input_rows.extend(run_rows)
        per_run_stats.append({
            "run_index": i,
            "out_file": str(run_out),
            **aggregate_rows(run_rows),
        })
        print(f"[repeat-verify] run {i}/{args.repeats}: rows={len(run_rows)}", flush=True)

    combined = aggregate_rows(aggregate_input_rows)
    summary = {
        "input_file": str(input_path),
        "repeats": args.repeats,
        "model": args.model,
        "subject": args.subject,
        "competition": args.competition,
        "generated_at": datetime.now().isoformat(),
        "combined_stats": combined,
        "per_run_stats": per_run_stats,
    }

    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[repeat-verify] wrote summary: {out_summary}", flush=True)

    if not args.keep_runs:
        for run in runs_dir.glob("*.jsonl"):
            run.unlink(missing_ok=True)
        try:
            runs_dir.rmdir()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
