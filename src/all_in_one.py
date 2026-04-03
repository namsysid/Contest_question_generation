#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    name: str
    script: str


STAGES: List[StageSpec] = [
    StageSpec("01", "enrich", "01_enrich_problem_schema_paper-2.py"),
    StageSpec("02", "embed", "02_embed_and_index.py"),
    StageSpec("03", "retrieve", "03_retrieve.py"),
    StageSpec("04", "generate_skeletons", "04_generate_skeletons.py"),
    StageSpec("05", "generate_questions", "05_generate_questions.py"),
]
STAGE_IDS = {stage.stage_id for stage in STAGES}


def parse_stage_id(raw: str) -> str:
    text = str(raw).strip()
    if text.isdigit():
        text = f"{int(text):02d}"
    if text not in STAGE_IDS:
        raise argparse.ArgumentTypeError(f"Unsupported stage '{raw}'. Expected one of: {', '.join(sorted(STAGE_IDS))}")
    return text


def default_domain_paths(domain: str) -> Dict[str, Path]:
    key = domain.strip().lower()
    if key in {"chem", "chemistry"}:
        base = ROOT / "chem_data"
    elif key in {"phys", "physics", "fma"}:
        base = ROOT / "data"
    else:
        base = ROOT / key

    return {
        "base": base,
        "input": base / "txts" / "all_questions.jsonl",
        "enriched": base / "enriched_schemae" / "enriched.jsonl",
        "skeleton_embedded": base / "skeleton_embedded.jsonl",
        "question_embedded": base / "question_embedded.jsonl",
        "anchors": base / "anchors.jsonl",
        "bundles": base / "retrieval_bundles.jsonl",
        "generated_skeletons": base / "generated_skeletons.jsonl",
        "generated_problems": base / "generated_problems.jsonl",
    }


def infer_stage_domain(domain: str) -> str:
    key = domain.strip().lower()
    if key in {"chem", "chemistry"}:
        return "chem"
    return "fma"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def run_stage(stage: StageSpec, args: List[str], dry_run: bool) -> None:
    cmd = [sys.executable, str(SRC_DIR / stage.script), *args]
    print(f"[{stage.stage_id}] {stage.name}: {shell_join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def build_stage_commands(args: argparse.Namespace) -> Dict[str, List[str]]:
    paths = default_domain_paths(args.domain)

    input_path = Path(args.input) if args.input else paths["input"]
    enriched_path = Path(args.enriched_out) if args.enriched_out else paths["enriched"]
    skeleton_embedded_path = Path(args.skeleton_embedded_out) if args.skeleton_embedded_out else paths["skeleton_embedded"]
    question_embedded_path = Path(args.question_embedded_out) if args.question_embedded_out else paths["question_embedded"]
    anchors_path = Path(args.anchors_out) if args.anchors_out else paths["anchors"]
    bundles_path = Path(args.bundles_out) if args.bundles_out else paths["bundles"]
    generated_skeletons_path = Path(args.generated_skeletons_out) if args.generated_skeletons_out else paths["generated_skeletons"]
    generated_problems_path = Path(args.generated_problems_out) if args.generated_problems_out else paths["generated_problems"]

    outputs = [
        enriched_path,
        skeleton_embedded_path,
        question_embedded_path,
        anchors_path,
        bundles_path,
        generated_skeletons_path,
        generated_problems_path,
    ]
    for path in outputs:
        ensure_parent(path)

    common_debug = ["--debug"] if args.debug else []
    shared_limit = ["--limit", str(args.limit)] if args.limit > 0 else []
    stage_domain = args.stage_domain or infer_stage_domain(args.domain)

    commands: Dict[str, List[str]] = {}

    commands["01"] = [
        "--input", str(input_path),
        "--out", str(enriched_path),
        "--model", args.enrich_model,
        "--corpus", args.corpus,
        "--year", str(args.year),
        "--variant", args.variant,
        "--grammar-mode", args.grammar_mode,
        "--max-depth", str(args.max_depth),
        "--max-branching", str(args.max_branching),
        "--max-nodes", str(args.max_nodes),
        "--max-edges", str(args.max_edges),
        *shared_limit,
        *common_debug,
    ]

    commands["02"] = [
        "--input", str(enriched_path),
        "--embed_model", args.embed_model,
        "--anchor_frac", str(args.anchor_frac),
        "--k_density", str(args.k_density),
        "--out_skel", str(skeleton_embedded_path),
        "--out_q", str(question_embedded_path),
        "--out_anchors", str(anchors_path),
    ]

    commands["03"] = [
        "--skeleton_embedded", str(skeleton_embedded_path),
        "--question_embedded", str(question_embedded_path),
        "--anchors", str(anchors_path),
        "--enriched", str(enriched_path),
        "--num_bundles", str(args.num_bundles),
        "--annulus_min", str(args.annulus_min),
        "--annulus_max", str(args.annulus_max),
        "--tail_frac", str(args.tail_frac),
        "--tail_min", str(args.tail_min),
        "--k_solution", str(args.k_solution),
        "--k_question", str(args.k_question),
        "--k_paired", str(args.k_paired),
        "--mmr_lambda_solution", str(args.mmr_lambda_solution),
        "--mmr_lambda_question", str(args.mmr_lambda_question),
        "--out", str(bundles_path),
        "--seed", str(args.seed),
    ]
    if args.require_skeleton_text:
        commands["03"].append("--require_skeleton_text")
    if args.require_insight_seed:
        commands["03"].append("--require_insight_seed")
    if args.require_insight_solution_exemplars:
        commands["03"].append("--require_insight_solution_exemplars")
    if args.min_seed_complexity > 0:
        commands["03"].extend(["--min_seed_complexity", str(args.min_seed_complexity)])
    if args.min_sol_ex_complexity > 0:
        commands["03"].extend(["--min_sol_ex_complexity", str(args.min_sol_ex_complexity)])

    commands["04"] = [
        "--bundles", str(bundles_path),
        "--out", str(generated_skeletons_path),
        "--model", args.graph_model,
        "--domain", stage_domain,
        *shared_limit,
        *common_debug,
    ]

    commands["05"] = [
        "--bundles", str(bundles_path),
        "--skeletons", str(generated_skeletons_path),
        "--out", str(generated_problems_path),
        "--model", args.question_model,
        "--max_q_exemplars", str(args.max_q_exemplars),
        "--max_paired_exemplars", str(args.max_paired_exemplars),
        "--repair_max", str(args.repair_max),
        "--sleep", str(args.sleep),
        *shared_limit,
        *common_debug,
    ]
    if args.verify_model:
        commands["05"].extend(["--verify_model", args.verify_model])

    return commands


def selected_stages(from_stage: str, to_stage: str) -> List[StageSpec]:
    start = next(i for i, stage in enumerate(STAGES) if stage.stage_id == from_stage)
    end = next(i for i, stage in enumerate(STAGES) if stage.stage_id == to_stage)
    if start > end:
        raise ValueError(f"from-stage {from_stage} must be <= to-stage {to_stage}")
    return STAGES[start : end + 1]


def cmd_list(_: argparse.Namespace) -> int:
    print("Available stages:")
    for stage in STAGES:
        print(f"  {stage.stage_id}  {stage.name:<20} {stage.script}")
    print("")
    print("Example:")
    print("  python3 src/all_in_one.py pipeline --domain phys")
    print("  python3 src/all_in_one.py pipeline --domain chem --from-stage 03 --to-stage 05")
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    commands = build_stage_commands(args)
    stages = selected_stages(args.from_stage, args.to_stage)
    for stage in stages:
        run_stage(stage, commands[stage.stage_id], dry_run=args.dry_run)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run stages 01-05 through a single entrypoint.")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="Show wrapped pipeline stages")
    list_parser.set_defaults(func=cmd_list)

    pipe = sub.add_parser("pipeline", help="Run any contiguous subset of stages 01-05")
    pipe.set_defaults(func=cmd_pipeline)

    pipe.add_argument("--domain", default="phys", help="Preset path family: phys -> data, chem -> chem_data")
    pipe.add_argument("--stage-domain", default="", help="Domain string passed into stage 04; default is inferred from --domain")
    pipe.add_argument("--from-stage", type=parse_stage_id, default="01")
    pipe.add_argument("--to-stage", type=parse_stage_id, default="05")
    pipe.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    pipe.add_argument("--debug", action="store_true", help="Pass --debug to stages that support it")
    pipe.add_argument("--limit", type=int, default=0, help="Apply the same row limit to stages 01, 04, and 05")

    pipe.add_argument("--input", default="", help="Override stage 01 input JSONL")
    pipe.add_argument("--enriched-out", default="", help="Override stage 01 output path")
    pipe.add_argument("--skeleton-embedded-out", default="", help="Override stage 02 structural embedding output")
    pipe.add_argument("--question-embedded-out", default="", help="Override stage 02 question embedding output")
    pipe.add_argument("--anchors-out", default="", help="Override stage 02 anchor output")
    pipe.add_argument("--bundles-out", default="", help="Override stage 03 retrieval bundle output")
    pipe.add_argument("--generated-skeletons-out", default="", help="Override stage 04 output path")
    pipe.add_argument("--generated-problems-out", default="", help="Override stage 05 output path")

    pipe.add_argument("--enrich-model", default="qwen2.5:7b-instruct")
    pipe.add_argument("--embed-model", default="qwen3-embedding")
    pipe.add_argument("--graph-model", default="qwen2.5:7b-instruct")
    pipe.add_argument("--question-model", default="qwen2.5:7b-instruct")
    pipe.add_argument("--verify-model", default="")

    pipe.add_argument("--corpus", default="unknown")
    pipe.add_argument("--year", type=int, default=0)
    pipe.add_argument("--variant", default="")
    pipe.add_argument("--grammar-mode", choices=["off", "warn", "strict"], default="strict")
    pipe.add_argument("--max-depth", type=int, default=4)
    pipe.add_argument("--max-branching", type=int, default=3)
    pipe.add_argument("--max-nodes", type=int, default=10)
    pipe.add_argument("--max-edges", type=int, default=14)

    pipe.add_argument("--anchor-frac", type=float, default=0.18)
    pipe.add_argument("--k-density", type=int, default=20)

    pipe.add_argument("--num-bundles", type=int, default=25)
    pipe.add_argument("--annulus-min", type=float, default=0.08)
    pipe.add_argument("--annulus-max", type=float, default=0.40)
    pipe.add_argument("--tail-frac", type=float, default=0.35)
    pipe.add_argument("--tail-min", type=float, default=0.30)
    pipe.add_argument("--k-solution", type=int, default=4)
    pipe.add_argument("--k-question", type=int, default=4)
    pipe.add_argument("--k-paired", type=int, default=3)
    pipe.add_argument("--mmr-lambda-solution", type=float, default=0.7)
    pipe.add_argument("--mmr-lambda-question", type=float, default=0.7)
    pipe.add_argument("--seed", type=int, default=0)
    pipe.add_argument("--require-skeleton-text", action=argparse.BooleanOptionalAction, default=True)
    pipe.add_argument("--require-insight-seed", action="store_true")
    pipe.add_argument("--require-insight-solution-exemplars", action="store_true")
    pipe.add_argument("--min-seed-complexity", type=int, default=0)
    pipe.add_argument("--min-sol-ex-complexity", type=int, default=0)

    pipe.add_argument("--max-q-exemplars", type=int, default=4)
    pipe.add_argument("--max-paired-exemplars", type=int, default=3)
    pipe.add_argument("--repair-max", type=int, default=1)
    pipe.add_argument("--sleep", type=float, default=0.0)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
