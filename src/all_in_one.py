#!/usr/bin/env python3
"""Unified entrypoint for all scripts in src/.

This file wraps the existing stage scripts so you can run the whole project from one place,
with consistent path management and domain presets.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


ROOT = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent

SCRIPT_REGISTRY: Dict[str, Path] = {
    "00": SRC / "00_pdf-to-txt.py",
    "01": SRC / "01_enrich_problem_schema_paper.py",
    "02": SRC / "02_embed_and_index.py",
    "03": SRC / "03_retrieve.py",
    "04": SRC / "04_generate_skeletons.py",
    "05": SRC / "05_generate_questions.py",
    "06": SRC / "06_verify_and_score.py",
    "07": SRC / "07_direct_generate.py",
    "07_test": SRC / "07_direct_and_test.py",
    "generate": SRC / "generate.py",
    "generate_anchor": SRC / "generate_anchor.py",
    "jsonl_to_embed_txt": SRC / "jsonl_to_embed_txt.py",
    "solution_pdf": SRC / "solution_pdf-to-jsonl.py",
    "skeleton_creation": SRC / "skeleton_creation",
}


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def _run(cmd: List[str], dry_run: bool = False, env_overrides: Optional[Dict[str, str]] = None) -> None:
    cmd_pretty = " ".join(shlex.quote(c) for c in cmd)
    print(f"[run] {cmd_pretty}")
    if env_overrides:
        shown = ", ".join(f"{k}={v}" for k, v in sorted(env_overrides.items()))
        print(f"[env] {shown}")
    if dry_run:
        return
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)


def _log(debug: bool, msg: str) -> None:
    if debug:
        print(f"[debug] {msg}")


def _count_jsonl(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _truncate_jsonl(in_path: Path, out_path: Path, max_rows: int, debug: bool = False) -> Tuple[int, List[str]]:
    _ensure_parent(out_path)
    kept = 0
    ids: List[str] = []
    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1
            if isinstance(obj, dict):
                rid = obj.get("id")
                if isinstance(rid, str) and rid:
                    ids.append(rid)
            if kept >= max_rows:
                break
    _log(debug, f"capped {in_path} -> {out_path} rows={kept}")
    return kept, ids


def _filter_jsonl_by_ids(in_path: Path, out_path: Path, keep_ids: Set[str], debug: bool = False) -> int:
    _ensure_parent(out_path)
    kept = 0
    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rid = obj.get("id") if isinstance(obj, dict) else None
            if isinstance(rid, str) and rid in keep_ids:
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
    _log(debug, f"filtered {in_path} -> {out_path} rows={kept}")
    return kept


def _filter_anchors_by_ids(in_path: Path, out_path: Path, keep_ids: Set[str], debug: bool = False) -> int:
    _ensure_parent(out_path)
    kept = 0
    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                rid = obj
                out = obj
            elif isinstance(obj, dict):
                rid = obj.get("id")
                out = obj
            else:
                continue
            if isinstance(rid, str) and rid in keep_ids:
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
                kept += 1
    _log(debug, f"filtered anchors {in_path} -> {out_path} rows={kept}")
    return kept


def _api_env(args: argparse.Namespace) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if getattr(args, "api_base_url", ""):
        env["OPENAI_BASE_URL"] = args.api_base_url
    if getattr(args, "api_key", ""):
        env["OPENAI_API_KEY"] = args.api_key
    return env


def _load_domain_paths(domain: str) -> Dict[str, Path]:
    if domain == "chem":
        base = ROOT / "chem_data"
        return {
            "raw_questions": base / "txts" / "all_questions.jsonl",
            "enriched": base / "enriched_schemae" / "enriched.jsonl",
            "enriched_for_embed": base / "enriched_schemae" / "enriched_for_embedding.jsonl",
            "skeleton_embedded": base / "skeleton_embedded.jsonl",
            "question_embedded": base / "question_embedded.jsonl",
            "anchors": base / "anchors.jsonl",
            "retrieval_bundles": base / "retrieval_bundles.jsonl",
            "generated_skeletons": base / "generated_skeletons.jsonl",
            "generated_problems": base / "generated_problems.jsonl",
            "scored": base / "scored.jsonl",
            "exemplars": base / "txts" / "all_questions.jsonl",
        }

    # default phys
    return {
        "raw_questions": ROOT / "data" / "text" / "exams.txt" / "all_questions.jsonl",
        "enriched": ROOT / "data" / "enriched_schemae" / "enriched.jsonl",
        "enriched_for_embed": ROOT / "data" / "enriched_schemae" / "enriched_for_embedding.jsonl",
        "skeleton_embedded": ROOT / "data" / "skeleton_embedded.jsonl",
        "question_embedded": ROOT / "data" / "question_embedded.jsonl",
        "anchors": ROOT / "data" / "anchors.jsonl",
        "retrieval_bundles": ROOT / "retrieval_bundles.jsonl",
        "generated_skeletons": ROOT / "generated_skeletons.jsonl",
        "generated_problems": ROOT / "generated_problems.jsonl",
        "scored": ROOT / "scored.jsonl",
        "exemplars": ROOT / "data" / "text" / "exam1-2015-1-8.jsonl",
    }


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _normalize_enriched_for_embedding(in_path: Path, out_path: Path) -> int:
    """Bridge schema differences between stage 01 and stage 02.

    Stage 01 writes nested schema fields under problem/analysis.
    Stage 02 expects top-level question/skeleton in legacy format.
    This creates a compatibility JSONL while preserving original keys.
    """
    _ensure_parent(out_path)
    n = 0
    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            problem = row.get("problem") or {}
            analysis = row.get("analysis") or {}
            source = row.get("source") or {}

            stem = (problem.get("stem") or "").strip()
            choices = problem.get("choices") or []
            if isinstance(choices, list) and choices:
                q_text = (stem + "\n" + "\n".join(str(c) for c in choices)).strip()
            else:
                q_text = stem

            enriched = {
                **row,
                "question": q_text,
                "skeleton": analysis.get("skeleton"),
                "topic": (analysis.get("concepts") or [None])[0],
                "difficulty": analysis.get("difficulty"),
                "source": source.get("corpus") or source.get("pdf"),
            }

            f_out.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            n += 1
    return n


def _run_pipeline(args: argparse.Namespace) -> None:
    env = _api_env(args)
    paths = _load_domain_paths(args.domain)

    # Explicit overrides
    if args.input:
        paths["raw_questions"] = Path(args.input)
    if args.out_prefix:
        pfx = Path(args.out_prefix)
        paths["enriched"] = pfx / "enriched.jsonl"
        paths["enriched_for_embed"] = pfx / "enriched_for_embedding.jsonl"
        paths["skeleton_embedded"] = pfx / "skeleton_embedded.jsonl"
        paths["question_embedded"] = pfx / "question_embedded.jsonl"
        paths["anchors"] = pfx / "anchors.jsonl"
        paths["retrieval_bundles"] = pfx / "retrieval_bundles.jsonl"
        paths["generated_skeletons"] = pfx / "generated_skeletons.jsonl"
        paths["generated_problems"] = pfx / "generated_problems.jsonl"
        paths["scored"] = pfx / "scored.jsonl"

    for k in ["enriched", "enriched_for_embed", "skeleton_embedded", "question_embedded", "anchors", "retrieval_bundles", "generated_skeletons", "generated_problems", "scored"]:
        _ensure_parent(paths[k])

    stage_order = ["01", "02", "03", "04", "05", "06"]
    start_i = stage_order.index(args.from_stage)
    end_i = stage_order.index(args.to_stage)
    if start_i > end_i:
        raise ValueError("--from-stage must be <= --to-stage")
    active_stages = stage_order[start_i : end_i + 1]
    _log(args.debug, f"active stages: {active_stages}")
    _log(args.debug, f"domain={args.domain} question_cap={args.question_cap} limit={args.limit}")

    if args.question_cap and args.question_cap > 0 and "01" not in active_stages:
        # When stage 01 is skipped, cap existing datasets before downstream stages.
        cap_n = args.question_cap
        cap_base = paths["enriched"].parent
        capped_enriched = cap_base / f"{paths['enriched'].stem}.capped{cap_n}.jsonl"
        if not args.dry_run:
            _, capped_ids = _truncate_jsonl(paths["enriched"], capped_enriched, cap_n, args.debug)
            keep_ids = set(capped_ids)
            paths["enriched"] = capped_enriched

            if keep_ids and "02" not in active_stages and "03" in active_stages:
                capped_skel = paths["skeleton_embedded"].parent / f"{paths['skeleton_embedded'].stem}.capped{cap_n}.jsonl"
                capped_q = paths["question_embedded"].parent / f"{paths['question_embedded'].stem}.capped{cap_n}.jsonl"
                capped_anchors = paths["anchors"].parent / f"{paths['anchors'].stem}.capped{cap_n}.jsonl"
                _filter_jsonl_by_ids(paths["skeleton_embedded"], capped_skel, keep_ids, args.debug)
                _filter_jsonl_by_ids(paths["question_embedded"], capped_q, keep_ids, args.debug)
                _filter_anchors_by_ids(paths["anchors"], capped_anchors, keep_ids, args.debug)
                paths["skeleton_embedded"] = capped_skel
                paths["question_embedded"] = capped_q
                paths["anchors"] = capped_anchors
        else:
            _log(args.debug, f"dry-run: would cap enriched to {capped_enriched} rows={cap_n}")

    if "01" in active_stages:
        stage01_limit = args.limit
        if args.question_cap and args.question_cap > 0:
            stage01_limit = args.question_cap if not stage01_limit else min(stage01_limit, args.question_cap)
            _log(args.debug, f"stage 01 effective limit={stage01_limit}")

        cmd_01 = [
            sys.executable,
            str(SCRIPT_REGISTRY["01"]),
            "--input", str(paths["raw_questions"]),
            "--out", str(paths["enriched"]),
            "--model", args.chat_model,
            "--max_steps", str(args.max_steps),
            "--allowed_ops", args.allowed_ops,
        ]
        if stage01_limit:
            cmd_01 += ["--limit", str(stage01_limit)]
        if args.debug:
            cmd_01 += ["--debug"]
        _run(cmd_01, args.dry_run, env)
        if not args.dry_run:
            _log(args.debug, f"stage 01 wrote rows={_count_jsonl(paths['enriched'])} file={_repo_rel(paths['enriched'])}")

    if "02" in active_stages:
        embed_input = paths["enriched"]
        if args.normalize_for_embed:
            if not args.dry_run:
                n = _normalize_enriched_for_embedding(paths["enriched"], paths["enriched_for_embed"])
                print(
                    f"[info] normalized {n} rows for embedding: "
                    f"{_repo_rel(paths['enriched_for_embed'])}"
                )
            else:
                print(
                    "[info] dry-run: would normalize schema for embedding to "
                    f"{_repo_rel(paths['enriched_for_embed'])}"
                )
            embed_input = paths["enriched_for_embed"]
        if args.question_cap and args.question_cap > 0 and "01" not in active_stages:
            cap_n = args.question_cap
            capped_embed = embed_input.parent / f"{embed_input.stem}.capped{cap_n}.jsonl"
            if not args.dry_run:
                _truncate_jsonl(embed_input, capped_embed, cap_n, args.debug)
            else:
                _log(args.debug, f"dry-run: would cap {embed_input} to {capped_embed} rows={cap_n}")
            embed_input = capped_embed
        _log(args.debug, f"stage 02 embed input: {_repo_rel(embed_input)}")

        cmd_02 = [
            sys.executable,
            str(SCRIPT_REGISTRY["02"]),
            "--input", str(embed_input),
            "--embed_model", args.embedding_model,
            "--out_skel", str(paths["skeleton_embedded"]),
            "--out_q", str(paths["question_embedded"]),
            "--out_anchors", str(paths["anchors"]),
        ]
        _run(cmd_02, args.dry_run, env)
        if not args.dry_run:
            _log(args.debug, f"stage 02 skeleton rows={_count_jsonl(paths['skeleton_embedded'])}")
            _log(args.debug, f"stage 02 question rows={_count_jsonl(paths['question_embedded'])}")
            _log(args.debug, f"stage 02 anchors rows={_count_jsonl(paths['anchors'])}")

    if "03" in active_stages:
        if args.question_cap and args.question_cap > 0 and args.num_bundles == 0:
            _log(args.debug, f"stage 03 auto-setting --num_bundles={args.question_cap} from question cap")
        cmd_03 = [
            sys.executable,
            str(SCRIPT_REGISTRY["03"]),
            "--skeleton_embedded", str(paths["skeleton_embedded"]),
            "--question_embedded", str(paths["question_embedded"]),
            "--anchors", str(paths["anchors"]),
            "--enriched", str(paths["enriched"]),
            "--out", str(paths["retrieval_bundles"]),
            "--require_skeleton_text",
        ]
        if args.num_bundles:
            cmd_03 += ["--num_bundles", str(args.num_bundles)]
        elif args.question_cap and args.question_cap > 0:
            cmd_03 += ["--num_bundles", str(args.question_cap)]
        if args.seed:
            cmd_03 += ["--seed", str(args.seed)]
        _run(cmd_03, args.dry_run, env)
        if not args.dry_run:
            _log(args.debug, f"stage 03 bundles rows={_count_jsonl(paths['retrieval_bundles'])}")

    if "04" in active_stages:
        cmd_04 = [
            sys.executable,
            str(SCRIPT_REGISTRY["04"]),
            "--bundles", str(paths["retrieval_bundles"]),
            "--out", str(paths["generated_skeletons"]),
            "--model", args.chat_model,
            "--max_steps", str(args.gen_skeleton_max_steps),
        ]
        if args.limit:
            cmd_04 += ["--limit", str(args.limit)]
        if args.debug:
            cmd_04 += ["--debug"]
        _run(cmd_04, args.dry_run, env)
        if not args.dry_run:
            _log(args.debug, f"stage 04 generated skeleton rows={_count_jsonl(paths['generated_skeletons'])}")

    if "05" in active_stages:
        cmd_05 = [
            sys.executable,
            str(SCRIPT_REGISTRY["05"]),
            "--bundles", str(paths["retrieval_bundles"]),
            "--skeletons", str(paths["generated_skeletons"]),
            "--out", str(paths["generated_problems"]),
            "--model", args.chat_model,
            "--repair_max", str(args.repair_max),
        ]
        if args.limit:
            cmd_05 += ["--limit", str(args.limit)]
        if args.debug:
            cmd_05 += ["--debug"]
        _run(cmd_05, args.dry_run, env)
        if not args.dry_run:
            _log(args.debug, f"stage 05 generated problem rows={_count_jsonl(paths['generated_problems'])}")

    if "06" in active_stages:
        cmd_06 = [
            sys.executable,
            str(SCRIPT_REGISTRY["06"]),
            "--input", str(paths["generated_problems"]),
            "--out", str(paths["scored"]),
            "--model", args.verify_model or args.chat_model,
            "--subject", args.subject,
            "--competition", args.competition,
            "--exemplars", str(paths["exemplars"]),
        ]
        if args.limit:
            cmd_06 += ["--limit", str(args.limit)]
        if args.debug:
            cmd_06 += ["--debug"]
        _run(cmd_06, args.dry_run, env)
        if not args.dry_run:
            _log(args.debug, f"stage 06 scored rows={_count_jsonl(paths['scored'])}")

    print("\nPipeline complete. Outputs:")
    print(f"- Enriched: {_repo_rel(paths['enriched'])}")
    if args.normalize_for_embed:
        print(f"- Embed-ready input: {_repo_rel(paths['enriched_for_embed'])}")
    print(f"- Retrieval bundles: {_repo_rel(paths['retrieval_bundles'])}")
    print(f"- Generated problems: {_repo_rel(paths['generated_problems'])}")
    print(f"- Scored: {_repo_rel(paths['scored'])}")


def _run_direct(args: argparse.Namespace) -> None:
    env = _api_env(args)
    cmd = [
        sys.executable,
        str(SCRIPT_REGISTRY["07"]),
        "--input", args.input,
        "--out", args.out,
        "--model", args.chat_model,
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.debug:
        cmd += ["--debug"]
    _run(cmd, args.dry_run, env)


def _run_pdf(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_REGISTRY["00"]),
        "--in", args.input_dir,
        "--out", args.out_dir,
    ]
    if args.recursive:
        cmd.append("--recursive")
    _run(cmd, args.dry_run)


def _run_solution_pdf(args: argparse.Namespace) -> None:
    script = SCRIPT_REGISTRY["solution_pdf"]
    cmd = [sys.executable, str(script), args.input_pdf, "-o", args.out]
    if args.debug:
        cmd.append("--debug")
    _run(cmd, args.dry_run)


def _run_script(args: argparse.Namespace) -> None:
    env = _api_env(args)
    script = SCRIPT_REGISTRY[args.name]
    cmd = [sys.executable, str(script)] + list(args.args)
    _run(cmd, args.dry_run, env)


def _run_direct_test(args: argparse.Namespace) -> None:
    env = _api_env(args)
    cmd = [
        sys.executable,
        str(SCRIPT_REGISTRY["07_test"]),
        "--input", args.input,
        "--generated-out", args.generated_out,
        "--scored-out", args.scored_out,
        "--chat-model", args.chat_model,
        "--verify-model", args.verify_model,
        "--subject", args.subject,
        "--competition", args.competition,
        "--exemplars",
    ] + list(args.exemplars)
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.debug:
        cmd += ["--debug"]
    _run(cmd, args.dry_run, env)


def _list_scripts(_: argparse.Namespace) -> None:
    print("Available script keys:")
    for key in sorted(SCRIPT_REGISTRY.keys()):
        print(f"- {key}: {_repo_rel(SCRIPT_REGISTRY[key])}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Single entrypoint for full pipeline + all src scripts."
    )
    sub = p.add_subparsers(dest="command", required=True)

    # full pipeline (01->06)
    p_pipeline = sub.add_parser("pipeline", help="Run stage pipeline 01->06")
    p_pipeline.add_argument("--domain", choices=["chem", "phys"], default="chem")
    p_pipeline.add_argument("--input", default="", help="Override raw questions JSONL")
    p_pipeline.add_argument("--out-prefix", default="", help="Write all outputs under this folder")
    p_pipeline.add_argument("--from-stage", choices=["01", "02", "03", "04", "05", "06"], default="01")
    p_pipeline.add_argument("--to-stage", choices=["01", "02", "03", "04", "05", "06"], default="06")
    p_pipeline.add_argument("--chat-model", default="qwen2.5:7b-instruct")
    p_pipeline.add_argument("--embedding-model", default="qwen3-embedding")
    p_pipeline.add_argument("--verify-model", default="")
    p_pipeline.add_argument("--api-base-url", default="http://192.168.50.186:11434/v1")
    p_pipeline.add_argument("--api-key", default="ollama")
    p_pipeline.add_argument("--subject", default="Chem")
    p_pipeline.add_argument("--competition", default="USNCO")
    p_pipeline.add_argument("--max-steps", type=int, default=8, help="Stage 01 max skeleton steps")
    p_pipeline.add_argument("--gen-skeleton-max-steps", type=int, default=10, help="Stage 04 max skeleton steps")
    p_pipeline.add_argument(
        "--allowed-ops",
        default="IDENTIFY_GIVENS,IDENTIFY_RELATION,APPLY_RELATION,INTRODUCE_AUX,CONSTRAINT_COUPLING,CASEWORK/REGIME,INVARIANT/SYMMETRY,CHECK/SANITY,SAVE_RESULT",
        help="Stage 01 allowed ops list",
    )
    p_pipeline.add_argument("--num-bundles", type=int, default=0)
    p_pipeline.add_argument("--seed", type=int, default=0)
    p_pipeline.add_argument("--repair-max", type=int, default=1)
    p_pipeline.add_argument("--limit", type=int, default=0)
    p_pipeline.add_argument(
        "--question-cap",
        type=int,
        default=0,
        help="Hard cap on number of questions considered end-to-end (0 = disabled).",
    )
    p_pipeline.add_argument("--normalize-for-embed", action="store_true", default=True)
    p_pipeline.add_argument("--no-normalize-for-embed", action="store_false", dest="normalize_for_embed")
    p_pipeline.add_argument("--debug", action="store_true")
    p_pipeline.add_argument("--dry-run", action="store_true")
    p_pipeline.set_defaults(func=_run_pipeline)

    # direct stage 07
    p_direct = sub.add_parser("direct", help="Run direct generation (07)")
    p_direct.add_argument("--input", required=True)
    p_direct.add_argument("--out", default=str(ROOT / "direct_generated.jsonl"))
    p_direct.add_argument("--chat-model", default="qwen2.5:7b-instruct")
    p_direct.add_argument("--api-base-url", default="http://192.168.50.186:11434/v1")
    p_direct.add_argument("--api-key", default="ollama")
    p_direct.add_argument("--limit", type=int, default=0)
    p_direct.add_argument("--debug", action="store_true")
    p_direct.add_argument("--dry-run", action="store_true")
    p_direct.set_defaults(func=_run_direct)

    p_direct_test = sub.add_parser("direct-test", help="Run direct generation (07) then score/test (06)")
    p_direct_test.add_argument("--input", required=True)
    p_direct_test.add_argument("--generated-out", default=str(ROOT / "direct_generated.jsonl"))
    p_direct_test.add_argument("--scored-out", default=str(ROOT / "direct_scored.jsonl"))
    p_direct_test.add_argument("--chat-model", default="qwen2.5:7b-instruct")
    p_direct_test.add_argument("--verify-model", default="")
    p_direct_test.add_argument("--api-base-url", default="http://192.168.50.186:11434/v1")
    p_direct_test.add_argument("--api-key", default="ollama")
    p_direct_test.add_argument("--subject", default="Chem")
    p_direct_test.add_argument("--competition", default="USNCO")
    p_direct_test.add_argument("--exemplars", nargs="*", default=["chem_data/txts/all_questions.jsonl"])
    p_direct_test.add_argument("--limit", type=int, default=0)
    p_direct_test.add_argument("--debug", action="store_true")
    p_direct_test.add_argument("--dry-run", action="store_true")
    p_direct_test.set_defaults(func=_run_direct_test)

    # pdf -> text/jsonl (00)
    p_pdf = sub.add_parser("pdf", help="Run PDF parsing stage (00)")
    p_pdf.add_argument("--input-dir", required=True)
    p_pdf.add_argument("--out-dir", required=True)
    p_pdf.add_argument("--recursive", action="store_true")
    p_pdf.add_argument("--dry-run", action="store_true")
    p_pdf.set_defaults(func=_run_pdf)

    # solution pdf parser
    p_solpdf = sub.add_parser("solution-pdf", help="Run solution PDF parser")
    p_solpdf.add_argument("--input-pdf", required=True)
    p_solpdf.add_argument("--out", required=True)
    p_solpdf.add_argument("--debug", action="store_true")
    p_solpdf.add_argument("--dry-run", action="store_true")
    p_solpdf.set_defaults(func=_run_solution_pdf)

    # passthrough script runner
    p_script = sub.add_parser("script", help="Run any script in src/ by key")
    p_script.add_argument("--api-base-url", default="http://192.168.50.186:11434/v1")
    p_script.add_argument("--api-key", default="ollama")
    p_script.add_argument("--dry-run", action="store_true")
    p_script.add_argument("name", choices=sorted(SCRIPT_REGISTRY.keys()))
    p_script.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the target script")
    p_script.set_defaults(func=_run_script)

    p_list = sub.add_parser("list", help="List available script keys")
    p_list.set_defaults(func=_list_scripts)

    return p


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.func(args)


if __name__ == "__main__":
    main()
