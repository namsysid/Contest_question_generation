#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    name: str
    script: str


@dataclass
class BucketGroup:
    bucket_ids: List[str]
    rows: List[Dict[str, Any]]
    centroid_breadth: float
    centroid_depth: float


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
    if key in {"phys", "physics", "fma"}:
        return "fma"
    return key or "fma"


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


def build_stage_commands_for_paths(
    args: argparse.Namespace,
    *,
    input_path: Path,
    enriched_path: Path,
    skeleton_embedded_path: Path,
    question_embedded_path: Path,
    anchors_path: Path,
    bundles_path: Path,
    generated_skeletons_path: Path,
    generated_problems_path: Path,
) -> Dict[str, List[str]]:
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
        *shared_limit,
        *common_debug,
    ]
    if args.max_depth is not None:
        commands["01"].extend(["--max-depth", str(args.max_depth)])
    if args.max_branching is not None:
        commands["01"].extend(["--max-branching", str(args.max_branching)])
    if args.max_nodes is not None:
        commands["01"].extend(["--max-nodes", str(args.max_nodes)])
    if args.max_edges is not None:
        commands["01"].extend(["--max-edges", str(args.max_edges)])

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


def build_stage01_command(args: argparse.Namespace, *, input_path: Path, enriched_path: Path) -> List[str]:
    ensure_parent(enriched_path)
    shared_limit = ["--limit", str(args.limit)] if args.limit > 0 else []
    common_debug = ["--debug"] if args.debug else []
    cmd = [
        "--input", str(input_path),
        "--out", str(enriched_path),
        "--model", args.enrich_model,
        "--corpus", args.corpus,
        "--year", str(args.year),
        "--variant", args.variant,
        "--grammar-mode", args.grammar_mode,
        *shared_limit,
        *common_debug,
    ]
    if args.max_depth is not None:
        cmd.extend(["--max-depth", str(args.max_depth)])
    if args.max_branching is not None:
        cmd.extend(["--max-branching", str(args.max_branching)])
    if args.max_nodes is not None:
        cmd.extend(["--max-nodes", str(args.max_nodes)])
    if args.max_edges is not None:
        cmd.extend(["--max-edges", str(args.max_edges)])
    return cmd


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

    return build_stage_commands_for_paths(
        args,
        input_path=input_path,
        enriched_path=enriched_path,
        skeleton_embedded_path=skeleton_embedded_path,
        question_embedded_path=question_embedded_path,
        anchors_path=anchors_path,
        bundles_path=bundles_path,
        generated_skeletons_path=generated_skeletons_path,
        generated_problems_path=generated_problems_path,
    )


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
    print("  python3 src/all_in_one.py bucketed-pipeline --domain phys")
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    commands = build_stage_commands(args)
    stages = selected_stages(args.from_stage, args.to_stage)
    for stage in stages:
        run_stage(stage, commands[stage.stage_id], dry_run=args.dry_run)
    return 0


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_optional_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def compute_breadth_depth_from_graph(graph: Dict[str, Any]) -> Tuple[int, int]:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return (0, 0)

    id_to_type: Dict[str, str] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id", "")).strip()
        node_type = str(node.get("type", "")).strip()
        if node_id:
            id_to_type[node_id] = node_type

    out_adj: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        src = str(edge.get("src", "")).strip()
        dst = str(edge.get("dst", "")).strip()
        if src in id_to_type and dst in id_to_type:
            out_adj[src].append(dst)

    max_branching = max((len(v) for v in out_adj.values()), default=0)
    sources = [nid for nid, ntype in id_to_type.items() if ntype == "Given"]
    targets = set(nid for nid, ntype in id_to_type.items() if ntype == "Target")

    target_depth = 0
    if sources and targets:
        node_limit = max(len(id_to_type), 1)
        for source in sources:
            dq = deque([(source, 0)])
            seen_depth: Dict[str, int] = {source: 0}
            while dq:
                cur, depth = dq.popleft()
                if cur in targets and depth > target_depth:
                    target_depth = depth
                if depth > node_limit:
                    continue
                for nxt in out_adj.get(cur, []):
                    next_depth = depth + 1
                    if next_depth > seen_depth.get(nxt, -1):
                        seen_depth[nxt] = next_depth
                        dq.append((nxt, next_depth))

    return (max(0, int(max_branching)), max(0, int(target_depth)))


def breadth_depth_combo(schema: Dict[str, Any]) -> Tuple[int, int]:
    analysis = schema.get("analysis")
    if not isinstance(analysis, dict):
        return (0, 0)

    metrics = analysis.get("graph_metrics")
    breadth = None
    depth = None
    if isinstance(metrics, dict):
        breadth = parse_optional_int(metrics.get("max_branching"))
        depth = parse_optional_int(metrics.get("target_depth"))

    if breadth is None or depth is None:
        graph = analysis.get("problem_graph")
        if isinstance(graph, dict):
            fallback_breadth, fallback_depth = compute_breadth_depth_from_graph(graph)
            if breadth is None:
                breadth = fallback_breadth
            if depth is None:
                depth = fallback_depth

    return (max(0, breadth or 0), max(0, depth or 0))


def bucket_sort_key(bucket_name: str) -> Tuple[int, int]:
    if not bucket_name.startswith("b") or "d" not in bucket_name:
        return (10**9, 10**9)
    raw_breadth, raw_depth = bucket_name[1:].split("d", 1)
    try:
        return (int(raw_breadth), int(raw_depth))
    except ValueError:
        return (10**9, 10**9)


def bucket_name_from_combo(breadth: int, depth: int) -> str:
    return f"b{breadth}d{depth}"


def parse_bucket_combo(bucket_name: str) -> Tuple[int, int]:
    breadth, depth = bucket_sort_key(bucket_name)
    if breadth == 10**9:
        return (0, 0)
    return (breadth, depth)


def merge_two_groups(a: BucketGroup, b: BucketGroup) -> BucketGroup:
    a_n = len(a.rows)
    b_n = len(b.rows)
    total = max(a_n + b_n, 1)
    return BucketGroup(
        bucket_ids=sorted(a.bucket_ids + b.bucket_ids, key=bucket_sort_key),
        rows=a.rows + b.rows,
        centroid_breadth=((a.centroid_breadth * a_n) + (b.centroid_breadth * b_n)) / total,
        centroid_depth=((a.centroid_depth * a_n) + (b.centroid_depth * b_n)) / total,
    )


def nearest_group_index(groups: List[BucketGroup], idx: int) -> Optional[int]:
    if not groups or idx < 0 or idx >= len(groups):
        return None
    src = groups[idx]
    best_i: Optional[int] = None
    best_key: Optional[Tuple[float, int, int, int]] = None
    for j, cand in enumerate(groups):
        if j == idx:
            continue
        dist = abs(src.centroid_breadth - cand.centroid_breadth) + abs(src.centroid_depth - cand.centroid_depth)
        cb = int(round(cand.centroid_breadth))
        cd = int(round(cand.centroid_depth))
        key = (dist, abs(cb - int(round(src.centroid_breadth))), abs(cd - int(round(src.centroid_depth))), -len(cand.rows))
        if best_key is None or key < best_key:
            best_key = key
            best_i = j
    return best_i


def merge_small_bucket_groups(
    bucketed: Dict[str, List[Dict[str, Any]]],
    min_bucket_size: int,
) -> List[BucketGroup]:
    groups: List[BucketGroup] = []
    for bucket_name, rows in bucketed.items():
        breadth, depth = parse_bucket_combo(bucket_name)
        groups.append(
            BucketGroup(
                bucket_ids=[bucket_name],
                rows=list(rows),
                centroid_breadth=float(breadth),
                centroid_depth=float(depth),
            )
        )

    if min_bucket_size <= 1:
        return sorted(groups, key=lambda g: bucket_sort_key(g.bucket_ids[0]))

    while True:
        small_indices = [i for i, g in enumerate(groups) if len(g.rows) < min_bucket_size]
        if not small_indices or len(groups) <= 1:
            break

        idx = min(small_indices, key=lambda i: len(groups[i].rows))
        near_idx = nearest_group_index(groups, idx)
        if near_idx is None:
            break

        a_i, b_i = sorted((idx, near_idx))
        merged = merge_two_groups(groups[a_i], groups[b_i])
        groups.pop(b_i)
        groups.pop(a_i)
        groups.append(merged)

    return sorted(
        groups,
        key=lambda g: (int(round(g.centroid_breadth)), int(round(g.centroid_depth)), g.bucket_ids[0]),
    )


def group_folder_name(group: BucketGroup, seq: int) -> str:
    if len(group.bucket_ids) == 1:
        return group.bucket_ids[0]
    cb = int(round(group.centroid_breadth))
    cd = int(round(group.centroid_depth))
    return f"merged_{seq:03d}_b{cb}d{cd}"


def sync_bucket_folders_from_enriched(run_root: Path, args: argparse.Namespace) -> None:
    if not args.bucket_enriched_input:
        return
    enriched_path = Path(args.bucket_enriched_input)
    if not enriched_path.is_file():
        raise ValueError(f"--bucket-enriched-input not found: {enriched_path}")

    rows = read_jsonl(enriched_path)
    bucketed: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        breadth, depth = breadth_depth_combo(row)
        bucketed[bucket_name_from_combo(breadth, depth)].append(row)
    groups = merge_small_bucket_groups(bucketed, args.min_bucket_size)

    buckets_root = run_root / "buckets"
    buckets_root.mkdir(parents=True, exist_ok=True)
    for idx, group in enumerate(groups, start=1):
        folder_name = group_folder_name(group, idx)
        bucket_dir = buckets_root / folder_name
        bucket_dir.mkdir(parents=True, exist_ok=True)
        schema_path = bucket_dir / f"schemae_{folder_name}.jsonl"
        # Preserve existing schema files in this run root; create only when missing.
        if not schema_path.exists():
            write_jsonl(schema_path, group.rows)
            print(
                f"[bucketed-resume] created missing bucket {folder_name} ({len(group.rows)} rows)",
            )


def first_schema_file_in_bucket(bucket_dir: Path) -> Optional[Path]:
    matches = sorted(bucket_dir.glob("schemae_*.jsonl"))
    if not matches:
        return None
    return matches[0]


def is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def count_jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def first_missing_stage_for_bucket(bucket_dir: Path, schema_path: Path) -> Optional[str]:
    schema_rows = count_jsonl_rows(schema_path)
    stage02_paths = [
        bucket_dir / "skeleton_embedded.jsonl",
        bucket_dir / "question_embedded.jsonl",
        bucket_dir / "anchors.jsonl",
    ]
    stage03_path = bucket_dir / "retrieval_bundles.jsonl"
    stage04_path = bucket_dir / "generated_skeletons.jsonl"
    stage05_path = bucket_dir / "generated_problems.jsonl"

    stage02_complete = (
        count_jsonl_rows(stage02_paths[0]) >= schema_rows > 0
        and count_jsonl_rows(stage02_paths[1]) >= schema_rows > 0
        and is_nonempty_file(stage02_paths[2])
    )
    if not stage02_complete:
        return "02"

    bundles_rows = count_jsonl_rows(stage03_path)
    if bundles_rows <= 0:
        return "03"

    if count_jsonl_rows(stage04_path) < bundles_rows:
        return "04"

    if count_jsonl_rows(stage05_path) < bundles_rows:
        return "05"
    return None


def cmd_bucketed_resume(args: argparse.Namespace, run_root: Path) -> int:
    sync_bucket_folders_from_enriched(run_root, args)

    bucket_root = run_root / "buckets"
    if not bucket_root.is_dir():
        raise ValueError(f"No bucket directory found at: {bucket_root}")

    bucket_dirs = sorted([p for p in bucket_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not bucket_dirs:
        raise ValueError(f"No bucket folders found under: {bucket_root}")

    print(f"[bucketed-resume] run_root={run_root}")
    for bucket_dir in bucket_dirs:
        schema_path = first_schema_file_in_bucket(bucket_dir)
        if schema_path is None:
            print(f"[bucketed-resume] skip {bucket_dir.name}: no schemae_*.jsonl found")
            continue

        start_stage = first_missing_stage_for_bucket(bucket_dir, schema_path)
        if start_stage is None:
            print(f"[bucketed-resume] skip {bucket_dir.name}: already complete through stage 05")
            continue

        per_bucket_commands = build_stage_commands_for_paths(
            args,
            input_path=schema_path,
            enriched_path=schema_path,
            skeleton_embedded_path=bucket_dir / "skeleton_embedded.jsonl",
            question_embedded_path=bucket_dir / "question_embedded.jsonl",
            anchors_path=bucket_dir / "anchors.jsonl",
            bundles_path=bucket_dir / "retrieval_bundles.jsonl",
            generated_skeletons_path=bucket_dir / "generated_skeletons.jsonl",
            generated_problems_path=bucket_dir / "generated_problems.jsonl",
        )

        start_idx = next(i for i, s in enumerate(STAGES) if s.stage_id == start_stage)
        print(f"[bucketed-resume] {bucket_dir.name}: resuming from stage {start_stage}")
        for stage in STAGES[start_idx:]:
            run_stage(stage, per_bucket_commands[stage.stage_id], dry_run=args.dry_run)

    return 0


def cmd_bucketed_pipeline(args: argparse.Namespace) -> int:
    if args.resume_run_root:
        run_root = Path(args.resume_run_root)
        return cmd_bucketed_resume(args, run_root)

    base = default_domain_paths(args.domain)["base"]
    run_root = Path(args.bucket_root) if args.bucket_root else (base / "bucketed_runs" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[bucketed] run_root={run_root}")

    if args.skip_enrich:
        if not args.bucket_enriched_input:
            raise ValueError("--bucket-enriched-input is required when --skip-enrich is set")
        enriched_all_path = Path(args.bucket_enriched_input)
    else:
        input_path = Path(args.input) if args.input else default_domain_paths(args.domain)["input"]
        enriched_all_path = run_root / "enriched" / "enriched.jsonl"
        run_stage(STAGES[0], build_stage01_command(args, input_path=input_path, enriched_path=enriched_all_path), dry_run=args.dry_run)
        if args.dry_run:
            print("[bucketed] dry-run enabled; stage 01 printed, bucketing skipped.")
            return 0

    all_rows = read_jsonl(enriched_all_path)
    bucketed: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        breadth, depth = breadth_depth_combo(row)
        bucketed[bucket_name_from_combo(breadth, depth)].append(row)

    summary: List[Dict[str, Any]] = []
    groups = merge_small_bucket_groups(bucketed, args.min_bucket_size)
    for idx, group in enumerate(groups, start=1):
        folder_name = group_folder_name(group, idx)
        bucket_rows = group.rows
        bucket_dir = run_root / "buckets" / folder_name
        bucket_enriched = bucket_dir / f"schemae_{folder_name}.jsonl"
        write_jsonl(bucket_enriched, bucket_rows)
        source_text = ", ".join(group.bucket_ids)
        print(f"[bucketed] {folder_name}: {len(bucket_rows)} schemas (from: {source_text}) -> {bucket_dir}")

        summary.append({
            "bucket": folder_name,
            "source_buckets": group.bucket_ids,
            "count": len(bucket_rows),
            "enriched": str(bucket_enriched),
        })

        per_bucket_commands = build_stage_commands_for_paths(
            args,
            input_path=bucket_enriched,
            enriched_path=bucket_enriched,
            skeleton_embedded_path=bucket_dir / "skeleton_embedded.jsonl",
            question_embedded_path=bucket_dir / "question_embedded.jsonl",
            anchors_path=bucket_dir / "anchors.jsonl",
            bundles_path=bucket_dir / "retrieval_bundles.jsonl",
            generated_skeletons_path=bucket_dir / "generated_skeletons.jsonl",
            generated_problems_path=bucket_dir / "generated_problems.jsonl",
        )
        for stage in STAGES[1:]:
            run_stage(stage, per_bucket_commands[stage.stage_id], dry_run=args.dry_run)

    summary_path = run_root / "bucket_summary.json"
    ensure_parent(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[bucketed] wrote summary: {summary_path}")
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
    pipe.add_argument("--max-depth", type=int, default=None, help="Optional cap for stage 01; unset = unbounded")
    pipe.add_argument("--max-branching", type=int, default=None, help="Optional cap for stage 01; unset = unbounded")
    pipe.add_argument("--max-nodes", type=int, default=None, help="Optional cap for stage 01; unset = unbounded")
    pipe.add_argument("--max-edges", type=int, default=None, help="Optional cap for stage 01; unset = unbounded")

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

    bucket = sub.add_parser(
        "bucketed-pipeline",
        help="Run stage 01 once, bucket enriched schemae by breadth-depth, then run stages 02-05 per bucket folder",
    )
    bucket.set_defaults(func=cmd_bucketed_pipeline)

    bucket.add_argument("--domain", default="phys", help="Preset path family: phys -> data, chem -> chem_data")
    bucket.add_argument("--stage-domain", default="", help="Domain string passed into stage 04; default is inferred from --domain")
    bucket.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    bucket.add_argument("--debug", action="store_true", help="Pass --debug to stages that support it")
    bucket.add_argument("--limit", type=int, default=0, help="Apply the same row limit to stages 01, 04, and 05")
    bucket.add_argument("--min-bucket-size", type=int, default=30, help="Merge nearest buckets until each has at least this many rows")
    bucket.add_argument("--bucket-root", default="", help="Root output folder for this bucketed run")
    bucket.add_argument("--resume-run-root", default="", help="Resume an existing bucketed run root and continue missing stages")
    bucket.add_argument("--skip-enrich", action="store_true", help="Skip stage 01 and bucket an existing enriched file")
    bucket.add_argument("--bucket-enriched-input", default="", help="Existing enriched JSONL for --skip-enrich, or for resume bucket-sync")

    bucket.add_argument("--input", default="", help="Override stage 01 input JSONL")
    bucket.add_argument("--enrich-model", default="qwen2.5:7b-instruct")
    bucket.add_argument("--embed-model", default="qwen3-embedding")
    bucket.add_argument("--graph-model", default="qwen2.5:7b-instruct")
    bucket.add_argument("--question-model", default="qwen2.5:7b-instruct")
    bucket.add_argument("--verify-model", default="")

    bucket.add_argument("--corpus", default="unknown")
    bucket.add_argument("--year", type=int, default=0)
    bucket.add_argument("--variant", default="")
    bucket.add_argument("--grammar-mode", choices=["off", "warn", "strict"], default="strict")
    bucket.add_argument("--max-depth", type=int, default=None, help="Optional cap for stage 01; unset = unbounded")
    bucket.add_argument("--max-branching", type=int, default=None, help="Optional cap for stage 01; unset = unbounded")
    bucket.add_argument("--max-nodes", type=int, default=None, help="Optional cap for stage 01; unset = unbounded")
    bucket.add_argument("--max-edges", type=int, default=None, help="Optional cap for stage 01; unset = unbounded")

    bucket.add_argument("--anchor-frac", type=float, default=0.18)
    bucket.add_argument("--k-density", type=int, default=20)

    bucket.add_argument("--num-bundles", type=int, default=25)
    bucket.add_argument("--annulus-min", type=float, default=0.08)
    bucket.add_argument("--annulus-max", type=float, default=0.40)
    bucket.add_argument("--tail-frac", type=float, default=0.35)
    bucket.add_argument("--tail-min", type=float, default=0.30)
    bucket.add_argument("--k-solution", type=int, default=4)
    bucket.add_argument("--k-question", type=int, default=4)
    bucket.add_argument("--k-paired", type=int, default=3)
    bucket.add_argument("--mmr-lambda-solution", type=float, default=0.7)
    bucket.add_argument("--mmr-lambda-question", type=float, default=0.7)
    bucket.add_argument("--seed", type=int, default=0)
    bucket.add_argument("--require-skeleton-text", action=argparse.BooleanOptionalAction, default=True)
    bucket.add_argument("--require-insight-seed", action="store_true")
    bucket.add_argument("--require-insight-solution-exemplars", action="store_true")
    bucket.add_argument("--min-seed-complexity", type=int, default=0)
    bucket.add_argument("--min-sol-ex-complexity", type=int, default=0)

    bucket.add_argument("--max-q-exemplars", type=int, default=4)
    bucket.add_argument("--max-paired-exemplars", type=int, default=3)
    bucket.add_argument("--repair-max", type=int, default=1)
    bucket.add_argument("--sleep", type=float, default=0.0)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
