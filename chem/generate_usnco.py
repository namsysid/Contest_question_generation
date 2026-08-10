#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from urllib import error, request


REPO_ROOT = Path(__file__).resolve().parents[1]
CHEM_DIR = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"

SOURCE_JSONL = CHEM_DIR / "data" / "usnco_questions.jsonl"
RANGES_PATH = CHEM_DIR / "ranges.json"
RAW_PIPELINE_JSONL = CHEM_DIR / "data" / "usnco_pipeline_raw.jsonl"
ENRICHED_JSONL = CHEM_DIR / "enriched_schemae" / "enriched.jsonl"
SKELETON_EMBEDDED_JSONL = CHEM_DIR / "skeleton_embedded.jsonl"
QUESTION_EMBEDDED_JSONL = CHEM_DIR / "question_embedded.jsonl"
ANCHORS_JSONL = CHEM_DIR / "anchors.jsonl"
TARGETS_DIR = CHEM_DIR / "targets"
OUTPUTS_DIR = CHEM_DIR / "outputs"
RANGE_RUNS_DIR = CHEM_DIR / "range_runs"


def range_slug(range_label: str) -> str:
    return range_label.replace("-", "_")


def artifact_paths(range_label: str) -> Dict[str, Path]:
    root = RANGE_RUNS_DIR / range_label
    return {
        "root": root,
        "raw": root / "data" / "usnco_pipeline_raw.jsonl",
        "enriched": root / "enriched_schemae" / "enriched.jsonl",
        "skeleton_embedded": root / "skeleton_embedded.jsonl",
        "question_embedded": root / "question_embedded.jsonl",
        "anchors": root / "anchors.jsonl",
        "targets": root / "targets" / f"usnco_{range_label}_targets.jsonl",
    }


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def load_ranges() -> Dict[str, Dict[str, Any]]:
    data = json.loads(RANGES_PATH.read_text(encoding="utf-8"))
    return {item["range"]: item for item in data["ranges"]}


def normalize_range(raw: str) -> str:
    parts = raw.replace("to", "-").replace(":", "-").replace(",", "-").split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid range '{raw}'. Use a range like 1-5 or 56-60.")
    start, end = int(parts[0].strip()), int(parts[1].strip())
    return f"{start}-{end}"


def choices_are_nonempty(choices: Any) -> bool:
    if not isinstance(choices, dict):
        return False
    return all(str(choices.get(letter, "")).strip() for letter in ("A", "B", "C", "D"))


def prepare_raw(
    question_numbers: set[int] | None = None,
    range_label: str | None = None,
    out_path: Path | None = None,
) -> List[Dict[str, Any]]:
    rows = read_jsonl(SOURCE_JSONL)
    out: List[Dict[str, Any]] = []
    skipped = 0
    skipped_out_of_range = 0

    for row in rows:
        content = row.get("content") or {}
        question_text = str(content.get("question_text") or "").strip()
        choices = content.get("choices") or {}
        if row.get("status") != "transcribed" or not question_text or not choices_are_nonempty(choices):
            skipped += 1
            continue

        answer = row.get("answer") or {}
        source = row.get("source") or {}
        section = row.get("section") or {}
        qnum = int(source.get("question_number") or 0)
        if question_numbers is not None and qnum not in question_numbers:
            skipped_out_of_range += 1
            continue

        year = source.get("year")
        exam_type = source.get("exam_type")
        exam_code = source.get("exam_code")

        out.append(
            {
                "id": row["id"],
                "domain": "chem",
                "question_text": question_text,
                "choices": {letter: str(choices.get(letter, "")).strip() for letter in ("A", "B", "C", "D")},
                "answer_key": answer.get("letter"),
                "source_pdf": f"USNCO {year} {exam_type} q{qnum}",
                "has_diagram": "shown" in question_text.lower() or "diagram" in question_text.lower(),
                "diagram_files": [],
                "topic": section.get("name"),
                "section": section,
                "source": source,
                "metadata": {
                    "program": source.get("program", "USNCO"),
                    "exam_code": exam_code,
                    "exam_type": exam_type,
                    "year": year,
                    "question_number": qnum,
                    "question_range": section.get("question_range"),
                    "section_name": section.get("name"),
                    "source_id": row["id"],
                },
            }
        )

    target_path = out_path or RAW_PIPELINE_JSONL
    write_jsonl(target_path, out)
    scope = f" for range {range_label}" if range_label else ""
    extra = f", skipped_out_of_range {skipped_out_of_range}" if question_numbers is not None else ""
    print(f"Prepared {len(out)} transcribed rows{scope} -> {target_path} (skipped {skipped}{extra})")
    return out


def ollama_base_url(args: argparse.Namespace) -> str:
    return (
        getattr(args, "ollama_base_url", "")
        or os.environ.get("OLLAMA_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "http://192.168.50.186:11434"
    ).rstrip("/")


def looks_like_openai_model(model: str) -> bool:
    name = (model or "").strip().lower()
    return name.startswith(("gpt-", "o1", "o3", "o4", "o5", "chatgpt-", "text-embedding-", "embedding-"))


def preflight_ollama(args: argparse.Namespace) -> None:
    models = [args.enrich_model, args.embed_model, args.gen_model]
    if args.verify_model:
        models.append(args.verify_model)
    if all(looks_like_openai_model(model) for model in models):
        return

    base_url = ollama_base_url(args)
    timeout = float(getattr(args, "ollama_timeout", 10.0))
    try:
        with request.urlopen(f"{base_url}/api/tags", timeout=timeout) as resp:
            resp.read(1)
    except error.URLError as exc:
        raise SystemExit(
            "Cannot reach Ollama before starting enrichment.\n"
            f"Tried: {base_url}/api/tags\n"
            f"Error: {exc.reason}\n\n"
            "Start Ollama or pass the correct server, for example:\n"
            "  python3 chem/run_usnco_all_in_one.py --rebuild --range 1-5 --num 10 --debug --ollama-base-url http://localhost:11434"
        ) from exc


def subprocess_env(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    if getattr(args, "ollama_base_url", ""):
        env["OLLAMA_BASE_URL"] = args.ollama_base_url.rstrip("/")
    return env


def run_cmd(cmd: Sequence[str], args: argparse.Namespace | None = None) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=(subprocess_env(args) if args else None))


def artifacts_exist(range_label: str | None = None) -> bool:
    if range_label:
        paths = artifact_paths(range_label)
        required = [
            paths["raw"],
            paths["enriched"],
            paths["skeleton_embedded"],
            paths["question_embedded"],
            paths["anchors"],
        ]
        return all(path.exists() and path.stat().st_size > 0 for path in required)

    required = [
        RAW_PIPELINE_JSONL,
        ENRICHED_JSONL,
        SKELETON_EMBEDDED_JSONL,
        QUESTION_EMBEDDED_JSONL,
        ANCHORS_JSONL,
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def build_pipeline(args: argparse.Namespace, range_label: str | None = None) -> None:
    question_numbers = None
    paths = None
    if range_label:
        ranges = load_ranges()
        question_numbers = set(ranges[range_label]["question_numbers"])
        paths = artifact_paths(range_label)

    raw_path = paths["raw"] if paths else RAW_PIPELINE_JSONL
    enriched_path = paths["enriched"] if paths else ENRICHED_JSONL
    skeleton_path = paths["skeleton_embedded"] if paths else SKELETON_EMBEDDED_JSONL
    question_path = paths["question_embedded"] if paths else QUESTION_EMBEDDED_JSONL
    anchors_path = paths["anchors"] if paths else ANCHORS_JSONL

    prepare_raw(question_numbers=question_numbers, range_label=range_label, out_path=raw_path)
    preflight_ollama(args)
    enriched_path.parent.mkdir(parents=True, exist_ok=True)
    skeleton_path.parent.mkdir(parents=True, exist_ok=True)
    question_path.parent.mkdir(parents=True, exist_ok=True)
    anchors_path.parent.mkdir(parents=True, exist_ok=True)

    run_cmd(
        [
            sys.executable,
            str(SRC_DIR / "01_enrich_problem_schema_paper-2.py"),
            "--input",
            str(raw_path),
            "--out",
            str(enriched_path),
            "--model",
            args.enrich_model,
            "--corpus",
            "usnco",
            "--grammar-mode",
            args.grammar_mode,
            "--json-retries",
            str(args.json_retries),
        ]
        + (["--debug"] if args.debug else [])
        + (["--limit", str(args.build_limit)] if args.build_limit else []),
        args=args,
    )

    run_cmd(
        [
            sys.executable,
            str(SRC_DIR / "02_embed_and_index.py"),
            "--input",
            str(enriched_path),
            "--embed_model",
            args.embed_model,
            "--out_skel",
            str(skeleton_path),
            "--out_q",
            str(question_path),
            "--out_anchors",
            str(anchors_path),
        ],
        args=args,
    )


def enriched_source_number(row: Dict[str, Any]) -> int | None:
    source = row.get("source") or {}
    pdf = str(source.get("pdf") or "")
    marker = " q"
    if marker in pdf:
        try:
            return int(pdf.rsplit(marker, 1)[1])
        except ValueError:
            return None
    return None


def question_view(row: Dict[str, Any]) -> str:
    problem = row.get("problem") or {}
    stem = str(problem.get("stem") or "").strip()
    choices = problem.get("choices") or []
    if isinstance(choices, list) and choices:
        return (stem + "\n" + "\n".join(str(choice) for choice in choices)).strip()
    return stem


def graph_text(row: Dict[str, Any]) -> str:
    analysis = row.get("analysis") or {}
    return str(analysis.get("graph_text") or analysis.get("skeleton") or "").strip()


def make_targets(range_label: str, count: int, max_exemplars: int, seed: int, start_index: int = 0) -> Path:
    ranges = load_ranges()
    range_info = ranges[range_label]
    allowed_numbers = set(range_info["question_numbers"])
    paths = artifact_paths(range_label)
    enriched = read_jsonl(paths["enriched"])
    candidates = [row for row in enriched if enriched_source_number(row) in allowed_numbers and graph_text(row)]

    if not candidates:
        raise RuntimeError(
            f"No enriched rows found for {range_label}. Run --build first, and check transcribed rows for this range."
        )

    # Deterministic rotation without importing random for easier reproducibility.
    offset = seed % len(candidates)
    ordered = candidates[offset:] + candidates[:offset]
    selected = [ordered[i % len(ordered)] for i in range(count)]

    all_exemplars = [row for row in enriched if question_view(row) and graph_text(row)]
    targets: List[Dict[str, Any]] = []
    for local_idx, seed_row in enumerate(selected, start=1):
        idx = start_index + local_idx
        seed_id = seed_row["id"]
        topic = (seed_row.get("analysis") or {}).get("concepts") or [range_info["title"]]
        seed_pdf = str((seed_row.get("source") or {}).get("pdf") or "")
        source_match = re.match(r"USNCO\s+(\d+)\s+(\S+)\s+q(\d+)", seed_pdf)
        seed_year = int(source_match.group(1)) if source_match else None
        seed_exam_type = source_match.group(2) if source_match else None
        exemplars = []
        for candidate in all_exemplars:
            if candidate["id"] == seed_id:
                continue
            exemplars.append(
                {
                    "id": candidate["id"],
                    "question_text": question_view(candidate),
                    "skeleton_text": graph_text(candidate),
                }
            )
            if len(exemplars) >= max_exemplars:
                break

        targets.append(
            {
                "id": f"usnco_{range_label.replace('-', '_')}_{idx:02d}",
                "skeleton_text": graph_text(seed_row),
                "exemplars": exemplars,
                "_chem": {
                    "range": range_label,
                    "range_title": range_info["title"],
                    "generated_question_number": range_info["question_numbers"][(idx - 1) % len(range_info["question_numbers"])],
                    "generation_index": idx,
                    "seed_id": seed_id,
                    "seed_question_number": enriched_source_number(seed_row),
                    "year": seed_year,
                    "exam_type": seed_exam_type,
                    "seed_topic": topic[0] if topic else range_info["title"],
                },
            }
        )

    out = paths["targets"]
    write_jsonl(out, targets)
    print(f"Wrote {len(targets)} range targets -> {out}")
    return out


def generate_from_pipeline(args: argparse.Namespace, range_label: str) -> Path:
    if not artifacts_exist(range_label):
        raise SystemExit("Missing chem pipeline artifacts. Run once with --build, or use --prepare-only to inspect converted rows.")

    out = args.out or (OUTPUTS_DIR / f"usnco_{range_label}_generated.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    existing_count = 0 if getattr(args, "overwrite_output", False) else count_jsonl(out)

    targets = make_targets(
        range_label=range_label,
        count=args.num,
        max_exemplars=args.max_exemplars,
        seed=args.seed + existing_count,
        start_index=existing_count,
    )
    temp_out = out.with_name(f".{out.stem}.tmp_new{out.suffix or '.jsonl'}")

    cmd = [
        sys.executable,
        str(SRC_DIR / "generate.py"),
        "--targets",
        str(targets),
        "--out",
        str(temp_out),
        "--model",
        args.gen_model,
        "--limit",
        str(args.num),
        "--max_exemplars",
        str(args.max_exemplars),
        "--temperature",
        str(args.temperature),
        "--num_choices",
        "4",
    ]
    if args.do_verify:
        cmd.append("--do_verify")
    if args.verify_model:
        cmd.extend(["--verify_model", args.verify_model])
    preflight_ollama(args)
    run_cmd(cmd, args=args)
    new_rows = read_jsonl(temp_out) if temp_out.exists() else []
    if getattr(args, "overwrite_output", False):
        write_jsonl(out, new_rows)
        print(f"Wrote {len(new_rows)} new rows -> {out}")
    else:
        appended = append_jsonl(out, new_rows)
        print(f"Appended {appended} new rows -> {out} (previously {existing_count})")
    try:
        temp_out.unlink()
    except FileNotFoundError:
        pass
    return out


def list_ranges() -> None:
    for label, info in load_ranges().items():
        print(f"{label}: {info['title']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and generate USNCO chemistry questions through the repo's src pipeline."
    )
    parser.add_argument("--range", dest="range_label", help="Required generation bucket, e.g. 1-6 or 55-60.")
    parser.add_argument("--list-ranges", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Only write chem/data/usnco_pipeline_raw.jsonl.")
    parser.add_argument("--build", action="store_true", help="Run src enrichment and embedding before generation.")
    parser.add_argument("--build-limit", type=int, default=0, help="Limit enrichment rows while testing. 0 means all.")
    parser.add_argument("--num", type=int, default=6, help="Questions to generate for the requested range.")
    parser.add_argument("--max-exemplars", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enrich-model", default="gpt-4o")
    parser.add_argument("--embed-model", default="text-embedding-3-large")
    parser.add_argument("--gen-model", default="gpt-4o")
    parser.add_argument("--verify-model", default="")
    parser.add_argument("--do-verify", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--grammar-mode", choices=["off", "warn", "strict"], default="strict")
    parser.add_argument("--json-retries", type=int, default=3)
    parser.add_argument("--debug", action="store_true", help="Pass debug logging through to src enrichment.")
    parser.add_argument("--ollama-base-url", default="", help="Override OLLAMA_BASE_URL for src model calls.")
    parser.add_argument("--ollama-timeout", type=float, default=10.0, help="Seconds for the Ollama preflight check.")
    parser.add_argument("--overwrite-output", action="store_true", help="Replace the output file instead of appending new generations.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.list_ranges:
        list_ranges()
        return

    if not args.range_label:
        valid = ", ".join(load_ranges())
        raise SystemExit(f"Choose a question range with --range. Valid ranges: {valid}")

    range_label = normalize_range(args.range_label)
    if range_label not in load_ranges():
        valid = ", ".join(load_ranges())
        raise SystemExit(f"Unsupported range '{range_label}'. Valid ranges: {valid}")

    if args.prepare_only:
        paths = artifact_paths(range_label)
        prepare_raw(
            question_numbers=set(load_ranges()[range_label]["question_numbers"]),
            range_label=range_label,
            out_path=paths["raw"],
        )
        return

    if args.build:
        build_pipeline(args, range_label=range_label)

    out = generate_from_pipeline(args, range_label)
    print(f"Generated via src pipeline -> {out}")


if __name__ == "__main__":
    main()
