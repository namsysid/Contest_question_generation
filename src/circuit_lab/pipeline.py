from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(module: str, arguments: list[str]) -> None:
    subprocess.run([sys.executable, "-m", module, *arguments], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete graph-first Circuit Lab pipeline")
    parser.add_argument("--input", required=True, help="Source Circuit Lab test PDF")
    parser.add_argument("--work-dir", default="science_olympiad/circuit_lab_b")
    parser.add_argument("--level", choices=["invitational", "regional", "state", "national"], default="regional")
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--source-limit", type=int, default=0)
    parser.add_argument("--mcq-only", action="store_true", help="Use every source MCQ and exclude multipart items")
    parser.add_argument("--source-response-type", choices=["multiple_choice", "multipart"], default="")
    parser.add_argument("--match-source-difficulty", action="store_true")
    parser.add_argument("--difficulty-floor", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--provider", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--generation-model", default="qwen2.5:7b-instruct")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--judge-model", default="qwen2.5:7b-instruct")
    args = parser.parse_args()
    if args.provider == "openai":
        if args.generation_model == "qwen2.5:7b-instruct":
            args.generation_model = "gpt-4.1-mini"
        if args.embedding_model == "nomic-embed-text":
            args.embedding_model = "text-embedding-3-small"
        if args.judge_model == "qwen2.5:7b-instruct":
            args.judge_model = "gpt-4.1-mini"
    root = Path(args.work_dir)
    corpus = root / "corpus" / "items.jsonl"
    enriched = root / "enriched" / "items.jsonl"
    question_index = root / "enriched" / "question_embeddings.jsonl"
    structure_index = root / "enriched" / "structure_embeddings.jsonl"
    anchors = root / "enriched" / "anchors.jsonl"
    bundles = root / "enriched" / "retrieval_bundles.jsonl"
    plans = root / "generated" / "reasoning_plans.jsonl"
    generated = root / "generated" / "items.jsonl"
    reports = root / "validation" / "item_reports.jsonl"
    run("src.circuit_lab.ingest", ["--input", args.input, "--out", str(corpus), "--asset-dir", str(root / "corpus" / "assets")])
    provider = ["--provider", args.provider]
    enrich_args = ["--input", str(corpus), "--out", str(enriched), "--model", args.generation_model, *provider]
    source_response_type = "multiple_choice" if args.mcq_only else args.source_response_type
    if source_response_type:
        enrich_args.extend(["--response-type", source_response_type])
    if args.source_limit:
        enrich_args.extend(["--limit", str(args.source_limit)])
    run("src.circuit_lab.enrich", enrich_args)
    embed_args = ["--input", str(enriched), "--question-out", str(question_index), "--structure-out", str(structure_index),
                  "--anchors-out", str(anchors), "--model", args.embedding_model, *provider]
    if source_response_type:
        embed_args.extend(["--anchor-fraction", "1.0"])
    run("src.circuit_lab.embed", embed_args)
    run("src.circuit_lab.retrieve", ["--enriched", str(enriched), "--question-index", str(question_index), "--structure-index", str(structure_index), "--anchors", str(anchors), "--out", str(bundles), "--count", str(args.count)])
    plan_args = ["--bundles", str(bundles), "--novelty-corpus", str(corpus), "--out", str(plans),
                 "--model", args.generation_model, "--level", args.level, "--difficulty-floor", str(args.difficulty_floor), *provider]
    if args.match_source_difficulty:
        plan_args.append("--match-source-difficulty")
    run("src.circuit_lab.plan", plan_args)
    run("src.circuit_lab.generate", ["--bundles", str(bundles), "--plans", str(plans), "--novelty-corpus", str(corpus), "--out", str(generated), "--model", args.generation_model, *provider])
    run("src.circuit_lab.validate", ["--input", str(generated), "--novelty-corpus", str(corpus), "--out", str(reports), "--model", args.judge_model, *provider])
    print(f"Complete: generated items at {generated}; validation reports at {reports}")


if __name__ == "__main__":
    main()
