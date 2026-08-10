#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import generate_usnco


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete USNCO chem pipeline: prepare, enrich, embed, target, and generate."
    )
    parser.add_argument("--range", dest="range_label", required=True, help="Six-question bucket, e.g. 1-6 or 55-60.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild enrichment and embeddings even if artifacts exist.")
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

    ranges = generate_usnco.load_ranges()
    range_label = generate_usnco.normalize_range(args.range_label)
    if range_label not in ranges:
        valid = ", ".join(ranges)
        raise SystemExit(f"Unsupported range '{range_label}'. Valid ranges: {valid}")

    if args.rebuild or not generate_usnco.artifacts_exist(range_label):
        generate_usnco.build_pipeline(args, range_label=range_label)
    else:
        paths = generate_usnco.artifact_paths(range_label)
        generate_usnco.prepare_raw(
            question_numbers=set(ranges[range_label]["question_numbers"]),
            range_label=range_label,
            out_path=paths["raw"],
        )
        print(f"Reusing existing chem artifacts for range {range_label}. Use --rebuild to refresh them.")

    out = generate_usnco.generate_from_pipeline(args, range_label)
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
