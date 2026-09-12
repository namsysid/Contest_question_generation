#!/usr/bin/env python3
"""Build topic-filtered all_in_one.py retrieval inputs from existing F=ma indices."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


TOPICS = {
    "kinematics": ("Kinematics", r"kinematic|projectile|velocity|acceleration|free[ -]?fall|trajectory|relative motion"),
    "forces": ("Forces", r"newton|friction|tension|normal force|drag|force|incline|pulley"),
    "energy": ("Energy", r"work.energy|mechanical energy|potential energy|kinetic energy|power|spring"),
    "momentum": ("Momentum", r"momentum|impulse|collision|center of mass"),
    "circular_gravity": ("Circular & gravity", r"circular|centripetal|gravitation|gravity|orbit|satellite|planet"),
    "rotation": ("Rotation", r"rotat|torque|angular|rolling|moment of inertia|static equilibrium"),
    "oscillations": ("Oscillations", r"oscillat|pendulum|simple harmonic|period|frequency|spring"),
    "fluids": ("Fluids", r"fluid|pressure|buoy|bernoulli|continuity|density|viscos|flow|archimedes"),
}
OFFICIAL_FMA_PDF_RE = re.compile(r"^\d{4}_fnet_ma_exam(?:_[AB])?\.pdf$", re.IGNORECASE)

# Ordered from specific mechanics families to broad fallback families. A source row must enter
# exactly one shard; the former independent regex filters put orbits, collisions, and pendulums
# into kinematics merely because their text also mentioned velocity or acceleration.
PRIMARY_PATTERNS = (
    ("fluids", r"fluid|buoy|bernoulli|continuity|viscos|archimedes|hydrostatic"),
    ("circular_gravity", r"orbit|satellite|planet|gravitation"),
    ("momentum", r"momentum|impulse|collision|center of mass"),
    ("oscillations", r"oscillat|pendulum|simple harmonic|resonance|spring.{0,40}(?:period|frequency)"),
    ("rotation", r"torque|angular momentum|angular acceleration|moment of inertia|rolling|rotational"),
    ("circular_gravity", r"centripetal|circular motion"),
    ("energy", r"work.energy|mechanical energy|potential energy|kinetic energy|power|spring"),
    ("forces", r"newton|friction|tension|normal force|drag|incline|pulley|force"),
    ("kinematics", r"kinematic|projectile|velocity|acceleration|free[ -]?fall|trajectory|relative motion"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def topic_text(row: dict[str, Any]) -> str:
    analysis = row.get("analysis") or {}
    profile = analysis.get("graph_profile") or {}
    problem = row.get("problem") or {}
    return " ".join(map(str, (
        problem.get("stem") or "", analysis.get("concepts") or [], analysis.get("skills") or [],
        profile.get("insight_type") or [], profile.get("target_summary") or "", analysis.get("graph_text") or "",
    ))).casefold()


def matches_topic(row: dict[str, Any], topic_key: str) -> bool:
    """Return whether a row materially matches a topic, allowing cross-topic membership."""
    problem = row.get("problem") if isinstance(row.get("problem"), dict) else {}
    stem = str(problem.get("stem") or "").casefold()
    return bool(re.search(TOPICS[topic_key][1], stem, flags=re.IGNORECASE))


def classify_topic(row: dict[str, Any]) -> str | None:
    text = topic_text(row)
    for topic_key, pattern in PRIMARY_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return topic_key
    return None


def is_official_fma_source(row: dict[str, Any]) -> bool:
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    return bool(OFFICIAL_FMA_PDF_RE.fullmatch(Path(str(source.get("pdf") or "")).name))


def dense_anchor_ids(rows: list[dict[str, Any]], fraction: float = 0.25) -> list[str]:
    matrix = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    similarities = matrix @ matrix.T
    np.fill_diagonal(similarities, -1)
    k = max(1, min(20, len(rows) - 1))
    density = np.partition(similarities, -k, axis=1)[:, -k:].mean(axis=1)
    count = max(4, int(round(len(rows) * fraction)))
    indices = np.argsort(-density)[:count]
    return [str(rows[index]["id"]) for index in indices]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enriched", type=Path, default=Path("data/enriched_schemae/enriched.jsonl"))
    parser.add_argument("--skeleton-embeddings", type=Path, default=Path("data/skeleton_embedded.jsonl"))
    parser.add_argument("--question-embeddings", type=Path, default=Path("data/question_embedded.jsonl"))
    parser.add_argument("--out-root", type=Path, default=Path("fma_solution_first/topics"))
    parser.add_argument("--min-topic-sources", type=int, default=12)
    parser.add_argument("--topic-key", choices=TOPICS)
    parser.add_argument("--official-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-multitopic", action="store_true",
        help="Allow a source row into each matching topic shard instead of forcing one label.",
    )
    args = parser.parse_args()
    enriched = read_jsonl(args.enriched)
    skeletons = {str(row["id"]): row for row in read_jsonl(args.skeleton_embeddings)}
    questions = {str(row["id"]): row for row in read_jsonl(args.question_embeddings)}
    selected_topics = (
        {args.topic_key: TOPICS[args.topic_key]} if args.topic_key else TOPICS
    )
    for topic_key, (label, _) in selected_topics.items():
        selected = [row for row in enriched if (
                        matches_topic(row, topic_key) if args.allow_multitopic else classify_topic(row) == topic_key
                    )
                    and (not args.official_only or is_official_fma_source(row))
                    and not bool((row.get("analysis") or {}).get("diagram_required"))
                    and str(row.get("id")) in skeletons and str(row.get("id")) in questions]
        if len(selected) < args.min_topic_sources:
            raise RuntimeError(f"{topic_key} has only {len(selected)} indexed source rows")
        directory = args.out_root / topic_key
        skel_rows = [skeletons[str(row["id"])] for row in selected]
        q_rows = [questions[str(row["id"])] for row in selected]
        write_jsonl(directory / "enriched.jsonl", selected)
        write_jsonl(directory / "skeleton_embedded.jsonl", skel_rows)
        write_jsonl(directory / "question_embedded.jsonl", q_rows)
        (directory / "anchors.json").write_text(
            json.dumps([{"id": value} for value in dense_anchor_ids(skel_rows)], indent=2), encoding="utf-8"
        )
        (directory / "topic.json").write_text(
            json.dumps({"topic_key": topic_key, "topic": label, "source_count": len(selected)}, indent=2),
            encoding="utf-8",
        )
        print(f"{topic_key}: {len(selected)} source rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
