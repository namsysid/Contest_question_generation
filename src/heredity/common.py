from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


TOPICS = {
    "mendelian_probability": ("punnett", "heterozyg", "homozyg", "segregation", "probability", "genotyp", "phenotyp"),
    "non_mendelian_inheritance": ("codomin", "incomplete dominance", "multiple allele", "blood type", "epistas"),
    "sex_linked_pedigrees": ("pedigree", "x-linked", "sex-linked", "carrier", "maternal", "y chromosome"),
    "cell_division_chromosomes": ("mitosis", "meiosis", "chromosome", "chromatid", "nondisjunction", "crossing over"),
    "dna_replication_structure": ("dna", "replication", "helicase", "ligase", "polymerase", "nucleotide", "chargaff"),
    "gene_expression_regulation": ("transcription", "translation", "codon", "trna", "rrna", "operon", "gene expression"),
    "mutations_biotechnology": ("mutation", "frameshift", "pcr", "electrophoresis", "restriction", "amplif"),
}
RESPONSE_TYPES = {"multiple_choice", "numeric", "short_answer"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def classify_topics(text: str) -> list[str]:
    lower = text.lower()
    found = [name for name, terms in TOPICS.items() if any(term in lower for term in terms)]
    return found or ["mendelian_probability"]


def graph_text(item: dict[str, Any]) -> str:
    graph = (item.get("analysis") or {}).get("reasoning_graph") or item.get("reasoning_graph") or {}
    nodes = " | ".join(f"{x.get('type')}:{x.get('label')}" for x in graph.get("nodes") or [])
    edges = " | ".join(f"{x.get('src')}-{x.get('type')}->{x.get('dst')}" for x in graph.get("edges") or [])
    return f"FORMAT:{item.get('response_type')} TOPICS:{','.join(item.get('topics') or [])} NODES:{nodes} EDGES:{edges}"


def hash_embedding(text: str, dimensions: int = 384) -> list[float]:
    """Stable dependency-free feature hashing for offline dual retrieval."""
    values = [0.0] * dimensions
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    for token in features:
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        raw = int.from_bytes(digest, "big")
        values[raw % dimensions] += 1.0 if raw & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def validate_item(item: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if item.get("response_type") not in RESPONSE_TYPES:
        errors.append("invalid response_type")
    if not str(item.get("prompt") or "").strip():
        errors.append("prompt required")
    if not str(item.get("solution") or "").strip():
        errors.append("worked solution required")
    if not isinstance(item.get("difficulty"), int) or not 1 <= item["difficulty"] <= 5:
        errors.append("difficulty must be 1-5")
    if not isinstance(item.get("points"), (int, float)) or item["points"] <= 0:
        errors.append("points must be positive")
    if item.get("response_type") == "multiple_choice":
        choices = item.get("choices") or {}
        if set(choices) != set("ABCD"):
            errors.append("MCQ requires exactly A-D")
        if item.get("answer") not in choices:
            errors.append("MCQ answer must be A-D")
        if len({re.sub(r"\s+", " ", str(v).strip().lower()) for v in choices.values()}) != 4:
            errors.append("choices must be distinct")
    elif item.get("answer") in (None, ""):
        errors.append("constructed response answer required")
    prompt = str(item.get("prompt") or "").lower()
    if any(p in prompt for p in ("shown below", "diagram above", "accompanying pedigree", "karyotype below")):
        errors.append("unavailable visual dependency")
    if not set(item.get("topics") or []).intersection(TOPICS):
        errors.append("recognized Heredity topic required")
    return errors


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
