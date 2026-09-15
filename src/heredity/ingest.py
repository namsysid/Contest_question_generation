from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import fitz

from .common import classify_topics, graph_text, write_jsonl


def pdf_text(path: Path) -> str:
    doc = fitz.open(path)
    try:
        return "\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()


def _reasoning_graph(prompt: str, topic: str) -> dict[str, Any]:
    nodes = [{"id": "g1", "type": "Given", "label": prompt[:180]},
             {"id": "l1", "type": "Law", "label": topic.replace("_", " ")},
             {"id": "t1", "type": "Target", "label": "keyed response"}]
    return {"nodes": nodes, "edges": [{"src": "g1", "dst": "t1", "type": "supports"},
                                        {"src": "l1", "dst": "t1", "type": "depends_on"}]}


def _answer_map(text: str) -> dict[int, str]:
    # Some PDFs place zero-width glyphs after each key letter; do not anchor the
    # match at end-of-line. The older key puts the number and letter on adjacent lines.
    return {int(n): a for n, a in re.findall(r"(?m)^\s*(\d+)\s*[-–]?\s*([A-E])", text)}


def extract_mcq_pair(test_path: Path, key_path: Path, source: str, max_choice: str = "D") -> list[dict[str, Any]]:
    text, keys = pdf_text(test_path), _answer_map(pdf_text(key_path))
    starts = list(re.finditer(r"(?m)^\s*(\d+)[.)][^\w]*", text))
    rows: list[dict[str, Any]] = []
    for index, match in enumerate(starts):
        number = int(match.group(1))
        if number not in keys:
            continue
        block = text[match.end(): starts[index + 1].start() if index + 1 < len(starts) else len(text)]
        parts = list(re.finditer(r"(?m)^\s*([A-E])[.)][^\w]*", block))
        if len(parts) < 4:
            continue
        stem = block[:parts[0].start()].strip()
        choices: dict[str, str] = {}
        for pidx, part in enumerate(parts):
            label = part.group(1)
            end = parts[pidx + 1].start() if pidx + 1 < len(parts) else len(block)
            choices[label] = re.sub(r"\s+", " ", block[part.end():end]).strip()
        if set(choices) != set("ABCD") or keys[number] not in choices or len(stem) < 8:
            continue
        topic = classify_topics(stem + " " + " ".join(choices.values()))[0]
        row = {"id": f"{source}-{number:03d}", "competition": "Science Olympiad", "event": "Heredity",
               "division": "B", "response_type": "multiple_choice", "prompt": re.sub(r"\s+", " ", stem),
               "choices": choices, "answer": keys[number], "solution": f"The paired answer key gives {keys[number]}.",
               "points": 1, "difficulty": 1 if number <= 2 else 2, "topics": [topic],
               "source": {"test": test_path.name, "key": key_path.name, "question_number": str(number)}}
        row["analysis"] = {"skills": [topic], "reasoning_graph": _reasoning_graph(row["prompt"], topic)}
        row["graph_text"] = graph_text(row)
        rows.append(row)
    return rows


BERKELEY = [
    (2, "short_answer", "For Aabb × AABb, give the genotypic ratio and phenotype ratio when A and B each show complete dominance and assort independently.", "Genotypic 1:1:1:1; phenotypic 1:1", 3, "mendelian_probability"),
    (4, "short_answer", "Two parents are heterozygous at 26 independently assorting loci. Give, as an expression, the probability that an offspring is heterozygous at all 26 loci.", "(1/2)^26", 3, "mendelian_probability"),
    (5, "numeric", "An I^A I^B parent and an I^B i parent have a child. What is the probability the child has blood type B?", {"value": 0.5, "unit": "probability", "tolerance": 0}, 2, "non_mendelian_inheritance"),
    (11, "numeric", "A diploid cell with 2n = 8 completes meiosis I and II normally. How many chromosomes are in each daughter cell?", {"value": 4, "unit": "chromosomes", "tolerance": 0}, 2, "cell_division_chromosomes"),
    (14, "short_answer", "What is the term for unequal chromosome segregation during anaphase?", "Nondisjunction", 1, "cell_division_chromosomes"),
    (20, "short_answer", "Write the antiparallel complementary strand to 5′-GATACAGATACA-3′, labeling both ends.", "3′-CTATGTCTATGT-5′", 2, "dna_replication_structure"),
    (24, "short_answer", "A double-stranded DNA sample is 32% adenine. State the percentages of thymine, guanine, cytosine, and uracil.", "T 32%, G 18%, C 18%, U 0%", 2, "dna_replication_structure"),
    (34, "numeric", "Starting with one double-stranded DNA molecule, how many double-stranded molecules result after five ideal PCR cycles?", {"value": 32, "unit": "DNA molecules", "tolerance": 0}, 2, "mutations_biotechnology"),
]


def curated_berkeley(test_path: Path, key_path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, response_type, prompt, answer, difficulty, topic in BERKELEY:
        row = {"id": f"berkeley-2026-{number:03d}", "competition": "Science Olympiad", "event": "Heredity",
               "division": "B", "response_type": response_type, "prompt": prompt, "answer": answer,
               "solution": f"Apply {topic.replace('_', ' ')}; the paired key gives {answer}.", "points": 2,
               "difficulty": difficulty, "topics": [topic],
               "source": {"test": test_path.name, "key": key_path.name, "question_number": str(number)}}
        row["analysis"] = {"skills": [topic], "reasoning_graph": _reasoning_graph(prompt, topic)}
        row["graph_text"] = graph_text(row)
        rows.append(row)
    return rows


def build_corpus(source_root: Path) -> list[dict[str, Any]]:
    rows = extract_mcq_pair(source_root / "bullso/bullso-test.pdf", source_root / "bullso/bullso-key.pdf", "bullso-2026")
    rows += curated_berkeley(source_root / "berkeley/berkeley-test.pdf", source_root / "berkeley/berkeley-key.pdf")
    rows += extract_mcq_pair(source_root / "sample/2014-test.pdf", source_root / "sample/2014-key.pdf", "ut-austin-2014")
    return rows


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="science_olympiad/heredity_b/sources")
    parser.add_argument("--out", default="science_olympiad/heredity_b/run/corpus/items.jsonl")
    args = parser.parse_args()
    rows = build_corpus(Path(args.source_dir))
    write_jsonl(args.out, rows)
    print(json.dumps({"items": len(rows), "sources": sorted({x['source']['test'] for x in rows})}))


if __name__ == "__main__":
    main()
