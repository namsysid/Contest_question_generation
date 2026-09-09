from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from .common import classify_topics, slug, write_jsonl

try:
    import fitz
except ModuleNotFoundError:
    fitz = None


MCQ_RE = re.compile(r"(?ms)^\s*(\d+)\.\s*(?:\[(\d+)\s*points?\]\s*)?(.*?)(?=^\s*\d+\.\s*(?:\[\d+\s*points?\]\s*)?|\Z)")
CHOICE_RE = re.compile(r"(?ms)^\s*([A-E])\.\s*(.*?)(?=^\s*[A-E]\.\s|\Z)")
PART_RE = re.compile(r"(?ms)^\s*\(([a-z])\)\s*(?:\[(\d+)\s*points?\]\s*)?(.*?)(?=^\s*\([a-z]\)\s|\Z)")
SECTION_RE = re.compile(r"(?m)^\s*(?:\d+\s*)?\n?(Multiple Choice|DC Circuit Analysis|Electrostatics|Hands On)\s*(?:\[\d+ points\])?\s*$", re.I)


def _clean(text: str) -> str:
    text = re.sub(r"(?m)^.*(?:Page \d+ of \d+|Science Olympiad @|DO NOT WRITE ON THIS EXAM|Exam Booklet|February \d+).*$", "", text)
    return " ".join(text.split())


def _page_assets(doc: Any, output: Path, stem: str) -> dict[int, str]:
    output.mkdir(parents=True, exist_ok=True)
    assets = {}
    for index, page in enumerate(doc):
        path = output / f"{stem}-page-{index + 1:02d}.png"
        page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(path)
        assets[index + 1] = str(path)
    return assets


def parse_pdf(path: str | Path, asset_dir: str | Path) -> list[dict[str, Any]]:
    if fitz is None:
        raise RuntimeError("PyMuPDF is required: pip install pymupdf")
    source = Path(path)
    doc = fitz.open(source)
    page_assets = _page_assets(doc, Path(asset_dir), slug(source.stem))
    pages = [page.get_text("text") for page in doc]
    full = "\n".join(pages)
    sections = list(SECTION_RE.finditer(full))
    rows: list[dict[str, Any]] = []
    for sec_index, match in enumerate(sections):
        name = match.group(1).lower()
        if name == "hands on":
            continue
        body = full[match.end(): sections[sec_index + 1].start() if sec_index + 1 < len(sections) else len(full)]
        if name == "multiple choice":
            for question in MCQ_RE.finditer(body):
                raw = question.group(3)
                choices_found = list(CHOICE_RE.finditer(raw))
                if len(choices_found) < 4:
                    continue
                prompt = _clean(raw[: choices_found[0].start()])
                choices = {c.group(1): _clean(c.group(2)) for c in choices_found}
                rows.append({
                    "id": f"{slug(source.stem)}-mc-{int(question.group(1)):02d}",
                    "event": "circuit_lab", "division": "B", "section": "multiple_choice",
                    "response_type": "multiple_choice", "prompt": prompt, "choices": choices,
                    "answer": None, "solution": None, "points": int(question.group(2) or 1),
                    "topics": classify_topics(prompt), "assets": [],
                    "source": {"pdf": source.name, "question_number": int(question.group(1))},
                })
        else:
            for question in MCQ_RE.finditer(body):
                raw = question.group(3)
                parts = []
                for part in PART_RE.finditer(raw):
                    prompt = _clean(part.group(3))
                    response_type = "numeric" if re.search(r"calculate|magnitude|position|how many|what is the (?:current|potential|power|energy|resistance)", prompt, re.I) else "short_answer"
                    parts.append({"id": part.group(1), "response_type": response_type, "prompt": prompt,
                                  "answer": None, "solution": None, "points": int(part.group(2) or 1)})
                if not parts:
                    continue
                shared = _clean(raw[: raw.find(f"({parts[0]['id']})")])
                points = sum(part["points"] for part in parts)
                rows.append({
                    "id": f"{slug(source.stem)}-{slug(name)}-{int(question.group(1)):02d}",
                    "event": "circuit_lab", "division": "B", "section": slug(name),
                    "response_type": "multipart", "prompt": shared, "parts": parts,
                    "points": points, "topics": classify_topics(shared + " " + " ".join(p["prompt"] for p in parts)),
                    "assets": list(page_assets.values()),
                    "source": {"pdf": source.name, "question_number": int(question.group(1))},
                })
    doc.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Circuit Lab written questions from a PDF")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--asset-dir", default="science_olympiad/circuit_lab_b/corpus/assets")
    args = parser.parse_args()
    rows = parse_pdf(args.input, args.asset_dir)
    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} items to {args.out}")


if __name__ == "__main__":
    main()
