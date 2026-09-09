#!/usr/bin/env python3
"""Download official ACS USNCO MCQ exams and convert usable questions to JSONL."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

try:
    import fitz
except ModuleNotFoundError:  # pragma: no cover - reported by the CLI
    fitz = None

from usnco_generation import USNCO_TOPIC_BY_KEY, infer_usnco_topic


DEFAULT_INDEX_URL = "https://www.acs.org/education/olympiad/prepare-for-exams.html"
USER_AGENT = "ContestQuestionGeneration/1.0 (official USNCO corpus builder)"
PDF_RE = re.compile(r"(?P<year>20\d{2})[^/?#]*?(?P<kind>local|national)[^/?#]*\.pdf", re.I)
QUESTION_RE = re.compile(r"(?m)^\s*(?P<number>[1-9]|[1-5]\d|60)\.\s+")
CHOICE_MARKER_RE = re.compile(r"\(\s*(?P<letter>[A-D])\s*\)")
KEY_PAIR_RE = re.compile(r"(?m)^\s*(?P<number>[1-9]|[1-5]\d|60)\.\s*(?P<letter>[A-D])\s*$")
ACS_PAST_EXAMS = "https://www.acs.org/content/dam/acsorg/education/students/highschool/olympiad/pastexams"

# ACS sometimes serves its index page through an anti-bot interstitial. These are exact public
# ACS links, used only when live link discovery yields nothing. Keep discovery as the first path
# so new years and filename changes are picked up automatically.
KNOWN_EXAM_PATHS = (
    (2026, "local", "2026-usnco-local-exam.pdf"),
    (2025, "local", "2025-usnco-local-exam.pdf"),
    (2024, "local", "2024-usnco-local-exam.pdf"),
    (2023, "local", "2023-usnco-local-exam-local-original.pdf"),
    (2023, "local", "2023-usnco-local-exam-new.pdf"),
    (2022, "local", "2022-usnco-local-exam.pdf"),
    (2021, "local", "2021-usnco-local-exam.pdf"),
    (2020, "local", "2020-usnco-local-exam.pdf"),
    (2019, "local", "2019-usnco-local-exam.pdf"),
    (2018, "local", "2018-local-olympiad-exam.pdf"),
    (2017, "local", "2017-local-olympiad-exam.pdf"),
    (2026, "national", "2026-usnco-national-exam-part-i.pdf"),
    (2025, "national", "2025-usnco-national-exam-part-i.pdf"),
    (2024, "national", "2024-usnco-national-exam-part-i.pdf"),
    (2023, "national", "2023-usnco-national-exam-part-i.pdf"),
    (2022, "national", "2022-usnco-exam-part-i.pdf"),
    (2021, "national", "2021-usnco-exam-part-i.pdf"),
    (2020, "national", "2020_usnco_exam_part_i.pdf"),
    (2019, "national", "2019-usnco-exam-part-i.pdf"),
    (2018, "national", "2018-usnco-exam-part-i.pdf"),
    (2017, "national", "2017-national-exam-part-i.pdf"),
)


@dataclass(frozen=True)
class ExamLink:
    year: int
    exam_type: str
    url: str

    @property
    def filename(self) -> str:
        return Path(urlparse(self.url).path).name


class AcsExamLinkParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[ExamLink] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href") or ""
        url = urljoin(self.base_url, href)
        match = PDF_RE.search(url)
        if not match:
            return
        lowered = url.casefold()
        exam_type = match.group("kind").casefold()
        if exam_type == "national" and not re.search(r"(?:part[-_ ]?i(?:[^i]|$)|part-1|part1)", lowered):
            return
        if "solution" in lowered or "annotat" in lowered:
            return
        self.links.append(ExamLink(int(match.group("year")), exam_type, url))


def fetch_bytes(url: str, timeout: float) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported URL scheme: {url}")
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout) as response:
        return response.read()


def discover_exam_links(index_url: str, timeout: float = 30.0) -> list[ExamLink]:
    parser = AcsExamLinkParser(index_url)
    try:
        parser.feed(fetch_bytes(index_url, timeout).decode("utf-8", errors="replace"))
    except OSError:
        # Offline runs can still enumerate the known official links; downloading them
        # later will surface the underlying network error in the usual way.
        pass
    links = parser.links or [
        ExamLink(year, exam_type, f"{ACS_PAST_EXAMS}/{filename}")
        for year, exam_type, filename in KNOWN_EXAM_PATHS
    ]
    deduped = {(link.year, link.exam_type, link.url): link for link in links}
    return sorted(deduped.values(), key=lambda link: (link.year, link.exam_type, link.url))


def extract_pdf_text(path: Path) -> str:
    if fitz is None:
        raise RuntimeError("PDF extraction requires PyMuPDF; install the `pymupdf` package")
    document = fitz.open(path)
    try:
        return "\n".join(page.get_text("text") or "" for page in document)
    finally:
        document.close()


def _clean_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\r", "\n")
    text = re.sub(r"(?m)^\s*(?:Property of ACS.*|Page \d+|Olympiad \d{4}.*|USNCO.*Exam.*)\s*$", "", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def parse_answer_key(text: str) -> dict[int, str]:
    return {int(match.group("number")): match.group("letter") for match in KEY_PAIR_RE.finditer(text)}


def parse_mcq_text(text: str, link: ExamLink) -> list[dict]:
    """Parse only complete four-choice blocks; ambiguous PDF extraction is discarded."""
    answer_key = parse_answer_key(text)
    matches = list(QUESTION_RE.finditer(text))
    best: dict[int, dict] = {}
    for index, match in enumerate(matches):
        number = int(match.group("number"))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end():end]
        choice_matches = list(CHOICE_MARKER_RE.finditer(block))
        if [item.group("letter") for item in choice_matches] != list("ABCD"):
            continue
        choices = {}
        for choice_index, item in enumerate(choice_matches):
            choice_end = (
                choice_matches[choice_index + 1].start()
                if choice_index + 1 < len(choice_matches) else len(block)
            )
            choices[item.group("letter")] = _clean_text(block[item.end():choice_end])
        if set(choices) != set("ABCD") or any(not value for value in choices.values()):
            continue
        stem = _clean_text(block[:choice_matches[0].start()])
        if len(stem) < 12:
            continue
        topic_key = infer_usnco_topic(number, stem)
        topic, _ = USNCO_TOPIC_BY_KEY[topic_key]
        row = {
            "id": f"usnco-{link.year}-{link.exam_type}-{number:02d}",
            "domain": "chemistry",
            "question": stem,
            "choices": choices,
            "answer": answer_key.get(number),
            "topic_key": topic_key,
            "topic": topic,
            "source": {
                "publisher": "American Chemical Society",
                "program": "USNCO",
                "year": link.year,
                "exam_type": link.exam_type,
                "question_number": number,
                "url": link.url,
            },
        }
        if number not in best or len(stem) > len(best[number]["question"]):
            best[number] = row
    return [best[number] for number in sorted(best)]


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("chem/data/usnco_official.jsonl"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("chem/raw_pdfs"))
    parser.add_argument("--index-url", default=DEFAULT_INDEX_URL)
    parser.add_argument("--years", nargs="+", type=int)
    parser.add_argument("--exam-type", choices=("local", "national", "both"), default="both")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    links = discover_exam_links(args.index_url, args.timeout)
    if args.years:
        links = [link for link in links if link.year in set(args.years)]
    if args.exam_type != "both":
        links = [link for link in links if link.exam_type == args.exam_type]
    if not links:
        raise RuntimeError("no matching official ACS USNCO exam PDFs were discovered")
    if args.dry_run:
        for link in links:
            print(f"{link.year} {link.exam_type}: {link.url}")
        return 0

    args.pdf_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for link in links:
        pdf_path = args.pdf_dir / link.filename
        if args.overwrite or not pdf_path.exists():
            payload = fetch_bytes(link.url, args.timeout)
            if not payload.startswith(b"%PDF"):
                raise ValueError(
                    f"download did not return a PDF (ACS may be serving an anti-bot page): {link.url}"
                )
            pdf_path.write_bytes(payload)
        rows.extend(parse_mcq_text(extract_pdf_text(pdf_path), link))
    count = write_jsonl(args.out, rows)
    print(f"Wrote {count} complete official USNCO MCQ exemplars -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
