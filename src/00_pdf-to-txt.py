#!/usr/bin/env python3
"""
Stage 00:
- download all F=net=ma exam PDFs from the AAPT past exams page
- extract combined text
- split questions into JSONL
- extract embedded / rendered figures
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

try:
    import fitz  # PyMuPDF
except ModuleNotFoundError:
    fitz = None


DEFAULT_INDEX_URL = "https://aapt.org/physicsteam/PT-exams.cfm"
USER_AGENT = "ContestQuestionGeneration/1.0 (+https://aapt.org/physicsteam/PT-exams.cfm)"
YEAR_RE = re.compile(r"^\s*(20\d{2}|19\d{2})\s*$")
QUESTION_RE = re.compile(r"\n\s*(\d{1,2})\.\s")
FIGURE_HINT_RE = re.compile(r"(figure|diagram|shown|shown below)", re.IGNORECASE)


@dataclass(frozen=True)
class ExamLink:
    year: int
    label: str
    url: str

    @property
    def filename(self) -> str:
        suffix = ""
        match = re.search(r"\b([AB])\b", self.label, re.IGNORECASE)
        if match:
            suffix = f"_{match.group(1).upper()}"
        return f"{self.year}_fnet_ma_exam{suffix}.pdf"


class PastExamsParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.current_heading_level: str | None = None
        self.current_heading_text: list[str] = []
        self.current_year: int | None = None
        self.in_anchor = False
        self.anchor_href: str | None = None
        self.anchor_text: list[str] = []
        self.links: list[ExamLink] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.current_heading_level = tag
            self.current_heading_text = []
        elif tag == "a":
            self.in_anchor = True
            self.anchor_href = attr_map.get("href")
            self.anchor_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag == self.current_heading_level:
            heading = " ".join("".join(self.current_heading_text).split())
            if YEAR_RE.match(heading):
                self.current_year = int(heading)
            self.current_heading_level = None
            self.current_heading_text = []
            return

        if tag == "a" and self.in_anchor:
            label = " ".join("".join(self.anchor_text).split())
            if self.current_year is not None and self.anchor_href and is_fnet_ma_label(label):
                self.links.append(
                    ExamLink(
                        year=self.current_year,
                        label=label,
                        url=urljoin(self.base_url, self.anchor_href),
                    )
                )
            self.in_anchor = False
            self.anchor_href = None
            self.anchor_text = []

    def handle_data(self, data: str) -> None:
        if self.current_heading_level is not None:
            self.current_heading_text.append(data)
        if self.in_anchor:
            self.anchor_text.append(data)


def is_fnet_ma_label(label: str) -> bool:
    normalized = label.casefold()
    if "solution" in normalized or "answer" in normalized:
        return False
    return "fnet=ma exam" in normalized


def fetch_text(url: str, timeout: float) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def fetch_binary(url: str, timeout: float) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def collect_exam_links(index_url: str, timeout: float) -> list[ExamLink]:
    parser = PastExamsParser(index_url)
    parser.feed(fetch_text(index_url, timeout))
    deduped: dict[tuple[int, str], ExamLink] = {}
    for link in parser.links:
        deduped[(link.year, link.filename)] = link
    return sorted(deduped.values(), key=lambda item: (item.year, item.filename))


def ensure_pdf_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme for PDF link: {url}")


def filter_years(links: Iterable[ExamLink], years: set[int] | None) -> list[ExamLink]:
    if years is None:
        return list(links)
    return [link for link in links if link.year in years]


def download_exam(link: ExamLink, raw_pdf_dir: Path, timeout: float, overwrite: bool) -> Path:
    ensure_pdf_url(link.url)
    out_path = raw_pdf_dir / link.filename
    if out_path.exists() and not overwrite:
        return out_path

    pdf_bytes = fetch_binary(link.url, timeout)
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError(f"Downloaded file does not look like a PDF: {link.url}")
    out_path.write_bytes(pdf_bytes)
    return out_path


def extract_page_text_blocks(page) -> str:
    blocks = page.get_text("blocks")
    blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))
    return "\n".join(b[4] for b in blocks if isinstance(b[4], str))


def render_question_crop(page, qnum: int, page_idx: int, out_dir: Path, name_prefix: str) -> List[str]:
    hits = page.search_for(f"{qnum}.")
    if not hits:
        return []

    rect = hits[0]
    for hit in hits[1:]:
        rect |= hit

    rect = (rect + (-1800, -5, 1800, 150)) & page.rect
    if rect.is_empty:
        return []

    name = f"{name_prefix}_page{page_idx + 1:02d}_q{qnum:02d}.png"
    out_path = out_dir / name
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
    pix.save(out_path)
    pix = None
    return [name]


def extract_page_images(doc, page_idx: int, out_dir: Path, name_prefix: str) -> List[str]:
    page = doc.load_page(page_idx)
    image_paths = []
    for img_idx, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        if pix.n > 4:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        name = f"{name_prefix}_page{page_idx + 1:02d}_img{img_idx + 1}.png"
        out_path = out_dir / name
        pix.save(out_path)
        pix = None
        image_paths.append(name)
    return image_paths


def split_questions(full_text: str) -> List[Dict]:
    splits = list(QUESTION_RE.finditer(full_text))
    questions = []
    for i, match in enumerate(splits):
        start = match.start()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(full_text)
        qnum = match.group(1)
        qtext = full_text[start:end].strip()
        questions.append(
            {
                "question_number": int(qnum),
                "question_text": qtext,
                "span": (start, end),
            }
        )
    return questions


def parse_pdf(pdf_path: Path, text_fh, jsonl_fh, image_dir: Path) -> None:
    doc = fitz.open(pdf_path)
    name_prefix = pdf_path.stem
    page_texts = []
    page_figures = {}

    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        page_text = extract_page_text_blocks(page)
        page_texts.append(page_text)
        page_figures[page_idx] = extract_page_images(doc, page_idx, image_dir, name_prefix)

    full_text = "\n\n".join(page_texts)
    page_ranges = []
    cursor = 0
    for idx, text in enumerate(page_texts):
        start = cursor
        cursor += len(text)
        end = cursor
        page_ranges.append((start, end))
        if idx < len(page_texts) - 1:
            cursor += 2

    text_fh.write(f"\n\n### source: {pdf_path.name}\n")
    text_fh.write(full_text)
    text_fh.flush()

    for question in split_questions(full_text):
        text = question["question_text"]
        uses_figure = bool(FIGURE_HINT_RE.search(text))
        q_start, q_end = question["span"]
        question_pages = [
            idx
            for idx, (p_start, p_end) in enumerate(page_ranges)
            if not (q_end <= p_start or q_start >= p_end)
        ]

        attached_figs: list[str] = []
        if uses_figure:
            for page_idx in question_pages:
                attached_figs.extend(page_figures.get(page_idx, []))
            if not attached_figs:
                for page_idx in question_pages:
                    attached_figs.extend(
                        render_question_crop(
                            doc.load_page(page_idx),
                            question["question_number"],
                            page_idx,
                            image_dir,
                            name_prefix,
                        )
                    )
            if not attached_figs:
                for page_idx in question_pages:
                    page = doc.load_page(page_idx)
                    name = f"{name_prefix}_page{page_idx + 1:02d}_render.png"
                    out_path = image_dir / name
                    if not out_path.exists():
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        pix.save(out_path)
                        pix = None
                    attached_figs.append(name)

        row = {
            "id": f"{pdf_path.stem}_Q{question['question_number']:02d}",
            "question_number": question["question_number"],
            "question_text": text,
            "has_diagram": uses_figure,
            "diagram_files": attached_figs,
            "source_pdf": pdf_path.name,
        }
        jsonl_fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    jsonl_fh.flush()
    doc.close()


def parse_downloaded_pdfs(raw_pdf_dir: Path, txt_dir: Path) -> None:
    if fitz is None:
        raise RuntimeError("PyMuPDF is required for PDF parsing. Install the `pymupdf` package.")
    image_dir = txt_dir / "figures"
    image_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(raw_pdf_dir.glob("*.pdf"))
    all_text_path = txt_dir / "all_text.txt"
    all_jsonl_path = txt_dir / "all_questions.jsonl"
    with all_text_path.open("w", encoding="utf-8") as text_fh, all_jsonl_path.open("w", encoding="utf-8") as jsonl_fh:
        for pdf_path in pdfs:
            parse_pdf(pdf_path, text_fh, jsonl_fh, image_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        dest="out_dir",
        required=True,
        help="Output root directory. The script writes raw_pdfs/ and txts/ under this path.",
    )
    parser.add_argument(
        "--index-url",
        default=DEFAULT_INDEX_URL,
        help=f"Page to scrape for past exam links. Defaults to {DEFAULT_INDEX_URL}.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        help="Optional list of years to download, e.g. --years 2024 2025.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download PDFs even if the destination file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the discovered downloads without writing files or parsing PDFs.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Network timeout in seconds for page fetches and downloads.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not fetch PDFs; parse whatever already exists in raw_pdfs/.",
    )
    parser.add_argument(
        "--skip-parse",
        action="store_true",
        help="Download PDFs only and skip text/JSONL extraction.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    raw_pdf_dir = out_dir / "raw_pdfs"
    txt_dir = out_dir / "txts"
    raw_pdf_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    links: list[ExamLink] = []
    if not args.skip_download:
        links = filter_years(
            collect_exam_links(args.index_url, args.timeout),
            set(args.years) if args.years else None,
        )
        if not links:
            print("No F=net=ma exam PDFs found for the requested criteria.", file=sys.stderr)
            return 1

        for link in links:
            print(f"{link.year}: {link.label} -> {link.filename}")
            if not args.dry_run:
                path = download_exam(link, raw_pdf_dir, args.timeout, args.overwrite)
                print(f"saved {path}")

    if args.dry_run:
        return 0

    if args.skip_parse:
        return 0

    if not any(raw_pdf_dir.glob("*.pdf")):
        print(f"No PDFs found in {raw_pdf_dir}", file=sys.stderr)
        return 1

    parse_downloaded_pdfs(raw_pdf_dir, txt_dir)
    print(f"wrote {txt_dir / 'all_text.txt'}")
    print(f"wrote {txt_dir / 'all_questions.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
