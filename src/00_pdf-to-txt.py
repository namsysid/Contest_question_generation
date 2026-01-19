#!/usr/bin/env python3
"""
Parse STEM contest PDFs into:
- text
- per-question JSONL
- extracted diagram images

Single approach: PyMuPDF text + image extraction only (NO OCR)
"""

from pathlib import Path
import re
import json
import fitz  # PyMuPDF
from typing import List, Dict

# ------------------ CONFIG ------------------

QUESTION_RE = re.compile(r"\n\s*(\d{1,2})\.\s")  # 1. 2. 3. ...
FIGURE_HINT_RE = re.compile(r"(figure|diagram|shown|shown below)", re.IGNORECASE)

# ------------------ CORE ------------------

def extract_page_text_blocks(page) -> str:
    blocks = page.get_text("blocks")
    blocks = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))
    return "\n".join(b[4] for b in blocks if isinstance(b[4], str))


def render_question_crop(page, qnum: int, page_idx: int, out_dir: Path, name_prefix: str) -> List[str]:
    """Render a clipped region around the question number to avoid full-page images."""
    hits = page.search_for(f"{qnum}.")
    if not hits:
        return []

    rect = hits[0]
    for r in hits[1:]:
        rect |= r

    # Wider crop, shallower height to include nearby diagram without full page
    pad_x = 1800
    pad_y = 5
    rect = (rect + (-pad_x, -pad_y, pad_x, pad_y * 30)) & page.rect
    if rect.is_empty:
        return []

    name = f"{name_prefix}_page{page_idx+1:02d}_q{qnum:02d}.png"
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

        if pix.n > 4:  # CMYK → RGB
            pix = fitz.Pixmap(fitz.csRGB, pix)

        name = f"{name_prefix}_page{page_idx+1:02d}_img{img_idx+1}.png"
        out_path = out_dir / name
        pix.save(out_path)
        pix = None

        image_paths.append(name)

    return image_paths


def split_questions(full_text: str) -> List[Dict]:
    splits = list(QUESTION_RE.finditer(full_text))
    questions = []

    for i, m in enumerate(splits):
        start = m.start()
        end = splits[i+1].start() if i+1 < len(splits) else len(full_text)
        qnum = m.group(1)
        qtext = full_text[start:end].strip()

        questions.append({
            "question_number": int(qnum),
            "question_text": qtext,
            "span": (start, end),
        })

    return questions


def parse_pdf(pdf_path: Path, text_fh, jsonl_fh, image_dir: Path):
    doc = fitz.open(pdf_path)

    name_prefix = pdf_path.stem

    page_texts = []
    page_figures = {}

    # ---- extract text + images per page ----
    for p in range(doc.page_count):
        page = doc.load_page(p)
        page_text = extract_page_text_blocks(page)
        page_texts.append(page_text)

        figs = extract_page_images(doc, p, image_dir, name_prefix)
        page_figures[p] = figs

    full_text = "\n\n".join(page_texts)
    page_ranges = []
    cursor = 0
    for i, text in enumerate(page_texts):
        start = cursor
        cursor += len(text)
        end = cursor
        page_ranges.append((start, end))
        if i < len(page_texts) - 1:
            cursor += 2  # account for the "\n\n" joiner

    text_fh.write(f"\n\n### source: {pdf_path.name}\n")
    text_fh.write(full_text)
    text_fh.flush()

    # ---- split into questions ----
    questions = split_questions(full_text)

    # ---- attach diagrams heuristically ----
    enriched = []
    for q in questions:
        text = q["question_text"]
        uses_figure = bool(FIGURE_HINT_RE.search(text))
        q_start, q_end = q["span"]

        # map question text span to page indices
        question_pages = [
            idx for idx, (p_start, p_end) in enumerate(page_ranges)
            if not (q_end <= p_start or q_start >= p_end)
        ]

        attached_figs = []
        if uses_figure:
            for p_idx in question_pages:
                attached_figs.extend(page_figures.get(p_idx, []))

            # If no embedded images were found, try cropped renders around the question number
            if not attached_figs:
                for p_idx in question_pages:
                    page = doc.load_page(p_idx)
                    crop_imgs = render_question_crop(
                        page, q["question_number"], p_idx, image_dir, name_prefix
                    )
                    attached_figs.extend(crop_imgs)

            # As a final fallback, render full pages for coverage (still avoids missing diagrams)
            if not attached_figs:
                for p_idx in question_pages:
                    page = doc.load_page(p_idx)
                    name = f"{name_prefix}_page{p_idx+1:02d}_render.png"
                    out_path = image_dir / name
                    if not out_path.exists():
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        pix.save(out_path)
                        pix = None
                    attached_figs.append(name)

        enriched.append({
            "id": f"{pdf_path.stem}_Q{q['question_number']:02d}",
            "question_number": q["question_number"],
            "question_text": text,
            "has_diagram": uses_figure,
            "diagram_files": attached_figs,
            "source_pdf": pdf_path.name
        })

    # ---- write JSONL ----
    for row in enriched:
        jsonl_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    jsonl_fh.flush()

    doc.close()


# ------------------ CLI ------------------

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="pdf_dir", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--recursive", action="store_true")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = out_dir / "figures"
    image_dir.mkdir(parents=True, exist_ok=True)

    pdfs = pdf_dir.rglob("*.pdf") if args.recursive else pdf_dir.glob("*.pdf")

    all_text_path = out_dir / "all_text.txt"
    all_jsonl_path = out_dir / "all_questions.jsonl"
    with all_text_path.open("w", encoding="utf-8") as text_fh, \
            all_jsonl_path.open("w", encoding="utf-8") as jsonl_fh:
        for pdf in pdfs:
            parse_pdf(pdf, text_fh, jsonl_fh, image_dir)

    print("Done.")


if __name__ == "__main__":
    main()
