# src/pdf_loader.py
"""
PDF loader for IEEE-style papers.

Body text and figures/tables are handled separately:
  1. Figure regions (images, vector drawings next to a "Fig." caption) and tables
     (detected with PyMuPDF's find_tables) are located on every page.
  2. Text inside those regions (axis labels, table cells) and the caption blocks
     are left out of the body text, so they cannot break sentences apart.
  3. Each caption is stored as a Figure, together with the table cells as Markdown.
     The body text keeps its "see Fig. 2" mentions, which src/figures.py uses to
     link a paragraph back to the right figure/table.
"""
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24.3 (new import name)
except ImportError:
    import fitz  # older PyMuPDF

from src.figures import Figure, parse_caption

# Margins (in points) that hold running headers and footers
HEADER_MARGIN = 30
FOOTER_MARGIN = 30
# Max vertical gap (points) between a caption block and the block that continues it
CAPTION_GAP = 6
# Max distance (points) between a figure's graphics and its caption
FIGURE_CAPTION_DISTANCE = 40
# Graphics smaller than this fraction of the page are icons/logos; larger than MAX are page backgrounds
MIN_GRAPHIC_AREA = 0.02
MAX_GRAPHIC_AREA = 0.9
# A text block that lies this much inside a figure/table region belongs to it
INSIDE_FRACTION = 0.6
# The heading that starts the bibliography; only searched for in the second half of the paper
REFERENCES_HEADING_RE = re.compile(r"(?:^|\n\n)\s*(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography)\b")


@dataclass
class ParsedPaper:
    text: str                                   # body text, without figures/tables and references
    figures: List[Figure] = field(default_factory=list)
    references: str = ""                        # the bibliography (and anything after it)

    def full_text(self) -> str:
        """Body text, then all figure/table captions (and table cells), then the references."""
        parts = [self.text]
        if self.figures:
            parts.append("--- FIGURES AND TABLES ---\n" + "\n\n".join(f.as_text() for f in self.figures))
        if self.references:
            parts.append("--- REFERENCES ---\n" + self.references)
        return "\n\n".join(parts)


def split_references(text: str):
    """
    Split text into (body, references) at the last references heading in the second half.
    References are kept out of the retrieval index: citation lists match many queries
    by keyword while answering none of them.
    """
    matches = [m for m in REFERENCES_HEADING_RE.finditer(text) if m.start() >= len(text) // 2]
    if not matches:
        return text, ""
    cut = matches[-1]
    body = text[:cut.start()].rstrip()
    references = re.sub(r"^\s*(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography)\s*", "", text[cut.start():])
    return body, references.strip()


def clean_text(text: str) -> str:
    """
    Clean text by joining broken lines and removing hyphenations.
    """
    # Turn ligatures such as the single "fi" glyph (U+FB01) into plain letters
    text = unicodedata.normalize("NFKC", text)
    # Remove line-break hyphenation ("tun-\ning" -> "tuning") but keep real hyphens ("DP-\nPSASC")
    text = re.sub(r'(\w+)-\s*\n\s*([a-z]\w*)', r'\1\2', text)
    text = re.sub(r'(\w)-\s*\n\s*(?=[A-Z0-9])', r'\1-', text)
    # Replace single newlines with a space
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Clean up multiple spaces
    text = re.sub(r' +', ' ', text)
    return text.strip()


def _area(r) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def _inside_fraction(inner, outer) -> float:
    """Fraction of rectangle `inner` that lies inside rectangle `outer`."""
    w = min(inner[2], outer[2]) - max(inner[0], outer[0])
    h = min(inner[3], outer[3]) - max(inner[1], outer[1])
    if w <= 0 or h <= 0:
        return 0.0
    return (w * h) / max(_area(inner), 1e-6)


def _horizontal_overlap(a, b) -> float:
    return min(a[2], b[2]) - max(a[0], b[0])


def _vertical_distance(a, b) -> float:
    """Gap between two rectangles along y (0 if they overlap vertically)."""
    return max(0.0, max(a[1], b[1]) - min(a[3], b[3]))


def _find_captions(blocks):
    """
    Find caption blocks and merge each one with the blocks that directly continue it.
    Returns a list of (kind, number, caption_text, bbox, used_block_ids).
    """
    captions = []
    used = set()
    for i, b in enumerate(blocks):
        parsed = parse_caption(b[4])
        if parsed is None:
            continue
        kind, number, body = parsed
        bbox = list(b[:4])
        parts = [body]
        ids = {i}
        # Absorb continuation blocks: just below, horizontally overlapping, not a caption themselves
        while True:
            nxt = None
            for j, c in enumerate(blocks):
                if j in ids or j in used or parse_caption(c[4]):
                    continue
                gap = c[1] - bbox[3]
                if -1 <= gap <= CAPTION_GAP and _horizontal_overlap(c, bbox) > 0.5 * (c[2] - c[0]):
                    if nxt is None or c[1] < blocks[nxt][1]:
                        nxt = j
            if nxt is None:
                break
            c = blocks[nxt]
            parts.append(c[4])
            ids.add(nxt)
            bbox = [min(bbox[0], c[0]), bbox[1], max(bbox[2], c[2]), c[3]]
        used |= ids
        captions.append((kind, number, clean_text("\n".join(parts)), bbox, ids))
    return captions


def _graphic_regions(page, captions):
    """
    Rectangles that hold figures: raster images, plus vector-drawing clusters that sit
    next to a figure caption (so ruled boxes around body text are not mistaken for figures).
    """
    page_area = _area(page.rect)

    def plausible(r):
        return MIN_GRAPHIC_AREA * page_area <= _area(r) <= MAX_GRAPHIC_AREA * page_area

    regions = [tuple(info["bbox"]) for info in page.get_image_info() if plausible(info["bbox"])]

    figure_captions = [c for c in captions if c[0] == "figure"]
    try:
        clusters = page.cluster_drawings()
    except Exception:
        clusters = []
    for r in clusters:
        r = tuple(r)
        if plausible(r) and any(
            _vertical_distance(r, cap[3]) <= FIGURE_CAPTION_DISTANCE and _horizontal_overlap(r, cap[3]) > 0
            for cap in figure_captions
        ):
            regions.append(r)
    return regions


def _tables(page):
    """Detected tables as (bbox, markdown)."""
    try:
        found = page.find_tables().tables
    except Exception:
        return []
    tables = []
    for t in found:
        try:
            md = t.to_markdown().replace("<br>", " ")
        except Exception:
            md = ""
        tables.append((tuple(t.bbox), unicodedata.normalize("NFKC", md).strip()))
    return tables


def _parse_page(page, page_number):
    """Split one page into body blocks (in reading order) and its figures/tables."""
    page_height = page.rect.height
    blocks = [b for b in page.get_text("blocks")
              if b[1] > HEADER_MARGIN and b[3] < page_height - FOOTER_MARGIN and b[4].strip()]

    captions = _find_captions(blocks)
    tables = _tables(page)
    graphics = _graphic_regions(page, captions)
    # A table also shows up as a drawing cluster; keep only one region for it
    graphics = [g for g in graphics if not any(_inside_fraction(g, t[0]) > 0.8 for t in tables)]

    # Match every table with the nearest table caption on the page
    figures = []
    unclaimed_tables = list(tables)
    for kind, number, caption, bbox, _ in captions:
        content = ""
        if kind == "table" and unclaimed_tables:
            nearest = min(unclaimed_tables, key=lambda t: _vertical_distance(t[0], bbox))
            if _vertical_distance(nearest[0], bbox) <= FIGURE_CAPTION_DISTANCE:
                content = nearest[1]
                unclaimed_tables.remove(nearest)
        figures.append(Figure(kind, number, page_number, caption, content))
    for _, md in unclaimed_tables:
        figures.append(Figure("table", None, page_number, "", md))

    # Body text = everything that is neither a caption nor inside a figure/table
    caption_ids = set().union(*(c[4] for c in captions)) if captions else set()
    regions = graphics + [t[0] for t in tables]
    body = [b for i, b in enumerate(blocks)
            if i not in caption_ids and not any(_inside_fraction(b, r) >= INSIDE_FRACTION for r in regions)]

    # Sorting logic for Spanning, Left, and Right columns
    page_width = page.rect.width
    middle = page_width / 2

    def get_block_order(b):
        x0, y0, x1, y1 = b[:4]
        width = max(x1 - x0, 1e-6)
        # Spanning block: very wide (abstract), or centered across the middle (title, authors)
        centered = min(middle - x0, x1 - middle) > 0.35 * width
        if width > (page_width * 0.8) or centered:
            column_id = 0
        elif (x0 + x1) / 2 < (page_width / 2):  # Left column
            column_id = 1
        else:  # Right column
            column_id = 2
        return (column_id, y0)

    return sorted(body, key=get_block_order), figures


def parse_pdf(pdf_path: str) -> ParsedPaper:
    """
    Parse a PDF into body text (column-sorted, header/footer filtered, smartly joined)
    and a list of its figures and tables.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    # Using a single string to accumulate all content for easier punctuation checking
    full_text = ""
    figures = []

    for page_num in range(len(doc)):
        body_blocks, page_figures = _parse_page(doc[page_num], page_num + 1)
        figures.extend(page_figures)

        # Smart joining logic for blocks within the current page
        for b in body_blocks:
            cleaned = clean_text(b[4])
            if not cleaned:
                continue
            if not full_text:
                full_text = cleaned
            elif full_text.endswith(('.', '!', '?', ':')):
                # Likely a paragraph end
                full_text += "\n\n" + cleaned
            else:
                # The sentence likely continues in the next block
                full_text += " " + cleaned

    doc.close()
    body, references = split_references(full_text)
    return ParsedPaper(body, figures, references)


if __name__ == "__main__":
    # Local module test: python -m src.pdf_loader [path/to/paper.pdf]
    import os
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    default_pdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sample_ieee.pdf")
    paper = parse_pdf(sys.argv[1] if len(sys.argv) > 1 else default_pdf)
    print(f"Body text: {len(paper.text)} characters, {len(paper.figures)} figures/tables, "
          f"references: {len(paper.references)} characters\n")
    for f in paper.figures:
        print(f.as_text()[:300], "\n")
    print("--- Preview (first 1500 chars of body text) ---")
    print(paper.text[:1500])
