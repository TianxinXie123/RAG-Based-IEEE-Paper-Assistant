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
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24.3 (new import name)
except ImportError:
    import fitz  # older PyMuPDF

from src.figures import Figure, parse_caption
from src.chunker import HEADING_MARK
from src.symbols import apply_symbol_map, reattach_accents

# PyMuPDF span flag for superscript text (bit 0)
SUPERSCRIPT_FLAG = 1
# Text extraction flags: PyMuPDF's defaults for dict/rawdict, minus embedded image data (which made
# pages with large images many times slower to read while only the text is used)
TEXT_FLAGS = fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES

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
# A figure region may contain at most this many words of body-size text (axis labels, legends)
MAX_BODY_WORDS_IN_FIGURE = 40
# Text within this many points of the body font size counts as body-size text
BODY_SIZE_TOLERANCE = 0.6
# Running headers/footers live in the top and bottom share of the page; a line found there on at least
# REPEATED_LINE_SHARE of the pages (digits ignored, so page numbers do not matter) is a running header/footer
MARGIN_ZONE = 0.1
REPEATED_LINE_SHARE = 0.3
# Copyright notices, license statements and the download watermark of digital libraries
BOILERPLATE_RE = re.compile(
    r"Authorized licensed use limited to|All rights reserved|Personal use is permitted|requires IEEE permission"
    r"|ieee\.org/publications/rights|Creative Commons Attribution|creativecommons\.org/licenses",
    re.IGNORECASE)
# The heading that starts the bibliography; only searched for in the second half of the paper
REFERENCES_HEADING_RE = re.compile(r"(?:^|\n\n)\s*(?:## )?(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography)\b")

# Section headings: set in a font at least this much larger than the body text...
HEADING_SIZE_RATIO = 1.15
# ...short, and (mostly) in capitals or numbered like "II. TITLE" / "B. TITLE"
MAX_HEADING_CHARS = 120
HEADING_UPPER_SHARE = 0.8
NUMBERED_HEADING_RE = re.compile(r"^(?:[IVXLC]+|[A-Z])\.\s+\S")
# Headings that always start a new section (never the second line of the previous heading)
UNNUMBERED_HEADINGS = {"ABSTRACT", "INDEX TERMS", "REFERENCES", "BIBLIOGRAPHY", "ACKNOWLEDGMENT",
                       "ACKNOWLEDGMENTS", "ACKNOWLEDGEMENT", "APPENDIX", "CONCLUSION", "CONCLUSIONS"}


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


def raw_text(pdf_path: str) -> str:
    """
    All text of the PDF, page by page, with no filtering at all (margins, figures, captions included).
    Used as the reference for evaluation labels, independent of what the parser keeps.
    """
    with fitz.open(pdf_path) as doc:
        return "\n\n".join(clean_text(page.get_text()) for page in doc)


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
    references = re.sub(r"^\s*(?:## )?(?:REFERENCES|References|BIBLIOGRAPHY|Bibliography)\s*", "", text[cut.start():])
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


def _body_words_inside(region, lines, body_size) -> int:
    """Number of words of body-size text whose line lies inside the region."""
    count = 0
    for bbox, size, words in lines:
        if abs(size - body_size) <= BODY_SIZE_TOLERANCE and _inside_fraction(bbox, region) >= INSIDE_FRACTION:
            count += words
    return count


def _graphic_regions(page, captions, body_size):
    """
    Rectangles that hold figures: raster images, plus vector-drawing clusters that sit
    next to a figure caption (so ruled boxes around body text are not mistaken for figures).
    A region that contains a paragraph's worth of body-size text is not a figure: magazine
    layouts merge figure drawings with page decorations into clusters spanning whole columns.
    """
    page_area = _area(page.rect)
    lines = []
    for block in page.get_text("dict", flags=TEXT_FLAGS)["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"])
            if text.strip():
                size = max(line["spans"], key=lambda span: len(span["text"].strip()))["size"]
                lines.append((tuple(line["bbox"]), size, len(text.split())))

    def plausible(r):
        return (MIN_GRAPHIC_AREA * page_area <= _area(r) <= MAX_GRAPHIC_AREA * page_area
                and _body_words_inside(r, lines, body_size) <= MAX_BODY_WORDS_IN_FIGURE)

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


def _format_span(span, symbol_map) -> str:
    """Text of one span, with repaired special characters and superscripts marked as ^x or ^(...)."""
    text = apply_symbol_map("".join(c["c"] for c in span["chars"]), span["font"], symbol_map)
    if span["flags"] & SUPERSCRIPT_FLAG and text.strip():
        core = text.strip()
        text = f"^{core}" if len(core) == 1 else f"^({core})"
    return text


def body_font_size(doc) -> float:
    """The font size used for most characters of the document (the body text size)."""
    sizes = Counter()
    for page in doc:
        for block in page.get_text("dict", flags=TEXT_FLAGS)["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    sizes[round(span["size"], 1)] += len(span["text"].strip())
    return sizes.most_common(1)[0][0] if sizes else 0.0


def is_heading_line(spans, text: str, body_size: float, first_in_block: bool) -> bool:
    """
    A section heading: short, contains letters, and either set in a larger font than the body
    and written in capitals or numbered ("II. TITLE"), or numbered and in capitals at the start
    of a block (headings set in the body size, e.g. small caps).
    """
    text = text.strip()
    if not text or len(text) > MAX_HEADING_CHARS or not any(c.isalpha() for c in text):
        return False
    letters = [c for c in text if c.isalpha()]
    mostly_upper = sum(c.isupper() for c in letters) >= HEADING_UPPER_SHARE * len(letters)
    numbered = bool(NUMBERED_HEADING_RE.match(text))
    sizes = [s["size"] for s in spans if "".join(c["c"] for c in s["chars"]).strip()]
    larger = bool(sizes) and min(sizes) >= HEADING_SIZE_RATIO * body_size
    return (larger and (mostly_upper or numbered)) or (first_in_block and numbered and mostly_upper)


def _continues_heading(text: str) -> bool:
    """Can this heading line be the second line of the previous heading (not a new heading)?"""
    return not NUMBERED_HEADING_RE.match(text) and text.strip().upper() not in UNNUMBERED_HEADINGS


def _normalize_margin_line(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "", text)).strip().lower()


def repeated_margin_lines(doc):
    """Normalized texts of lines that appear in the top/bottom zone of many pages (running headers/footers)."""
    pages_with = Counter()
    for page in doc:
        height = page.rect.height
        seen = set()
        for block in page.get_text("dict", flags=TEXT_FLAGS)["blocks"]:
            for line in block.get("lines", []):
                y0, y1 = line["bbox"][1], line["bbox"][3]
                if y1 <= MARGIN_ZONE * height or y0 >= (1 - MARGIN_ZONE) * height:
                    norm = _normalize_margin_line("".join(span["text"] for span in line["spans"]))
                    if norm:
                        seen.add(norm)
        pages_with.update(seen)
    min_pages = max(2, math.ceil(REPEATED_LINE_SHARE * len(doc)))
    return {norm for norm, n in pages_with.items() if n >= min_pages}


def _is_margin_noise(bbox, text, page_height, repeated) -> bool:
    """A running header/footer, a page number, a copyright notice or a download watermark."""
    in_zone = bbox[3] <= MARGIN_ZONE * page_height or bbox[1] >= (1 - MARGIN_ZONE) * page_height
    if not in_zone:
        return False
    norm = _normalize_margin_line(text)
    return not norm.strip(" .,-|") or norm in repeated or bool(BOILERPLATE_RE.search(text))


def _text_blocks(page, symbol_map, body_size, repeated=frozenset()):
    """
    Text blocks as (x0, y0, x1, y1, text) tuples, like page.get_text("blocks"), but built
    character by character so that wrongly mapped special characters can be repaired, and
    with section headings written as Markdown "## TITLE" paragraphs.
    """
    blocks = []
    for block in page.get_text("rawdict", flags=TEXT_FLAGS)["blocks"]:
        if block.get("type") != 0:  # 0 = text; 1 = image
            continue
        parts = []  # ("heading" | "body", text), consecutive lines of the same kind merged
        for i, line in enumerate(block["lines"]):
            text = reattach_accents("".join(_format_span(s, symbol_map) for s in line["spans"]))
            if _is_margin_noise(line["bbox"], text, page.rect.height, repeated):
                continue
            if is_heading_line(line["spans"], text, body_size, first_in_block=(i == 0)):
                if parts and parts[-1][0] == "heading" and _continues_heading(text):
                    parts[-1] = ("heading", f"{parts[-1][1]} {text.strip()}")
                else:
                    parts.append(("heading", text.strip()))
            elif parts and parts[-1][0] == "body":
                parts[-1] = ("body", f"{parts[-1][1]}\n{text}")
            else:
                parts.append(("body", text))
        text = "\n\n".join(HEADING_MARK + t if kind == "heading" else t for kind, t in parts)
        blocks.append((*block["bbox"], text + "\n"))
    return blocks


def _append_block(full_text: str, cleaned: str) -> str:
    """Smart joining: add a cleaned block to the text, as a new paragraph or a continuation."""
    if not full_text:
        return cleaned
    last_paragraph = full_text.rsplit("\n\n", 1)[-1]
    if cleaned.startswith(HEADING_MARK):
        # A heading split over two blocks ("III. A FRAMEWORK FOR ..." + "SCALING METHODS")
        title, sep, rest = cleaned.partition("\n\n")
        title = title[len(HEADING_MARK):]
        if last_paragraph.startswith(HEADING_MARK) and _continues_heading(title):
            return f"{full_text} {title}{sep}{rest}"
        return f"{full_text}\n\n{cleaned}"
    if last_paragraph.startswith(HEADING_MARK) or full_text.endswith(('.', '!', '?', ':')):
        # After a heading, or likely a paragraph end
        return f"{full_text}\n\n{cleaned}"
    # The sentence likely continues in the next block
    return f"{full_text} {cleaned}"


def _parse_page(page, page_number, symbol_map, body_size, repeated=frozenset()):
    """Split one page into body blocks (in reading order) and its figures/tables."""
    page_height = page.rect.height
    blocks = [b for b in _text_blocks(page, symbol_map, body_size, repeated)
              if b[1] > HEADER_MARGIN and b[3] < page_height - FOOTER_MARGIN and b[4].strip()]

    captions = _find_captions(blocks)
    # Table detection is the slowest step of parsing, so it only runs on pages with a table caption
    tables = _tables(page) if any(c[0] == "table" for c in captions) else []
    graphics = _graphic_regions(page, captions, body_size)
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


def parse_pdf(pdf_path: str, symbol_map: Optional[Dict[str, str]] = None) -> ParsedPaper:
    """
    Parse a PDF into body text (column-sorted, header/footer filtered, smartly joined)
    and a list of its figures and tables.
    `symbol_map` ({glyph key: correct character}, see src/symbols.py) repairs special
    characters whose Unicode mapping in the PDF is wrong.
    """
    symbol_map = symbol_map or {}
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    # Using a single string to accumulate all content for easier punctuation checking
    full_text = ""
    figures = []

    body_size = body_font_size(doc)
    repeated = repeated_margin_lines(doc)
    for page_num in range(len(doc)):
        body_blocks, page_figures = _parse_page(doc[page_num], page_num + 1, symbol_map, body_size, repeated)
        figures.extend(page_figures)

        # Smart joining logic for blocks within the current page
        for b in body_blocks:
            cleaned = clean_text(b[4])
            if cleaned:
                full_text = _append_block(full_text, cleaned)

    doc.close()
    body, references = split_references(full_text)
    return ParsedPaper(body, figures, references)


if __name__ == "__main__":
    # Local module test: python -m src.pdf_loader path/to/paper.pdf
    import os
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m src.pdf_loader path/to/paper.pdf")
    paper = parse_pdf(sys.argv[1])
    print(f"Body text: {len(paper.text)} characters, {len(paper.figures)} figures/tables, "
          f"references: {len(paper.references)} characters\n")
    for f in paper.figures:
        print(f.as_text()[:300], "\n")
    print("--- Preview (first 1500 chars of body text) ---")
    print(paper.text[:1500])
