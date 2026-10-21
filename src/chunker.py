# src/chunker.py
"""
Token-aware, section-aware chunking.

The embedding model only reads the first `max_seq_length` tokens of a text (256 for
all-MiniLM-L6-v2) and silently drops the rest, so every chunk must fit in that budget.
Text is split into sentences, sentences are packed into chunks of at most `max_tokens`,
paragraph boundaries are preferred as cut points, and each chunk starts with the last
sentence(s) of the previous one (the overlap) to keep context across the cut.

Section headings (Markdown lines "## II. TITLE", as written by the PDF loader) are hard
boundaries: a chunk never mixes two sections, and the overlap never crosses a heading.
Optionally every chunk starts with its section path, e.g.
"[IV. CASE STUDY > B. EXPERIMENTAL SETUP AND ANALYSIS]" (a "contextual chunk header").
It is off by default: on the evaluation set it did not improve Recall@3 for any retrieval
mode and made BM25 slightly worse (the repeated heading words dilute the keyword scores).
"""
import re
from typing import Callable, List, Optional, Tuple

# A sentence ends with . ! or ? followed by whitespace
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Section headings are Markdown level-2 headings
HEADING_MARK = "## "
# "II. TITLE" is a section, "B. TITLE" a subsection; unnumbered headings (ABSTRACT) are sections
_SECTION_RE = re.compile(r"^[IVXLC]+\.\s")
_SUBSECTION_RE = re.compile(r"^[A-Z]\.\s")

# Start a new chunk at a paragraph boundary once the current chunk is this full
PARAGRAPH_BREAK_FILL = 0.6
# The section header may use at most this share of a chunk's token budget
MAX_HEADER_SHARE = 0.4


def approx_token_count(text: str) -> int:
    """Rough token estimate (about 4 characters per token) when no tokenizer is given."""
    return max(1, len(text) // 4)


def _split_long_sentence(sentence: str, max_tokens: int, count: Callable[[str], int]) -> List[str]:
    """Split a single sentence that is longer than max_tokens at word boundaries."""
    pieces, current = [], ""
    for word in sentence.split(" "):
        candidate = f"{current} {word}".strip()
        if current and count(candidate) > max_tokens:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _join(units: List[Tuple[str, int, bool]]) -> str:
    """Join sentences; a sentence that starts a paragraph is preceded by a blank line."""
    text = ""
    for sentence, _, starts_paragraph in units:
        if not text:
            text = sentence
        else:
            text += ("\n\n" if starts_paragraph else " ") + sentence
    return text


def _heading_level(title: str) -> int:
    return 2 if _SUBSECTION_RE.match(title) and not _SECTION_RE.match(title) else 1


def _split_sections(text: str) -> List[Tuple[List[str], List[str]]]:
    """Split text into (section path, paragraphs) pairs at Markdown headings."""
    sections = []
    path: List[Tuple[int, str]] = []
    paragraphs: List[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if paragraph.startswith(HEADING_MARK):
            if paragraphs:
                sections.append(([t for _, t in path], paragraphs))
                paragraphs = []
            title = paragraph[len(HEADING_MARK):].strip()
            level = _heading_level(title)
            path = [(lvl, t) for lvl, t in path if lvl < level] + [(level, title)]
        else:
            paragraphs.append(paragraph)
    if paragraphs:
        sections.append(([t for _, t in path], paragraphs))
    return sections


def _pack(paragraphs: List[str], max_tokens: int, overlap_tokens: int, count: Callable[[str], int]) -> List[str]:
    """Pack the sentences of some paragraphs into chunks of at most max_tokens, with overlap."""
    # 1. Break the paragraphs into sentence units: (sentence, token_count, starts_paragraph)
    units = []
    for paragraph in paragraphs:
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        first = True
        for sentence in SENTENCE_RE.split(paragraph):
            if not sentence:
                continue
            for piece in _split_long_sentence(sentence, max_tokens, count):
                units.append((piece, count(piece), first))
                first = False

    # 2. Pack the units into chunks
    chunks = []
    current, current_tokens = [], 0
    for unit in units:
        _, tokens, starts_paragraph = unit
        too_full = current and current_tokens + tokens > max_tokens
        good_break = current and starts_paragraph and current_tokens >= PARAGRAPH_BREAK_FILL * max_tokens
        if too_full or good_break:
            chunks.append(_join(current))
            # Overlap: carry over whole trailing sentences, as long as the next chunk still fits
            overlap, overlap_size = [], 0
            for prev in reversed(current):
                if overlap_size + prev[1] > overlap_tokens:
                    break
                overlap.insert(0, prev)
                overlap_size += prev[1]
            if overlap_size + tokens > max_tokens:
                overlap, overlap_size = [], 0
            current, current_tokens = overlap, overlap_size
        current.append(unit)
        current_tokens += tokens

    if current:
        chunks.append(_join(current))
    return chunks


def smart_paragraph_chunker(text: str, max_tokens: int = 240, overlap_tokens: int = 40,
                            count_tokens: Optional[Callable[[str], int]] = None,
                            section_headers: bool = False) -> List[str]:
    """
    Split text into chunks of at most `max_tokens` tokens (as counted by `count_tokens`).

    Args:
        text: the document text; paragraphs are separated by blank lines, and "## " lines
            are section headings.
        max_tokens: hard upper bound for each chunk (section header included).
        overlap_tokens: up to this many tokens of whole trailing sentences are repeated
            at the start of the next chunk of the same section.
        count_tokens: function returning the token count of a string; pass the embedding
            model's tokenizer for exact counts. Defaults to approx_token_count.
        section_headers: start every chunk with its section path in brackets; when False,
            a section's heading becomes the first line of its first chunk instead.
    """
    count = count_tokens or approx_token_count
    chunks = []
    for path, paragraphs in _split_sections(text):
        if not path:
            chunks.extend(_pack(paragraphs, max_tokens, overlap_tokens, count))
            continue
        if not section_headers:
            chunks.extend(_pack([path[-1]] + paragraphs, max_tokens, overlap_tokens, count))
            continue
        header = "[" + " > ".join(path) + "]"
        if count(header) > MAX_HEADER_SHARE * max_tokens:
            header = "[" + path[-1] + "]"
            if count(header) > MAX_HEADER_SHARE * max_tokens:
                header = _split_long_sentence(header, int(MAX_HEADER_SHARE * max_tokens), count)[0] + "]"
        budget = max_tokens - count(header) - 1
        chunks.extend(f"{header}\n{body}" for body in _pack(paragraphs, budget, overlap_tokens, count))
    return chunks
