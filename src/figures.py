# src/figures.py
"""
Figures and tables of a paper, and the links between them and the body text.

The PDF loader keeps figures/tables out of the body text and stores each one as a
Figure (caption + table cells). Body text refers to them as "Fig. 2", "Figure 17.5",
"Table I", ...; find_references() turns such mentions into keys like "figure:2" or
"table:I", which are used to attach the right figure/table to a retrieved chunk.
"""
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Set

# A figure/table number: "2", "17.5", "3a", or a Roman numeral as used for IEEE tables ("IV")
_NUMBER = r"(?:\d+(?:\.\d+)*(?:[a-z](?![a-z]))?|[IVXLC]+(?![A-Za-z]))"

# A caption block starts with the label and a number followed by "." / ":" / a line break,
# e.g. "FIG. 1. A conceptual ...", "Figure 17.5: ...", "TABLE I\nSIMULATION PARAMETERS".
# Body sentences such as "Fig. 2 reveals why ..." do not match.
CAPTION_RE = re.compile(r"^\s*(?i:(fig(?:ure)?s?\.?|table))\s*(\d+(?:\.\d+)*|[IVXLC]+)\s*(?:[.:](?!\d)|\n|$)")

# A mention in body text: "Fig. 2", "Figs. 1 and 2", "Figure 3(a)", "Table I", "Tables II-IV"
REFERENCE_RE = re.compile(
    rf"(?i:\b(figs?\.|figures?|tables?))\s*({_NUMBER}(?:\([a-z]\))?(?:\s*(?:,|and|&|[-\u2013])\s*{_NUMBER}(?:\([a-z]\))?)*)"
)
_SINGLE_NUMBER_RE = re.compile(r"\d+(?:\.\d+)*|[IVXLC]+")


def _kind_of(label_word: str) -> str:
    return "table" if label_word.lower().startswith("table") else "figure"


def make_key(kind: str, number: Optional[str]) -> Optional[str]:
    return f"{kind}:{number.upper()}" if number else None


@dataclass
class Figure:
    kind: str              # "figure" or "table"
    number: Optional[str]  # "2", "17.5", "I"; None for a table found without a caption
    page: int              # 1-based page number
    caption: str           # caption text without the "FIG. 2." prefix
    content: str = ""      # table cells as Markdown; empty for figures

    @property
    def key(self) -> Optional[str]:
        return make_key(self.kind, self.number)

    @property
    def label(self) -> str:
        if self.number is None:
            return "Table (unlabeled)" if self.kind == "table" else "Figure (unlabeled)"
        return f"Table {self.number}" if self.kind == "table" else f"Fig. {self.number}"

    def as_text(self) -> str:
        """The figure/table as one self-contained chunk of text."""
        text = f"[{self.label}, page {self.page}] {self.caption}".rstrip()
        if self.content:
            text += "\n" + self.content
        return text


def parse_caption(text: str):
    """Return (kind, number, caption_body) if the text block is a figure/table caption, else None."""
    m = CAPTION_RE.match(text)
    if not m:
        return None
    return _kind_of(m.group(1)), m.group(2).upper(), text[m.end():].strip()


def iter_references(text: str) -> Iterator[str]:
    """Figure/table keys mentioned in a piece of body text, in order of appearance (may repeat)."""
    for m in REFERENCE_RE.finditer(text):
        kind = _kind_of(m.group(1))
        for number in _SINGLE_NUMBER_RE.findall(m.group(2)):
            yield make_key(kind, number)


def find_references(text: str) -> Set[str]:
    """All figure/table keys mentioned in a piece of body text."""
    return set(iter_references(text))


def attach_referenced_figures(results: List[str], chunks: List[str], figure_index: Dict[str, int],
                              limit: int = 3) -> List[str]:
    """
    Append the figures/tables that the retrieved chunks mention (at most `limit`),
    so the LLM sees e.g. Table I whenever a retrieved paragraph says "as shown in Table I".
    """
    extras = []
    for chunk in results:
        for key in iter_references(chunk):
            idx = figure_index.get(key)
            if idx is None:
                continue
            figure_chunk = chunks[idx]
            if figure_chunk not in results and figure_chunk not in extras:
                extras.append(figure_chunk)
    return results + extras[:limit]
