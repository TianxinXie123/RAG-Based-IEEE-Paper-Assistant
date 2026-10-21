# src/symbols.py
"""
Repair math/special characters that a PDF maps to the wrong Unicode text.

A PDF stores two things for each character: the glyph that is drawn, and (through the font's
ToUnicode table) the Unicode text it stands for. Publishers' math fonts often get the table
wrong: in the sample paper "ε = 8, δ = 10⁻⁵" is extracted as "2 ¼ 8, d ¼ 10-5". The wrong
characters are ordinary ones ("2", "d"), so they cannot be found by looking for "unknown"
characters; they are found by their FONT instead:

  1. Detect: characters drawn with a math/symbol font, plus characters that are never
     real text (control characters, private-use code points, U+FFFD).
  2. Deduplicate by (font, extracted character): the same glyph in the same font is always
     extracted the same way, so each one is recognized once per paper.
  3. Recognize: render the line that contains the glyph, mark the glyph with a red box, and
     ask a vision model which character is printed there.
  4. Map: the result (font, extracted char) -> correct char is cached as JSON and applied
     everywhere in the paper by the PDF loader.
"""
import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24.3 (new import name)
except ImportError:
    import fitz  # older PyMuPDF

# Text extraction flags: dict/rawdict defaults without embedded image data (only the text is needed)
TEXT_FLAGS = fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES

# Fonts whose characters are often mapped to the wrong Unicode text: Symbol-encoded fonts (Symbol,
# SymbolMT, SymbolTigerExpert: "m" is drawn as "μ", "D" as "Δ") and the publisher symbol fonts of IEEE
# magazines ("AdvP..." are symbol fonts; "AdvOT..." are normal text fonts). TeX and MathType math fonts
# (CMMI, CMSY, MTSY, ...) are NOT listed: their glyphs have standard names that PyMuPDF maps correctly,
# and flagging them sent hundreds of correct characters of a math-heavy paper to the vision model.
SYMBOL_FONT_RE = re.compile(r"^(?:AdvP(?!OT)|Symbol)", re.IGNORECASE)

# Math extension fonts (TeX CMEX, MathType MT Extra / ...EX..., BLEX) hold only large delimiters, braces and
# integral signs; their control-character codes are dropped instead of being sent to the vision model
EXTENSION_FONT_RE = re.compile(r"^(?:CMEX|EUEX|MT\w*EX|BLEX)", re.IGNORECASE)

# At most this many distinct glyphs are sent to the vision model per paper, so that an unexpected font
# can never trigger a long series of requests; the rest are kept as extracted
MAX_RECOGNITIONS_PER_PAPER = 30
# PDF subset fonts are named like "ABCDEF+CMMI10"
_SUBSET_PREFIX_RE = re.compile(r"^[A-Z]{6}\+")

# Spacing accents and the combining marks they stand for. In older PDFs an accent is drawn as
# its own glyph BEFORE the letter it sits on ("H¨ubler"); moving the combining mark after the
# letter lets Unicode normalization compose it ("Hübler").
SPACING_TO_COMBINING = {
    "¨": "̈",  # diaeresis  ¨
    "´": "́",  # acute      ´
    "`": "̀",       # grave
    "ˆ": "̂",  # circumflex ˆ
    "˜": "̃",  # tilde      ˜
    "¸": "̧",  # cedilla    ¸
    "˚": "̊",  # ring       ˚
    "ˇ": "̌",  # caron      ˇ
}

# The longest reply accepted from the recognizer (one symbol, or a short sequence like "≤")
MAX_SYMBOL_LENGTH = 3


def font_base_name(font: str) -> str:
    return _SUBSET_PREFIX_RE.sub("", font)


def is_symbol_font(font: str) -> bool:
    return bool(SYMBOL_FONT_RE.match(font_base_name(font)))


def is_suspicious(font: str, char: str) -> bool:
    """Is this extracted character likely to be wrong?"""
    if not char or char.isspace() or EXTENSION_FONT_RE.match(font_base_name(font)):
        return False
    category = unicodedata.category(char[0])
    if category in ("Cc", "Co") or char == "�":  # control, private use, replacement character
        return True
    return is_symbol_font(font)


def glyph_key(font: str, char: str) -> str:
    return f"{font_base_name(font)}|{char}"


@dataclass
class GlyphSample:
    """The first place in the paper where a suspicious glyph appears."""
    key: str
    font: str
    char: str        # what the PDF's text layer says
    page: int        # 0-based page index
    char_bbox: tuple
    line_bbox: tuple
    line_text: str   # the extracted text of the whole line, for context
    count: int = 1   # how often the glyph appears in the paper


def find_suspicious_glyphs(pdf_path: str) -> Dict[str, GlyphSample]:
    """All distinct suspicious glyphs in the PDF, keyed by glyph_key()."""
    samples: Dict[str, GlyphSample] = {}
    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            for block in page.get_text("rawdict", flags=TEXT_FLAGS)["blocks"]:
                for line in block.get("lines", []):
                    line_text = "".join(c["c"] for s in line["spans"] for c in s["chars"])
                    for span in line["spans"]:
                        for ch in span["chars"]:
                            if not is_suspicious(span["font"], ch["c"]):
                                continue
                            key = glyph_key(span["font"], ch["c"])
                            if key in samples:
                                samples[key].count += 1
                            else:
                                samples[key] = GlyphSample(key, span["font"], ch["c"], page_index,
                                                           tuple(ch["bbox"]), tuple(line["bbox"]), line_text)
    return samples


def render_glyph_in_context(pdf_path: str, sample: GlyphSample, dpi: int = 300) -> bytes:
    """PNG of the line containing the glyph, with the glyph marked by a red box."""
    with fitz.open(pdf_path) as doc:  # a private copy: the red box is never saved to the file
        page = doc[sample.page]
        char_rect = fitz.Rect(sample.char_bbox)
        page.draw_rect(char_rect + (-0.8, -0.8, 0.8, 0.8), color=(1, 0, 0), width=0.6)
        # Some margin around the line, so that accents and superscripts are not cut off
        clip = fitz.Rect(sample.line_bbox) + (-6, -6, 6, 6)
        return page.get_pixmap(clip=clip, dpi=dpi).tobytes("png")


def clean_recognized_symbol(reply: str) -> Optional[str]:
    """Validate a recognizer reply: a short symbol on the first line, else None."""
    first_line = next((line.strip() for line in reply.splitlines() if line.strip()), "")
    first_line = first_line.strip("`'\"")
    if not first_line or len(first_line) > MAX_SYMBOL_LENGTH:
        return None
    return first_line


def apply_symbol_map(text: str, font: str, symbol_map: Dict[str, str]) -> str:
    """Replace each character of a span that the map knows (only suspicious ones are in the map)."""
    out = []
    for char in text:
        replacement = symbol_map.get(glyph_key(font, char)) if is_suspicious(font, char) else None
        if replacement is not None:
            out.append(SPACING_TO_COMBINING.get(replacement, replacement))
        elif unicodedata.category(char) == "Cc" and not char.isspace():
            continue  # an unrecognized control character is never real text
        else:
            out.append(char)
    return "".join(out)


def reattach_accents(text: str) -> str:
    """Move each combining mark that precedes a letter (PDF drawing order) behind it, then compose."""
    chars = list(text)
    i = 0
    while i < len(chars) - 1:
        if unicodedata.combining(chars[i]) and chars[i + 1].isalpha():
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            i += 2
        else:
            i += 1
    return unicodedata.normalize("NFC", "".join(chars))


def symbol_cache_path(pdf_path: str, cache_dir: str, pdf_hash: str) -> str:
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    return os.path.join(cache_dir, "symbols", f"{stem}_{pdf_hash}.json")


# Bump when the cache format or the detection rules change, so cached scans are redone
SYMBOL_CACHE_VERSION = 2


def _read_cache(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_symbol_cache(path: str) -> Dict[str, dict]:
    """The recognized glyphs stored in a symbol cache file ({} if there is none)."""
    return _read_cache(path).get("glyphs", {})


def save_symbol_cache(path: str, glyphs: Dict[str, dict], pending: Dict[str, GlyphSample], skipped: Dict[str, str]):
    """
    Store recognized glyphs, glyphs still waiting for recognition (with everything needed to render
    them, so the PDF does not have to be scanned again) and glyphs that will not be retried.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"version": SYMBOL_CACHE_VERSION, "glyphs": glyphs,
            "pending": {k: asdict(s) for k, s in pending.items()}, "skipped": skipped}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resolve_symbol_map(pdf_path: str, cache_path: str,
                       recognize: Optional[Callable[[bytes, GlyphSample], str]]) -> Dict[str, str]:
    """
    Return {glyph_key: correct character} for the PDF.

    The PDF is scanned for suspicious glyphs once; the result is cached, so later runs start fast.
    Glyphs not recognized yet are sent to `recognize(png_bytes, sample) -> reply text` (skipped if None),
    most frequent first and at most MAX_RECOGNITIONS_PER_PAPER per paper. A recognition that fails
    because of an error (no network, no key) is retried next time; an unclear answer is not.
    """
    data = _read_cache(cache_path)
    glyphs = data.get("glyphs", {})
    skipped = data.get("skipped", {})
    changed = False
    if data.get("version") == SYMBOL_CACHE_VERSION:
        pending = {k: GlyphSample(**v) for k, v in data.get("pending", {}).items()}
    else:
        samples = find_suspicious_glyphs(pdf_path)
        pending = {k: s for k, s in samples.items() if k not in glyphs and k not in skipped}
        changed = True

    if pending and recognize is not None:
        ordered = sorted(pending.values(), key=lambda s: s.count, reverse=True)
        for sample in ordered[MAX_RECOGNITIONS_PER_PAPER:]:
            skipped[sample.key] = "over the per-paper recognition limit"
            del pending[sample.key]
        if len(ordered) > MAX_RECOGNITIONS_PER_PAPER:
            print(f"[WARN] {len(ordered)} special characters found; only the {MAX_RECOGNITIONS_PER_PAPER} "
                  f"most frequent are recognized, the rest are kept as extracted.")
        todo = ordered[:MAX_RECOGNITIONS_PER_PAPER]
        print(f"[INFO] Recognizing {len(todo)} special character(s) with the vision model ...")
        for n, sample in enumerate(todo, 1):
            try:
                reply = recognize(render_glyph_in_context(pdf_path, sample), sample)
            except Exception as e:
                print(f"[WARN] Could not recognize special characters ({type(e).__name__}: {e}); "
                      f"they are kept as extracted for now.")
                break
            changed = True
            del pending[sample.key]
            symbol = clean_recognized_symbol(reply)
            if symbol is None:
                skipped[sample.key] = f"unclear answer: {reply[:40]!r}"
                print(f"[WARN]   ({n}/{len(todo)}) unclear answer for {sample.char!r} on page {sample.page + 1}")
                continue
            glyphs[sample.key] = {"char": symbol, "extracted": sample.char, "page": sample.page + 1,
                                  "count": sample.count, "line": sample.line_text}
            print(f"[INFO]   ({n}/{len(todo)}) page {sample.page + 1}: {sample.char!r} -> {symbol!r}")
    elif pending:
        print(f"[INFO] {len(pending)} special character(s) could not be checked (no vision model available).")

    if changed:
        save_symbol_cache(cache_path, glyphs, pending, skipped)
    return {key: entry["char"] for key, entry in glyphs.items()}
