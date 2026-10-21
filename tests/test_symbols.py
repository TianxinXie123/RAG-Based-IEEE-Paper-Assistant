import os

import pytest

from src.pdf_loader import parse_pdf
from src.symbols import (apply_symbol_map, clean_recognized_symbol, is_suspicious, load_symbol_cache,
                         reattach_accents, resolve_symbol_map)

SAMPLE_PDF = os.path.join(os.path.dirname(__file__), "..", "data", "papers", "sample_ieee.pdf")
# What the printed glyphs of the sample paper really are
SAMPLE_ANSWERS = {"\x01": "•", "2": "∈", "¼": "=", "d": "δ", "€": "¨"}


def test_suspicious_means_unreliable_font_or_never_real_text():
    assert is_suspicious("AdvP4C4E74", "2")             # publisher symbol font
    assert is_suspicious("ABCDEF+SymbolMT", "m")        # Symbol encoding: "m" is drawn as mu
    assert not is_suspicious("ABCDEF+CMMI10", "x")      # TeX math italic: mapped correctly
    assert not is_suspicious("AdvOTed04a3c9", "2")      # normal text font
    assert is_suspicious("AnyFont", "")             # control character
    assert is_suspicious("AnyFont", "")           # private use area
    assert not is_suspicious("CMEX10", "")          # extension font: large delimiters, just dropped
    assert not is_suspicious("AdvP4C4E74", " ")         # whitespace is never repaired


def test_apply_symbol_map_replaces_known_and_drops_unknown_control_characters():
    symbol_map = {"AdvP4C4E74|2": "∈", "AdvP4C4E74|¼": "="}
    assert apply_symbol_map("2 ¼", "AdvP4C4E74", symbol_map) == "∈ ="
    assert apply_symbol_map("a\x01b", "AdvOTed04a3c9", {}) == "ab"
    # The same character in a normal text font is left alone
    assert apply_symbol_map("2", "AdvOTed04a3c9", symbol_map) == "2"


def test_accent_drawn_before_its_letter_is_composed():
    symbol_map = {"AdvP4C4E59|€": "¨"}
    line = (apply_symbol_map("H", "AdvOTed04a3c9", symbol_map)
            + apply_symbol_map("€", "AdvP4C4E59", symbol_map)
            + apply_symbol_map("ubler", "AdvOTed04a3c9", symbol_map))
    assert reattach_accents(line) == "Hübler"


def test_clean_recognized_symbol():
    assert clean_recognized_symbol("∈") == "∈"
    assert clean_recognized_symbol("`δ`\n") == "δ"
    assert clean_recognized_symbol("The character is epsilon") is None
    assert clean_recognized_symbol("") is None


@pytest.fixture
def sample_pdf():
    if not os.path.exists(SAMPLE_PDF):
        pytest.skip("sample PDF not available")
    return SAMPLE_PDF


def test_resolve_symbol_map_recognizes_each_glyph_once_and_caches(sample_pdf, tmp_path):
    cache_path = str(tmp_path / "symbols.json")
    calls = []

    def fake_recognizer(png, sample):
        assert png.startswith(b"\x89PNG")
        calls.append(sample.key)
        return SAMPLE_ANSWERS[sample.char]

    symbol_map = resolve_symbol_map(sample_pdf, cache_path, fake_recognizer)
    assert sorted(symbol_map.values()) == sorted(SAMPLE_ANSWERS.values())
    assert len(calls) == len(set(calls)) == 5  # each distinct glyph exactly once

    # Second run: everything comes from the cache, no recognition at all
    def must_not_be_called(png, sample):
        raise AssertionError("glyph should come from the cache")
    assert resolve_symbol_map(sample_pdf, cache_path, must_not_be_called) == symbol_map


def test_failed_recognition_is_not_cached(sample_pdf, tmp_path):
    cache_path = str(tmp_path / "symbols.json")

    def offline(png, sample):
        raise ConnectionError("no network")

    assert resolve_symbol_map(sample_pdf, cache_path, offline) == {}
    assert load_symbol_cache(cache_path) == {}


def test_sample_paper_text_is_repaired(sample_pdf, tmp_path):
    symbol_map = resolve_symbol_map(sample_pdf, str(tmp_path / "s.json"), lambda png, s: SAMPLE_ANSWERS[s.char])
    paper = parse_pdf(sample_pdf, symbol_map)
    assert "(∈ = 8, δ = 10^(-5))" in paper.text
    assert "• Inner Momentum" in paper.text
    assert "Hübler" in paper.references


def test_recognitions_are_capped_and_the_scan_is_cached(sample_pdf, tmp_path, monkeypatch):
    from src import symbols
    monkeypatch.setattr(symbols, "MAX_RECOGNITIONS_PER_PAPER", 2)
    cache_path = str(tmp_path / "symbols.json")
    calls = []

    def recognizer(png, sample):
        calls.append(sample.key)
        return SAMPLE_ANSWERS[sample.char]

    assert len(resolve_symbol_map(sample_pdf, cache_path, recognizer)) == 2
    assert len(calls) == 2
    # The most frequent glyph (the bullet, 9 times) is recognized first
    assert "AdvP4C4E74|" in calls

    # The next run reads the cached scan instead of scanning the PDF again
    monkeypatch.setattr(symbols, "find_suspicious_glyphs", lambda path: (_ for _ in ()).throw(AssertionError))
    assert len(resolve_symbol_map(sample_pdf, cache_path, recognizer)) == 2
