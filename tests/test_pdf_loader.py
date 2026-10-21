import os

import pytest

from src.pdf_loader import _is_margin_noise, clean_text, parse_pdf, split_references

SAMPLE_PDF = os.path.join(os.path.dirname(__file__), "..", "data", "papers", "sample_ieee.pdf")


def test_clean_text_joins_lines_and_fixes_hyphenation():
    assert clean_text("fine-\ntuning the\nmodel") == "finetuning the model"
    assert clean_text("DP-\nSGD works") == "DP-SGD works"
    assert clean_text("ﬁne") == "fine"  # ligature


def test_margin_noise_is_headers_page_numbers_and_boilerplate():
    height, repeated = 800, {"ieee transactions on wireless communications, vol. ,"}
    top, bottom, middle = (40, 30, 300, 38), (40, 770, 300, 778), (40, 400, 300, 410)
    assert _is_margin_noise(top, "IEEE TRANSACTIONS ON WIRELESS COMMUNICATIONS, VOL. 25, 2026", height, repeated)
    assert _is_margin_noise(bottom, "11177", height, repeated)
    assert _is_margin_noise(bottom, "Authorized licensed use limited to: Some University.", height, set())
    # The same text in the middle of the page is body text
    assert not _is_margin_noise(middle, "11177", height, repeated)
    assert not _is_margin_noise(top, "A normal first line of a column", height, repeated)


def test_split_references_uses_the_heading_in_the_second_half():
    body = "Intro text. " * 50
    text = f"{body}\n\nREFERENCES [1] A. Author, Some paper."
    split_body, refs = split_references(text)
    assert split_body == body.rstrip()
    assert refs == "[1] A. Author, Some paper."


def test_split_references_ignores_an_early_mention():
    text = "References to prior work are discussed below. " + "More text. " * 50
    assert split_references(text) == (text, "")


@pytest.fixture(scope="module")
def paper():
    if not os.path.exists(SAMPLE_PDF):
        pytest.skip("sample PDF not available")
    return parse_pdf(SAMPLE_PDF)


def test_sample_paper_figures_and_tables(paper):
    assert [f.key for f in paper.figures] == ["figure:1", "table:I", "figure:2", "figure:3"]
    table = next(f for f in paper.figures if f.key == "table:I")
    # Table rows come out in the right order, with their numbers
    assert table.content.index("DP-SGD") < table.content.index("Auto-S") < table.content.index("DP-PSAC|")
    assert "92.65" in table.content


def test_sample_paper_body_has_no_table_cells_or_captions(paper):
    assert "97.35" not in paper.text           # a table cell
    assert "FIG. 1." not in paper.text         # a caption
    assert "Fig. 2 reveals why" in paper.text  # body text that mentions a figure stays


def test_sample_paper_headings_are_detected_and_merged_across_lines(paper):
    headings = [p for p in paper.text.split("\n\n") if p.startswith("## ")]
    assert "## III. A FRAMEWORK FOR ANALYZING ADAPTIVE SCALING METHODS" in headings  # two lines, two blocks
    assert "## B. EXPERIMENTAL SETUP AND ANALYSIS" in headings
    assert "## ABSTRACT" in headings
    assert len(headings) == 14
    # A body sentence that mentions a section is not a heading
    assert not any("We" in h for h in headings)


def test_sample_paper_title_is_complete_and_references_are_split_off(paper):
    assert paper.text.startswith("## ADAPTIVE GRADIENT METHODS FOR DIFFERENTIALLY PRIVATE TINYML IN 6G")
    assert paper.references.startswith("[1]")
    assert "Deep leakage from gradients" not in paper.text
    assert "--- REFERENCES ---" in paper.full_text()
