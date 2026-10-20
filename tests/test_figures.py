from src.figures import Figure, attach_referenced_figures, find_references, parse_caption


def test_caption_with_period():
    assert parse_caption("FIG. 1. A conceptual framework") == ("figure", "1", "A conceptual framework")


def test_caption_with_decimal_number_and_colon():
    assert parse_caption("Figure 17.5: Pipeline overview") == ("figure", "17.5", "Pipeline overview")


def test_ieee_table_caption_on_its_own_line():
    assert parse_caption("TABLE I\nSIMULATION PARAMETERS") == ("table", "I", "SIMULATION PARAMETERS")


def test_body_sentences_are_not_captions():
    assert parse_caption("Fig. 2 reveals why the method works.") is None
    assert parse_caption("Fig. 2.5 shows the result.") is None
    assert parse_caption("Table In this case nothing.") is None


def test_find_single_and_multiple_references():
    assert find_references("as shown in Fig. 2.") == {"figure:2"}
    assert find_references("Figs. 1 and 2") == {"figure:1", "figure:2"}
    assert find_references("see Fig. 3(a) and Table II") == {"figure:3", "table:II"}
    assert find_references("Figure 17.5 shows") == {"figure:17.5"}


def test_words_starting_with_roman_letters_are_not_references():
    assert find_references("Table In the next section") == set()


def test_figure_labels_and_text():
    table = Figure("table", "I", 4, "Comparison of methods.", "|a|b|")
    assert table.key == "table:I"
    assert table.as_text() == "[Table I, page 4] Comparison of methods.\n|a|b|"
    assert Figure("table", None, 2, "").key is None


def test_attach_referenced_figures_adds_missing_ones_only():
    chunks = ["Results are in Table I and Fig. 2.", "[Table I, page 4] ...", "[Fig. 2, page 5] ..."]
    figure_index = {"table:I": 1, "figure:2": 2}

    # Both referenced figures are added after the retrieved chunk
    assert attach_referenced_figures([chunks[0]], chunks, figure_index) == chunks

    # A figure that was already retrieved is not added twice
    assert attach_referenced_figures([chunks[0], chunks[1]], chunks, figure_index) == chunks

    # The limit caps how many figures are added
    assert attach_referenced_figures([chunks[0]], chunks, figure_index, limit=1) == chunks[:2]
