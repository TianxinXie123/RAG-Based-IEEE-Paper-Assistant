from src.chunker import smart_paragraph_chunker


def count_words(text):
    """Simple token counter for tests: one token per word."""
    return len(text.split())


def sentence(i, words=10):
    return " ".join(f"w{i}" for _ in range(words - 1)) + f" end{i}."


def test_short_text_is_one_chunk():
    assert smart_paragraph_chunker("One sentence. Two sentences.", count_tokens=count_words) == [
        "One sentence. Two sentences."
    ]


def test_every_chunk_respects_the_token_limit():
    text = " ".join(sentence(i) for i in range(50))
    chunks = smart_paragraph_chunker(text, max_tokens=35, overlap_tokens=10, count_tokens=count_words)
    assert len(chunks) > 1
    assert all(count_words(c) <= 35 for c in chunks)


def test_overlap_repeats_the_last_whole_sentence():
    text = " ".join(sentence(i) for i in range(6))
    chunks = smart_paragraph_chunker(text, max_tokens=30, overlap_tokens=10, count_tokens=count_words)
    last_sentence_of_first = chunks[0].split(". ")[-1]
    assert chunks[1].startswith(last_sentence_of_first.rstrip("."))


def test_sentence_longer_than_limit_is_split():
    long_sentence = " ".join(f"word{i}" for i in range(100)) + "."
    chunks = smart_paragraph_chunker(long_sentence, max_tokens=30, overlap_tokens=0, count_tokens=count_words)
    assert all(count_words(c) <= 30 for c in chunks)
    assert " ".join(chunks).split() == long_sentence.split()


def test_no_empty_chunks_and_no_duplicate_chunks():
    # A heading without punctuation used to be duplicated into the next chunk
    text = "A TITLE WITHOUT PUNCTUATION\n\n" + " ".join(sentence(i) for i in range(10))
    chunks = smart_paragraph_chunker(text, max_tokens=30, overlap_tokens=10, count_tokens=count_words)
    assert all(c.strip() for c in chunks)
    assert len(chunks) == len(set(chunks))
    assert sum(c.count("A TITLE WITHOUT PUNCTUATION") for c in chunks) == 1


def test_headings_are_hard_chunk_boundaries():
    text = "## I. INTRO\n\nIntro one. Intro two.\n\n## II. METHOD\n\nMethod one. Method two."
    assert smart_paragraph_chunker(text, count_tokens=count_words) == [
        "I. INTRO\n\nIntro one. Intro two.",
        "II. METHOD\n\nMethod one. Method two.",
    ]


def test_overlap_never_crosses_a_heading():
    text = "## I. INTRO\n\n" + " ".join(sentence(i) for i in range(3)) + "\n\n## II. METHOD\n\n" + sentence(9)
    chunks = smart_paragraph_chunker(text, max_tokens=100, overlap_tokens=20, count_tokens=count_words)
    assert chunks[-1] == "II. METHOD\n\n" + sentence(9)


def test_section_headers_show_the_section_path_and_fit_the_budget():
    text = ("## IV. CASE STUDY\n\nIntro.\n\n## B. EXPERIMENTAL SETUP\n\n"
            + " ".join(sentence(i) for i in range(10)))
    chunks = smart_paragraph_chunker(text, max_tokens=40, overlap_tokens=0, count_tokens=count_words,
                                     section_headers=True)
    assert chunks[0] == "[IV. CASE STUDY]\nIntro."
    assert all(c.startswith("[IV. CASE STUDY > B. EXPERIMENTAL SETUP]\n") for c in chunks[1:])
    assert all(count_words(c) <= 40 for c in chunks)


def test_a_new_section_replaces_the_previous_subsection_in_the_path():
    text = "## II. A\n\nx.\n\n## B. SUB\n\ny.\n\n## III. C\n\nz."
    assert smart_paragraph_chunker(text, count_tokens=count_words, section_headers=True) == [
        "[II. A]\nx.", "[II. A > B. SUB]\ny.", "[III. C]\nz."
    ]


def test_paragraph_breaks_are_kept_inside_a_chunk():
    text = "First paragraph here.\n\nSecond paragraph here."
    assert smart_paragraph_chunker(text, count_tokens=count_words) == [
        "First paragraph here.\n\nSecond paragraph here."
    ]
