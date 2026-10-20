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


def test_paragraph_breaks_are_kept_inside_a_chunk():
    text = "First paragraph here.\n\nSecond paragraph here."
    assert smart_paragraph_chunker(text, count_tokens=count_words) == [
        "First paragraph here.\n\nSecond paragraph here."
    ]
