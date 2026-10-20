import pickle

import numpy as np

from src.vector_store import cache_paths_for, dense_search, load_vector_db, save_vector_db


class FakeModel:
    """Stands in for the embedding model: every text maps to a fixed vector."""

    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts):
        return np.array([self.vectors[t] for t in texts], dtype="float32")


VECTORS = {
    "about cats": [1.0, 0.0, 0.0],
    "about dogs": [0.0, 1.0, 0.0],
    "about fish": [0.0, 0.0, 1.0],
    "cat question": [0.9, 0.1, 0.0],
}
CHUNKS = ["about cats", "about dogs", "about fish"]


def build(tmp_path):
    index_path, chunks_path = str(tmp_path / "index.faiss"), str(tmp_path / "chunks.pkl")
    model = FakeModel(VECTORS)
    save_vector_db(model.encode(CHUNKS), CHUNKS, index_path, chunks_path, {"table:I": 2})
    return model, index_path, chunks_path


def test_save_and_load_round_trip(tmp_path):
    _, index_path, chunks_path = build(tmp_path)
    index, chunks, figure_index = load_vector_db(index_path, chunks_path)
    assert index.ntotal == 3
    assert chunks == CHUNKS
    assert figure_index == {"table:I": 2}


def test_search_returns_most_similar_first(tmp_path):
    model, index_path, chunks_path = build(tmp_path)
    index, chunks, _ = load_vector_db(index_path, chunks_path)
    ids, scores = dense_search("cat question", model, index, k=2)
    assert chunks[ids[0]] == "about cats"
    assert scores[0] > scores[1]


def test_k_larger_than_index_is_capped(tmp_path):
    model, index_path, chunks_path = build(tmp_path)
    index, chunks, _ = load_vector_db(index_path, chunks_path)
    ids, _ = dense_search("cat question", model, index, k=10)
    assert sorted(ids) == [0, 1, 2]


def test_missing_or_old_format_cache_is_ignored(tmp_path):
    assert load_vector_db(str(tmp_path / "nope.faiss"), str(tmp_path / "nope.pkl")) == (None, None, None)
    _, index_path, chunks_path = build(tmp_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(CHUNKS, f)  # the old format was a bare list
    assert load_vector_db(index_path, chunks_path) == (None, None, None)


def test_cache_key_changes_with_pdf_model_and_version(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 one")
    base = cache_paths_for(pdf, "model-a", tmp_path, 1)
    assert cache_paths_for(pdf, "model-a", tmp_path, 1) == base
    assert cache_paths_for(pdf, "model-b", tmp_path, 1) != base
    assert cache_paths_for(pdf, "model-a", tmp_path, 2) != base
    pdf.write_bytes(b"%PDF-1.4 two")
    assert cache_paths_for(pdf, "model-a", tmp_path, 1) != base
