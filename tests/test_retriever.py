import numpy as np
import pytest

from src import retriever as retriever_module
from src.bm25 import BM25, tokenize
from src.retriever import Retriever, reciprocal_rank_fusion
from src.vector_store import load_vector_db, save_vector_db

CHUNKS = [
    "Static clipping uses one fixed threshold for every gradient.",
    "DP-PSASC introduces a tunable coefficient for scaling.",
    "The O-RAN RIC sends policies through the A1 interface.",
]
# Fake embeddings: the dense model "thinks" every query is closest to chunk 0
VECTORS = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


class FakeModel:
    def encode(self, texts):
        return np.array([[1.0, 0.1, 0.0] for _ in texts], dtype="float32")


@pytest.fixture
def index(tmp_path):
    index_path, chunks_path = str(tmp_path / "i.faiss"), str(tmp_path / "c.pkl")
    save_vector_db(np.array(VECTORS, dtype="float32"), CHUNKS, index_path, chunks_path)
    return load_vector_db(index_path, chunks_path)[0]


def test_tokenize_drops_stopwords_and_keeps_hyphenated_terms_whole_and_split():
    assert tokenize("What is the core innovation of DP-PSASC*?") == ["core", "innovation", "dp-psasc*", "dp", "psasc"]


def test_tokenize_indexes_greek_letters_by_name_too():
    assert tokenize("δ = 10") == ["δ", "delta", "10"]
    assert tokenize("∈") == []  # a math symbol, not a letter


def test_bm25_ranks_rare_term_matches_first():
    bm25 = BM25(CHUNKS)
    results = bm25.search("tunable coefficient", k=3)
    assert results[0][0] == 1
    # Chunks without any query term are not returned
    assert [i for i, _ in results] == [1]


def test_bm25_prefers_shorter_chunk_for_same_term_count():
    bm25 = BM25(["interface", "interface " + "filler " * 20])
    scores = bm25.scores("interface")
    assert scores[0] > scores[1] > 0


def test_reciprocal_rank_fusion_rewards_agreement():
    # Chunk 2 is 2nd in both lists, chunks 0 and 1 are 1st in one list but missing from the other
    assert reciprocal_rank_fusion([[0, 2], [1, 2]])[0] == 2


def test_dense_mode_follows_embeddings(index):
    r = Retriever(FakeModel(), index, CHUNKS, mode="dense", rerank=False)
    results, best = r.search("anything about tunable coefficients", k=1)
    assert results == [CHUNKS[0]]
    assert best > 0.9


def test_hybrid_mode_lets_keywords_win_when_dense_is_wrong(index):
    r = Retriever(FakeModel(), index, CHUNKS, mode="hybrid", rerank=False)
    results, best = r.search("A1 interface", k=3)
    # The keyword match (chunk 2) is fused near the top even though dense ranks it last
    assert CHUNKS[2] in results[:2]
    # The reported score is always the dense one, used by the relevance threshold
    assert best > 0.9


def test_reranker_reorders_candidates(index, monkeypatch):
    class FakeReranker:
        def predict(self, pairs):
            # Prefer the chunk that mentions the RIC
            return [1.0 if "RIC" in chunk else 0.0 for _, chunk in pairs]

    monkeypatch.setattr(retriever_module, "get_reranker", lambda name=None: FakeReranker())
    r = Retriever(FakeModel(), index, CHUNKS, mode="dense", rerank=True)
    results, _ = r.search("policies", k=1)
    assert results == [CHUNKS[2]]


def test_unknown_mode_raises(index):
    with pytest.raises(ValueError):
        Retriever(FakeModel(), index, CHUNKS, mode="magic", rerank=False)
