# src/retriever.py
"""
Retrieval strategies, behind one interface:

  - "dense":  embedding cosine similarity (FAISS)
  - "bm25":   keyword search
  - "hybrid": dense + BM25 merged with Reciprocal Rank Fusion (RRF)

Optionally a cross-encoder reranker re-scores the top candidates. A bi-encoder (the embedding
model) encodes query and chunk separately, which is fast but coarse; a cross-encoder reads the
query and a chunk together, which is much more accurate but too slow to run on every chunk.
So the usual two-stage design is: cheap retrieval of ~20 candidates, then rerank them.
"""
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

from src.bm25 import BM25
from src.vector_store import dense_search

RETRIEVAL_MODES = ("dense", "bm25", "hybrid")
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Number of candidates each first-stage retriever contributes before fusion/reranking
DEFAULT_CANDIDATES = 20
# Standard RRF constant: dampens the advantage of the very top ranks
RRF_K = 60


@lru_cache(maxsize=None)
def get_reranker(model_name: str = DEFAULT_RERANKER_MODEL):
    """Load the cross-encoder once (the first call downloads it, ~90MB)."""
    from sentence_transformers import CrossEncoder
    print(f"[INFO] Loading reranker: {model_name} ...")
    return CrossEncoder(model_name)


def reciprocal_rank_fusion(rankings: List[List[int]], k: int = RRF_K) -> List[int]:
    """
    Merge several ranked lists of chunk ids: score(id) = sum over lists of 1 / (k + rank).
    Uses only ranks, so scores on different scales (cosine vs BM25) never need normalizing.
    """
    fused: Dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused, key=lambda i: fused[i], reverse=True)


class Retriever:
    def __init__(self, embed_model, index, chunks: List[str], mode: str = "hybrid", rerank: bool = True,
                 candidates: int = DEFAULT_CANDIDATES, reranker_model: str = DEFAULT_RERANKER_MODEL):
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"Unknown retrieval mode '{mode}'. Choose from: {', '.join(RETRIEVAL_MODES)}")
        self.embed_model = embed_model
        self.index = index
        self.chunks = chunks
        self.mode = mode
        self.candidates = candidates
        self.bm25 = BM25(chunks) if mode in ("bm25", "hybrid") else None
        self.reranker = get_reranker(reranker_model) if rerank else None

    def search(self, query: str, k: int = 3) -> Tuple[List[str], float]:
        """
        Return (top-k chunks, best dense similarity score).
        The dense score is returned whatever the mode, because the relevance threshold is
        calibrated on cosine similarity (BM25 and RRF scores have no fixed scale).
        """
        pool = max(k, self.candidates) if (self.reranker or self.mode == "hybrid") else k
        dense_ids, dense_scores = dense_search(query, self.embed_model, self.index, pool)
        best_dense = dense_scores[0] if dense_scores else 0.0

        if self.mode == "dense":
            ranked = dense_ids
        else:
            bm25_ids = [i for i, _ in self.bm25.search(query, pool)]
            ranked = bm25_ids if self.mode == "bm25" else reciprocal_rank_fusion([dense_ids, bm25_ids])

        if self.reranker:
            candidates = ranked[:self.candidates]
            if candidates:
                rerank_scores = self.reranker.predict([(query, self.chunks[i]) for i in candidates])
                ranked = [candidates[j] for j in np.argsort(-np.asarray(rerank_scores))]

        return [self.chunks[i] for i in ranked[:k]], best_dense
