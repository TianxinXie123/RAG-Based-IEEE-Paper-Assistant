# src/bm25.py
"""
Okapi BM25 keyword search.

Dense embeddings capture meaning but blur rare exact terms ("DP-PSASC", "Imagenette", "A1");
BM25 scores a chunk by the query terms it contains, weighted by how rare each term is (IDF)
and normalized by chunk length:

    score(q, d) = sum over terms t in q of  IDF(t) * f(t, d) * (k1 + 1) / (f(t, d) + k1 * (1 - b + b * |d| / avgdl))

where f(t, d) is the term frequency in chunk d, |d| its length and avgdl the average chunk length.
"""
import math
import re
from collections import Counter
from typing import List, Tuple

# Words too common to carry meaning on their own
STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have how if in into is it its
of on or so such than that the their them then there these they this to was were what when where which
while who why will with would we our you your
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-*][a-z0-9*]+)*\*?")


def tokenize(text: str) -> List[str]:
    """
    Lowercase word tokens without stopwords. A hyphenated term is kept whole AND split into
    its parts, so "DP-PSASC*" matches both "DP-PSASC*" and "PSASC".
    """
    tokens = []
    for term in _TOKEN_RE.findall(text.lower()):
        if term in STOPWORDS:
            continue
        tokens.append(term)
        parts = [p for p in re.split(r"[-*]", term) if p]
        if len(parts) > 1:
            tokens.extend(p for p in parts if p not in STOPWORDS)
    return tokens


class BM25:
    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_terms = [Counter(tokenize(d)) for d in documents]
        self.doc_lengths = [sum(c.values()) for c in self.doc_terms]
        self.avg_length = sum(self.doc_lengths) / max(len(documents), 1)

        # IDF with the usual +1 inside the log, so it is never negative
        n = len(documents)
        document_frequency = Counter(term for terms in self.doc_terms for term in terms)
        self.idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in document_frequency.items()}

    def scores(self, query: str) -> List[float]:
        query_terms = tokenize(query)
        result = []
        for terms, length in zip(self.doc_terms, self.doc_lengths):
            score = 0.0
            for t in query_terms:
                f = terms.get(t, 0)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * length / max(self.avg_length, 1e-9))
                    score += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            result.append(score)
        return result

    def search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Top-k (chunk_id, score) pairs; chunks that share no term with the query are left out."""
        ranked = sorted(enumerate(self.scores(query)), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked[:k] if s > 0]
