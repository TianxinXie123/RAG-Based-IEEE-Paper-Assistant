# src/evaluation.py
"""
Retrieval evaluation: does the retriever find the chunk that contains the answer?

Each test question lists evidence phrases copied from the paper. The question is answered
at rank r if the r-th retrieved chunk contains any of them. Labelling by phrase instead of
by chunk id keeps the test set valid when the chunking strategy changes.

Metrics:
  - Recall@k (hit rate): fraction of answerable questions whose evidence is in the top k chunks
  - MRR (mean reciprocal rank): average of 1/rank of the first hit (0 if not found)
Unanswerable questions (no evidence) are used to choose the relevance threshold: the best
retrieval score should be high for answerable questions and low for unanswerable ones.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from src.figures import attach_referenced_figures


def normalize(text: str) -> str:
    """Lowercase, unify quotes and collapse whitespace, so evidence matching is robust to extraction noise."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def contains_evidence(chunk: str, evidence: List[str]) -> bool:
    chunk = normalize(chunk)
    return any(normalize(e) in chunk for e in evidence)


@dataclass
class Question:
    id: str
    question: str
    evidence: List[str]

    @property
    def answerable(self) -> bool:
        return bool(self.evidence)


@dataclass
class QuestionResult:
    question: Question
    rank: Optional[int]                 # 1-based rank of the first chunk with the evidence, None if not found
    top_score: float                    # similarity score of the best retrieved chunk
    hit_with_figures: bool = False      # evidence found in the top k after attaching referenced figures
    retrieved: List[str] = field(default_factory=list)


def load_questions(path: str) -> List[Question]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [Question(q["id"], q["question"], q.get("evidence", [])) for q in data["questions"]]


def check_labels(questions: List[Question], chunks: List[str]) -> List[str]:
    """Evidence phrases that appear in no chunk at all (a labelling mistake, not a retrieval miss)."""
    problems = []
    for q in questions:
        for e in q.evidence:
            if not any(contains_evidence(c, [e]) for c in chunks):
                problems.append(f"{q.id}: evidence not found in any chunk: {e!r}")
    return problems


def evaluate(questions: List[Question], search: Callable, chunks: List[str], figure_index: Dict[str, int],
             max_k: int, figure_k: int = 3) -> List[QuestionResult]:
    """
    Run every question through `search(question_text, k) -> (results, top_score)`.
    `figure_k` is the k used for the "with figure linking" variant (same as the app's default top-k).
    """
    results = []
    for q in questions:
        retrieved, top_score = search(q.question, max_k)
        rank = None
        if q.answerable:
            rank = next((i + 1 for i, c in enumerate(retrieved) if contains_evidence(c, q.evidence)), None)
        with_figures = attach_referenced_figures(retrieved[:figure_k], chunks, figure_index)
        hit_with_figures = q.answerable and any(contains_evidence(c, q.evidence) for c in with_figures)
        results.append(QuestionResult(q, rank, float(top_score), hit_with_figures, retrieved))
    return results


def recall_with_figures(results: List[QuestionResult]) -> float:
    answerable = [r for r in results if r.question.answerable]
    return sum(r.hit_with_figures for r in answerable) / max(len(answerable), 1)


def recall_at_k(results: List[QuestionResult], k: int) -> float:
    answerable = [r for r in results if r.question.answerable]
    return sum(1 for r in answerable if r.rank is not None and r.rank <= k) / max(len(answerable), 1)


def mean_reciprocal_rank(results: List[QuestionResult]) -> float:
    answerable = [r for r in results if r.question.answerable]
    return sum(1 / r.rank for r in answerable if r.rank) / max(len(answerable), 1)


def threshold_report(results: List[QuestionResult], thresholds: List[float]):
    """
    For each candidate threshold: how many answerable questions would be wrongly rejected
    (top score below it) and how many unanswerable ones would be correctly rejected.
    Returns a list of (threshold, wrongly_rejected, correctly_rejected).
    """
    answerable = [r.top_score for r in results if r.question.answerable]
    unanswerable = [r.top_score for r in results if not r.question.answerable]
    return [(t, sum(s < t for s in answerable), sum(s < t for s in unanswerable)) for t in thresholds]
