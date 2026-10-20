# evaluate.py
"""
Measure retrieval quality on a labelled question set (no LLM calls, so it is free and repeatable).

Every retrieval configuration (dense, BM25, hybrid, with/without reranker) is compared side by side,
then the per-question details are shown for one configuration (the app's default unless chosen).

Examples:
    python evaluate.py                                         # default question set for sample_ieee.pdf
    python evaluate.py --questions eval/my_questions.json      # another question set
    python evaluate.py --mode dense --no-rerank --show-misses  # details for one configuration
"""
import argparse
import json
import os
import statistics
import sys
import time

from src.embedder import DEFAULT_EMBEDDING_MODEL, get_embedding_model
from src.evaluation import (check_labels, evaluate, load_questions, mean_reciprocal_rank, recall_at_k,
                            recall_with_figures, threshold_report)
from src.pipeline import load_or_build_index
from src.retriever import RETRIEVAL_MODES, Retriever

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_DIR, "data", "cache")
DEFAULT_QUESTIONS = os.path.join(PROJECT_DIR, "eval", "sample_ieee_questions.json")
K_VALUES = (1, 3, 5)
CANDIDATE_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
# (mode, rerank) pairs compared in the summary table
CONFIGURATIONS = [("dense", False), ("bm25", False), ("hybrid", False), ("dense", True), ("hybrid", True)]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval on a labelled question set.")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS, help="Question set (JSON)")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the index even if a cached one exists")
    parser.add_argument("--mode", choices=RETRIEVAL_MODES, default="hybrid",
                        help="Configuration to show per-question details for (default: hybrid)")
    parser.add_argument("--no-rerank", action="store_true", help="Show details for the configuration without reranker")
    parser.add_argument("--show-misses", action="store_true", help="Show the retrieved chunks for each miss")
    return parser.parse_args()


def config_name(mode, rerank):
    return f"{mode} + rerank" if rerank else mode


def run_configuration(questions, embed_model, index, chunks, figure_index, mode, rerank, max_k):
    """Evaluate one configuration; returns (results, average latency per question in ms)."""
    retriever = Retriever(embed_model, index, chunks, mode=mode, rerank=rerank)
    retriever.search("warm-up query", max_k)  # exclude one-off model warm-up from the timing
    start = time.perf_counter()
    results = evaluate(questions, retriever.search, chunks, figure_index, max_k)
    latency_ms = (time.perf_counter() - start) / max(len(questions), 1) * 1000
    return results, latency_ms


def print_details(results, max_k, show_misses):
    answerable = [r for r in results if r.question.answerable]
    unanswerable = [r for r in results if not r.question.answerable]

    print(f"{'id':<5} {'rank':>4} {'score':>6}  question")
    for r in results:
        rank = "-" if not r.question.answerable else (str(r.rank) if r.rank else "miss")
        print(f"{r.question.id:<5} {rank:>4} {r.top_score:>6.3f}  {r.question.question}")

    if answerable and unanswerable:
        a_scores = [r.top_score for r in answerable]
        u_scores = [r.top_score for r in unanswerable]
        print("\n--- Best dense-match score (used by the relevance threshold): answerable vs unanswerable ---")
        print(f"answerable:   min {min(a_scores):.3f}  median {statistics.median(a_scores):.3f}  max {max(a_scores):.3f}")
        print(f"unanswerable: min {min(u_scores):.3f}  median {statistics.median(u_scores):.3f}  max {max(u_scores):.3f}")
        print(f"\n{'threshold':>9}  {'answerable wrongly rejected':>27}  {'unanswerable correctly rejected':>31}")
        for t, wrong, right in threshold_report(results, CANDIDATE_THRESHOLDS):
            print(f"{t:>9.2f}  {wrong:>16}/{len(answerable):<10}  {right:>20}/{len(unanswerable):<10}")

    if show_misses:
        for r in answerable:
            if r.rank is None:
                print(f"\n--- MISS {r.question.id}: {r.question.question}")
                print(f"expected evidence: {r.question.evidence}")
                for i, c in enumerate(r.retrieved[:max_k], 1):
                    print(f"  {i}. {c[:160]!r}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()

    with open(args.questions, encoding="utf-8") as f:
        pdf_path = os.path.join(PROJECT_DIR, json.load(f)["pdf"])
    questions = load_questions(args.questions)

    embed_model = get_embedding_model(DEFAULT_EMBEDDING_MODEL)
    index, chunks, figure_index = load_or_build_index(pdf_path, embed_model, CACHE_DIR, args.rebuild)

    problems = check_labels(questions, chunks)
    for p in problems:
        print(f"[LABEL ERROR] {p}")

    max_k = max(K_VALUES)
    detail_config = (args.mode, not args.no_rerank)
    configurations = CONFIGURATIONS if detail_config in CONFIGURATIONS else CONFIGURATIONS + [detail_config]

    all_results = {}
    rows = []
    for mode, rerank in configurations:
        results, latency_ms = run_configuration(questions, embed_model, index, chunks, figure_index,
                                                mode, rerank, max_k)
        all_results[(mode, rerank)] = results
        rows.append((config_name(mode, rerank), [recall_at_k(results, k) for k in K_VALUES],
                     recall_with_figures(results), mean_reciprocal_rank(results), latency_ms))

    n_answerable = sum(q.answerable for q in questions)
    print("\n" + "=" * 86)
    print(f"Retrieval evaluation: {n_answerable} answerable + {len(questions) - n_answerable} unanswerable "
          f"questions, {len(chunks)} chunks")
    print("=" * 86)
    header = "".join(f"{'R@' + str(k):>8}" for k in K_VALUES)
    print(f"{'configuration':<18}{header}{'R@3+fig':>9}{'MRR@' + str(max_k):>8}{'ms/query':>10}")
    for name, recalls, with_fig, mrr, latency_ms in rows:
        cells = "".join(f"{r:>8.1%}" for r in recalls)
        print(f"{name:<18}{cells}{with_fig:>9.1%}{mrr:>8.3f}{latency_ms:>10.1f}")

    print(f"\n--- Details: {config_name(*detail_config)} ---")
    print_details(all_results[detail_config], max_k, args.show_misses)

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
