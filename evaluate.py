# evaluate.py
"""
Measure retrieval quality on labelled question sets (no LLM calls, so it is free and repeatable).

Each JSON file in eval/ holds the questions for one paper. Every retrieval configuration (dense,
BM25, hybrid, with/without reranker) is run on every paper; the script prints the overall results
per configuration, then the per-paper results for one configuration (the app's default unless chosen).

Examples:
    python evaluate.py                                          # all question sets in eval/
    python evaluate.py --questions eval/twc_haps_islands.json   # one paper
    python evaluate.py --mode dense --no-rerank --details       # per-question details for one configuration
    python evaluate.py --show-misses                            # what was retrieved for each miss
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time

from src.embedder import DEFAULT_EMBEDDING_MODEL, get_embedding_model
from src.evaluation import (check_labels, evaluate, load_questions, mean_reciprocal_rank, recall_at_k,
                            recall_with_figures, threshold_report)
from src.pdf_loader import raw_text
from src.pipeline import load_or_build_index
from src.retriever import RETRIEVAL_MODES, Retriever

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_DIR, "data", "cache")
EVAL_DIR = os.path.join(PROJECT_DIR, "eval")
K_VALUES = (1, 3, 5)
CANDIDATE_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
# (mode, rerank) pairs compared in the summary table
CONFIGURATIONS = [("dense", False), ("bm25", False), ("hybrid", False), ("dense", True), ("hybrid", True)]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval on labelled question sets.")
    parser.add_argument("--questions", nargs="*", default=None,
                        help="Question set files (JSON); default: every file in eval/")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the indexes even if cached ones exist")
    parser.add_argument("--mode", choices=RETRIEVAL_MODES, default="hybrid",
                        help="Configuration for the per-paper table and details (default: hybrid)")
    parser.add_argument("--no-rerank", action="store_true", help="Use the configuration without reranker for details")
    parser.add_argument("--details", action="store_true", help="Print every question with its rank and score")
    parser.add_argument("--show-misses", action="store_true", help="Show the retrieved chunks for each miss")
    return parser.parse_args()


def config_name(mode, rerank):
    return f"{mode} + rerank" if rerank else mode


def answerable(results):
    return [r for r in results if r.question.answerable]


def metrics_row(results):
    """[R@1, R@3, R@5], R@3 with figure linking, MRR for the answerable questions of `results`."""
    a = answerable(results)
    return [recall_at_k(a, k) for k in K_VALUES], recall_with_figures(a), mean_reciprocal_rank(a)


def print_threshold_report(results):
    a_scores = [r.top_score for r in results if r.question.answerable]
    u_scores = [r.top_score for r in results if not r.question.answerable]
    if not a_scores or not u_scores:
        return
    print("\n--- Best dense-match score (used by the relevance threshold): answerable vs unanswerable ---")
    print(f"answerable:   min {min(a_scores):.3f}  median {statistics.median(a_scores):.3f}  max {max(a_scores):.3f}")
    print(f"unanswerable: min {min(u_scores):.3f}  median {statistics.median(u_scores):.3f}  max {max(u_scores):.3f}")
    print(f"\n{'threshold':>9}  {'answerable wrongly rejected':>27}  {'unanswerable correctly rejected':>31}")
    for t, wrong, right in threshold_report(results, CANDIDATE_THRESHOLDS):
        print(f"{t:>9.2f}  {wrong:>16}/{len(a_scores):<10}  {right:>20}/{len(u_scores):<10}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()

    question_files = args.questions or sorted(glob.glob(os.path.join(EVAL_DIR, "*.json")))
    if not question_files:
        sys.exit(f"[ERROR] No question sets found in {EVAL_DIR}")
    detail_config = (args.mode, not args.no_rerank)
    configurations = CONFIGURATIONS if detail_config in CONFIGURATIONS else CONFIGURATIONS + [detail_config]

    embed_model = get_embedding_model(DEFAULT_EMBEDDING_MODEL)
    max_k = max(K_VALUES)
    # results[config][paper] = list of QuestionResult; latency[config] = list of seconds per question
    results = {c: {} for c in configurations}
    latency = {c: [] for c in configurations}
    problems = []

    for path in question_files:
        paper = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            pdf_path = os.path.join(PROJECT_DIR, json.load(f)["pdf"])
        questions = load_questions(path)
        index, chunks, figure_index = load_or_build_index(pdf_path, embed_model, CACHE_DIR, args.rebuild)
        problems += [f"{paper}/{p}" for p in check_labels(questions, chunks, [raw_text(pdf_path)])]

        for mode, rerank in configurations:
            retriever = Retriever(embed_model, index, chunks, mode=mode, rerank=rerank)
            retriever.search("warm-up query", max_k)  # keep one-off model warm-up out of the timing
            start = time.perf_counter()
            results[(mode, rerank)][paper] = evaluate(questions, retriever.search, chunks, figure_index, max_k)
            latency[(mode, rerank)].append((time.perf_counter() - start) / max(len(questions), 1))

    for p in problems:
        print(f"[LABEL ERROR] {p}")

    papers = list(results[configurations[0]])
    pooled = {c: [r for paper in papers for r in results[c][paper]] for c in configurations}
    n_answerable = len(answerable(pooled[configurations[0]]))
    n_total = len(pooled[configurations[0]])

    print("\n" + "=" * 86)
    print(f"Retrieval evaluation: {len(papers)} papers, {n_answerable} answerable + "
          f"{n_total - n_answerable} unanswerable questions")
    print("=" * 86)
    header = "".join(f"{'R@' + str(k):>8}" for k in K_VALUES)
    print(f"{'configuration':<18}{header}{'R@3+fig':>9}{'MRR@' + str(max_k):>8}{'ms/query':>10}")
    for c in configurations:
        recalls, with_fig, mrr = metrics_row(pooled[c])
        ms = statistics.mean(latency[c]) * 1000
        print(f"{config_name(*c):<18}{''.join(f'{r:>8.1%}' for r in recalls)}{with_fig:>9.1%}{mrr:>8.3f}{ms:>10.1f}")

    print(f"\n--- Per paper: {config_name(*detail_config)} ---")
    print(f"{'paper':<28}{'questions':>10}{header}{'MRR':>8}")
    for paper in papers:
        recalls, _, mrr = metrics_row(results[detail_config][paper])
        n = len(answerable(results[detail_config][paper]))
        print(f"{paper:<28}{n:>10}{''.join(f'{r:>8.1%}' for r in recalls)}{mrr:>8.3f}")

    if args.details:
        print(f"\n--- Details: {config_name(*detail_config)} ---")
        for paper in papers:
            for r in results[detail_config][paper]:
                rank = "-" if not r.question.answerable else (str(r.rank) if r.rank else "miss")
                print(f"{paper[:16]:<17}{r.question.id:<5} {rank:>4} {r.top_score:>6.3f}  {r.question.question}")

    print_threshold_report(pooled[detail_config])

    if args.show_misses:
        for paper in papers:
            for r in answerable(results[detail_config][paper]):
                if r.rank is None:
                    print(f"\n--- MISS {paper}/{r.question.id}: {r.question.question}")
                    print(f"expected evidence: {r.question.evidence}")
                    for i, c in enumerate(r.retrieved[:max_k], 1):
                        print(f"  {i}. {c[:160]!r}")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
