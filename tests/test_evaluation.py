import json

from src.evaluation import (Question, check_labels, contains_evidence, evaluate, load_questions,
                            mean_reciprocal_rank, recall_at_k, threshold_report)

CHUNKS = [
    "Static clipping uses a fixed threshold.",
    "Auto-S divides every gradient by its own norm.",
    "[Table I, page 4] Accuracy table\n|DP-SGD|97.35|",
    "Results are summarized in Table I.",
]
FIGURE_INDEX = {"table:I": 2}


def fake_search(order_by_question):
    """A search function that returns chunks in a fixed order per question."""
    def search(question, k):
        order = order_by_question[question][:k]
        return [CHUNKS[i] for i in order], 0.9
    return search


def test_contains_evidence_ignores_case_quotes_and_whitespace():
    assert contains_evidence("The model’s   FIXED\nthreshold", ["model's fixed threshold"])
    assert not contains_evidence("something else", ["fixed threshold"])


def test_ranks_metrics_and_figure_linking():
    questions = [
        Question("a", "static?", ["fixed threshold"]),     # found at rank 1
        Question("b", "auto?", ["its own norm"]),          # found at rank 2
        Question("c", "accuracy?", ["97.35"]),             # not retrieved, but linked via "Table I"
        Question("u", "cake?", []),                        # unanswerable
    ]
    search = fake_search({"static?": [0, 1], "auto?": [3, 1], "accuracy?": [3, 0], "cake?": [0, 1]})
    results = evaluate(questions, search, CHUNKS, FIGURE_INDEX, max_k=2, figure_k=2)

    assert [r.rank for r in results] == [1, 2, None, None]
    assert recall_at_k(results, 1) == 1 / 3
    assert recall_at_k(results, 2) == 2 / 3
    assert abs(mean_reciprocal_rank(results) - (1 + 0.5) / 3) < 1e-9
    # Question c is answered once the table referenced by the retrieved chunk is attached
    assert results[2].hit_with_figures
    assert not results[3].hit_with_figures


def test_threshold_report_counts_rejections():
    questions = [Question("a", "static?", ["fixed threshold"]), Question("u", "cake?", [])]
    search = lambda q, k: ([CHUNKS[0]], 0.5 if q == "static?" else 0.1)
    results = evaluate(questions, search, CHUNKS, FIGURE_INDEX, max_k=1)
    assert threshold_report(results, [0.05, 0.2, 0.6]) == [(0.05, 0, 0), (0.2, 0, 1), (0.6, 1, 1)]


def test_check_labels_reports_evidence_that_exists_nowhere():
    questions = [Question("a", "q", ["fixed threshold"]), Question("b", "q", ["not in the paper"])]
    assert check_labels(questions, CHUNKS) == ["b: evidence not found in the paper: 'not in the paper'"]
    # Evidence the parser lost is still a valid label when it is in the PDF's own text
    assert check_labels(questions, CHUNKS, ["... not in the paper ..."]) == []


def test_load_questions(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps({"pdf": "x.pdf", "questions": [
        {"id": "q1", "question": "What?", "evidence": ["e"]},
        {"id": "u1", "question": "Cake?"},
    ]}), encoding="utf-8")
    questions = load_questions(str(path))
    assert [q.answerable for q in questions] == [True, False]
