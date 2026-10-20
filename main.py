# main.py
"""
Ask questions about an (IEEE-style) PDF paper.

Examples:
    python main.py                                   # uses data/sample_ieee.pdf with Claude
    python main.py --pdf data/my_paper.pdf           # any other paper (index is built automatically)
    python main.py --pdf data/my_paper.pdf --rebuild # force re-indexing
    python main.py --model claude-opus-5-5           # a stronger (more expensive) Claude model
    python main.py --provider gemini                 # answer with Gemini instead of Claude

Inside the program, besides normal questions you can type:
    /summary   structured summary of the whole paper
    /suggest   reviewer-style strengths, weaknesses and suggestions
    /reset     forget the conversation so far
    /help      list the commands
"""
import argparse
import os
import sys

from src.embedder import DEFAULT_EMBEDDING_MODEL, get_embedding_model
from src.figures import attach_referenced_figures
from src.llm_engine import (DEFAULT_PROVIDER, PROVIDERS, SUGGESTIONS_TASK, SUMMARY_TASK, LLMError,
                            get_document_response, get_llm_response, rewrite_query)
from src.pdf_loader import parse_pdf
from src.pipeline import load_or_build_index
from src.retriever import RETRIEVAL_MODES, Retriever

# Resolve paths relative to this file, so the program works from any working directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PDF = os.path.join(PROJECT_DIR, "data", "sample_ieee.pdf")
CACHE_DIR = os.path.join(PROJECT_DIR, "data", "cache")

# Number of earlier question/answer turns sent along with a new question
MAX_HISTORY_TURNS = 3

# Below this best-match cosine similarity the question is treated as off-topic and no LLM call is made.
# Chosen with evaluate.py: on the sample set every answerable question scored >= 0.19, while off-topic
# questions scored <= 0.18. On-topic questions the paper cannot answer score as high as answerable ones,
# so those are left to the LLM, which is instructed to say it doesn't know.
MIN_RELEVANCE_SCORE = 0.15
# Starting a question with this prefix skips the relevance check
FORCE_PREFIX = "!"


def parse_args():
    parser = argparse.ArgumentParser(description="RAG question answering over a PDF paper.")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="Path to the PDF (default: data/sample_ieee.pdf)")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the index even if a cached one exists")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default=os.getenv("RAG_PROVIDER", DEFAULT_PROVIDER),
                        help=f"LLM provider (default: {DEFAULT_PROVIDER}, or the RAG_PROVIDER env var)")
    parser.add_argument("--model", default=None,
                        help="LLM model name (default depends on provider: "
                             + ", ".join(f"{p}={c['default_model']}" for p, c in PROVIDERS.items()) + ")")
    parser.add_argument("-k", "--top-k", type=int, default=3, help="Number of chunks to retrieve (default: 3)")
    # Defaults chosen with evaluate.py: hybrid had the best Recall@1/MRR; the reranker did not improve
    # Recall@3 on the sample set and added ~0.8 s per question, so it is opt-in.
    parser.add_argument("--retrieval", choices=RETRIEVAL_MODES, default="hybrid",
                        help="Retrieval mode: dense (embeddings), bm25 (keywords) or hybrid (both, default)")
    parser.add_argument("--rerank", action="store_true", help="Rerank the candidates with a cross-encoder")
    return parser.parse_args()


# Whole-paper commands: command -> (task prompt, status message)
DOCUMENT_COMMANDS = {
    "/summary": (SUMMARY_TASK, "Summarizing the whole paper..."),
    "/suggest": (SUGGESTIONS_TASK, "Reviewing the paper and drafting suggestions..."),
}

HELP_TEXT = """Commands:
  /summary   structured summary of the whole paper
  /suggest   strengths, weaknesses and suggestions for the work
  /reset     forget the conversation so far (follow-up questions use the last few turns)
  /help      show this list
  exit       quit
Anything else is treated as a question about the paper.
Questions that look unrelated to the paper are not sent to the LLM; start one with '!' to ask anyway."""


def stream_answer(generate):
    """
    Print an answer while it streams in. `generate` takes an on_text callback and returns the full text.
    Returns the full answer, or None if it failed or was cancelled with Ctrl+C.
    """
    print("\n" + "-" * 20 + " AI ANSWER " + "-" * 20)
    try:
        answer = generate(lambda text: print(text, end="", flush=True))
    except LLMError as e:
        print(f"\n[ERROR] {e}")
        print("You can try again, or restart with a different --provider / --model.")
        answer = None
    except KeyboardInterrupt:
        print("\n[Cancelled]")
        answer = None
    print("\n" + "-" * 51)
    return answer


def main():
    # Answers can contain symbols such as Greek letters; never crash on a console that cannot show them
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    pdf_path = os.path.abspath(args.pdf)
    if not os.path.exists(pdf_path):
        sys.exit(f"[ERROR] PDF not found: {pdf_path}")

    # Load the embedding model ONCE; it is used both for indexing and for queries
    embed_model = get_embedding_model(DEFAULT_EMBEDDING_MODEL)

    # Each PDF (by content) gets its own cached index
    try:
        index, chunks, figure_index = load_or_build_index(pdf_path, embed_model, CACHE_DIR, args.rebuild)
    except Exception as e:
        sys.exit(f"[ERROR] Could not read {os.path.basename(pdf_path)}: {type(e).__name__}: {e}")
    retriever = Retriever(embed_model, index, chunks, mode=args.retrieval, rerank=args.rerank)

    model_name = args.model or PROVIDERS[args.provider]["default_model"]
    print("\n" + "=" * 50)
    print(f"READY: Ask me anything about {os.path.basename(pdf_path)}")
    print(f"LLM: {args.provider} / {model_name}   Retrieval: {args.retrieval}{' + rerank' if args.rerank else ''}")
    print("Type a question, /summary, /suggest, /reset, /help, or 'exit' to quit")
    print("=" * 50)

    full_text = None  # the full paper text, extracted only when a whole-paper command is used
    history = []      # earlier (question, answer) turns, for follow-up questions

    while True:
        try:
            user_query = input("\n[Q]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_query:
            continue
        if user_query.lower() in ("exit", "quit"):
            break

        command = user_query.lower()
        if command == "/help":
            print(HELP_TEXT)
            continue
        if command == "/reset":
            history.clear()
            print("Conversation cleared.")
            continue
        if command in DOCUMENT_COMMANDS:
            task, status = DOCUMENT_COMMANDS[command]
            if full_text is None:
                full_text = parse_pdf(pdf_path).full_text()
            print(status)
            stream_answer(lambda on_text: get_document_response(
                task, full_text, provider=args.provider, model=args.model, on_text=on_text))
            continue
        if command.startswith("/"):
            print(f"Unknown command '{user_query}'. Type /help for the list of commands.")
            continue

        force = user_query.startswith(FORCE_PREFIX)
        if force:
            user_query = user_query[len(FORCE_PREFIX):].strip()
            if not user_query:
                continue

        # 1. Follow-up questions ("and on CIFAR10?") are rewritten into standalone ones for retrieval
        search_query = user_query
        if history:
            try:
                search_query = rewrite_query(user_query, history, provider=args.provider, model=args.model)
            except KeyboardInterrupt:
                print("\n[Cancelled]")
                continue
            if search_query != user_query:
                print(f"Searching for: {search_query}")

        # 2. Retrieval: find the most relevant chunks
        print("Finding relevant snippets...")
        results, best_score = retriever.search(search_query, k=args.top_k)
        if best_score < MIN_RELEVANCE_SCORE and not force:
            print(f"Nothing in the paper looks relevant to this question (best match score {best_score:.2f} "
                  f"< {MIN_RELEVANCE_SCORE}).")
            print(f"Try rephrasing it, or start the question with '{FORCE_PREFIX}' to ask anyway.")
            continue
        # Add the figures/tables that the retrieved paragraphs refer to ("see Table I")
        retrieved_count = len(results)
        results = attach_referenced_figures(results, chunks, figure_index)
        for extra in results[retrieved_count:]:
            print(f"Also using {extra.split(']')[0].lstrip('[')} (referenced in the retrieved text)")

        # 3. Generation: send them (and the recent conversation) to the LLM
        print("AI is thinking...")
        answer = stream_answer(lambda on_text: get_llm_response(
            user_query, results, provider=args.provider, model=args.model, history=history, on_text=on_text))
        if answer is not None:
            history.append((user_query, answer))
            del history[:-MAX_HISTORY_TURNS]


if __name__ == "__main__":
    main()
