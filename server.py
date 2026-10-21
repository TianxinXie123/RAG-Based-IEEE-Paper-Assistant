# server.py
"""
Web interface for the paper assistant: a small Flask API around the same pipeline as main.py,
plus the page in docs/ (which GitHub Pages can also host on its own).

    python server.py                         # then open http://localhost:8000
    python server.py --model claude-opus-5-5 --port 8080

API:
    GET  /api/health   backend status and the paper that is loaded
    POST /api/upload   multipart form with a "file" field: save the PDF and index it
    POST /api/ask      {"question": "...", "task": null | "summary" | "suggest", "history": [[q, a], ...]}
                       answered as a stream of JSON events, one per line (NDJSON):
                       status / sources / text / done, or notice / error
"""
import argparse
import hashlib
import json
import os
import queue
import shutil
import sys
import threading
from dataclasses import dataclass
from urllib.parse import urlparse

from flask import Flask, Response, jsonify, request
from werkzeug.utils import secure_filename

from main import CACHE_DIR, FORCE_PREFIX, MAX_HISTORY_TURNS, MIN_RELEVANCE_SCORE, PROJECT_DIR
from src.embedder import DEFAULT_EMBEDDING_MODEL, get_embedding_model
from src.figures import attach_referenced_figures
from src.llm_engine import (DEFAULT_PROVIDER, PROVIDERS, SUGGESTIONS_TASK, SUMMARY_TASK, LLMError,
                            get_document_response, get_llm_response, rewrite_query)
from src.pipeline import load_full_text, load_or_build_index
from src.retriever import DEFAULT_RERANKER_MODEL, Retriever, get_reranker

WEB_DIR = os.path.join(PROJECT_DIR, "docs")
UPLOAD_DIR = os.path.join(PROJECT_DIR, "data", "uploads")
MAX_UPLOAD_MB = 50

# Whole-paper tasks, as /summary and /suggest in main.py
TASKS = {"summary": SUMMARY_TASK, "suggest": SUGGESTIONS_TASK}

# Pages served from this computer may call the API, and so may these sites (the GitHub Pages copy of
# docs/). More can be added with the RAG_ALLOWED_ORIGINS env var, comma-separated.
ALLOWED_ORIGINS = {"https://tianxinxie123.github.io"} | {
    origin.strip() for origin in os.getenv("RAG_ALLOWED_ORIGINS", "").split(",") if origin.strip()}
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")

app = Flask(__name__, static_folder=WEB_DIR, static_url_path="")
app.config.update(MAX_CONTENT_LENGTH=MAX_UPLOAD_MB * 1024 * 1024,
                  PROVIDER=DEFAULT_PROVIDER, MODEL=None, TOP_K=3)


@dataclass
class Paper:
    """The paper the questions are about, with everything needed to answer them."""
    name: str            # file name as uploaded, shown on the page
    path: str
    retriever: Retriever
    chunks: list
    figure_index: dict
    full_text: str | None = None  # extracted only when a whole-paper task is asked


current_paper = None
# Only one paper is indexed at a time
indexing_lock = threading.Lock()


def origin_allowed(origin):
    if origin is None:  # same-origin GET requests and non-browser clients send no Origin header
        return True
    return urlparse(origin).hostname in LOCAL_HOSTS or origin in ALLOWED_ORIGINS


@app.before_request
def reject_other_websites():
    # Without this, any website open in the browser could upload files or spend API credits
    if not origin_allowed(request.headers.get("Origin")):
        return jsonify(error="This website is not allowed to use the assistant."), 403


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin and origin_allowed(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Vary"] = "Origin"
    return response


@app.errorhandler(413)
def file_too_large(_):
    return jsonify(error=f"The file is larger than {MAX_UPLOAD_MB} MB."), 413


def paper_info(paper):
    return {"name": paper.name, "chunks": len(paper.chunks), "figures": len(paper.figure_index)}


def upload_path(filename, data):
    """
    Where an upload is saved: data/uploads/<content hash>/<name>.pdf. A different paper with the same
    name never overwrites the loaded one, and the name is kept so the index cache of main.py is reused.
    secure_filename drops characters it cannot keep (e.g. Chinese ones), hence the fallback name.
    """
    stem = secure_filename(os.path.splitext(filename)[0]) or "paper"
    return os.path.join(UPLOAD_DIR, hashlib.sha256(data).hexdigest()[:12], f"{stem}.pdf")


@app.get("/")
def index_page():
    return app.send_static_file("index.html")


@app.get("/api/health")
def health():
    model = app.config["MODEL"] or PROVIDERS[app.config["PROVIDER"]]["default_model"]
    return jsonify(app="ieee-paper-assistant", model=f"{app.config['PROVIDER']} / {model}",
                   paper=paper_info(current_paper) if current_paper else None)


@app.post("/api/upload")
def upload():
    global current_paper
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify(error="No file was uploaded."), 400
    data = file.read()
    if b"%PDF" not in data[:1024]:
        return jsonify(error="That file is not a PDF."), 400
    if not indexing_lock.acquire(blocking=False):
        return jsonify(error="Another paper is being indexed. Wait for it to finish, then try again."), 409

    try:
        name = os.path.basename(file.filename)
        path = upload_path(name, data)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        embed_model = get_embedding_model(DEFAULT_EMBEDDING_MODEL)
        try:
            index, chunks, figure_index = load_or_build_index(path, embed_model, CACHE_DIR)
        except Exception as e:
            print(f"[ERROR] Could not read {name}: {type(e).__name__}: {e}")
            shutil.rmtree(os.path.dirname(path), ignore_errors=True)
            return jsonify(error=f"Could not read {name}. It may be damaged, password-protected, "
                                 "or a scanned image without a text layer."), 422
        current_paper = Paper(name, path, Retriever(embed_model, index, chunks), chunks, figure_index)
    finally:
        indexing_lock.release()
    return jsonify(paper=paper_info(current_paper))


@app.post("/api/ask")
def ask():
    body = request.get_json(silent=True) or {}
    question = str(body.get("question") or "").strip()
    task = body.get("task")
    history = [(str(turn[0]), str(turn[1])) for turn in body.get("history") or []
               if isinstance(turn, list) and len(turn) == 2][-MAX_HISTORY_TURNS:]

    paper = current_paper
    if paper is None:
        return jsonify(error="Upload a paper first."), 409
    if task is not None and task not in TASKS:
        return jsonify(error=f"Unknown task '{task}'."), 400
    if task is None and not question:
        return jsonify(error="The question is empty."), 400

    events = answer_task(paper, task) if task else answer_question(paper, question, history)
    return Response(ndjson(events), mimetype="application/x-ndjson")


def ndjson(events):
    """One JSON event per line; an unexpected error becomes an error event instead of a broken stream."""
    try:
        for event in events:
            yield json.dumps(event) + "\n"
    except Exception as e:
        yield json.dumps({"type": "error", "text": f"{type(e).__name__}: {e}"}) + "\n"


def stream_llm(generate):
    """
    Yield the answer of an LLM call piece by piece, then a "done" (or "error") event.
    `generate` takes an on_text callback, like in main.stream_answer. It runs in a thread
    because the pieces arrive through that callback, which cannot yield them itself.
    """
    events = queue.Queue()

    def run():
        try:
            generate(lambda text: events.put({"type": "text", "text": text}))
            events.put({"type": "done"})
        except Exception as e:
            events.put({"type": "error", "text": str(e) if isinstance(e, LLMError) else f"{type(e).__name__}: {e}"})
        finally:
            events.put(None)

    threading.Thread(target=run, daemon=True).start()
    while (event := events.get()) is not None:
        yield event


def answer_question(paper, question, history):
    """The steps of a question in main.py: rewrite a follow-up, retrieve, check relevance, generate."""
    provider, model = app.config["PROVIDER"], app.config["MODEL"]
    force = question.startswith(FORCE_PREFIX)
    if force:
        question = question[len(FORCE_PREFIX):].strip()
        if not question:
            yield {"type": "notice", "text": "The question is empty."}
            return

    search_query = question
    if history:
        yield {"type": "status", "text": "Reading the conversation..."}
        search_query = rewrite_query(question, history, provider=provider, model=model)
        if search_query != question:
            yield {"type": "status", "text": f"Searching for: {search_query}"}

    yield {"type": "status", "text": "Finding relevant snippets..."}
    results, best_score = paper.retriever.search(search_query, k=app.config["TOP_K"])
    if best_score < MIN_RELEVANCE_SCORE and not force:
        yield {"type": "notice",
               "text": f"Nothing in the paper looks relevant to this question (best match score {best_score:.2f} "
                       f"< {MIN_RELEVANCE_SCORE}). Try rephrasing it, or start the question with "
                       f"'{FORCE_PREFIX}' to ask anyway."}
        return
    # Add the figures/tables that the retrieved paragraphs refer to ("see Table I")
    results = attach_referenced_figures(results, paper.chunks, paper.figure_index)
    yield {"type": "sources", "sources": results}

    yield {"type": "status", "text": "Writing the answer..."}
    yield from stream_llm(lambda on_text: get_llm_response(
        question, results, provider=provider, model=model, history=history, on_text=on_text))


def answer_task(paper, task):
    yield {"type": "status", "text": "Reading the whole paper..."}
    if paper.full_text is None:
        paper.full_text = load_full_text(paper.path, CACHE_DIR)
    yield from stream_llm(lambda on_text: get_document_response(
        TASKS[task], paper.full_text, provider=app.config["PROVIDER"], model=app.config["MODEL"], on_text=on_text))


def parse_args():
    parser = argparse.ArgumentParser(description="Web interface for the paper assistant.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default=os.getenv("RAG_PROVIDER", DEFAULT_PROVIDER),
                        help=f"LLM provider (default: {DEFAULT_PROVIDER}, or the RAG_PROVIDER env var)")
    parser.add_argument("--model", default=None, help="LLM model name (default depends on provider)")
    parser.add_argument("-k", "--top-k", type=int, default=3, help="Number of chunks to retrieve (default: 3)")
    return parser.parse_args()


def main():
    # File names and answers can contain symbols the console cannot show; never crash on them
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    app.config.update(PROVIDER=args.provider, MODEL=args.model, TOP_K=args.top_k)

    # Load the models now, so the first upload and the first question don't wait for them
    # (with the same arguments as later calls: they are cached per argument)
    get_embedding_model(DEFAULT_EMBEDDING_MODEL)
    get_reranker(DEFAULT_RERANKER_MODEL)

    print(f"\nOpen http://localhost:{args.port} in your browser (Ctrl+C to stop)\n")
    # 127.0.0.1: reachable from this computer only
    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
