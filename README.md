IEEE Document Knowledge Base & RAG System
An intelligent assistant that transforms complex academic papers into a searchable knowledge base for instant, direct answers.

Overview
This project is a high-performance RAG (Retrieval-Augmented Generation) pipeline specifically optimized for academic research papers. It overcomes the common "cross-column reading" issue in IEEE-style PDFs, enabling accurate information retrieval and synthesis using state-of-the-art AI models.


Key Features

Coordinate-Aware PDF Parsing: Reconstructs logical reading order from double-column IEEE layouts using coordinate-based heuristic sorting.

Hybrid Search Engine: Combines semantic retrieval (FAISS + Sentence-Transformers all-MiniLM-L6-v2) with BM25 keyword search through Reciprocal Rank Fusion, with an optional cross-encoder reranker.

Token-Aware Chunking: Sentence-aligned chunks that fit the embedding model's token limit, with whole-sentence overlap.

Claude Integration: Uses the Anthropic Claude API to synthesize professional, context-aware answers in English (Google Gemini is supported as an optional alternative).

Whole-Paper Summary and Review: The /summary and /suggest commands send the full paper to the LLM for a structured summary or reviewer-style suggestions.

Persistent Vector Store: Local caching of embeddings in .faiss and .pkl formats to eliminate redundant API costs and computation.

Tech Stack
Language: Python

Vector DB: FAISS (Facebook AI Similarity Search)

LLM API: Google Gemini 2.0 API

Embedding Model: Sentence-Transformers

PDF Processing: PyMuPDF (fitz)


How It Works

Ingestion: The system parses the PDF and sorts text blocks by their (x, y) coordinates to handle multi-column formats.Embedding: Text is split into chunks and converted into 384-dimensional vectors.
Figures and Tables: Text inside figures and tables is kept out of the body text. Each caption (plus the table cells, as Markdown) becomes its own chunk, labeled like "[Table I, page 4]". When a retrieved paragraph mentions "Fig. 2" or "Table I", that figure/table is added to the context automatically.
Chunking: Body text is split into sentence-aligned chunks that fit the embedding model's 256-token limit (counted with its own tokenizer), with whole-sentence overlap. The references section is kept out of the index.
Indexing: Vectors are stored in a FAISS index for efficient similarity searching.
Querying: When a user asks a question, the system retrieves the most relevant chunks (hybrid search) and sends them to Claude to generate a final answer with [Source N] citations.

Setup

pip install -r requirements.txt

Get the sample paper. It is not included in this repository because it is copyrighted by IEEE:
  C. Hou et al., "Adaptive Gradient Methods for Differentially Private TinyML in 6G,"
  IEEE Wireless Communications, 2025. DOI: 10.1109/MWC.2025.3640093
  https://doi.org/10.1109/MWC.2025.3640093
Download the PDF (e.g. through your university's IEEE Xplore access) and save it as data/sample_ieee.pdf.
It is the default paper for main.py and the paper the evaluation set in eval/ was written for.
Any other paper works with: python main.py --pdf path/to/paper.pdf
PDFs in data/ are ignored by git, so your papers are never committed.

Set the API key for the LLM you want to use (Windows PowerShell):
  $env:GEMINI_API_KEY = "your-key"       # for Gemini
  $env:ANTHROPIC_API_KEY = "your-key"    # for Claude

Usage

python main.py                                    # sample_ieee.pdf, answered by Claude (claude-haiku-5-5)
python main.py --pdf data/my_paper.pdf            # any other paper; its index is built and cached automatically
python main.py --pdf data/my_paper.pdf --rebuild  # force re-indexing
python main.py --model claude-opus-5-5            # a stronger (more expensive) Claude model
python main.py --provider gemini                  # answer with Gemini (gemini-2.5-flash)
python main.py -k 5                               # retrieve 5 chunks instead of 3
python main.py --retrieval dense                  # embeddings only (also: bm25, hybrid = default)
python main.py --rerank                           # rerank candidates with a cross-encoder
python list_models.py                             # list the models your API keys can use

Commands inside the program:
  /summary   structured summary of the whole paper
  /suggest   strengths, weaknesses and concrete suggestions for the work
  /reset     forget the conversation so far
  /help      list the commands
  exit       quit
Anything else is answered from the most relevant retrieved chunks, with [Source N] citations.
Answers stream in as they are generated. Follow-up questions ("And on Imagenette?") use the last
3 turns: the question is first rewritten into a standalone one for retrieval.

Questions that look unrelated to the paper (best retrieval score below 0.15) are not sent to the LLM;
start a question with '!' to ask anyway.

Evaluation

python evaluate.py                                # retrieval quality on eval/sample_ieee_questions.json (no API calls)
python evaluate.py --show-misses                  # also print what was retrieved for each miss

The question set lists, for each question, evidence phrases copied from the paper; a question is answered
at rank r if the r-th retrieved chunk contains one of them. The script reports Recall@1/3/5, MRR, the
effect of figure linking, and the score distribution of answerable vs unanswerable questions that was used
to choose the relevance threshold. Results on the sample paper (30 answerable questions, 38 chunks):

configuration       R@1     R@3     R@5   R@3+fig   MRR@5   ms/query
dense             56.7%   70.0%   80.0%    73.3%    0.651        6
bm25              76.7%   93.3%   96.7%    93.3%    0.857        6
hybrid            80.0%   93.3%   93.3%    93.3%    0.856        6
dense + rerank    60.0%   93.3%   96.7%    93.3%    0.762      900
hybrid + rerank   60.0%   93.3%   96.7%    93.3%    0.762      780

The app uses hybrid retrieval (dense + BM25 merged with Reciprocal Rank Fusion) by default. The cross-encoder
reranker (cross-encoder/ms-marco-MiniLM-L-6-v2) did not improve Recall@3 here, lowered Recall@1 and added
~0.8 s per question, so it is opt-in (--rerank). Caveat: the questions were written while reading the paper,
so they share wording with it, which favors keyword search; 30 questions also means one question = 3.3%.

Tests

python -m pytest                                  # unit tests (no API key needed)

Indexes are cached per PDF in data/cache/<pdf-name>_<hash>/ (the hash covers the PDF content and the embedding model), so switching papers never reuses a stale index.
