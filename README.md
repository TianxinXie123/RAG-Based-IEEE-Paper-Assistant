IEEE Document Knowledge Base & RAG System
An intelligent assistant that transforms complex academic papers into a searchable knowledge base for instant, direct answers.

Overview
This project is a high-performance RAG (Retrieval-Augmented Generation) pipeline specifically optimized for academic research papers. It overcomes the common "cross-column reading" issue in IEEE-style PDFs, enabling accurate information retrieval and synthesis using state-of-the-art AI models.


Key Features

Coordinate-Aware PDF Parsing: Reconstructs logical reading order from double-column IEEE layouts using coordinate-based heuristic sorting.

Hybrid Search Engine: Combines semantic retrieval (FAISS + Sentence-Transformers all-MiniLM-L6-v2) with BM25 keyword search through Reciprocal Rank Fusion, followed by a cross-encoder reranker.

Section-Aware Chunking: Section headings are detected from font size and numbering and used as hard chunk boundaries; chunks are sentence-aligned, fit the embedding model's token limit and overlap by whole sentences.

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

Ingestion: The system parses the PDF and sorts text blocks by their (x, y) coordinates to handle multi-column formats. Running headers and footers (lines repeated in the page margins), page numbers, copyright notices and download watermarks are removed.
Embedding: Text is split into chunks and converted into 384-dimensional vectors.
Special Characters: Publishers' math fonts often map glyphs to the wrong Unicode text (the sample paper's "ε = 8, δ = 10⁻⁵" extracts as "2 ¼ 8, d ¼ 10-5"). Characters drawn with fonts known to be mapped unreliably (Symbol-encoded and publisher symbol fonts; TeX/MathType fonts are trusted) are detected by font, deduplicated per glyph, rendered with a red box around them and identified once with Claude vision; the resulting map is cached in data/cache/symbols/ and applied to the whole paper. Superscripts are written as ^x / ^(...), and accents drawn before their letter are recomposed ("Hübler"). At most 30 glyphs are recognized per paper. Without ANTHROPIC_API_KEY this step is skipped.
Figures and Tables: Text inside figures and tables is kept out of the body text. Each caption (plus the table cells, as Markdown) becomes its own chunk, labeled like "[Table I, page 4]". When a retrieved paragraph mentions "Fig. 2" or "Table I", that figure/table is added to the context automatically.
Chunking: Body text is split at section headings, then into sentence-aligned chunks that fit the embedding model's 256-token limit (counted with its own tokenizer), with whole-sentence overlap within a section. The references section is kept out of the index.
Indexing: Vectors are stored in a FAISS index for efficient similarity searching.
Querying: When a user asks a question, the system retrieves the most relevant chunks (hybrid search) and sends them to Claude to generate a final answer with [Source N] citations.

Setup

pip install -r requirements.txt

Put the papers you want to use in data/papers/ (for example, PDFs downloaded through your university's
IEEE Xplore access). They are not included in this repository because they are copyrighted, and PDFs in
data/ are ignored by git, so your papers are never committed. Each question set in eval/ names the paper
it was written for (its "pdf" field) and the paper's title and venue, so the same PDFs can be downloaded again.

Set the API key for the LLM you want to use (Windows PowerShell):
  $env:GEMINI_API_KEY = "your-key"       # for Gemini
  $env:ANTHROPIC_API_KEY = "your-key"    # for Claude

Usage

python main.py                                    # choose a paper from data/papers/, answered by Claude (claude-haiku-5-5)
python main.py --pdf data/papers/my_paper.pdf     # a specific paper; its index is built and cached automatically
python main.py --pdf data/papers/my_paper.pdf --rebuild  # force re-indexing
python main.py --model claude-opus-5-5            # a stronger (more expensive) Claude model
python main.py --provider gemini                  # answer with Gemini (gemini-2.5-flash)
python main.py -k 5                               # retrieve 5 chunks instead of 3
python main.py --retrieval dense                  # embeddings only (also: bm25, hybrid = default)
python main.py --no-rerank                        # skip the cross-encoder reranker (faster, less accurate)
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

python evaluate.py                                # retrieval quality on every question set in eval/ (no API calls)
python evaluate.py --show-misses                  # also print what was retrieved for each miss

The question set lists, for each question, evidence phrases copied from the paper; a question is answered
at rank r if the r-th retrieved chunk contains one of them. The script reports Recall@1/3/5, MRR, the
effect of figure linking, and the score distribution of answerable vs unanswerable questions that was used
to choose the relevance threshold.

Test set: 11 IEEE papers covering 7 templates (Transactions, Letters, magazines, IEEE Access, an Open
Journal, a conference paper), 111 answerable + 18 unanswerable questions. Results with the default
configuration (hybrid + rerank) before and after making the parser generalize beyond the first paper:

                                   R@1     R@3     R@5    MRR@5
first paper only (31 questions)  74.2%  100.0%  100.0%   0.860
all 11 papers, before            57.7%   78.4%   82.0%   0.677
all 11 papers, after             64.0%   87.4%   92.8%   0.756

The fixes that closed most of the gap: magazine layouts merged page decorations with figure drawings,
so ~58% of their body text was dropped as "figure text" (now a figure region may not contain a paragraph
of body-size text; Recall@3 on the two magazines rose from 25% to 87.5%), and running headers, page
numbers, copyright notices and download watermarks are removed. All configurations on all 11 papers:

configuration       R@1     R@3     R@5   R@3+fig   MRR@5   ms/query
dense             32.4%   57.7%   70.3%    58.6%    0.463       20
bm25              57.7%   73.9%   87.4%    73.9%    0.676       17
hybrid            47.7%   72.1%   80.2%    72.1%    0.601       17
dense + rerank    63.1%   83.8%   89.2%    83.8%    0.737      420
hybrid + rerank   64.0%   87.4%   92.8%    87.4%    0.756      480

History on the first paper (hybrid, Recall@3): token-aware chunking + hybrid search 70.0% (dense) -> 93.3%;
section headings as chunk boundaries -> 93.5%; with one-topic chunks the cross-encoder reranker
(cross-encoder/ms-marco-MiniLM-L-6-v2) reaches 100%, so the app uses hybrid + rerank by default. Adding the
section path to every chunk ("contextual chunk header") did not help, so it is off. Caveat: the questions
were written while reading the papers, so they share wording with them, which favors keyword search.

Tests

python -m pytest                                  # unit tests (no API key needed)

Indexes are cached per PDF in data/cache/<pdf-name>_<hash>/ (the hash covers the PDF content and the embedding model), so switching papers never reuses a stale index.
