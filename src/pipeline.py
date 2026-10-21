# src/pipeline.py
"""
The indexing pipeline shared by main.py (interactive Q&A) and evaluate.py (retrieval evaluation):
PDF -> special-character repair -> body text + figures/tables -> chunks -> embeddings -> FAISS index,
cached on disk per PDF.
"""
import hashlib
import json
import os

from src.chunker import smart_paragraph_chunker
from src.embedder import DEFAULT_EMBEDDING_MODEL, chunk_token_limit, generate_embeddings, token_counter
from src.llm_engine import identify_symbol
from src.pdf_loader import parse_pdf
from src.symbols import resolve_symbol_map, symbol_cache_path
from src.vector_store import cache_paths_for, file_hash, load_vector_db, save_vector_db

# Bump this whenever PDF parsing or chunking changes, so cached indexes are rebuilt
INDEX_VERSION = 5


def _claude_recognizer():
    """A recognizer for src.symbols that asks Claude vision, or None if no API key is set."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    return lambda png, sample: identify_symbol(png, sample.char, sample.line_text)


def get_symbol_map(pdf_path, cache_dir, recognize=True):
    """
    {glyph key: correct character} for the PDF's wrongly mapped special characters.
    Recognition runs once per new glyph (Claude vision); results are cached in cache_dir/symbols/.
    """
    cache_path = symbol_cache_path(pdf_path, cache_dir, file_hash(pdf_path))
    return resolve_symbol_map(pdf_path, cache_path, _claude_recognizer() if recognize else None)


def parse_paper(pdf_path, cache_dir, recognize_symbols=True):
    """Parse the PDF with its special characters repaired."""
    return parse_pdf(pdf_path, get_symbol_map(pdf_path, cache_dir, recognize_symbols))


def build_index(paper, pdf_path, embed_model, index_path, chunks_path):
    """
    Index a parsed paper: body text + figures/tables -> chunks -> embeddings -> FAISS index on disk.
    Body text is chunked to fit the embedding model, with section headings as hard chunk boundaries;
    every figure/table becomes one chunk of its own.
    References are not indexed.
    """
    print(f"[INFO] Building index for {os.path.basename(pdf_path)} ...")
    if not paper.text.strip():
        raise RuntimeError("No text could be extracted from the PDF (is it a scanned image?).")

    text_chunks = smart_paragraph_chunker(paper.text, max_tokens=chunk_token_limit(embed_model),
                                          count_tokens=token_counter(embed_model))
    chunks = text_chunks + [f.as_text() for f in paper.figures]
    # Figure/table key ("table:I") -> position of its chunk, used to link body text back to it
    figure_index = {}
    for i, f in enumerate(paper.figures):
        if f.key:
            figure_index.setdefault(f.key, len(text_chunks) + i)
    print(f"[INFO] Extracted {len(paper.text)} characters -> {len(text_chunks)} text chunks "
          f"+ {len(paper.figures)} figures/tables (references not indexed)")

    embeddings = generate_embeddings(chunks, model=embed_model)
    index = save_vector_db(embeddings, chunks, index_path, chunks_path, figure_index)
    return index, chunks, figure_index


def load_or_build_index(pdf_path, embed_model, cache_dir, rebuild=False, recognize_symbols=True):
    """Return (index, chunks, figure_index) for the PDF, from the cache when possible."""
    symbol_map = get_symbol_map(pdf_path, cache_dir, recognize_symbols)
    # The repaired characters change the text, so they are part of the index cache key
    symbols_digest = hashlib.sha256(json.dumps(symbol_map, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    index_path, chunks_path = cache_paths_for(pdf_path, DEFAULT_EMBEDDING_MODEL, cache_dir,
                                              f"{INDEX_VERSION}-symbols-{symbols_digest}")
    if not rebuild:
        index, chunks, figure_index = load_vector_db(index_path, chunks_path)
        if index is not None:
            print(f"[INFO] Loaded cached index ({len(chunks)} chunks) for {os.path.basename(pdf_path)}")
            return index, chunks, figure_index
    paper = parse_pdf(pdf_path, symbol_map)
    return build_index(paper, pdf_path, embed_model, index_path, chunks_path)
