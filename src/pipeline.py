# src/pipeline.py
"""
The indexing pipeline shared by main.py (interactive Q&A) and evaluate.py (retrieval evaluation):
PDF -> body text + figures/tables -> chunks -> embeddings -> FAISS index, cached on disk per PDF.
"""
import os

from src.chunker import smart_paragraph_chunker
from src.embedder import DEFAULT_EMBEDDING_MODEL, chunk_token_limit, generate_embeddings, token_counter
from src.pdf_loader import parse_pdf
from src.vector_store import cache_paths_for, load_vector_db, save_vector_db

# Bump this whenever PDF parsing or chunking changes, so cached indexes are rebuilt
INDEX_VERSION = 3


def build_index(pdf_path, embed_model, index_path, chunks_path):
    """
    Full ingestion pipeline: PDF -> body text + figures/tables -> chunks -> embeddings -> FAISS index on disk.
    Body text is chunked to fit the embedding model; every figure/table becomes one chunk of its own.
    References are not indexed.
    """
    print(f"[INFO] Building index for {os.path.basename(pdf_path)} ...")
    paper = parse_pdf(pdf_path)
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


def load_or_build_index(pdf_path, embed_model, cache_dir, rebuild=False):
    """Return (index, chunks, figure_index) for the PDF, from the cache when possible."""
    index_path, chunks_path = cache_paths_for(pdf_path, DEFAULT_EMBEDDING_MODEL, cache_dir, INDEX_VERSION)
    if not rebuild:
        index, chunks, figure_index = load_vector_db(index_path, chunks_path)
        if index is not None:
            print(f"[INFO] Loaded cached index ({len(chunks)} chunks) for {os.path.basename(pdf_path)}")
            return index, chunks, figure_index
    return build_index(pdf_path, embed_model, index_path, chunks_path)
