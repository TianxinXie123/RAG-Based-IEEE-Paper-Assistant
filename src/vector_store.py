# src/vector_store.py
import hashlib
import os
import pickle
from pathlib import Path

import faiss
import numpy as np


def cache_paths_for(pdf_path, embedding_model_name, cache_dir="data/cache", parser_version=1):
    """
    Build cache file paths that are unique to (PDF content, embedding model, parser version).

    The key is a SHA-256 hash of the PDF bytes plus the model name and parser version, so:
      - a different PDF never reuses another paper's index
      - editing/replacing a PDF with the same filename triggers a rebuild
      - switching embedding models or changing the PDF parser triggers a rebuild
    """
    pdf_path = Path(pdf_path)
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    h.update(embedding_model_name.encode("utf-8"))
    h.update(f"parser-v{parser_version}".encode("utf-8"))
    key = h.hexdigest()[:12]

    folder = Path(cache_dir) / f"{pdf_path.stem}_{key}"
    return str(folder / "index.faiss"), str(folder / "chunks.pkl")


def save_vector_db(embeddings, chunks, index_path, chunks_path, figure_index=None):
    """
    Save the FAISS index and the chunk texts.
    `figure_index` maps figure/table keys ("table:I") to the position of their chunk.
    """
    # Convert to float32 and NORMALIZE the vectors
    # Normalizing makes inner product equivalent to cosine similarity
    embeddings = np.asarray(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    # IndexFlatIP (Inner Product) on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump({"chunks": chunks, "figure_index": figure_index or {}}, f)
    print(f"[INFO] Index saved to {os.path.dirname(index_path)}")
    return index


def load_vector_db(index_path, chunks_path):
    """Return (index, chunks, figure_index), or (None, None, None) if there is no usable cache."""
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return None, None, None
    with open(chunks_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        # Cache written by an older version of this program: rebuild it
        return None, None, None
    index = faiss.read_index(index_path)
    return index, data["chunks"], data["figure_index"]


def dense_search(query_text, model, index, k=3):
    """Return (chunk_ids, cosine_scores) of the k chunks most similar to the query, best first."""
    # Encode and NORMALIZE the query vector as well
    query_vector = np.asarray(model.encode([query_text]), dtype="float32")
    faiss.normalize_L2(query_vector)

    # Never ask for more results than there are chunks
    k = min(k, index.ntotal)

    # Inner Product returns similarity scores (higher = more similar)
    scores, indices = index.search(query_vector, k)

    # FAISS uses -1 for "no result"; skip those
    ids, kept_scores = [], []
    for i, s in zip(indices[0], scores[0]):
        if i != -1:
            ids.append(int(i))
            kept_scores.append(float(s))
    return ids, kept_scores
