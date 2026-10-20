# src/embedder.py
from functools import lru_cache
from typing import List, Optional

from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=None)
def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Load the embedding model once and reuse it everywhere (indexing + querying).
    The first call downloads the model (~80MB); later calls return the cached object.
    """
    print(f"[INFO] Loading embedding model: {model_name} ...")
    return SentenceTransformer(model_name)


# Tokens kept free below the model's limit for the special tokens ([CLS], [SEP]) and
# small differences between counting sentences separately and counting the joined chunk
TOKEN_LIMIT_MARGIN = 16


def chunk_token_limit(model: SentenceTransformer) -> int:
    """Largest chunk (in tokens) the model can embed without silently truncating it."""
    return model.max_seq_length - TOKEN_LIMIT_MARGIN


def token_counter(model: SentenceTransformer):
    """A function that counts tokens exactly as the embedding model's tokenizer does."""
    tokenizer = model.tokenizer
    return lambda text: len(tokenizer.tokenize(text))


def generate_embeddings(chunks: List[str], model: Optional[SentenceTransformer] = None,
                        model_name: str = DEFAULT_EMBEDDING_MODEL):
    """
    Convert list of text chunks into numerical vectors (embeddings).
    Pass an already-loaded `model` to avoid loading it a second time.
    """
    if model is None:
        model = get_embedding_model(model_name)

    print(f"[INFO] Generating embeddings for {len(chunks)} chunks...")
    return model.encode(chunks, show_progress_bar=True)


if __name__ == "__main__":
    # Local module test
    vectors = generate_embeddings(["Hello world", "This is a test for embeddings"])
    print(f"Embedding Shape: {vectors.shape}")  # Should be (2, 384)
