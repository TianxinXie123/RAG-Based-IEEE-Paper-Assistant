# src/embedder.py
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

def generate_embeddings(chunks: List[str], model_name: str = 'all-MiniLM-L6-v2'):
    """
    Convert list of text chunks into numerical vectors (embeddings).
    """
    # English Comment: Load the pre-trained transformer model
    # The first time you run this, it will download the model (~80MB)
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    # English Comment: Perform the encoding (text to vector)
    print(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    return embeddings

if __name__ == "__main__":
    # English Comment: Local module test
    test_chunks = ["Hello world", "This is a test for embeddings"]
    vectors = generate_embeddings(test_chunks)
    print(f"Embedding Shape: {vectors.shape}") # Should be (2, 384)