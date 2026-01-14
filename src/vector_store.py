# src/vector_store.py
import faiss
import pickle
import os
import numpy as np

def save_vector_db(embeddings, chunks, index_path="data/index.faiss", chunks_path="data/chunks.pkl"):
    # English Comment: Convert to float32 and NORMALIZE the vectors
    # Normalizing makes L2 distance equivalent to Cosine Similarity logic
    embeddings = embeddings.astype('float32')
    faiss.normalize_L2(embeddings)
    
    dimension = embeddings.shape[1]
    
    # English Comment: Use IndexFlatIP (Inner Product) for Cosine Similarity
    # This is much more robust for semantic search than raw L2 distance
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"[INFO] Normalized Index saved to {index_path}")

def load_vector_db(index_path="data/index.faiss", chunks_path="data/chunks.pkl"):
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        return None, None
    index = faiss.read_index(index_path)
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    return index, chunks

def search_in_db(query_text, model, index, chunks, k=3):
    # English Comment: Encode and NORMALIZE the query vector as well
    query_vector = model.encode([query_text])
    query_vector = np.array(query_vector).astype('float32')
    faiss.normalize_L2(query_vector)
    
    # English Comment: Inner Product (IP) returns Similarity Scores (Higher is better)
    # Note: Unlike L2, for IP, higher scores mean more similarity!
    scores, indices = index.search(query_vector, k)
    
    results = [chunks[i] for i in indices[0]]
    return results, scores[0]