# main.py
from src.pdf_loader import load_pdf_text
from src.chunker import smart_paragraph_chunker
from src.embedder import generate_embeddings
from src.vector_store import load_vector_db, save_vector_db, search_in_db
from src.llm_engine import get_gemini_response # New import
import os
from sentence_transformers import SentenceTransformer

def main():
    # ... (Step 0 to Step 4 are the same as before) ...
    # English Comment: Assuming index and chunks are already loaded/created
    index, chunks = load_vector_db() 
    if not index:
        # (Run the pipeline to create them if they don't exist)
        pass

    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("\n" + "="*50)
    print("READY: Ask me anything about the paper!")
    print("="*50)

    while True:
        user_query = input("\n[Q]: ")
        if user_query.lower() in ['exit', 'quit']: break

        # 1. Retrieval: Find the raw chunks
        print("Finding relevant snippets...")
        results, scores = search_in_db(user_query, model, index, chunks, k=3)

        # 2. Generation: Send to Gemini
        print("AI is thinking...")
        final_answer = get_gemini_response(user_query, results)

        print("\n" + "-"*20 + " AI ANSWER " + "-"*20)
        print(final_answer)
        print("-" * 51)

if __name__ == "__main__":
    main()