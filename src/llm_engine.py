# src/llm_engine.py
import google.generativeai as genai
import os

def get_gemini_response(query, context_chunks):
    """
    Send the query and retrieved context to Gemini to generate a coherent answer.
    """
    # English Comment: Configure the API key from environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables."

    genai.configure(api_key=api_key)
    
    # English Comment: Use the Gemini 2.0 Flash model (Fast and Free)
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
    except:
        # English Comment: Fallback to the known stable 2.0 model name if the shorthand fails
        model = genai.GenerativeModel('models/gemini-2.0-flash')
    # English Comment: Combine the retrieved chunks into a single context string
    context_text = "\n\n".join([f"Source {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])

    # English Comment: Craft the prompt (System Instruction + Context + Question)
    prompt = f"""
    You are a helpful research assistant. Answer the user's question based ONLY on the provided research paper snippets. 
    If the answer is not in the context, say that you don't know. 
    Keep the answer professional, concise, and structured.

    --- CONTEXT FROM PAPER ---
    {context_text}

    --- USER QUESTION ---
    {query}

    --- FINAL ANSWER (in Chinese) ---
    """

    # English Comment: Generate the response
    response = model.generate_content(prompt)
    return response.text