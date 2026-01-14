IEEE Document Knowledge Base & RAG System
An intelligent assistant that transforms complex academic papers into a searchable knowledge base for instant, direct answers.

Overview
This project is a high-performance RAG (Retrieval-Augmented Generation) pipeline specifically optimized for academic research papers. It overcomes the common "cross-column reading" issue in IEEE-style PDFs, enabling accurate information retrieval and synthesis using state-of-the-art AI models.


Key Features

Coordinate-Aware PDF Parsing: Reconstructs logical reading order from double-column IEEE layouts using coordinate-based heuristic sorting.

Semantic Search Engine: Utilizes FAISS and Sentence-Transformers (all-MiniLM-L6-v2) for sub-second semantic retrieval.Dynamic Semantic Chunking: Implements sentence-level overlap to maintain contextual integrity across text fragments.

Gemini 2.0 Integration: Leverages Google Gemini 2.0 Flash to synthesize professional, context-aware answers in Chinese or English.

Persistent Vector Store: Local caching of embeddings in .faiss and .pkl formats to eliminate redundant API costs and computation.

Tech Stack
Language: Python

Vector DB: FAISS (Facebook AI Similarity Search)

LLM API: Google Gemini 2.0 API

Embedding Model: Sentence-Transformers

PDF Processing: PyMuPDF (fitz)


How It Works

Ingestion: The system parses the PDF and sorts text blocks by their (x, y) coordinates to handle multi-column formats.Embedding: Text is split into chunks and converted into 384-dimensional vectors.
Indexing: Vectors are stored in a FAISS index for efficient similarity searching.
Querying: When a user asks a question, the system finds the most relevant snippets and sends them to Gemini 2.0 to generate a final answer.