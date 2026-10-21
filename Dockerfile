# Dockerfile - the web interface (server.py) as a Hugging Face Docker Space; see README.md, "Hugging Face Space"
FROM python:3.13-slim

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
# Created by the user (WORKDIR alone may create it as root), so server.py can write data/ in it
RUN mkdir -p $HOME/app
WORKDIR $HOME/app

# CPU-only PyTorch first: the default wheel brings CUDA libraries the Space has no use for (gigabytes)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Put the embedding and reranker models in the image, so a restart does not download them again
COPY --chown=user src/ src/
RUN python -c "from src.embedder import get_embedding_model; from src.retriever import get_reranker; \
get_embedding_model(); get_reranker()"
ENV HF_HUB_OFFLINE=1

COPY --chown=user . .

# The API key comes from the Space's secrets (ANTHROPIC_API_KEY), never from this file
EXPOSE 7860
CMD ["python", "server.py", "--public", "--port", "7860"]
