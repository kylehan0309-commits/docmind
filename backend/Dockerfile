# DocMind backend: FastAPI + local embeddings, optionally talking to Ollama /
# Claude / Gemini. Python 3.13 (3.14 has no pymupdf/torch wheels yet).
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    DATABASE_URL=sqlite+aiosqlite:////data/docmind.db \
    UPLOAD_DIR=/data/uploads

WORKDIR /app

# CPU-only torch first, from the PyTorch CPU index, so `pip install
# sentence-transformers` below doesn't drag in the ~2 GB CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the MiniLM embedding model into the image so the container runs
# offline and the first request isn't a download.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY app ./app

# SQLite db + uploaded PDFs live here; mount a volume to persist them.
RUN mkdir -p /data/uploads
VOLUME ["/data"]

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
