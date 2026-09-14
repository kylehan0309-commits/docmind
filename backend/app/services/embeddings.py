"""
Local embedding service.

Uses sentence-transformers (all-MiniLM-L6-v2 by default) to turn chunk / query
text into vectors. Runs on CPU, no API keys. The model is downloaded on first
use (~90MB) and cached under ~/.cache/huggingface.

Vectors are L2-normalized, so cosine similarity between two vectors is just their
dot product (see services/retrieval.py).
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of strings -> (n, dim) float32 array of normalized vectors."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vecs.astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string -> (dim,) float32 normalized vector."""
    return embed_texts([text])[0]


def to_bytes(vec: np.ndarray) -> bytes:
    """Serialize a vector for storage in the Chunk.embedding BLOB column."""
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_bytes(blob: bytes) -> np.ndarray:
    """Inverse of to_bytes()."""
    return np.frombuffer(blob, dtype=np.float32)
