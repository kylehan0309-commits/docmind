"""
Brute-force cosine-similarity retrieval over stored chunk embeddings.

Fine for an MVP with a handful of documents (hundreds–low thousands of chunks):
one matrix-vector product per query, sub-millisecond. Swap in a vector DB
(Chroma / pgvector / sqlite-vec) here if the corpus grows.
"""
import numpy as np

from app.models.document import Chunk
from app.services.embeddings import from_bytes


def rank_chunks(
    query_vec: np.ndarray, chunks: list[Chunk], top_k: int
) -> list[tuple[Chunk, float]]:
    """Return the top_k (chunk, score) pairs by cosine similarity, best first.

    Embedding-service vectors are L2-normalized, so cosine similarity is just a
    dot product. Chunks with no stored embedding are ignored.
    """
    embedded = [(c, from_bytes(c.embedding)) for c in chunks if c.embedding is not None]
    if not embedded:
        return []

    matrix = np.vstack([vec for _, vec in embedded])
    scores = matrix @ query_vec

    ranked = [(chunk, float(score)) for (chunk, _), score in zip(embedded, scores)]
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]
