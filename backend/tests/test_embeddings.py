import numpy as np
import pytest

from app.services.embeddings import from_bytes, to_bytes


def test_to_bytes_from_bytes_roundtrip():
    v = np.random.default_rng(0).random(384).astype("float32")
    out = from_bytes(to_bytes(v))
    assert out.dtype == np.float32
    assert np.array_equal(out, v)


def test_to_bytes_length_is_four_bytes_per_dim():
    v = np.zeros(384, dtype="float32")
    assert len(to_bytes(v)) == 384 * 4


def test_to_bytes_accepts_non_float32_input():
    v = np.arange(10, dtype="float64")
    assert from_bytes(to_bytes(v)).tolist() == v.astype("float32").tolist()


@pytest.mark.slow
def test_real_model_embeds_and_ranks_by_meaning():
    from app.services.embeddings import embed_query, embed_texts

    vecs = embed_texts(["a small kitten", "a large truck"])
    assert vecs.shape == (2, 384)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)

    q = embed_query("a tiny cat")
    sims = vecs @ q
    assert sims[0] > sims[1]  # kitten closer to cat than truck is
