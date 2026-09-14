import numpy as np
import pytest

from app.routers.graph import _excerpt
from app.services.graph_extraction import (
    differ_only_by_number,
    normalize_name,
    resolve_entity,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("OpenAI, Inc.", "openai inc"),
        ("  The  DATA  ", "the data"),
        ("PyMuPDF", "pymupdf"),
        ("GPT-4", "gpt 4"),
        ("all-MiniLM-L6-v2", "all minilm l6 v2"),
    ],
)
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("GPT-4", "GPT-3", True),
        ("llama3.1", "llama3.2", True),
        ("Windows 10", "Windows 11", True),
        ("Q3 revenue", "Q4 revenue", True),
        ("OpenAI", "OpenAI, Inc.", False),
        ("Northfield University", "the Northfield University", False),
        ("DocMind", "DocMind", False),  # no differing tokens at all
    ],
)
def test_differ_only_by_number(a, b, expected):
    assert differ_only_by_number(a, b) is expected


def _unit(*xs) -> np.ndarray:
    v = np.array(xs, dtype="float32")
    return v / np.linalg.norm(v)


def test_resolve_entity_matches_close_same_type_vector():
    v = _unit(1, 0, 0)
    known = [("e1", "ORGANIZATION", "OpenAI", v)]
    assert resolve_entity("OpenAI Inc", v.copy(), "ORGANIZATION", known, 0.87) == "e1"


def test_resolve_entity_rejects_different_type():
    v = _unit(1, 0, 0)
    known = [("e1", "ORGANIZATION", "OpenAI", v)]
    assert resolve_entity("OpenAI Inc", v.copy(), "PERSON", known, 0.87) is None


def test_resolve_entity_rejects_below_threshold():
    known = [("e1", "CONCEPT", "alpha", _unit(1, 0, 0))]
    assert resolve_entity("beta", _unit(0, 1, 0), "CONCEPT", known, 0.87) is None


def test_resolve_entity_never_merges_numeric_siblings():
    v = _unit(1, 0, 0)
    known = [("e1", "TECHNOLOGY", "GPT-4", v)]
    # identical vector, same type, but names differ only by a number
    assert resolve_entity("GPT-3", v.copy(), "TECHNOLOGY", known, 0.87) is None


def test_excerpt_truncates_long_text():
    out = _excerpt("x" * 400, width=260)
    assert out.endswith("…") and len(out) == 261


def test_excerpt_centres_on_the_term():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    out = _excerpt(text, "epsilon", width=20)
    assert "epsilon" in out
    assert len(out) <= 22  # width + up to two ellipses
