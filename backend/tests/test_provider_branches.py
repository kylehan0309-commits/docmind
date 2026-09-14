"""Unit tests for the per-provider dispatch in llm.py / graph_extraction.py /
vision.py - the branches that build the Claude / Gemini / Ollama SDK calls and
pull the answer back out. The SDK client objects are mocked (no network); the
real service functions run."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.config import settings
from app.services import provider
from app.services.graph_extraction import (
    ChunkExtraction,
    ConsolidationResult,
    consolidate_entities,
    extract_from_chunk,
)
from app.services.llm import SYSTEM_PROMPT, generate_answer, stream_answer
from app.services.vision import describe_page


def _force(monkeypatch, name: str):
    monkeypatch.setattr("app.services.provider._active", name)
    if name == "anthropic":
        monkeypatch.setattr("app.config.settings.anthropic_api_key", "sk-test")
    elif name == "gemini":
        monkeypatch.setattr("app.config.settings.gemini_api_key", "g-test")


def _chunk(text="chunk text", filename="d.pdf", page=1):
    return SimpleNamespace(
        document=SimpleNamespace(filename=filename), page_number=page, text=text
    )


def _text_blocks(*texts):
    return [SimpleNamespace(type="text", text=t) for t in texts]


@pytest.fixture
def fc(monkeypatch):
    """Fake Anthropic / Gemini / Ollama clients wired into every call site."""
    a, g, o = MagicMock(), MagicMock(), MagicMock()
    monkeypatch.setattr("app.services.provider._anthropic", a)
    monkeypatch.setattr("app.services.provider._gemini", g)
    for mod in ("app.services.llm", "app.services.graph_extraction", "app.services.vision"):
        monkeypatch.setattr(f"{mod}._get_client", lambda _o=o: _o)
    return SimpleNamespace(anthropic=a, gemini=g, ollama=o)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)


# --------------------------------------------------------------------------
# provider.py core
# --------------------------------------------------------------------------
def test_available_reflects_keys(monkeypatch):
    assert provider.available() == {"local": True, "anthropic": False, "gemini": False}
    monkeypatch.setattr("app.config.settings.anthropic_api_key", "x")
    monkeypatch.setattr("app.config.settings.gemini_api_key", "y")
    assert provider.available() == {"local": True, "anthropic": True, "gemini": True}


def test_get_provider_falls_back_to_local_without_key(monkeypatch):
    monkeypatch.setattr("app.services.provider._active", "gemini")
    assert provider.get_provider() == "local"
    monkeypatch.setattr("app.config.settings.gemini_api_key", "g-test")
    assert provider.get_provider() == "gemini"


# --------------------------------------------------------------------------
# generate_answer
# --------------------------------------------------------------------------
def test_generate_answer_anthropic(fc, monkeypatch):
    _force(monkeypatch, "anthropic")
    fc.anthropic.messages.create.return_value = SimpleNamespace(
        content=_text_blocks("Claude ", "answer")
    )

    out = generate_answer("what is X?", [_chunk()])
    assert out == "Claude answer"
    kw = fc.anthropic.messages.create.call_args.kwargs
    assert kw["model"] == settings.anthropic_model
    assert kw["system"] == SYSTEM_PROMPT
    assert "what is X?" in kw["messages"][0]["content"]


def test_generate_answer_gemini(fc, monkeypatch):
    _force(monkeypatch, "gemini")
    fc.gemini.models.generate_content.return_value = SimpleNamespace(text="  Gemini answer  ")

    out = generate_answer("what is X?", [_chunk()])
    assert out == "Gemini answer"
    kw = fc.gemini.models.generate_content.call_args.kwargs
    assert kw["model"] == settings.gemini_model
    assert kw["config"].system_instruction == SYSTEM_PROMPT
    assert "what is X?" in kw["contents"]


def test_generate_answer_local_ollama(fc, monkeypatch):
    _force(monkeypatch, "local")
    fc.ollama.chat.return_value = {"message": {"content": "ollama answer\n"}}

    out = generate_answer("q", [_chunk()])
    assert out == "ollama answer"
    kw = fc.ollama.chat.call_args.kwargs
    assert kw["model"] == settings.ollama_model
    assert kw["messages"][0]["role"] == "system"


def test_generate_answer_retries_transient_provider_error(fc, monkeypatch):
    _force(monkeypatch, "anthropic")
    attempts = []

    def flaky(**_kw):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("ServerError: 503 overloaded")
        return SimpleNamespace(content=_text_blocks("recovered"))

    fc.anthropic.messages.create.side_effect = flaky
    assert generate_answer("q", [_chunk()]) == "recovered"
    assert len(attempts) == 2


# --------------------------------------------------------------------------
# stream_answer
# --------------------------------------------------------------------------
def test_stream_answer_anthropic(fc, monkeypatch):
    _force(monkeypatch, "anthropic")
    cm = MagicMock()
    cm.__enter__.return_value = SimpleNamespace(text_stream=["Hel", "lo"])
    fc.anthropic.messages.stream.return_value = cm

    assert list(stream_answer("q", [_chunk()])) == ["Hel", "lo"]


def test_stream_answer_gemini_skips_empty_deltas(fc, monkeypatch):
    _force(monkeypatch, "gemini")
    fc.gemini.models.generate_content_stream.return_value = [
        SimpleNamespace(text="a"),
        SimpleNamespace(text=None),
        SimpleNamespace(text="b"),
    ]
    assert list(stream_answer("q", [_chunk()])) == ["a", "b"]


def test_stream_answer_local_skips_empty_deltas(fc, monkeypatch):
    _force(monkeypatch, "local")
    fc.ollama.chat.return_value = [
        {"message": {"content": "x"}},
        {"message": {"content": ""}},
        {"message": {"content": "y"}},
    ]
    assert list(stream_answer("q", [_chunk()])) == ["x", "y"]
    assert fc.ollama.chat.call_args.kwargs["stream"] is True


# --------------------------------------------------------------------------
# extract_from_chunk
# --------------------------------------------------------------------------
_SAMPLE = ChunkExtraction(entities=[], relationships=[])


def test_extract_from_chunk_anthropic(fc, monkeypatch):
    _force(monkeypatch, "anthropic")
    fc.anthropic.messages.parse.return_value = SimpleNamespace(parsed_output=_SAMPLE)

    assert extract_from_chunk("text") is _SAMPLE
    assert fc.anthropic.messages.parse.call_args.kwargs["output_format"] is ChunkExtraction


def test_extract_from_chunk_gemini_uses_parsed(fc, monkeypatch):
    _force(monkeypatch, "gemini")
    fc.gemini.models.generate_content.return_value = SimpleNamespace(parsed=_SAMPLE, text=None)
    assert extract_from_chunk("text") is _SAMPLE


def test_extract_from_chunk_gemini_falls_back_to_json_text(fc, monkeypatch):
    _force(monkeypatch, "gemini")
    fc.gemini.models.generate_content.return_value = SimpleNamespace(
        parsed=None, text='{"entities": [], "relationships": []}'
    )
    out = extract_from_chunk("text")
    assert isinstance(out, ChunkExtraction) and out.entities == []


def test_extract_from_chunk_local(fc, monkeypatch):
    _force(monkeypatch, "local")
    fc.ollama.chat.return_value = {"message": {"content": _SAMPLE.model_dump_json()}}
    out = extract_from_chunk("text")
    assert isinstance(out, ChunkExtraction)


def test_consolidate_entities_anthropic(fc, monkeypatch):
    _force(monkeypatch, "anthropic")
    result = ConsolidationResult(groups=[])
    fc.anthropic.messages.parse.return_value = SimpleNamespace(parsed_output=result)
    assert consolidate_entities([("id1", "A", "OTHER"), ("id2", "B", "OTHER")]) is result
    assert fc.anthropic.messages.parse.call_args.kwargs["output_format"] is ConsolidationResult


# --------------------------------------------------------------------------
# describe_page
# --------------------------------------------------------------------------
def test_describe_page_anthropic(fc, monkeypatch):
    _force(monkeypatch, "anthropic")
    fc.anthropic.messages.create.return_value = SimpleNamespace(
        content=_text_blocks("a bar chart of sales")
    )
    assert describe_page(b"\x89PNG...") == "a bar chart of sales"
    content = fc.anthropic.messages.create.call_args.kwargs["messages"][0]["content"]
    assert any(part["type"] == "image" for part in content)


def test_describe_page_gemini(fc, monkeypatch):
    _force(monkeypatch, "gemini")
    fc.gemini.models.generate_content.return_value = SimpleNamespace(text="a flowchart")
    assert describe_page(b"png") == "a flowchart"


def test_describe_page_local(fc, monkeypatch):
    _force(monkeypatch, "local")
    fc.ollama.chat.return_value = {"message": {"content": "a scatter plot"}}
    assert describe_page(b"png") == "a scatter plot"


def test_describe_page_returns_empty_for_none_or_tiny(fc, monkeypatch):
    _force(monkeypatch, "local")
    fc.ollama.chat.return_value = {"message": {"content": "NONE"}}
    assert describe_page(b"png") == ""
    fc.ollama.chat.return_value = {"message": {"content": "x"}}
    assert describe_page(b"png") == ""
