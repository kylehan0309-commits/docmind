import json

import pytest

pytestmark = pytest.mark.usefixtures("fake_embeddings")


async def _upload(client, make_pdf):
    files = {"file": ("doc.pdf", make_pdf(), "application/pdf")}
    return (await client.post("/documents/upload", files=files)).json()["id"]


def _parse_sse(text: str) -> list[dict]:
    frames = []
    for raw in text.strip().split("\n\n"):
        if not raw.strip():
            continue
        lines = raw.split("\n")
        event = next(l[len("event: "):] for l in lines if l.startswith("event: "))
        data = next(l[len("data: "):] for l in lines if l.startswith("data: "))
        frames.append({"event": event, "data": json.loads(data)})
    return frames


async def test_stream_rejects_empty_question(client):
    resp = await client.post("/chat/stream", json={"question": "   "})
    assert resp.status_code == 400


async def test_stream_404_when_no_documents(client):
    resp = await client.post("/chat/stream", json={"question": "anything?"})
    assert resp.status_code == 404


async def test_stream_sends_citations_then_tokens_then_done(client, make_pdf, monkeypatch):
    await _upload(client, make_pdf)

    def fake_stream(question, chunks):
        yield "Hello"
        yield " "
        yield "world"

    monkeypatch.setattr("app.routers.chat.stream_answer", fake_stream)

    resp = await client.post("/chat/stream", json={"question": "what is this?"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    frames = _parse_sse(resp.text)
    assert frames[0]["event"] == "citations"
    assert isinstance(frames[0]["data"]["citations"], list) and frames[0]["data"]["citations"]

    token_texts = [f["data"]["text"] for f in frames if f["event"] == "token"]
    assert token_texts == ["Hello", " ", "world"]

    assert frames[-1]["event"] == "done"


async def test_stream_surfaces_error_after_partial_tokens(client, make_pdf, monkeypatch):
    await _upload(client, make_pdf)

    def fake_stream(question, chunks):
        yield "partial answer"
        raise RuntimeError("simulated provider failure")

    monkeypatch.setattr("app.routers.chat.stream_answer", fake_stream)

    resp = await client.post("/chat/stream", json={"question": "q"})
    assert resp.status_code == 200  # headers were already sent before the failure

    frames = _parse_sse(resp.text)
    assert frames[0]["event"] == "citations"
    assert frames[1] == {"event": "token", "data": {"text": "partial answer"}}
    assert frames[-1]["event"] == "error"
    assert "simulated provider failure" in frames[-1]["data"]["detail"]
    assert not any(f["event"] == "done" for f in frames)


async def test_stream_fails_before_any_tokens(client, make_pdf, monkeypatch):
    await _upload(client, make_pdf)

    def fake_stream(question, chunks):
        if True:  # noqa: SIM108 - keep this an obvious "raises before yielding" generator
            raise RuntimeError("boom before first token")
        yield "unreachable"

    monkeypatch.setattr("app.routers.chat.stream_answer", fake_stream)

    resp = await client.post("/chat/stream", json={"question": "q"})
    frames = _parse_sse(resp.text)
    assert frames[0]["event"] == "citations"
    assert frames[1]["event"] == "error"
    assert "boom before first token" in frames[1]["data"]["detail"]


async def test_stream_respects_top_k(client, make_pdf, monkeypatch):
    await _upload(client, make_pdf)
    monkeypatch.setattr("app.routers.chat.stream_answer", lambda question, chunks: iter(["ok"]))

    resp = await client.post("/chat/stream", json={"question": "q", "top_k": 1})
    frames = _parse_sse(resp.text)
    assert len(frames[0]["data"]["citations"]) <= 1
