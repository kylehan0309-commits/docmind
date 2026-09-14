import pytest

pytestmark = pytest.mark.usefixtures("fake_embeddings", "mock_llm")


async def _upload(client, make_pdf):
    files = {"file": ("doc.pdf", make_pdf(), "application/pdf")}
    return (await client.post("/documents/upload", files=files)).json()["id"]


async def test_chat_rejects_empty_question(client):
    resp = await client.post("/chat", json={"question": "   "})
    assert resp.status_code == 400


async def test_chat_404_when_no_documents(client):
    resp = await client.post("/chat", json={"question": "anything?"})
    assert resp.status_code == 404


async def test_chat_returns_answer_and_citation_shape(client, make_pdf):
    await _upload(client, make_pdf)

    resp = await client.post("/chat", json={"question": "what is this about?"})
    assert resp.status_code == 200
    body = resp.json()

    assert "Stub answer" in body["answer"]
    assert isinstance(body["citations"], list) and body["citations"]
    for c in body["citations"]:
        assert c.keys() >= {
            "document_id",
            "filename",
            "page_number",
            "chunk_index",
            "score",
            "text",
        }


async def test_chat_respects_top_k(client, make_pdf):
    await _upload(client, make_pdf)
    resp = await client.post("/chat", json={"question": "q", "top_k": 1})
    assert resp.status_code == 200
    assert len(resp.json()["citations"]) <= 1
