import pytest
from sqlalchemy import func, select

from app.models.document import Chunk, Document

pytestmark = pytest.mark.usefixtures("fake_embeddings")


async def _upload(client, make_pdf, **kw):
    files = {"file": ("doc.pdf", make_pdf(**kw), "application/pdf")}
    return await client.post("/documents/upload", files=files)


async def test_upload_parses_chunks_and_embeds(client, make_pdf, db):
    resp = await _upload(client, make_pdf, pages=2)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["page_count"] == 2

    rows = (
        await db.execute(select(Chunk).where(Chunk.document_id == body["id"]))
    ).scalars().all()
    assert len(rows) >= 1
    assert all(c.embedding is not None for c in rows)


async def test_upload_rejects_non_pdf(client):
    files = {"file": ("notes.txt", b"plain text", "text/plain")}
    resp = await client.post("/documents/upload", files=files)
    assert resp.status_code == 400


async def test_list_documents(client, make_pdf):
    await _upload(client, make_pdf)
    await _upload(client, make_pdf)
    resp = await client.get("/documents/")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


async def test_delete_removes_document_and_chunks(client, make_pdf, db):
    doc_id = (await _upload(client, make_pdf)).json()["id"]

    resp = await client.delete(f"/documents/{doc_id}")
    assert resp.status_code == 200
    assert resp.json() == {"deleted": doc_id}

    assert (await client.get("/documents/")).json() == []
    remaining = (
        await db.execute(select(func.count()).select_from(Chunk).where(Chunk.document_id == doc_id))
    ).scalar_one()
    assert remaining == 0


async def test_delete_missing_document_is_404(client):
    assert (await client.delete("/documents/nope")).status_code == 404
