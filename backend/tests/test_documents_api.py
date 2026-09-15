import pytest
from sqlalchemy import func, select

from app.models.document import Chunk, Document

pytestmark = pytest.mark.usefixtures("fake_embeddings")


async def _upload(client, make_pdf, **kw):
    files = {"file": ("doc.pdf", make_pdf(**kw), "application/pdf")}
    return await client.post("/documents/upload", files=files)


async def test_upload_returns_immediately_then_processes_in_the_background(
    client, make_pdf, db, finish_upload
):
    resp = await _upload(client, make_pdf, pages=2)
    assert resp.status_code == 200
    body = resp.json()
    # parsing/chunking/embedding hasn't run yet - the request didn't wait for it
    assert body["status"] == "processing"
    assert body["page_count"] == 0

    await finish_upload(body["id"])

    doc = await db.get(Document, body["id"])
    assert doc.status == "ready"
    assert doc.page_count == 2
    assert doc.error is None

    rows = (
        await db.execute(select(Chunk).where(Chunk.document_id == body["id"]))
    ).scalars().all()
    assert len(rows) >= 1
    assert all(c.embedding is not None for c in rows)


async def test_upload_failure_is_recorded_on_the_document(
    client, make_pdf, db, finish_upload, monkeypatch
):
    def boom(*_a, **_kw):
        raise RuntimeError("simulated parse failure")

    monkeypatch.setattr("app.routers.documents.extract_pages", boom)

    resp = await _upload(client, make_pdf)
    doc_id = resp.json()["id"]
    await finish_upload(doc_id)

    doc = await db.get(Document, doc_id)
    assert doc.status == "failed"
    assert "simulated parse failure" in doc.error


async def test_upload_rejects_non_pdf(client):
    files = {"file": ("notes.txt", b"plain text", "text/plain")}
    resp = await client.post("/documents/upload", files=files)
    assert resp.status_code == 400


async def test_list_documents(client, make_pdf, finish_upload):
    a = (await _upload(client, make_pdf)).json()["id"]
    b = (await _upload(client, make_pdf)).json()["id"]
    await finish_upload(a)
    await finish_upload(b)
    resp = await client.get("/documents/")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


async def test_delete_removes_document_and_chunks(client, make_pdf, db, finish_upload):
    doc_id = (await _upload(client, make_pdf)).json()["id"]
    await finish_upload(doc_id)

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


async def test_delete_cancels_an_in_flight_upload(client, make_pdf, monkeypatch):
    """Deleting a document mid-parse must not race the background task into
    committing chunks for a document that's about to be gone."""
    import time

    from app.routers import documents

    def slow_extract(*_a, **_kw):
        time.sleep(0.3)  # guarantees the task is still running when we delete
        return [{"page_number": 1, "text": "slow page"}]

    monkeypatch.setattr("app.routers.documents.extract_pages", slow_extract)

    doc_id = (await _upload(client, make_pdf)).json()["id"]
    assert doc_id in documents._upload_tasks
    assert not documents._upload_tasks[doc_id].done()

    resp = await client.delete(f"/documents/{doc_id}")
    assert resp.status_code == 200
    assert doc_id not in documents._upload_tasks

    assert (await client.get("/documents/")).json() == []
