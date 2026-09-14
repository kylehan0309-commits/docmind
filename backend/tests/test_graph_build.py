import pytest
from sqlalchemy import func, select

from app.models.document import Chunk, Document, Entity, Relationship
from app.services.graph_extraction import (
    ChunkExtraction,
    ExtractedEntity,
    ExtractedRelationship,
)

pytestmark = pytest.mark.usefixtures("fake_embeddings")


def _extraction(entities: list[tuple[str, str]], rels: list[tuple[str, str, str]]) -> ChunkExtraction:
    return ChunkExtraction(
        entities=[ExtractedEntity(name=n, type=t) for n, t in entities],
        relationships=[ExtractedRelationship(source=s, target=t, label=lbl) for s, t, lbl in rels],
    )


async def _seed_doc(db, texts: list[str], filename: str = "t.pdf") -> str:
    doc = Document(filename=filename, status="ready", page_count=len(texts))
    db.add(doc)
    await db.flush()
    for i, text in enumerate(texts):
        db.add(
            Chunk(
                document_id=doc.id,
                page_number=i + 1,
                chunk_index=i,
                text=text,
                graph_extracted=False,
            )
        )
    await db.commit()
    return doc.id


async def test_build_creates_entities_relationships_and_dedupes(
    client, db, monkeypatch, finish_build
):
    doc_id = await _seed_doc(db, ["chunk ONE", "chunk TWO"])

    def fake_extract(text):
        if "ONE" in text:
            return _extraction(
                [("DocMind", "TECHNOLOGY"), ("FastAPI", "TECHNOLOGY")],
                [("DocMind", "FastAPI", "built with")],
            )
        return _extraction(
            [("DocMind", "TECHNOLOGY"), ("SQLite", "TECHNOLOGY")],
            [("DocMind", "SQLite", "stores in")],
        )

    monkeypatch.setattr("app.routers.graph.extract_from_chunk", fake_extract)

    r = await client.post(f"/graph/build?document_ids={doc_id}")
    assert r.json() == {"status": "started", "total_chunks": 2}
    await finish_build()

    status = (await client.get("/graph/build/status")).json()
    assert status["processed"] == 2 and status["failed"] == 0 and status["error"] is None

    graph = (await client.get("/graph")).json()
    assert sorted(n["name"] for n in graph["nodes"]) == ["DocMind", "FastAPI", "SQLite"]
    assert len(graph["edges"]) == 2  # DocMind appeared in both chunks -> one node

    # every chunk marked done
    pending = (
        await db.execute(
            select(func.count()).select_from(Chunk).where(Chunk.graph_extracted.is_(False))
        )
    ).scalar_one()
    assert pending == 0


async def test_build_tolerates_a_failing_chunk(client, db, monkeypatch, finish_build):
    doc_id = await _seed_doc(db, ["good chunk", "BOOM chunk"])

    def fake_extract(text):
        if "BOOM" in text:
            raise RuntimeError("simulated provider 503")
        return _extraction([("Kept", "OTHER")], [])

    monkeypatch.setattr("app.routers.graph.extract_from_chunk", fake_extract)

    await client.post(f"/graph/build?document_ids={doc_id}")
    await finish_build()

    status = (await client.get("/graph/build/status")).json()
    assert status["processed"] == 1
    assert status["failed"] == 1
    assert "simulated provider 503" in status["last_error"]
    assert status["error"] is None  # the run itself didn't crash

    cov = {c["filename"]: c for c in (await client.get("/graph/coverage")).json()}
    assert cov["t.pdf"]["extracted_chunks"] == 1  # the failed chunk stays unmapped


async def test_build_is_incremental_across_documents(client, db, monkeypatch, finish_build):
    a = await _seed_doc(db, ["alpha"], filename="a.pdf")
    b = await _seed_doc(db, ["beta"], filename="b.pdf")

    calls: list[str] = []

    def fake_extract(text):
        calls.append(text)
        return _extraction([("E", "OTHER")], [])

    monkeypatch.setattr("app.routers.graph.extract_from_chunk", fake_extract)

    await client.post(f"/graph/build?document_ids={a}")
    await finish_build()
    assert calls == ["alpha"]

    # build "everything" - a.pdf is already done, so only b.pdf's chunk runs
    calls.clear()
    await client.post("/graph/build")
    await finish_build()
    assert calls == ["beta"]

    # nothing left
    r = await client.post("/graph/build")
    assert r.json()["status"] == "nothing_to_do"


async def test_force_remaps_and_orphan_cleans(client, db, monkeypatch, finish_build):
    doc_id = await _seed_doc(db, ["only chunk"])

    monkeypatch.setattr(
        "app.routers.graph.extract_from_chunk",
        lambda text: _extraction(
            [("Old", "OTHER"), ("Shared", "OTHER")], [("Old", "Shared", "rel")]
        ),
    )
    await client.post(f"/graph/build?document_ids={doc_id}")
    await finish_build()
    assert {n["name"] for n in (await client.get("/graph")).json()["nodes"]} == {"Old", "Shared"}

    # re-map with a different extraction
    monkeypatch.setattr(
        "app.routers.graph.extract_from_chunk",
        lambda text: _extraction(
            [("New", "OTHER"), ("Shared", "OTHER")], [("New", "Shared", "rel")]
        ),
    )
    r = await client.post(f"/graph/build?document_ids={doc_id}&force=true")
    assert r.json()["status"] == "started"
    await finish_build()

    names = {e.name for e in (await db.execute(select(Entity))).scalars().all()}
    assert names == {"New", "Shared"}  # "Old" had no refs left -> deleted


async def test_force_without_document_ids_is_400(client):
    assert (await client.post("/graph/build?force=true")).status_code == 400


async def test_build_conflict_when_already_running(client, monkeypatch):
    from app.routers import graph

    monkeypatch.setitem(graph._build, "running", True)
    assert (await client.post("/graph/build")).status_code == 409
