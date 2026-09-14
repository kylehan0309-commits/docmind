import pytest_asyncio

from app.models.document import Chunk, Document, Entity, EntityMention, Relationship

CHUNK_TEXT = (
    "DocMind is a retrieval system built with FastAPI and served locally. "
    "It stores chunks in SQLite. " + "filler " * 20 + "The core team works out of Rivertown."
)


@pytest_asyncio.fixture
async def seeded(db):
    doc = Document(filename="d.pdf", status="ready", page_count=1)
    db.add(doc)
    await db.flush()
    chunk = Chunk(
        document_id=doc.id, page_number=1, chunk_index=0, graph_extracted=True, text=CHUNK_TEXT
    )
    db.add(chunk)
    await db.flush()

    ids: dict[str, str] = {}
    for name, typ in [
        ("DocMind", "TECHNOLOGY"),
        ("FastAPI", "TECHNOLOGY"),
        ("Rivertown", "LOCATION"),
        ("Lonely", "OTHER"),  # no edges - must not appear in /graph
    ]:
        e = Entity(name=name, norm_name=name.lower(), type=typ)
        db.add(e)
        await db.flush()
        ids[name] = e.id

    db.add(
        Relationship(
            source_id=ids["DocMind"],
            target_id=ids["FastAPI"],
            label="built with",
            chunk_id=chunk.id,
            document_id=doc.id,
        )
    )
    for name in ("DocMind", "FastAPI", "Rivertown"):
        db.add(
            EntityMention(
                entity_id=ids[name], chunk_id=chunk.id, document_id=doc.id, surface_form=name
            )
        )
    await db.commit()
    return {"doc_id": doc.id, "ids": ids}


async def test_get_graph_only_returns_connected_entities(client, seeded):
    g = (await client.get("/graph")).json()
    assert sorted(n["name"] for n in g["nodes"]) == ["DocMind", "FastAPI"]
    assert len(g["edges"]) == 1
    assert g["edges"][0]["label"] == "built with"


async def test_coverage(client, seeded):
    cov = (await client.get("/graph/coverage")).json()
    assert cov == [
        {"document_id": seeded["doc_id"], "filename": "d.pdf", "total_chunks": 1, "extracted_chunks": 1}
    ]


async def test_entity_detail(client, seeded):
    d = (await client.get(f"/graph/entities/{seeded['ids']['DocMind']}")).json()
    assert d["name"] == "DocMind"
    assert d["degree"] == 1
    assert len(d["relationships"]) == 1
    rel = d["relationships"][0]
    assert rel["direction"] == "out"
    assert rel["other_name"] == "FastAPI"
    assert rel["filename"] == "d.pdf"
    assert rel["page_number"] == 1
    assert len(d["mentions"]) == 1


async def test_entity_detail_404(client, seeded):
    assert (await client.get("/graph/entities/nope")).status_code == 404


async def test_search_by_name_and_connected_entity(client, seeded):
    hits = (await client.get("/graph/search?q=fastapi")).json()
    by_name = {h["name"]: h for h in hits}
    assert by_name["FastAPI"]["match_field"] == "name"
    # DocMind surfaces because it's connected to FastAPI
    assert by_name["DocMind"]["match_field"] == "relationship"


async def test_search_by_type(client, seeded):
    hits = (await client.get("/graph/search?q=location")).json()
    assert [h["name"] for h in hits] == ["Rivertown"]
    assert hits[0]["match_field"] == "type"


async def test_search_by_relationship_label(client, seeded):
    names = {h["name"] for h in (await client.get("/graph/search?q=built with")).json()}
    assert names == {"DocMind", "FastAPI"}


async def test_search_ranks_name_before_relationship(client, seeded):
    hits = (await client.get("/graph/search?q=DocMind")).json()
    assert hits[0]["name"] == "DocMind"
    assert hits[0]["match_field"] == "name"


async def test_search_no_match_is_empty(client, seeded):
    assert (await client.get("/graph/search?q=zzznotpresent")).json() == []
