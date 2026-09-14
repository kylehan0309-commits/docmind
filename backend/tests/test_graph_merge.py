import pytest_asyncio
from sqlalchemy import select

from app.models.document import Chunk, Document, Entity, EntityMention, Relationship


@pytest_asyncio.fixture
async def three_entities(db):
    doc = Document(filename="d.pdf", status="ready", page_count=1)
    db.add(doc)
    await db.flush()
    chunk = Chunk(document_id=doc.id, page_number=1, chunk_index=0, text="x", graph_extracted=True)
    db.add(chunk)
    await db.flush()

    ids: dict[str, str] = {}
    for name in ("A", "B", "C"):
        e = Entity(name=name, norm_name=name.lower(), type="OTHER")
        db.add(e)
        await db.flush()
        ids[name] = e.id

    def rel(s, t, label):
        return Relationship(
            source_id=ids[s], target_id=ids[t], label=label,
            chunk_id=chunk.id, document_id=doc.id,
        )

    db.add_all([
        rel("A", "C", "points at"),   # after merge B->A: A points at C
        rel("B", "C", "points at"),   #   ... becomes a duplicate of the above
        rel("A", "B", "knows"),       #   ... becomes a self-loop A->A, dropped
    ])
    db.add(EntityMention(entity_id=ids["B"], chunk_id=chunk.id, document_id=doc.id, surface_form="B"))
    await db.commit()
    return ids


async def test_merge_repoints_edges_drops_selfloops_and_dupes(client, db, three_entities):
    ids = three_entities
    resp = await client.post(
        "/graph/entities/merge", json={"keep_id": ids["A"], "merge_ids": [ids["B"]]}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["kept"] == ids["A"] and body["merged"] == [ids["B"]]
    assert body["removed_duplicate_edges"] >= 1

    entities = {e.name for e in (await db.execute(select(Entity))).scalars().all()}
    assert entities == {"A", "C"}  # B is gone

    rels = (await db.execute(select(Relationship))).scalars().all()
    assert all(r.source_id != r.target_id for r in rels)  # no self-loop
    assert [(r.source_id, r.target_id, r.label) for r in rels] == [(ids["A"], ids["C"], "points at")]

    mention = (await db.execute(select(EntityMention))).scalars().one()
    assert mention.entity_id == ids["A"]  # B's mention re-pointed


async def test_merge_rejects_empty_merge_ids(client, three_entities):
    ids = three_entities
    resp = await client.post(
        "/graph/entities/merge", json={"keep_id": ids["A"], "merge_ids": [ids["A"]]}
    )
    assert resp.status_code == 400


async def test_merge_rejects_unknown_id(client, three_entities):
    ids = three_entities
    resp = await client.post(
        "/graph/entities/merge", json={"keep_id": ids["A"], "merge_ids": ["ghost"]}
    )
    assert resp.status_code == 404
