import pytest_asyncio
from sqlalchemy import select

from app.models.document import Chunk, Document, Entity, EntityMention, Relationship


@pytest_asyncio.fixture
async def scenario(db):
    """A, B, C entities. B has a mention and two relationships:
    - B -> C label "rel"   (will collide with A -> C label "rel" after merge -> duplicate, deleted)
    - D -> B label "knows" (will become D -> A after merge - not a self-loop)
    - A -> B label "owns"  (will become a self-loop A -> A after merge - deleted)
    A -> C label "rel" already exists, so merging B into A creates the duplicate case.
    """
    doc = Document(filename="d.pdf", status="ready", page_count=1)
    db.add(doc)
    await db.flush()
    chunk = Chunk(document_id=doc.id, page_number=1, chunk_index=0, text="x", graph_extracted=True)
    db.add(chunk)
    await db.flush()

    ids: dict[str, str] = {}
    for name in ("A", "B", "C", "D"):
        e = Entity(name=name, norm_name=name.lower(), type="OTHER")
        db.add(e)
        await db.flush()
        ids[name] = e.id

    def rel(s, t, label):
        return Relationship(
            source_id=ids[s], target_id=ids[t], label=label, chunk_id=chunk.id, document_id=doc.id
        )

    db.add_all([
        rel("A", "C", "rel"),
        rel("B", "C", "rel"),
        rel("D", "B", "knows"),
        rel("A", "B", "owns"),
    ])
    db.add(EntityMention(entity_id=ids["B"], chunk_id=chunk.id, document_id=doc.id, surface_form="B"))
    await db.commit()
    return {"doc_id": doc.id, "chunk_id": chunk.id, **ids}


async def test_merge_response_includes_merge_id(client, scenario):
    resp = await client.post(
        "/graph/entities/merge", json={"keep_id": scenario["A"], "merge_ids": [scenario["B"]]}
    )
    assert resp.status_code == 200
    assert resp.json()["merge_id"]


async def test_undo_restores_entity_relationships_and_mention(client, db, scenario):
    resp = await client.post(
        "/graph/entities/merge", json={"keep_id": scenario["A"], "merge_ids": [scenario["B"]]}
    )
    merge_id = resp.json()["merge_id"]

    # sanity: B is gone, and we lost a relationship to the self-loop drop + the duplicate collapse
    assert (await db.execute(select(Entity).where(Entity.id == scenario["B"]))).scalar_one_or_none() is None
    remaining = len((await db.execute(select(Relationship))).scalars().all())
    assert remaining == 2  # A->C survives (dup collapsed), D->A survives; B->C dup and A->A self-loop gone

    undo_resp = await client.post(f"/graph/merges/{merge_id}/undo")
    assert undo_resp.status_code == 200
    assert undo_resp.json()["restored_entities"] == 1

    b = (await db.execute(select(Entity).where(Entity.id == scenario["B"]))).scalar_one()
    assert b.name == "B" and b.type == "OTHER"

    rels = {
        (r.source_id, r.target_id, r.label)
        for r in (await db.execute(select(Relationship))).scalars().all()
    }
    assert rels == {
        (scenario["A"], scenario["C"], "rel"),
        (scenario["B"], scenario["C"], "rel"),
        (scenario["D"], scenario["B"], "knows"),
        (scenario["A"], scenario["B"], "owns"),
    }

    mention = (await db.execute(select(EntityMention))).scalars().one()
    assert mention.entity_id == scenario["B"]


async def test_undo_twice_is_409(client, scenario):
    merge_id = (
        await client.post(
            "/graph/entities/merge", json={"keep_id": scenario["A"], "merge_ids": [scenario["B"]]}
        )
    ).json()["merge_id"]

    assert (await client.post(f"/graph/merges/{merge_id}/undo")).status_code == 200
    assert (await client.post(f"/graph/merges/{merge_id}/undo")).status_code == 409


async def test_undo_unknown_id_is_404(client):
    resp = await client.post("/graph/merges/nope/undo")
    assert resp.status_code == 404


async def test_undo_refused_once_keep_entity_is_itself_merged_away(client, scenario):
    first_merge_id = (
        await client.post(
            "/graph/entities/merge", json={"keep_id": scenario["A"], "merge_ids": [scenario["B"]]}
        )
    ).json()["merge_id"]

    # now merge A (the keeper of the first merge) into C
    await client.post("/graph/entities/merge", json={"keep_id": scenario["C"], "merge_ids": [scenario["A"]]})

    resp = await client.post(f"/graph/merges/{first_merge_id}/undo")
    assert resp.status_code == 409


async def test_list_merges_reflects_undo_state(client, scenario):
    merge_id = (
        await client.post(
            "/graph/entities/merge", json={"keep_id": scenario["A"], "merge_ids": [scenario["B"]]}
        )
    ).json()["merge_id"]

    listing = (await client.get("/graph/merges")).json()
    assert len(listing) == 1
    assert listing[0]["id"] == merge_id
    assert listing[0]["keep_name"] == "A"
    assert listing[0]["merged"] == [{"id": scenario["B"], "name": "B"}]
    assert listing[0]["undone"] is False

    await client.post(f"/graph/merges/{merge_id}/undo")
    listing = (await client.get("/graph/merges")).json()
    assert listing[0]["undone"] is True
