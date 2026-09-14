import pytest
from sqlalchemy import func, select

from app.demo_seed import DEMO_FILENAME, seed_demo
from app.models.document import Chunk, Document, Entity, EntityMention, Relationship

pytestmark = pytest.mark.usefixtures("patch_db", "fake_embeddings")


async def _count(db, model) -> int:
    return (await db.execute(select(func.count()).select_from(model))).scalar_one()


async def test_seed_demo_populates_document_and_graph(db):
    await seed_demo()

    docs = (await db.execute(select(Document))).scalars().all()
    assert [d.filename for d in docs] == [DEMO_FILENAME]

    assert await _count(db, Entity) == 12
    assert await _count(db, Relationship) == 14
    assert await _count(db, EntityMention) == 13

    chunks = (await db.execute(select(Chunk))).scalars().all()
    assert len(chunks) == 2
    assert all(c.graph_extracted for c in chunks)
    assert all(c.embedding is not None for c in chunks)


async def test_seed_demo_is_idempotent(db):
    await seed_demo()
    await seed_demo()
    assert await _count(db, Document) == 1
    assert await _count(db, Entity) == 12
