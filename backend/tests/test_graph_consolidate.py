from sqlalchemy import select

from app.models.document import Entity
from app.services.graph_extraction import ConsolidationGroup, ConsolidationResult


async def _make_entities(db, specs: list[tuple[str, str]]) -> dict[str, str]:
    """specs: list of (name, type). Returns name -> id."""
    ids: dict[str, str] = {}
    for name, typ in specs:
        e = Entity(name=name, norm_name=name.lower(), type=typ)
        db.add(e)
        await db.flush()
        ids[name] = e.id
    await db.commit()
    return ids


def _mock_result(client_monkeypatch, groups: list[ConsolidationGroup]):
    client_monkeypatch.setattr(
        "app.routers.graph.consolidate_entities",
        lambda entities: ConsolidationResult(groups=groups),
    )


def _merged_names(body: dict) -> set[str]:
    return {m["name"] for g in body["groups"] for m in g["merged"]}


async def test_noop_with_fewer_than_two_entities(client, db):
    await _make_entities(db, [("Solo", "OTHER")])
    resp = await client.post("/graph/consolidate")
    assert resp.status_code == 200
    assert resp.json() == {"provider": "local", "groups": [], "removed_duplicate_edges": 0}


async def test_valid_group_is_merged(client, db, monkeypatch):
    ids = await _make_entities(db, [("RAG", "CONCEPT"), ("Retrieval-Augmented Generation", "CONCEPT")])
    _mock_result(
        monkeypatch,
        [ConsolidationGroup(keep_id=ids["Retrieval-Augmented Generation"], merge_ids=[ids["RAG"]])],
    )

    resp = await client.post("/graph/consolidate")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["groups"]) == 1
    assert body["groups"][0]["keep_name"] == "Retrieval-Augmented Generation"
    assert _merged_names(body) == {"RAG"}
    assert body["removed_duplicate_edges"] == 0

    names = {e.name for e in (await db.execute(select(Entity))).scalars().all()}
    assert names == {"Retrieval-Augmented Generation"}


async def test_unknown_merge_id_is_dropped(client, db, monkeypatch):
    ids = await _make_entities(db, [("A", "OTHER"), ("B", "OTHER")])
    _mock_result(monkeypatch, [ConsolidationGroup(keep_id=ids["A"], merge_ids=["ghost-id"])])

    resp = await client.post("/graph/consolidate")
    assert resp.json()["groups"] == []
    assert len((await db.execute(select(Entity))).scalars().all()) == 2


async def test_cross_type_group_is_dropped(client, db, monkeypatch):
    ids = await _make_entities(db, [("OpenAI", "ORGANIZATION"), ("Sam Altman", "PERSON")])
    _mock_result(monkeypatch, [ConsolidationGroup(keep_id=ids["OpenAI"], merge_ids=[ids["Sam Altman"]])])

    resp = await client.post("/graph/consolidate")
    assert resp.json()["groups"] == []
    assert len((await db.execute(select(Entity))).scalars().all()) == 2


async def test_numeric_sibling_group_is_dropped(client, db, monkeypatch):
    ids = await _make_entities(db, [("GPT-4", "TECHNOLOGY"), ("GPT-3", "TECHNOLOGY")])
    _mock_result(monkeypatch, [ConsolidationGroup(keep_id=ids["GPT-4"], merge_ids=[ids["GPT-3"]])])

    resp = await client.post("/graph/consolidate")
    assert resp.json()["groups"] == []
    assert len((await db.execute(select(Entity))).scalars().all()) == 2


async def test_overlapping_groups_dont_double_merge(client, db, monkeypatch):
    ids = await _make_entities(db, [("A", "OTHER"), ("B", "OTHER"), ("C", "OTHER")])
    _mock_result(
        monkeypatch,
        [
            ConsolidationGroup(keep_id=ids["A"], merge_ids=[ids["B"]]),
            ConsolidationGroup(keep_id=ids["C"], merge_ids=[ids["B"]]),  # B already consumed
        ],
    )

    resp = await client.post("/graph/consolidate")
    body = resp.json()
    assert len(body["groups"]) == 1
    assert _merged_names(body) == {"B"}

    names = {e.name for e in (await db.execute(select(Entity))).scalars().all()}
    assert names == {"A", "C"}  # B merged into A; C untouched


async def test_provider_error_is_502(client, db, monkeypatch):
    await _make_entities(db, [("A", "OTHER"), ("B", "OTHER")])

    def boom(entities):
        raise RuntimeError("simulated provider failure")

    monkeypatch.setattr("app.routers.graph.consolidate_entities", boom)
    resp = await client.post("/graph/consolidate")
    assert resp.status_code == 502
