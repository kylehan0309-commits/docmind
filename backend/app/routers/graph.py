import asyncio
import base64
import json
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import async_session, get_db
from app.models.document import (
    Chunk,
    Document,
    Entity,
    EntityMention,
    MergeHistory,
    Relationship,
)
from app.services.embeddings import embed_query, from_bytes, to_bytes
from app.services.graph_extraction import (
    ConsolidationResult,
    consolidate_entities,
    differ_only_by_number,
    extract_from_chunk,
    normalize_name,
    resolve_entity,
)
from app.services.provider import get_provider

router = APIRouter(prefix="/graph", tags=["graph"])


# --------------------------------------------------------------------------
# Background build job (one LLM call per chunk is slow, so /build returns
# immediately and progress is polled at /build/status). State is in-memory:
# on a server restart the job is forgotten, but it's resumable anyway via the
# Chunk.graph_extracted flag - just POST /build again.
# --------------------------------------------------------------------------
_build: dict = {
    "running": False,
    "processed": 0,
    "failed": 0,
    "total": 0,
    "new_entities": 0,
    "new_relationships": 0,
    "remaining_chunks": None,
    "error": None,       # fatal error that ended the run
    "last_error": None,  # last per-chunk error (run continued)
    "started_at": None,
    "finished_at": None,
}
_build_task: asyncio.Task | None = None


async def _pending_chunk_query(
    db: AsyncSession, document_ids: list[str] | None, limit: int | None
):
    stmt = select(Chunk).where(Chunk.graph_extracted.is_(False))
    if document_ids:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))
    stmt = stmt.order_by(Chunk.document_id, Chunk.chunk_index)
    if limit:
        stmt = stmt.limit(limit)
    return list((await db.execute(stmt)).scalars().all())


async def _reset_documents(db: AsyncSession, document_ids: list[str]) -> None:
    """Wipe extraction for the given documents so a build re-maps them from scratch."""
    await db.execute(
        delete(Relationship).where(Relationship.document_id.in_(document_ids))
    )
    await db.execute(
        delete(EntityMention).where(EntityMention.document_id.in_(document_ids))
    )
    await db.execute(
        update(Chunk).where(Chunk.document_id.in_(document_ids)).values(graph_extracted=False)
    )
    await db.flush()
    # drop entities that are now unreferenced everywhere
    orphan_ids = (
        await db.execute(
            select(Entity.id).where(
                ~select(Relationship.id)
                .where(
                    or_(
                        Relationship.source_id == Entity.id,
                        Relationship.target_id == Entity.id,
                    )
                )
                .exists(),
                ~select(EntityMention.id)
                .where(EntityMention.entity_id == Entity.id)
                .exists(),
            )
        )
    ).scalars().all()
    if orphan_ids:
        await db.execute(delete(Entity).where(Entity.id.in_(orphan_ids)))
    await db.commit()


async def _run_build(document_ids: list[str] | None, limit: int | None) -> None:
    try:
        async with async_session() as db:
            chunks = await _pending_chunk_query(db, document_ids, limit)

            # Load every existing entity once: exact-match index + vectors for fuzzy match.
            norm_to_id: dict[str, str] = {}
            known: list[tuple[str, str, str, np.ndarray]] = []  # (id, type, name, name vector)
            for ent in (await db.execute(select(Entity))).scalars().all():
                norm_to_id[ent.norm_name] = ent.id
                if ent.name_embedding is not None:
                    known.append((ent.id, ent.type, ent.name, from_bytes(ent.name_embedding)))

            threshold = settings.entity_merge_threshold

            async def get_or_create_entity(name: str, etype: str) -> str | None:
                norm = normalize_name(name)
                if not norm:
                    return None
                # 1. exact normalized-string match
                if norm in norm_to_id:
                    return norm_to_id[norm]
                # 2. fuzzy match against existing same-type entities
                vec = embed_query(name)
                match_id = resolve_entity(name, vec, etype, known, threshold)
                if match_id is not None:
                    norm_to_id[norm] = match_id  # alias this surface form to the match
                    return match_id
                # 3. new entity
                entity = Entity(
                    name=name.strip(), norm_name=norm, type=etype,
                    name_embedding=to_bytes(vec),
                )
                db.add(entity)
                await db.flush()
                norm_to_id[norm] = entity.id
                known.append((entity.id, etype, entity.name, vec))
                _build["new_entities"] += 1
                return entity.id

            for chunk in chunks:
                try:
                    result = await asyncio.to_thread(extract_from_chunk, chunk.text)
                except Exception as e:  # noqa: BLE001 - one bad chunk shouldn't kill the run
                    await db.rollback()
                    _build["failed"] += 1
                    _build["last_error"] = f"{type(e).__name__}: {e}"
                    continue

                type_by_norm = {normalize_name(e.name): e.type for e in result.entities}
                mentions: dict[str, str] = {}  # entity_id -> surface form as written here

                for e in result.entities:
                    eid = await get_or_create_entity(e.name, e.type)
                    if eid:
                        mentions.setdefault(eid, e.name.strip())

                for rel in result.relationships:
                    src_id = await get_or_create_entity(
                        rel.source, type_by_norm.get(normalize_name(rel.source), "OTHER")
                    )
                    tgt_id = await get_or_create_entity(
                        rel.target, type_by_norm.get(normalize_name(rel.target), "OTHER")
                    )
                    if src_id:
                        mentions.setdefault(src_id, rel.source.strip())
                    if tgt_id:
                        mentions.setdefault(tgt_id, rel.target.strip())
                    if not src_id or not tgt_id or src_id == tgt_id:
                        continue
                    db.add(
                        Relationship(
                            source_id=src_id,
                            target_id=tgt_id,
                            label=rel.label.strip()[:80],
                            chunk_id=chunk.id,
                            document_id=chunk.document_id,
                        )
                    )
                    _build["new_relationships"] += 1

                for eid, surface in mentions.items():
                    db.add(
                        EntityMention(
                            entity_id=eid,
                            chunk_id=chunk.id,
                            document_id=chunk.document_id,
                            surface_form=surface[:120],
                        )
                    )

                chunk.graph_extracted = True
                await db.commit()
                _build["processed"] += 1

            _build["remaining_chunks"] = (
                await db.execute(
                    select(func.count())
                    .select_from(Chunk)
                    .where(Chunk.graph_extracted.is_(False))
                )
            ).scalar_one()
    except Exception as e:  # noqa: BLE001 - surface any failure via status
        _build["error"] = f"{type(e).__name__}: {e}"
    finally:
        _build["running"] = False
        _build["finished_at"] = datetime.now(timezone.utc).isoformat()


@router.post("/build")
async def build_graph(
    document_ids: list[str] = Query(
        default=[], description="restrict to these documents (repeat the param); omit for all"
    ),
    force: bool = Query(
        False, description="re-map the selected documents from scratch (needs document_ids)"
    ),
    limit: int | None = Query(None, description="max chunks to process this run"),
    db: AsyncSession = Depends(get_db),
):
    """Extract entities/relationships for not-yet-processed chunks.

    Normally incremental - chunks already mapped (`graph_extracted`) are skipped,
    so re-running only picks up new documents. `force` + `document_ids` clears
    those documents first and re-maps them. Returns immediately; poll
    GET /graph/build/status.
    """
    global _build_task
    if _build["running"]:
        raise HTTPException(status_code=409, detail="A graph build is already running")
    if force and not document_ids:
        raise HTTPException(status_code=400, detail="force=true requires document_ids")

    if force:
        await _reset_documents(db, document_ids)

    total = len(await _pending_chunk_query(db, document_ids, limit))
    if total == 0:
        return {"status": "nothing_to_do", "total_chunks": 0}

    _build.update(
        running=True,
        processed=0,
        failed=0,
        total=total,
        new_entities=0,
        new_relationships=0,
        remaining_chunks=None,
        error=None,
        last_error=None,
        started_at=datetime.now(timezone.utc).isoformat(),
        finished_at=None,
    )
    _build_task = asyncio.create_task(_run_build(document_ids, limit))
    return {"status": "started", "total_chunks": total}


@router.get("/build/status")
async def build_status():
    return _build


@router.get("/coverage")
async def graph_coverage(db: AsyncSession = Depends(get_db)):
    """Per-document graph-extraction progress: how many chunks are mapped."""
    rows = (
        await db.execute(
            select(
                Document.id,
                Document.filename,
                func.count(Chunk.id),
                func.count(Chunk.id).filter(Chunk.graph_extracted.is_(True)),
            )
            .join(Chunk, Chunk.document_id == Document.id, isouter=True)
            .group_by(Document.id)
            .order_by(Document.created_at)
        )
    ).all()
    return [
        {
            "document_id": r[0],
            "filename": r[1],
            "total_chunks": r[2] or 0,
            "extracted_chunks": r[3] or 0,
        }
        for r in rows
    ]


class MergeRequest(BaseModel):
    keep_id: str
    merge_ids: list[str]


def _rel_snapshot(r: Relationship) -> dict:
    return {
        "id": r.id, "source_id": r.source_id, "target_id": r.target_id, "label": r.label,
        "chunk_id": r.chunk_id, "document_id": r.document_id,
        "created_at": r.created_at.isoformat(),
    }


def _entity_snapshot(e: Entity) -> dict:
    return {
        "id": e.id, "name": e.name, "norm_name": e.norm_name, "type": e.type,
        "name_embedding": base64.b64encode(e.name_embedding).decode() if e.name_embedding else None,
        "created_at": e.created_at.isoformat(),
    }


async def _merge_entities(db: AsyncSession, keep_id: str, merge_ids: list[str]) -> tuple[int, str]:
    """Fold `merge_ids` into `keep_id`: re-point their edges + mentions, drop the
    self-loops and duplicate edges that creates, then delete the merged entities.
    Caller must already have validated that every id exists. Commits.

    Snapshots everything it's about to touch into a MergeHistory row *before*
    mutating, so POST /graph/merges/{id}/undo can restore it later - a merge is
    otherwise irreversible (rows are overwritten/deleted in place, nothing else
    remembers their old values). Returns (duplicate edges collapsed, history id).
    """
    keep = await db.get(Entity, keep_id)
    merged_entities = (
        await db.execute(select(Entity).where(Entity.id.in_(merge_ids)))
    ).scalars().all()
    entities_snap = [_entity_snapshot(e) for e in merged_entities]

    # Every relationship/mention that currently points at a merge_id - these are
    # the ones about to be repointed (and are snapshotted at their PRE-merge
    # values, which is what undo needs regardless of what happens to them next).
    rel_snap: dict[str, dict] = {
        r.id: _rel_snapshot(r)
        for r in (
            await db.execute(
                select(Relationship).where(
                    or_(Relationship.source_id.in_(merge_ids), Relationship.target_id.in_(merge_ids))
                )
            )
        ).scalars().all()
    }
    mention_snap = [
        {"id": m.id, "entity_id": m.entity_id}
        for m in (
            await db.execute(select(EntityMention).where(EntityMention.entity_id.in_(merge_ids)))
        ).scalars().all()
    ]

    await db.execute(
        update(Relationship).where(Relationship.source_id.in_(merge_ids)).values(source_id=keep_id)
    )
    await db.execute(
        update(Relationship).where(Relationship.target_id.in_(merge_ids)).values(target_id=keep_id)
    )
    await db.flush()
    await db.execute(
        update(EntityMention).where(EntityMention.entity_id.in_(merge_ids)).values(entity_id=keep_id)
    )
    await db.flush()

    # self-loops created by the repoint - snapshot (if not already) before dropping
    self_loops = (
        await db.execute(select(Relationship).where(Relationship.source_id == Relationship.target_id))
    ).scalars().all()
    for r in self_loops:
        rel_snap.setdefault(r.id, _rel_snapshot(r))
    await db.execute(delete(Relationship).where(Relationship.source_id == Relationship.target_id))

    # collapse now-identical edges (same source, target, case-folded label) -
    # snapshot (if not already) before dropping. This can occasionally catch an
    # edge that predates this merge entirely (ordering quirk); snapshotting its
    # current values here is still correct for restoring it.
    seen: set[tuple[str, str, str]] = set()
    dup_ids: list[str] = []
    for r in (
        await db.execute(select(Relationship).order_by(Relationship.created_at))
    ).scalars().all():
        key = (r.source_id, r.target_id, r.label.lower())
        if key in seen:
            dup_ids.append(r.id)
            rel_snap.setdefault(r.id, _rel_snapshot(r))
        else:
            seen.add(key)
    if dup_ids:
        await db.execute(delete(Relationship).where(Relationship.id.in_(dup_ids)))

    await db.execute(delete(Entity).where(Entity.id.in_(merge_ids)))

    history = MergeHistory(
        keep_id=keep_id,
        keep_name=keep.name if keep else keep_id,
        entities_json=json.dumps(entities_snap),
        relationships_json=json.dumps(list(rel_snap.values())),
        mentions_json=json.dumps(mention_snap),
    )
    db.add(history)
    await db.commit()
    return len(dup_ids), history.id


@router.post("/entities/merge")
async def merge_entities(req: MergeRequest, db: AsyncSession = Depends(get_db)):
    """Manually fold `merge_ids` into `keep_id`."""
    merge_ids = [m for m in dict.fromkeys(req.merge_ids) if m != req.keep_id]
    if not merge_ids:
        raise HTTPException(status_code=400, detail="merge_ids is empty after removing keep_id")

    if not await db.get(Entity, req.keep_id):
        raise HTTPException(status_code=404, detail=f"keep_id {req.keep_id} not found")
    found = set(
        (await db.execute(select(Entity.id).where(Entity.id.in_(merge_ids)))).scalars().all()
    )
    if missing := set(merge_ids) - found:
        raise HTTPException(status_code=404, detail=f"unknown entity ids: {sorted(missing)}")

    removed, history_id = await _merge_entities(db, req.keep_id, merge_ids)
    return {
        "kept": req.keep_id,
        "merged": merge_ids,
        "removed_duplicate_edges": removed,
        "merge_id": history_id,
    }


@router.post("/consolidate")
async def consolidate_graph(db: AsyncSession = Depends(get_db)):
    """Second-pass dedup over the *whole* entity list (one LLM call), for
    acronym/synonym fragmentation the per-chunk embedding dedup can't catch
    (e.g. "RAG" vs "retrieval-augmented generation"). Applies via the same
    merge path as POST /entities/merge.

    Guards catch mechanical hallucinations - unknown ids, cross-type groups,
    numeric-sibling names (GPT-4/GPT-3) - but NOT a same-type semantic mistake
    (observed in testing: a weak local model merged two distinct organizations
    that happened to both be type ORGANIZATION). There's no undo once applied,
    so the response lists exactly what was merged - check it - and a cloud
    provider (Claude/Gemini) is markedly more reliable here than the local model.
    """
    entities = (await db.execute(select(Entity))).scalars().all()
    if len(entities) < 2:
        return {"provider": get_provider(), "groups": [], "removed_duplicate_edges": 0}

    by_id = {e.id: e for e in entities}
    try:
        result: ConsolidationResult = await asyncio.to_thread(
            consolidate_entities, [(e.id, e.name, e.type) for e in entities]
        )
    except Exception as e:  # noqa: BLE001 - surface provider errors cleanly
        raise HTTPException(status_code=502, detail=f"LLM provider error: {e}")

    consumed: set[str] = set()  # ids already folded away by an earlier group this run
    applied: list[dict] = []
    dup_edges_removed = 0

    for group in result.groups:
        keep = by_id.get(group.keep_id)
        if keep is None or group.keep_id in consumed:
            continue

        valid_merge_ids = []
        for mid in dict.fromkeys(group.merge_ids):
            if mid == group.keep_id or mid in consumed:
                continue
            other = by_id.get(mid)
            if other is None:
                continue  # hallucinated id
            if other.type != keep.type:
                continue  # cross-type merge - don't trust it
            if differ_only_by_number(keep.name, other.name):
                continue  # e.g. GPT-4 / GPT-3
            valid_merge_ids.append(mid)

        if not valid_merge_ids:
            continue

        removed, history_id = await _merge_entities(db, group.keep_id, valid_merge_ids)
        dup_edges_removed += removed
        consumed.update(valid_merge_ids)
        applied.append(
            {
                "keep_id": group.keep_id,
                "keep_name": keep.name,
                "merged": [{"id": mid, "name": by_id[mid].name} for mid in valid_merge_ids],
                "merge_id": history_id,
            }
        )

    return {
        "provider": get_provider(),
        "groups": applied,
        "removed_duplicate_edges": dup_edges_removed,
    }


@router.get("/merges")
async def list_merges(limit: int = Query(20, le=100), db: AsyncSession = Depends(get_db)):
    """Recent merge history (manual or from /consolidate), most recent first,
    each with an `undone` flag and enough to render an undo button."""
    rows = (
        await db.execute(
            select(MergeHistory).order_by(MergeHistory.created_at.desc()).limit(limit)
        )
    ).scalars().all()
    return [
        {
            "id": h.id,
            "keep_id": h.keep_id,
            "keep_name": h.keep_name,
            "merged": [{"id": e["id"], "name": e["name"]} for e in json.loads(h.entities_json)],
            "created_at": h.created_at.isoformat(),
            "undone": h.undone,
        }
        for h in rows
    ]


@router.post("/merges/{merge_id}/undo")
async def undo_merge(merge_id: str, db: AsyncSession = Depends(get_db)):
    """Reverse one merge: re-insert the deleted entities and restore every
    relationship/mention it touched to its pre-merge value (re-inserting any
    that were deleted as a self-loop or duplicate). Won't undo a merge twice,
    and refuses if `keep_id` no longer exists (it was itself merged away by a
    *later* merge - undo that one first).
    """
    history = await db.get(MergeHistory, merge_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"merge {merge_id} not found")
    if history.undone:
        raise HTTPException(status_code=409, detail="this merge was already undone")
    if not await db.get(Entity, history.keep_id):
        raise HTTPException(
            status_code=409,
            detail=f"keep entity {history.keep_id!r} no longer exists (merged away by a later "
            "merge) - undo that one first",
        )

    for e in json.loads(history.entities_json):
        db.add(
            Entity(
                id=e["id"],
                name=e["name"],
                norm_name=e["norm_name"],
                type=e["type"],
                name_embedding=base64.b64decode(e["name_embedding"]) if e["name_embedding"] else None,
                created_at=datetime.fromisoformat(e["created_at"]),
            )
        )
    await db.flush()

    for r in json.loads(history.relationships_json):
        existing = await db.get(Relationship, r["id"])
        if existing:
            existing.source_id = r["source_id"]
            existing.target_id = r["target_id"]
            existing.label = r["label"]
        else:
            db.add(
                Relationship(
                    id=r["id"], source_id=r["source_id"], target_id=r["target_id"],
                    label=r["label"], chunk_id=r["chunk_id"], document_id=r["document_id"],
                    created_at=datetime.fromisoformat(r["created_at"]),
                )
            )

    for m in json.loads(history.mentions_json):
        existing = await db.get(EntityMention, m["id"])
        if existing:
            existing.entity_id = m["entity_id"]

    history.undone = True
    await db.commit()
    return {"undone": merge_id, "restored_entities": len(json.loads(history.entities_json))}


@router.get("")
async def get_graph(
    document_id: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Return the graph as {nodes, edges}. Only entities that have an edge appear."""
    rel_stmt = select(Relationship)
    if document_id:
        rel_stmt = rel_stmt.where(Relationship.document_id == document_id)
    rels = list((await db.execute(rel_stmt)).scalars().all())

    entity_ids = {r.source_id for r in rels} | {r.target_id for r in rels}
    entities = (
        list(
            (
                await db.execute(select(Entity).where(Entity.id.in_(entity_ids)))
            ).scalars().all()
        )
        if entity_ids
        else []
    )

    # Collapse duplicate (source, target, label) edges; weight = how many chunks said it.
    weight: dict[tuple[str, str, str], int] = {}
    for r in rels:
        k = (r.source_id, r.target_id, r.label.lower())
        weight[k] = weight.get(k, 0) + 1

    seen: set[tuple[str, str, str]] = set()
    edges = []
    for r in rels:
        k = (r.source_id, r.target_id, r.label.lower())
        if k in seen:
            continue
        seen.add(k)
        edges.append(
            {
                "id": r.id,
                "source": r.source_id,
                "target": r.target_id,
                "label": r.label,
                "weight": weight[k],
            }
        )

    return {
        "nodes": [{"id": e.id, "name": e.name, "type": e.type} for e in entities],
        "edges": edges,
    }


def _excerpt(text: str, term: str | None = None, width: int = 260) -> str:
    """A short single-line snippet of `text`, centered on `term` if it occurs."""
    text = " ".join(text.split())
    if term:
        i = text.lower().find(term.lower())
        if i != -1:
            start = max(0, i - width // 3)
            end = start + width
            return ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")
    return text[:width] + ("…" if len(text) > width else "")


@router.get("/search")
async def search_graph(
    q: str = Query(..., min_length=1, description="text to look for"),
    db: AsyncSession = Depends(get_db),
):
    """Find entities whose *content* contains `q` - name, type, a relationship
    label, a connected entity's name, a mention surface form, or the text of a
    chunk that mentions it. Each hit says which field matched and why.
    Ordered name -> type -> relationship -> mention.
    """
    ql = q.lower()
    entities = list((await db.execute(select(Entity))).scalars().all())
    by_id = {e.id: e for e in entities}
    hits: dict[str, dict] = {}

    def add(eid: str, field: str, snippet: str, rank: int) -> None:
        ent = by_id.get(eid)
        if ent is None:
            return
        cur = hits.get(eid)
        if cur is None or rank < cur["_rank"]:
            hits[eid] = {
                "id": eid,
                "name": ent.name,
                "type": ent.type,
                "match_field": field,
                "snippet": snippet,
                "_rank": rank,
            }

    for e in entities:
        if ql in e.name.lower():
            add(e.id, "name", e.name, 0)
        elif ql in e.type.lower():
            add(e.id, "type", e.type, 1)

    for r in (await db.execute(select(Relationship))).scalars().all():
        src, tgt = by_id.get(r.source_id), by_id.get(r.target_id)
        if not src or not tgt:
            continue
        phrase = f"{src.name} —{r.label}→ {tgt.name}"
        label_hit = ql in r.label.lower()
        if label_hit or ql in tgt.name.lower():
            add(r.source_id, "relationship", phrase, 2)
        if label_hit or ql in src.name.lower():
            add(r.target_id, "relationship", phrase, 2)

    # Mention match: the text *around* where this entity is named contains `q`
    # (not just anywhere in the chunk - that returns every entity on the page).
    mention_rows = (
        await db.execute(
            select(EntityMention.entity_id, EntityMention.surface_form, Chunk.text)
            .join(Chunk, Chunk.id == EntityMention.chunk_id)
        )
    ).all()
    for eid, surface, text in mention_rows:
        text = text or ""
        surface = surface or ""
        if surface and ql in surface.lower():
            add(eid, "mention", surface, 1)
            continue
        pos = text.lower().find(surface.lower()) if surface else -1
        window = text[max(0, pos - 120) : pos + len(surface) + 120] if pos != -1 else ""
        if window and ql in window.lower():
            add(eid, "mention", _excerpt(window, q), 3)

    ordered = sorted(hits.values(), key=lambda h: (h["_rank"], h["name"].lower()))
    for h in ordered:
        h.pop("_rank", None)
    return ordered


@router.get("/entities/{entity_id}")
async def entity_detail(entity_id: str, db: AsyncSession = Depends(get_db)):
    """Everything the UI needs when a graph node is clicked: the entity, its
    relationships (with the source chunk), and every chunk that mentions it."""
    entity = await db.get(Entity, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="entity not found")

    rels = list(
        (
            await db.execute(
                select(Relationship).where(
                    or_(Relationship.source_id == entity_id, Relationship.target_id == entity_id)
                )
            )
        ).scalars().all()
    )
    mentions = list(
        (
            await db.execute(select(EntityMention).where(EntityMention.entity_id == entity_id))
        ).scalars().all()
    )

    chunk_ids = {r.chunk_id for r in rels} | {m.chunk_id for m in mentions}
    doc_ids = {r.document_id for r in rels} | {m.document_id for m in mentions}
    other_ids = ({r.source_id for r in rels} | {r.target_id for r in rels}) - {entity_id}
    chunks = {
        c.id: c
        for c in (await db.execute(select(Chunk).where(Chunk.id.in_(chunk_ids)))).scalars()
    }
    docs = {
        d.id: d
        for d in (await db.execute(select(Document).where(Document.id.in_(doc_ids)))).scalars()
    }
    others = {
        e.id: e
        for e in (await db.execute(select(Entity).where(Entity.id.in_(other_ids)))).scalars()
    }

    def _fname(doc_id: str) -> str:
        return docs[doc_id].filename if doc_id in docs else "?"

    connected: set[str] = set()
    relationships = []
    for r in rels:
        is_out = r.source_id == entity_id
        other_id = r.target_id if is_out else r.source_id
        other_name = others[other_id].name if other_id in others else "?"
        connected.add(other_id)
        ch = chunks.get(r.chunk_id)
        relationships.append(
            {
                "direction": "out" if is_out else "in",
                "label": r.label,
                "other_id": other_id,
                "other_name": other_name,
                "document_id": r.document_id,
                "filename": _fname(r.document_id),
                "page_number": ch.page_number if ch else None,
                # center the snippet on the other entity, which does appear in the text
                "excerpt": _excerpt(ch.text, other_name) if ch else "",
            }
        )

    mention_list = []
    for m in mentions:
        ch = chunks.get(m.chunk_id)
        mention_list.append(
            {
                "document_id": m.document_id,
                "filename": _fname(m.document_id),
                "page_number": ch.page_number if ch else None,
                "surface_form": m.surface_form,
                "excerpt": _excerpt(ch.text, m.surface_form) if ch else "",
            }
        )

    return {
        "id": entity.id,
        "name": entity.name,
        "type": entity.type,
        "degree": len(connected),
        "relationships": relationships,
        "mentions": mention_list,
    }


@router.get("/view", response_class=HTMLResponse)
async def graph_view():
    """A tiny self-contained viewer. Local dev only - pulls Cytoscape.js from a CDN."""
    return _VIEWER_HTML


_VIEWER_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>DocMind knowledge graph</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.30.2/cytoscape.min.js"></script>
<style>
  html, body { margin: 0; height: 100%; font-family: system-ui, sans-serif; }
  #cy { position: absolute; top: 44px; bottom: 0; left: 0; right: 0; }
  #bar { height: 44px; display: flex; align-items: center; gap: 12px; padding: 0 12px;
         border-bottom: 1px solid #ddd; font-size: 14px; }
  #bar b { font-weight: 600; }
  .legend span { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
                 margin: 0 4px 0 10px; vertical-align: middle; }
</style>
</head>
<body>
<div id="bar">
  <b>DocMind knowledge graph</b>
  <span id="stats"></span>
  <span class="legend" id="legend"></span>
  <button onclick="location.reload()">reload</button>
</div>
<div id="cy"></div>
<script>
const COLORS = {
  PERSON: "#e15759", ORGANIZATION: "#4e79a7", CONCEPT: "#59a14f",
  TECHNOLOGY: "#f28e2b", LOCATION: "#b07aa1", EVENT: "#edc948", OTHER: "#9c9c9c"
};

async function load() {
  const res = await fetch("/graph");
  const g = await res.json();
  document.getElementById("stats").textContent =
    g.nodes.length + " entities, " + g.edges.length + " relationships";
  document.getElementById("legend").innerHTML = Object.keys(COLORS)
    .map(k => '<span style="background:' + COLORS[k] + '"></span>' + k).join("");

  const elements = [
    ...g.nodes.map(n => ({ data: { id: n.id, label: n.name, type: n.type } })),
    ...g.edges.map(e => ({ data: {
      id: e.id, source: e.source, target: e.target,
      label: e.label + (e.weight > 1 ? " (" + e.weight + ")" : "")
    } })),
  ];

  cytoscape({
    container: document.getElementById("cy"),
    elements,
    style: [
      { selector: "node", style: {
          "background-color": ele => COLORS[ele.data("type")] || COLORS.OTHER,
          "label": "data(label)", "font-size": 11, "color": "#222",
          "text-valign": "center", "text-halign": "right", "text-margin-x": 4,
          "width": 18, "height": 18 } },
      { selector: "edge", style: {
          "width": 1.5, "line-color": "#bbb", "target-arrow-color": "#bbb",
          "target-arrow-shape": "triangle", "curve-style": "bezier",
          "label": "data(label)", "font-size": 9, "color": "#777",
          "text-rotation": "autorotate" } },
    ],
    layout: { name: "cose", animate: false, padding: 30, nodeRepulsion: 8000 },
  });
}
load();
</script>
</body>
</html>"""
