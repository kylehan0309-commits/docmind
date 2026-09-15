import asyncio
import contextlib
import os

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from sqlalchemy import select, delete, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import async_session, get_db
from app.models.document import Document, Chunk, Entity, EntityMention, Relationship
from app.services.pdf_parser import extract_pages, chunk_page_text
from app.services.embeddings import embed_texts, to_bytes

router = APIRouter(prefix="/documents", tags=["documents"])

os.makedirs(settings.upload_dir, exist_ok=True)

# Parsing/chunking/embedding (and any figure-description VLM calls inside
# extract_pages) runs in the background so POST /upload returns immediately
# regardless of PDF size - the frontend polls GET /documents/ for the
# processing -> ready/failed transition, same idiom as the graph build.
# Tracked per document (not a single global like the graph build) since
# multiple uploads can be in flight independently; also lets delete_document
# cancel one cleanly instead of racing it.
_upload_tasks: dict[str, asyncio.Task] = {}


async def _process_upload(document_id: str, save_path: str, describe_figures: bool) -> None:
    try:
        async with async_session() as db:
            document = await db.get(Document, document_id)
            if document is None:
                return  # deleted before this ever ran

            try:
                # extract_pages may call the vision model (blocking) - keep it off the loop.
                pages = await asyncio.to_thread(extract_pages, save_path, describe_figures)
                document.page_count = len(pages)

                chunk_rows: list[Chunk] = []
                for page in pages:
                    for chunk_text in chunk_page_text(page["text"]):
                        chunk_rows.append(Chunk(
                            document_id=document.id,
                            page_number=page["page_number"],
                            text=chunk_text,
                            chunk_index=len(chunk_rows),
                        ))

                # Embed every chunk in one batch and attach the vectors.
                if chunk_rows:
                    vectors = embed_texts([c.text for c in chunk_rows])
                    for chunk_row, vec in zip(chunk_rows, vectors):
                        chunk_row.embedding = to_bytes(vec)

                db.add_all(chunk_rows)
                document.status = "ready"
                document.error = None
                await db.commit()
            except Exception as e:  # noqa: BLE001 - surfaced via document.error, not a 500
                document.status = "failed"
                document.error = str(e)[:2000]
                await db.commit()
    finally:
        _upload_tasks.pop(document_id, None)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    describe_figures: bool = Query(
        False,
        description="Run each figure-bearing page through the local vision model "
        "and fold its description into the text. Slow (one VLM call per such page).",
    ),
    db: AsyncSession = Depends(get_db),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported right now")

    document = Document(filename=file.filename, status="processing")
    db.add(document)
    await db.commit()
    await db.refresh(document)

    save_path = os.path.join(settings.upload_dir, f"{document.id}.pdf")
    with open(save_path, "wb") as f:
        f.write(await file.read())

    _upload_tasks[document.id] = asyncio.create_task(
        _process_upload(document.id, save_path, describe_figures)
    )

    return {
        "id": document.id,
        "filename": document.filename,
        "status": document.status,
        "page_count": document.page_count,
        "error": document.error,
    }


@router.get("/")
async def list_documents(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document))
    docs = result.scalars().all()
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "status": d.status,
            "page_count": d.page_count,
            "error": d.error,
        }
        for d in docs
    ]


@router.delete("/{document_id}")
async def delete_document(document_id: str, db: AsyncSession = Depends(get_db)):
    document = await db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Stop an in-flight upload cleanly before deleting, rather than racing it -
    # otherwise it could still be mid-parse and commit chunks for a document
    # this request is about to remove.
    task = _upload_tasks.pop(document_id, None)
    if task and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    save_path = os.path.join(settings.upload_dir, f"{document_id}.pdf")
    if os.path.exists(save_path):
        os.remove(save_path)

    # Drop this document's graph edges + mentions, then any entity that is now
    # neither in an edge nor mentioned anywhere.
    await db.execute(delete(Relationship).where(Relationship.document_id == document_id))
    await db.execute(delete(EntityMention).where(EntityMention.document_id == document_id))
    await db.flush()
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

    await db.delete(document)  # chunks cascade via the Document.chunks relationship
    await db.commit()
    return {"deleted": document_id}
