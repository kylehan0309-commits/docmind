import asyncio
import json
import threading
from typing import AsyncIterator, Callable, Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.database import get_db
from app.models.document import Chunk
from app.services.embeddings import embed_query
from app.services.llm import generate_answer, stream_answer
from app.services.retrieval import rank_chunks

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    document_ids: list[str] | None = None  # optional filter; None = search all docs
    top_k: int | None = None               # overrides settings.retrieval_top_k


async def _retrieve(req: ChatRequest, db: AsyncSession) -> list[tuple[Chunk, float]]:
    """Shared by /chat and /chat/stream: validate the question and rank chunks."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    stmt = (
        select(Chunk)
        .options(selectinload(Chunk.document))
        .where(Chunk.embedding.is_not(None))
    )
    if req.document_ids:
        stmt = stmt.where(Chunk.document_id.in_(req.document_ids))

    chunks = list((await db.execute(stmt)).scalars().all())
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No embedded chunks found. Upload a document first.",
        )

    top_k = req.top_k or settings.retrieval_top_k
    query_vec = await asyncio.to_thread(embed_query, question)
    return rank_chunks(query_vec, chunks, top_k)


def _citation(chunk: Chunk, score: float) -> dict:
    return {
        "document_id": chunk.document_id,
        "filename": chunk.document.filename,
        "page_number": chunk.page_number,
        "chunk_index": chunk.chunk_index,
        "score": round(score, 4),
        "text": chunk.text,
    }


@router.post("")
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    ranked = await _retrieve(req, db)

    # generate_answer makes a blocking HTTP call (Ollama / Claude / Gemini) - keep
    # it off the event loop. The google-genai sync client in particular deadlocks
    # if called while a loop is running.
    try:
        answer = await asyncio.to_thread(
            generate_answer, req.question.strip(), [chunk for chunk, _ in ranked]
        )
    except Exception as e:  # noqa: BLE001 - surface provider errors cleanly
        raise HTTPException(status_code=502, detail=f"LLM provider error: {e}")

    return {
        "answer": answer,
        "citations": [_citation(chunk, score) for chunk, score in ranked],
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _bridge_sync_generator(factory: Callable[[], Iterator[str]]) -> AsyncIterator[str]:
    """Run a blocking generator (the provider SDK call) in a thread and re-yield
    its items on the event loop, so a slow/streaming HTTP call never blocks it."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    done = object()

    def worker():
        try:
            for item in factory():
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as e:  # noqa: BLE001 - re-raised on the event loop below
            loop.call_soon_threadsafe(queue.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = await queue.get()
        if item is done:
            return
        if isinstance(item, Exception):
            raise item
        yield item


@router.post("/stream")
async def chat_stream(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    """Like POST /chat, but sent as Server-Sent Events: a `citations` event as
    soon as retrieval finishes, then a `token` event per text delta, then `done`
    (or `error` if the provider call fails partway through).
    """
    ranked = await _retrieve(req, db)
    question = req.question.strip()
    top_chunks = [chunk for chunk, _ in ranked]

    async def event_stream() -> AsyncIterator[str]:
        yield _sse("citations", {"citations": [_citation(c, s) for c, s in ranked]})
        try:
            async for delta in _bridge_sync_generator(lambda: stream_answer(question, top_chunks)):
                yield _sse("token", {"text": delta})
            yield _sse("done", {})
        except Exception as e:  # noqa: BLE001 - surface provider errors as a stream event
            yield _sse("error", {"detail": f"LLM provider error: {e}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
