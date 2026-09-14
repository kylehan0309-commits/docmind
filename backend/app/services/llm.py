"""
Answer generation.

Local (default): the Ollama server, model `settings.ollama_model`.
Cloud: Claude / Gemini when that's the active provider.
The retrieved chunks and citation contract are identical either way.

Two entry points: `generate_answer` (blocking, returns the full text - used by
POST /chat) and `stream_answer` (a sync generator of text deltas - used by
POST /chat/stream). Streaming calls are NOT wrapped in call_with_retry: a retry
after tokens have already been yielded would duplicate text on the client, so a
mid-stream provider error is instead surfaced as a stream error event.
"""
from typing import Iterator

from ollama import Client

from app.config import settings
from app.models.document import Chunk
from app.services.provider import (
    anthropic_client,
    call_with_retry,
    gemini_client,
    get_provider,
)

_client: Client | None = None


def _get_client() -> Client:
    global _client
    if _client is None:
        _client = Client(host=settings.ollama_base_url)
    return _client


SYSTEM_PROMPT = (
    "You are DocMind, a research assistant. Answer the question using ONLY the "
    "provided document excerpts. Each excerpt has a number like [1]. After any "
    "statement that draws on an excerpt, cite its number(s) in square brackets. "
    "If the excerpts do not contain the answer, say so plainly rather than guessing."
)


def _build_context(chunks: list[Chunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{i}] ({chunk.document.filename}, page {chunk.page_number})\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def _user_message(question: str, chunks: list[Chunk]) -> str:
    return f"Document excerpts:\n\n{_build_context(chunks)}\n\nQuestion: {question}"


def generate_answer(question: str, chunks: list[Chunk]) -> str:
    """Send the retrieved chunks + question to the active LLM and return its answer."""
    user_msg = _user_message(question, chunks)
    provider = get_provider()

    if provider == "anthropic":
        resp = call_with_retry(
            anthropic_client().messages.create,
            model=settings.anthropic_model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    if provider == "gemini":
        from google.genai import types

        resp = call_with_retry(
            gemini_client().models.generate_content,
            model=settings.gemini_model,
            contents=user_msg,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT, max_output_tokens=2048
            ),
        )
        return (resp.text or "").strip()

    resp = _get_client().chat(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        options={"temperature": 0.2},
    )
    return resp["message"]["content"].strip()


def stream_answer(question: str, chunks: list[Chunk]) -> Iterator[str]:
    """Like generate_answer, but yields text deltas as they arrive."""
    user_msg = _user_message(question, chunks)
    provider = get_provider()

    if provider == "anthropic":
        with anthropic_client().messages.stream(
            model=settings.anthropic_model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            yield from stream.text_stream
        return

    if provider == "gemini":
        from google.genai import types

        for chunk in gemini_client().models.generate_content_stream(
            model=settings.gemini_model,
            contents=user_msg,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT, max_output_tokens=2048
            ),
        ):
            if chunk.text:
                yield chunk.text
        return

    for chunk in _get_client().chat(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        options={"temperature": 0.2},
        stream=True,
    ):
        text = chunk["message"]["content"]
        if text:
            yield text
