"""
Figure understanding via a local vision-language model (Ollama).

Given a rendered page image, ask the VLM to describe any charts, diagrams,
tables-as-graphics or images on it, so that visual content becomes searchable
text that flows into chunking, embeddings, chat and graph extraction.
"""
import base64

from ollama import Client

from app.config import settings
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


PROMPT = (
    "This is a page from a PDF. Describe ONLY the non-text visual content: "
    "charts, plots, diagrams, flowcharts, tables drawn as graphics, photos and "
    "illustrations. For a chart, state what it measures, its axes, and the trend "
    "or key values. For a diagram, name the components and how they connect. For "
    "a graphic table, give the columns and summarise the rows. Ignore ordinary "
    "body paragraphs. If the page has no such visual content, reply with exactly: NONE"
)


def describe_page(png_bytes: bytes) -> str:
    """Text description of the figures on a page image, or '' if there are none."""
    provider = get_provider()

    if provider == "anthropic":
        resp = call_with_retry(
            anthropic_client().messages.create,
            model=settings.anthropic_model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.standard_b64encode(png_bytes).decode(),
                            },
                        },
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
    elif provider == "gemini":
        from google.genai import types

        resp = call_with_retry(
            gemini_client().models.generate_content,
            model=settings.gemini_model,
            contents=[
                types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                PROMPT,
            ],
            config=types.GenerateContentConfig(max_output_tokens=1024),
        )
        text = (resp.text or "").strip()
    else:
        resp = _get_client().chat(
            model=settings.vision_model,
            messages=[{"role": "user", "content": PROMPT, "images": [png_bytes]}],
            options={"temperature": 0},
        )
        text = resp["message"]["content"].strip()

    if len(text) < 3 or text.upper().strip(".") == "NONE":
        return ""
    return text
