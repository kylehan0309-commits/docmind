"""
Which model provider is active right now.

`local`     -> Ollama models (llama3.2 / qwen2.5 / granite-vision)
`anthropic` -> Claude
`gemini`    -> Google Gemini (has a free tier)

Cloud providers cover chat, graph extraction and figure description; embeddings
are always local (retrieval must stay comparable across a session, and neither
cloud provider is wired for embeddings here).

The choice is process-wide and in-memory: it starts from `settings.llm_provider`
and the UI flips it via PUT /settings. Not persisted - a restart returns to the
configured default.
"""
from anthropic import Anthropic
from google import genai

from app.config import settings

_PROVIDERS = ("local", "anthropic", "gemini")
_active: str = settings.llm_provider if settings.llm_provider in _PROVIDERS else "local"

_anthropic: Anthropic | None = None
_gemini: "genai.Client | None" = None


def anthropic_available() -> bool:
    return bool(settings.anthropic_api_key)


def gemini_available() -> bool:
    return bool(settings.gemini_api_key)


def available() -> dict[str, bool]:
    return {"local": True, "anthropic": anthropic_available(), "gemini": gemini_available()}


def get_provider() -> str:
    """Effective provider - falls back to local if the selected one lost its key."""
    if not available().get(_active, False):
        return "local"
    return _active


def set_provider(name: str) -> None:
    if name not in _PROVIDERS:
        raise ValueError(f"unknown provider {name!r}; expected one of {_PROVIDERS}")
    if not available()[name]:
        raise ValueError(f"{name} is not configured (missing API key)")
    global _active
    _active = name


def anthropic_client() -> Anthropic:
    global _anthropic
    if _anthropic is None:
        _anthropic = Anthropic(api_key=settings.anthropic_api_key)
    return _anthropic


def gemini_client() -> "genai.Client":
    global _gemini
    if _gemini is None:
        _gemini = genai.Client(api_key=settings.gemini_api_key)
    return _gemini


def fresh_client():
    """A brand-new, uncached client for the active cloud provider (None if local).

    The cached singletons above are fine for one call at a time, but graph
    build runs several extraction calls concurrently on a cloud provider
    (see routers/graph.py) - sharing one client across threads corrupted the
    Gemini SDK's transport ("client has been closed" mid-batch, confirmed by
    reproducing it). Each concurrent caller gets its own client instead.
    """
    active = get_provider()
    if active == "anthropic":
        return Anthropic(api_key=settings.anthropic_api_key)
    if active == "gemini":
        return genai.Client(api_key=settings.gemini_api_key)
    return None


_TRANSIENT = ("ServerError", "APIStatusError", "InternalServerError", "RateLimitError",
             "APIConnectionError", "overloaded", "UNAVAILABLE", "RESOURCE_EXHAUSTED")


def call_with_retry(fn, *args, attempts: int = 4, base_delay: float = 1.5, **kwargs):
    """Call a cloud-provider function, retrying transient 429/5xx/overload errors
    with exponential backoff. Free tiers (Gemini especially) 503 under load."""
    import time

    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            blob = f"{type(e).__name__} {e}"
            transient = any(t in blob for t in _TRANSIENT)
            if not transient or i == attempts - 1:
                raise
            time.sleep(base_delay * (2**i))
