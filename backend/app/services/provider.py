"""
Which model provider is active right now.

`local`     -> Ollama models (llama3.2 / qwen2.5 / granite-vision)
`anthropic` -> Claude
`gemini`    -> Google Gemini (has a free tier)

Cloud providers cover chat, graph extraction and figure description; embeddings
are always local (retrieval must stay comparable across a session, and neither
cloud provider is wired for embeddings here).

The active choice lives in-memory (`_active`) for fast, synchronous reads from
every call site (llm.py, graph_extraction.py, vision.py all read it on every
call) - it starts from `settings.llm_provider`. On top of that, PUT /settings
also durably persists the choice to a `Setting` DB row, and app startup
(main.py's lifespan) loads it back via `load_persisted_provider()` - so a
backend restart resumes on whatever you last picked instead of silently
reverting to the .env default. Persistence is best-effort: if the persisted
provider's key is no longer configured, startup just keeps the .env default.
"""
from anthropic import Anthropic
from google import genai

from app.config import settings
from app.database import async_session
from app.models.document import Setting

_PROVIDERS = ("local", "anthropic", "gemini")
_active: str = settings.llm_provider if settings.llm_provider in _PROVIDERS else "local"

_PERSIST_KEY = "llm_provider"

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


async def persist_provider(name: str) -> None:
    """Durably remember the active provider so a restart resumes on it. Called
    by PUT /settings after set_provider() succeeds - keep the two separate so
    every other call site (tests especially) can keep using the fast, sync
    set_provider() without needing a DB or an event loop."""
    async with async_session() as db:
        row = await db.get(Setting, _PERSIST_KEY)
        if row is None:
            db.add(Setting(key=_PERSIST_KEY, value=name))
        else:
            row.value = name
        await db.commit()


async def load_persisted_provider() -> None:
    """Called once at startup: apply a previously-persisted provider choice,
    if there is one and its key is still configured. Best-effort - a missing
    key (e.g. GEMINI_API_KEY was removed since) just keeps the .env default
    rather than failing startup."""
    async with async_session() as db:
        row = await db.get(Setting, _PERSIST_KEY)
    if row is None:
        return
    try:
        set_provider(row.value)
    except ValueError:
        pass


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
