"""
Shared test fixtures.

- `client`  - an httpx AsyncClient wired to the FastAPI app with a fresh in-memory
              SQLite database (the app's db plumbing is monkeypatched to it, and
              the `get_db` dependency is overridden).
- `db`      - an AsyncSession on that same database, for seeding / assertions.
- `fake_embeddings` - replaces the sentence-transformers calls with cheap
              deterministic vectors so tests don't download or run the model.
- `mock_llm` - replaces chat answer generation with a canned string.
- `make_pdf` - builds a tiny PDF (bytes) with real text.

The app lifespan is not run under the test transport, so `init_db` and the demo
seed do NOT fire automatically - tests start from an empty schema.
"""
import hashlib

import numpy as np
import pytest
import pytest_asyncio
import pymupdf
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import app.models.document  # noqa: F401 - register tables on Base.metadata
from app.database import Base

_EMBED_DIM = 16


@pytest_asyncio.fixture
async def db_engine():
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
def sessionmaker_(db_engine):
    return async_sessionmaker(db_engine, expire_on_commit=False)


@pytest.fixture(autouse=True)
def _tmp_upload_dir(tmp_path, monkeypatch):
    d = tmp_path / "uploads"
    d.mkdir()
    monkeypatch.setattr("app.config.settings.upload_dir", str(d))


@pytest.fixture
def patch_db(db_engine, sessionmaker_, monkeypatch):
    """Point every place that grabs a DB session at the test database."""
    monkeypatch.setattr("app.database.engine", db_engine)
    monkeypatch.setattr("app.database.async_session", sessionmaker_)
    monkeypatch.setattr("app.routers.graph.async_session", sessionmaker_, raising=False)
    monkeypatch.setattr("app.routers.documents.async_session", sessionmaker_, raising=False)
    monkeypatch.setattr("app.demo_seed.async_session", sessionmaker_, raising=False)
    return sessionmaker_


@pytest_asyncio.fixture
async def client(patch_db):
    from app.database import get_db
    from app.main import app

    async def _get_db():
        async with patch_db() as session:
            yield session

    app.dependency_overrides[get_db] = _get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def db(sessionmaker_):
    async with sessionmaker_() as session:
        yield session


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """The graph build status dict and the active-provider flag are module
    globals - reset them so tests don't leak state into each other."""
    from app.routers import documents, graph

    graph._build.update(
        running=False, processed=0, failed=0, total=0, new_entities=0,
        new_relationships=0, remaining_chunks=None, error=None, last_error=None,
        started_at=None, finished_at=None,
    )
    monkeypatch.setattr("app.routers.graph._build_task", None, raising=False)
    documents._upload_tasks.clear()
    monkeypatch.setattr("app.services.provider._active", "local", raising=False)
    monkeypatch.setattr("app.services.provider._anthropic", None, raising=False)
    monkeypatch.setattr("app.services.provider._gemini", None, raising=False)

    # Tests must not depend on whatever is in the developer's .env.
    monkeypatch.setattr("app.config.settings.anthropic_api_key", "", raising=False)
    monkeypatch.setattr("app.config.settings.gemini_api_key", "", raising=False)


@pytest.fixture
def finish_build():
    """Returns a coroutine that awaits the background task from POST /graph/build."""

    async def _finish():
        from app.routers import graph

        if graph._build_task is not None:
            await graph._build_task

    return _finish


@pytest.fixture
def finish_upload():
    """Returns a coroutine that awaits POST /documents/upload's background
    parse/chunk/embed task for a given document id (a no-op if it's already
    finished or unknown)."""

    async def _finish(document_id: str):
        from app.routers import documents

        task = documents._upload_tasks.get(document_id)
        if task is not None:
            await task

    return _finish


def _fake_vec(text: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.md5(text.encode()).digest()[:4], "little")
    v = np.random.default_rng(seed).standard_normal(_EMBED_DIM).astype("float32")
    return v / (float(np.linalg.norm(v)) or 1.0)


@pytest.fixture
def fake_embeddings(monkeypatch):
    def embed_texts(texts):
        if not texts:
            return np.empty((0, _EMBED_DIM), dtype="float32")
        return np.stack([_fake_vec(t) for t in texts])

    def embed_query(text):
        return _fake_vec(text)

    for target in (
        "app.services.embeddings.embed_texts",
        "app.routers.documents.embed_texts",
        "app.demo_seed.embed_texts",
    ):
        monkeypatch.setattr(target, embed_texts, raising=False)
    for target in (
        "app.services.embeddings.embed_query",
        "app.routers.chat.embed_query",
        "app.routers.graph.embed_query",
    ):
        monkeypatch.setattr(target, embed_query, raising=False)


@pytest.fixture
def mock_llm(monkeypatch):
    def generate_answer(question, chunks):
        return f"Stub answer for {question!r} using [1] ({len(chunks)} chunks)."

    monkeypatch.setattr("app.routers.chat.generate_answer", generate_answer)
    return generate_answer


@pytest.fixture
def make_pdf():
    def _make(text: str = "Hello world. DocMind test document with some words.", pages: int = 1) -> bytes:
        doc = pymupdf.open()
        for i in range(pages):
            page = doc.new_page()
            page.insert_textbox(pymupdf.Rect(40, 40, 550, 780), f"Page {i + 1}. {text}", fontsize=11)
        data = doc.tobytes()
        doc.close()
        return data

    return _make
