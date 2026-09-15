from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

engine = create_async_engine(settings.database_url, echo=False)

if engine.dialect.name == "sqlite":
    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, connection_record):
        """WAL lets a read (e.g. a chat request) proceed while a write (e.g. the
        background graph build, which commits once per chunk) is in progress,
        instead of blocking behind the whole file; NORMAL synchronous trades a
        sliver of durability-on-power-loss for materially faster commits - a
        fine trade for a local dev/demo database. No-op on the in-memory DB the
        test suite uses (SQLite doesn't support WAL there; it just stays put)."""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

async_session = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session
