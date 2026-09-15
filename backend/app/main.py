from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings as cfg
from app.database import init_db
from app.demo_seed import seed_demo
from app.routers import chat, documents, graph, settings
from app.services.provider import load_persisted_provider


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await load_persisted_provider()
    if cfg.seed_demo:
        try:
            await seed_demo()
        except Exception as e:  # noqa: BLE001 - a failed seed must not block startup
            print(f"demo seed skipped: {e}")
    yield


app = FastAPI(title="DocMind API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(graph.router)
app.include_router(settings.router)


@app.get("/")
async def root():
    return {"service": "DocMind API", "docs": "/docs", "health": "/health"}


@app.get("/health")
async def health():
    return {"status": "ok"}
