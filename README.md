# DocMind

A local-first RAG app: upload PDFs, chat and get cited answers, and
explore a knowledge graph of how entities connect across every document you
uploaded. Runs fully offline on local models (Ollama), or with Claude/Gemini
as a drop-in swap.

## Features

- **Upload and chat with citations** — PDFs are parsed, chunked, and embedded
  locally. Answers stream in token-by-token and cite the exact source chunks
  they're found in. Citation markers and entity names in the answer are
  clickable and jump straight to the source or the graph.
- **Cross-document knowledge graph** — entities and relationships are
  extracted from every chunk and deduplicated across documents (exact match →
  embedding similarity → an on-demand LLM consolidation pass for
  acronym/synonym duplicates). 
- **Interactive graph UI** — force-directed view with search, draggable nodes,
  and a detail panel per entity (relationships, source chunks, jump to chat).
- **Persists across reloads** — chat history and graph node positions (saved
  in the browser) and the active LLM provider (saved server-side) all survive
  a page refresh or a full stack restart.
- **Three interchangeable LLM providers** — local Ollama, Claude, or Gemini,
  for chat, extraction, and figure understanding. Embeddings always stay
  local. A provider is only offered in the UI once its API key is configured.
- **Dockerized** — one command brings up the backend, frontend, and a
  persistent volume. No local Python/Node/Ollama install required.

## Quick start

```bash
cp .env.example .env     # optional: add ANTHROPIC_API_KEY / GEMINI_API_KEY
docker compose up --build
```

Open **http://localhost:8080**. A demo document + prebuilt knowledge graph is
seeded on first start, so every feature — chat, graph search, entity
merging — is testable with no new document upload.

- **Cloud provider (recommended):** set `LLM_PROVIDER=gemini` (free or paid) or `anthropic` in `.env` alongside the key. Chat/graph/figures use it;
  embeddings stay local.
- **Fully local:** `docker compose --profile local up --build` also starts an
  Ollama container (pull models into it with `docker compose exec ollama
  ollama pull llama3.2:3b`). CPU-only inference is slow — a cloud provider is
  far faster for the graph-build step in particular.

## Tech stack

**Backend:** FastAPI, async SQLAlchemy + SQLite, PyMuPDF, sentence-transformers
(local embeddings), Anthropic + Google GenAI SDKs, pytest.
**Frontend:** React 19, TypeScript, Vite, Tailwind, react-force-graph-2d.
**Infra:** Docker Compose, nginx.

## How it works

- **Retrieval:** every chunk is embedded with `all-MiniLM-L6-v2`; a query is
  ranked against all chunks by cosine similarity (brute-force NumPy — no
  vector DB, fine up to a few thousand chunks) and the top matches are handed
  to the LLM as grounding context with numbered citations.
- **Streaming:** each provider's SDK exposes a different flavor of streaming
  (Ollama, Anthropic, Gemini all differ); a small bridge runs the blocking
  call in a background thread and re-yields it as Server-Sent Events, so a
  slow model never blocks the server.
- **Graph extraction:** one LLM call per chunk pulls out entities and
  relationships as structured output. New entities are deduped against
  existing ones in three passes — exact normalized name, embedding
  similarity, and an on-demand LLM pass that catches acronym/synonym
  duplicates the first two can't. Every merge snapshots the pre-merge state
  first, so it can be undone.
- **Provider dispatch:** a single active provider (local/Claude/Gemini) is
  shared by chat, extraction, and vision, with automatic fallback to local if
  a cloud key isn't configured. Embeddings never switch, so retrieval stays
  comparable within a session.

## Local development

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload   # http://localhost:8000, docs at /docs

# Frontend (separate terminal)
cd frontend
npm install
npm run dev   # http://localhost:5173, proxies /api to :8000
```

Needs Python 3.13 and, for local models, [Ollama](https://ollama.com) with
`ollama pull llama3.2:3b` and `ollama pull qwen2.5:7b`. Without any cloud key
the app runs 100% locally.

### Tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest -m "not slow"
```

~100 tests: the chunker, entity dedup, retrieval ranking, every API route,
graph build/merge/undo, SSE streaming, and a per-provider dispatch suite that
mocks the Claude/Gemini/Ollama clients and exercises the real request/response
handling for each.

## Project layout

```
backend/app/
  routers/       documents, chat (+ streaming), graph, settings
  services/      pdf parsing + chunking, embeddings, retrieval, LLM calls,
                 graph extraction, provider dispatch, vision
  models/        Document, Chunk, Entity, Relationship, EntityMention, MergeHistory
frontend/src/
  components/    DocumentsView, ChatView, GraphView, EntityPanel
  api.ts         typed client for the backend
```

## Notable design decisions

- Chunking is structure-aware (paragraph → sentence → word), not a blind
  character sliding window, so a chunk only breaks mid-sentence when a single
  sentence itself exceeds the budget.
- Streaming calls skip the retry-on-error wrapper used everywhere else,
  retrying after tokens have already reached the client would duplicate text,
  so a mid-stream failure surfaces as its own SSE `error` event instead.
- Entity-merge undo works by snapshotting every row a merge touches *before*
  mutating anything, not by trying to reverse-engineer the change after.
- Retrieval and the graph store are intentionally simple (NumPy cosine, SQLite)
  with a clear swap-in point (`services/retrieval.py`, `DATABASE_URL`) if the
  corpus outgrows them, no need to over-build for an MVP.
- Upload parsing/chunking/embedding runs in a background task, not the request -
  `POST /upload` returns as soon as the file is saved, and the UI polls the
  `processing → ready/failed` transition (`document.error` carries the reason
  on a failure), same idiom as the graph build's own progress polling.
- The active provider is kept in-memory for fast, synchronous reads on every
  LLM call, but `PUT /settings` also writes it to a one-row `Setting` table so
  a restart resumes on it instead of reverting to `.env`'s `LLM_PROVIDER`.

## License

MIT — see [LICENSE](LICENSE).
