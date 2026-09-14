# DocMind

Upload PDFs, chat with them, and (soon) see a knowledge graph of how
concepts connect across documents.

## Current status (Milestones 1–4: done)
- FastAPI backend
- Upload a PDF -> extract text per page (PyMuPDF) -> chunk -> embed -> store in SQLite
- List / delete documents
- **RAG chat with citations** — ask a question, get an answer grounded in the
  uploaded docs plus the chunks (filename + page) it used
- **Cross-document knowledge graph** — extract entities + relationships from every
  chunk with a local LLM, dedupe entities across documents
- **React frontend** (`frontend/`) — documents, chat, and an interactive
  force-directed graph; click a node for its type, connections, and source chunks
- **Local / Claude / Gemini** — a header toggle switches chat, graph extraction
  and figure description between the Ollama models, Claude, and Google Gemini
  (embeddings stay local). A cloud provider is only offered when its key is set.

Runs fully local by default: `all-MiniLM-L6-v2` for embeddings, cosine similarity
for retrieval, `llama3.2:3b` for chat, and `qwen2.5:7b` for graph extraction —
all via Ollama, no API keys. Set `ANTHROPIC_API_KEY` or `GEMINI_API_KEY`
(Gemini has a free tier) to enable the other providers.

## Quick start (Docker)

One command, no Python/Node/Ollama install:

```bash
cp .env.example .env     # optional: add ANTHROPIC_API_KEY / GEMINI_API_KEY, pick a provider
docker compose up --build
```

Then open **http://localhost:8080**. The backend, the built frontend (served by
nginx, which also proxies `/api` → backend), and a persistent volume for the
SQLite DB + uploads all come up together. The demo document + knowledge graph is
seeded on first start, so every feature is usable immediately.

- **Cloud provider (recommended):** put a key in `.env`, set `LLM_PROVIDER=gemini`
  (or `anthropic`), `docker compose up`. Chat/graph/figures use it; embeddings
  stay local. Gemini's free tier needs no card.
- **Fully local:** `docker compose --profile local up --build` also starts an
  `ollama` container. Pull models into it once:
  ```bash
  docker compose exec ollama ollama pull llama3.2:3b
  docker compose exec ollama ollama pull qwen2.5:7b
  ```
  (CPU-only inference is slow — minutes per graph chunk. A cloud provider is far
  faster.) To reuse Ollama already running on your host instead, skip the profile
  and set `OLLAMA_BASE_URL=http://host.docker.internal:11434` in `.env`.
- `WEB_PORT` in `.env` changes the `8080` host port.
- `docker compose down` stops it; `docker compose down -v` also wipes the DB +
  uploads volume.

The rest of this README covers running the two services directly for development.

## Prerequisites
- Python 3.13 (3.14 has no pymupdf/scipy/torch wheels yet)
- [Ollama](https://ollama.com) running, with both models pulled:
  ```bash
  ollama pull llama3.2:3b     # chat
  ollama pull qwen2.5:7b      # knowledge-graph extraction
  ```

## Running the backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # defaults are fine for local dev
uvicorn app.main:app --reload
```

First run downloads the embedding model (~90MB) on the first upload.

To enable a cloud provider, set its key in `backend/.env`:
- `ANTHROPIC_API_KEY` (+ optional `ANTHROPIC_MODEL=claude-sonnet-5` for ~2.5× lower
  cost than the `claude-opus-5` default) — pay-as-you-go
- `GEMINI_API_KEY` from [aistudio.google.com](https://aistudio.google.com) —
  **free tier**, no card; `GEMINI_MODEL` defaults to `gemini-3.6-flash`

Without any key the app is 100% local and both cloud toggles are disabled.
Restart the backend after editing `.env` (`--reload` doesn't re-read it).

Cloud calls retry transient 429/5xx with backoff (`call_with_retry` in
`provider.py`) — the Gemini free tier returns `503 "high demand"` fairly often,
and a graph build tolerates per-chunk failures (`failed` / `last_error` in the
build status), so re-running picks up whatever didn't land.

> **Windows Smart App Control note:** if SAC is on, the *first* import of a
> freshly installed native package (scipy, torch, …) can fail with
> "An Application Control policy has blocked this file." Re-run the command —
> SAC resolves the file's reputation after the first hit and subsequent runs pass.

### Tests
```bash
cd backend
pip install -r requirements-dev.txt
pytest -m "not slow"    # fast: no model download, ~10s
pytest                  # also runs the real-embedding-model test
```
`tests/conftest.py` gives each test a fresh in-memory SQLite DB and stubs the
embedding model + the LLM, so nothing hits the network (and blanks the `.env`
API keys so results don't depend on your local config). ~100 tests covering:
the structure-aware chunker (budget, sentence boundaries, overlap, hard-split) ·
entity-dedup helpers · `_excerpt` · vector (de)serialization ·
documents CRUD + cascade delete · chat (citations, errors) · **graph build**
(dedup, `force` re-map, per-chunk failure tolerance, incremental) · `/graph`,
`/coverage`, `/search` (each match type + ranking), `/entities/{id}` ·
`/entities/merge` (edge re-point, self-loop/dup removal, orphan cleanup) ·
`/settings` provider validation · the demo seed · `/chat/stream` SSE framing
(citations-then-tokens-then-done, a mid-stream provider error surfacing as an
`error` event instead of breaking the connection, and a failure before any
token is sent) · `/graph/consolidate` (a valid merge applies; unknown ids,
cross-type groups, numeric-sibling names, and overlapping groups are all
dropped without crashing; a provider error is a 502) · merge undo (a merge with
a self-loop and a duplicate-edge collapse restores byte-for-byte on undo,
including the mention; can't undo twice; refuses once the keep entity has
itself been merged away) · **provider dispatch** — `test_provider_branches.py`
mocks the Claude / Gemini / Ollama SDK client objects and runs the real
`generate_answer` / `stream_answer` / `extract_from_chunk` / `consolidate_entities`
/ `describe_page`, checking each routes to the right branch, builds the call with
the expected model + system prompt + message/image shape, pulls the answer back
out of that provider's response object, retries a transient error, and falls
back to local when a cloud key is missing.

Server runs at http://localhost:8000. Interactive API docs at
http://localhost:8000/docs.

## Running the frontend

```bash
cd frontend
npm install
npm run dev
```

Opens at http://localhost:5173 and proxies `/api/*` to the backend on :8000
(configured in `vite.config.ts` — no CORS or hard-coded host). Three tabs:

- **Documents** — upload, list, delete
- **Chat** — ask a question; the answer **streams in token by token** (Server-Sent
  Events), with its source chunks shown as soon as retrieval finishes, before
  the model has said a word. The `[1]` / `[2, 4]` citation markers in the answer
  are **clickable** — each scrolls to its source card, rings it, and expands the
  full chunk text; source cards also toggle expand on click. Entity names in the
  answer that exist in the knowledge graph are **also links** — click one to jump
  to the graph tab with that node selected and centred
- **Knowledge graph** — a document picker (with per-doc "X/Y mapped" badges) plus
  "Build selected" / "re-map" runs extraction with a live progress bar. In the
  force-directed view: **search** entities by name, type, a relationship label, a
  connected entity, or the text around where they're mentioned — matches are
  ringed, the rest dimmed, and a results list lets you pick which one to focus
  (it centers the graph on that node and opens its panel); **drag** an individual
  node to reposition it (the layout freezes
  after the first settle, so the rest of the graph stays put — hit **re-layout**
  to re-run it); **click** a node for a panel with its type, degree, every
  relationship (source doc + page + text snippet), and every chunk that mentions
  it. Clicking a related entity in the panel jumps to that node; **"Ask in chat ↗"**
  in the panel switches to the Chat tab and asks about that entity.
- header **Local / Claude / Gemini** toggle — switches provider for all three tabs

On first startup the backend seeds a **"DocMind demo.pdf"** document with a
ready-made 12-entity / 14-relationship graph (and chunk embeddings, so chat works
on it too), so every UI feature is testable without an upload + build. Delete
that document from the Documents tab to remove it, or set `SEED_DEMO=false` in
`backend/.env` to stop it being re-created.

`npm run build` type-checks and produces a static bundle in `frontend/dist/`.

### Try it
```bash
curl -X POST -F "file=@yourfile.pdf" http://localhost:8000/documents/upload
curl http://localhost:8000/documents/
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What does this document say about X?"}'
```

`/chat` body: `{"question": str, "document_ids": [str]?, "top_k": int?}`.
Omit `document_ids` to search every uploaded doc.

### Streaming chat
The frontend uses `POST /chat/stream` instead of `/chat` - same body, but the
response is Server-Sent Events instead of one JSON blob:
```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What does this document say about X?"}'
```
```
event: citations
data: {"citations": [...]}

event: token
data: {"text": "The"}

event: token
data: {"text": " document"}
...
event: done
data: {}
```
Citations arrive as soon as retrieval finishes (before the model has said a
word); `token` events carry each text delta; a mid-generation provider failure
arrives as an `error` event instead of breaking the connection. `/chat` (the
non-streaming JSON endpoint) is unchanged and still used by the test suite.

### Figure understanding (experimental)
`extract_pages` normally reads only the PDF text layer, so charts, diagrams and
scanned images contribute nothing. Upload with `?describe_figures=true` (or tick
the box in the Documents tab) to render each figure-bearing page and pass it to a
local vision model (`VISION_MODEL`, default `granite3.2-vision`); its description
is appended to that page's text and flows into chunking, chat and the graph.

```bash
curl -X POST -F "file=@chart.pdf" "http://localhost:8000/documents/upload?describe_figures=true"
```

**Status: wired up, slow and weak on a CPU-only machine with the local model.**
`granite3.2-vision` on CPU took 3–7 min/page and misread a simple bar chart.
`describe_page` follows the provider toggle. With **Gemini** it read a bar chart
correctly (axes, every bar value, trend) in ~6s vs. 3–7 min and wrong locally —
switch to Gemini or Claude for figures. For a local-only setup you'd want GPU
acceleration for Ollama (Intel `ipex-llm` for Arc, or NVIDIA).

### Knowledge graph
```bash
curl -X POST "http://localhost:8000/graph/build"                     # build all pending chunks
curl -X POST "http://localhost:8000/graph/build?document_ids=<id>&document_ids=<id>"   # only these
curl -X POST "http://localhost:8000/graph/build?document_ids=<id>&force=true"          # re-map from scratch
curl "http://localhost:8000/graph/build/status"                      # poll progress
curl "http://localhost:8000/graph/coverage"                          # per-doc mapped/total chunks
curl "http://localhost:8000/graph"                                   # {nodes, edges} JSON
# the frontend Knowledge-graph tab has a document picker + progress for all of this
```
`build` runs one LLM call per chunk (~30s each on CPU, seconds with the Claude
toggle), so it returns immediately and runs in the background. **Incremental by
default** — chunks already mapped (`graph_extracted`) are skipped, so re-running
only touches new documents. `force=true` (with `document_ids`) clears those docs'
entities/edges/mentions and re-maps them. `qwen2.5:7b` is the default local
extractor; `EXTRACTION_MODEL=llama3.2:3b` for a faster, rougher pass.

Entities are deduped across documents in two passes: exact normalized-name match,
then embedding similarity (cosine ≥ `ENTITY_MERGE_THRESHOLD`, default 0.87, same
type only, never when the names differ solely by a number like `GPT-4`/`GPT-3`).
Borderline pairs it won't auto-merge (`"OpenAI"` vs `"OpenAI, Inc."` ≈ 0.85) are
handled manually:
```bash
curl -X POST http://localhost:8000/graph/entities/merge \
  -H "Content-Type: application/json" \
  -d '{"keep_id": "<id>", "merge_ids": ["<id>", "<id>"]}'
```
which re-points the merged entities' edges onto `keep_id`, drops the resulting
self-loops and duplicate edges, and deletes them. The response includes a
`merge_id` - see Undo below.

### Entity consolidation
Those two passes only catch surface variation. A third, on-demand pass closes
the acronym/synonym gap they can't - "RAG" fusing with "retrieval-augmented
generation", "NFU" with "Northfield University":
```bash
curl -X POST http://localhost:8000/graph/consolidate
```
One LLM call sees every entity's `{id, name, type}` at once and proposes merge
groups; each is re-validated before being applied through the same path as
`/entities/merge` - unknown ids, cross-type groups, and numeric-sibling names
(`GPT-4`/`GPT-3`) are all dropped. **That only guards against mechanical
mistakes, not a wrong-but-plausible same-type merge.** In testing, the local
`qwen2.5:7b` merged two of the demo graph's entities that share nothing but a
type - two distinct organizations - while Claude and Gemini got the same
entity list right every time tried. **Use a cloud provider for this**,
and read the response, which lists exactly what got merged into what:
```json
{"provider": "gemini", "groups": [{"keep_name": "Retrieval-Augmented Generation", "merged": [{"name": "RAG"}], "merge_id": "<id>"}], "removed_duplicate_edges": 0}
```
The frontend's **Consolidate synonyms** button (Knowledge graph tab) shows the
same merge list, flagged amber when it ran on the local provider, and opens the
history panel below so you can undo on the spot.

### Undoing a merge
Every merge - manual or from consolidation - is reversible:
```bash
curl -X POST http://localhost:8000/graph/merges/<merge_id>/undo
curl http://localhost:8000/graph/merges          # recent merges, most recent first, with an `undone` flag
```
A merge overwrites/deletes rows in place, so undo works by snapshotting the
pre-merge state (the deleted entities, and every relationship/mention the merge
touched) into a `merge_history` row *before* mutating anything, then restoring
from that snapshot on request - re-inserting the entity, restoring repointed
relationships and mentions to their original ids, re-inserting any relationship
that got deleted as a self-loop or duplicate. Can't undo the same merge twice,
and refuses (409) if the entity it was merged into has itself since been merged
away by a *later* merge - undo that one first. The frontend's **History**
button (Knowledge graph tab) lists recent merges with an Undo button on each.

## Project layout
```
backend/
  app/
    main.py               # FastAPI app + CORS + startup (runs demo_seed)
    config.py             # settings (env-driven)
    database.py           # async SQLAlchemy engine/session
    demo_seed.py          # "DocMind demo.pdf" + prebuilt graph, seeded once on startup
    models/document.py    # Document, Chunk, Entity, Relationship, EntityMention, MergeHistory
    services/
      provider.py         # active provider (local | anthropic | gemini) + clients
      pdf_parser.py       # text extraction + chunking (+ optional figure description)
      embeddings.py       # sentence-transformers wrapper + vector (de)serialization
      retrieval.py        # brute-force cosine similarity ranking
      llm.py              # chat answer - Ollama / Claude / Gemini
      graph_extraction.py # per-chunk extraction (all 3 providers) + entity-name dedup
      vision.py           # figure description - Ollama VLM / Claude / Gemini
    routers/
      documents.py        # upload (parse+chunk+embed) / list / delete
      chat.py             # POST /chat -> {answer, citations}; POST /chat/stream -> SSE
      graph.py            # /graph/build (+document_ids,force), /coverage, /search, /graph, /entities/{id},
                          #   /entities/merge, /consolidate, /merges, /merges/{id}/undo, /view
      settings.py         # GET/PUT /settings - read + switch the provider
  tests/                  # pytest: conftest fixtures + unit + API tests
  requirements.txt
  requirements-dev.txt
frontend/
  src/
    api.ts               # typed fetch client (talks to /api/*)
    types.ts             # shared API types
    App.tsx              # tab layout
    components/
      DocumentsView.tsx  ChatView.tsx  GraphView.tsx  EntityPanel.tsx
  vite.config.ts         # Tailwind plugin + /api -> :8000 proxy
```

## Provider switching
`services/provider.py` holds the active provider (`local` | `anthropic` |
`gemini`) as process-wide in-memory state, seeded from `LLM_PROVIDER` and flipped
by `PUT /settings`. `llm.py`, `graph_extraction.py` and `vision.py` each branch on
`get_provider()`; a provider whose key isn't set falls back to `local`.
Embeddings never switch — retrieval must stay comparable across a session, and
neither cloud SDK is wired for embeddings here. Claude uses `messages.create` /
`messages.parse`; Gemini uses `generate_content` with `response_schema` for
extraction. The switch is not persisted; a restart returns to the `.env` default.
Existing graph edges are **not** re-extracted on switch — use `force=true` on
`/graph/build` to redo specific documents with the new provider.

## Design notes for future-me
- Streaming (`services/llm.py::stream_answer`, `routers/chat.py`): each provider's
  SDK streaming call (Ollama `stream=True`, Anthropic `messages.stream()`, Gemini
  `generate_content_stream`) is a *blocking* sync generator. `_bridge_sync_generator`
  runs it in a background thread and re-yields items onto the event loop via an
  `asyncio.Queue`, so a slow model never blocks other requests. Streaming calls
  are **not** wrapped in `call_with_retry` - retrying after tokens have already
  reached the client would duplicate text - so a mid-stream provider error
  becomes an `error` SSE event instead. `POST /chat` (non-streaming) keeps its
  retry-on-transient-error behavior.
- `pdf_parser.py` uses PyMuPDF for speed. If table/layout accuracy
  becomes a problem (it will, for anything with tables), swap in
  Docling — same function signature, just change the internals.
- Chunking (`pdf_parser.py::chunk_page_text`) is structure-aware: it greedily
  packs whole paragraphs — then whole sentences, for an over-budget paragraph —
  up to a 1000-char budget, so a chunk ends mid-sentence only when a single
  sentence is itself longer than the budget. Consecutive chunks overlap by
  ~150 chars of whole trailing sentences. Sized in characters (no tokenizer
  dependency); 1000 keeps typical prose inside all-MiniLM-L6-v2's 256-token
  window. A true token count (the model's own tokenizer) would be the next
  refinement if truncation ever bites.
- Retrieval is brute-force cosine in NumPy (`services/retrieval.py`). Fine for
  hundreds–thousands of chunks. Swap in Chroma / pgvector / sqlite-vec there
  if the corpus grows — nothing else needs to change.
- Embeddings are computed synchronously inside the upload request. If uploads
  get slow, move that to a background task and let `status` go
  `processing -> ready`.
- Embedding vectors live in the `Chunk.embedding` BLOB column. There's no
  migration tooling — schema changes currently mean deleting the dev
  `docmind.db` and letting `init_db()` recreate it.
- Graph extraction is LLM-based (`graph_extraction.py`). Entity dedup is three
  passes now: exact-name, embedding-similarity, and on-demand LLM consolidation
  (see the graph section above) — the last one closes the acronym/synonym gap
  but is only as reliable as the model running it (verified: local `qwen2.5:7b`
  produced a false-positive merge that Claude/Gemini didn't). `qwen2.5:7b`
  isn't consistent about edge-label style (mixes "built with" and `BUILT_WITH`);
  normalize in `get_graph` if it bothers you.
- Graph build state (`/graph/build/status`) is in-memory, so it resets on server
  restart. That's fine because the job is resumable via `Chunk.graph_extracted`.
- SQLite is intentionally the DB for now — zero setup. Swap
  `DATABASE_URL` for Postgres (e.g. Supabase) later without touching
  model code, since we're using SQLAlchemy.
