"""
A ready-made demo document + knowledge graph, inserted on startup (once) so the
graph UI - search, drag, the entity panel - can be exercised without uploading a
PDF and waiting for a build. Controlled by settings.seed_demo. Delete the
"DocMind demo.pdf" document from the UI to remove it.
"""
from sqlalchemy import select

from app.database import async_session
from app.models.document import Chunk, Document, Entity, EntityMention, Relationship
from app.services.graph_extraction import normalize_name

DEMO_FILENAME = "DocMind demo.pdf"

_CHUNKS = [
    (
        "DocMind demo - architecture. DocMind is a retrieval-augmented generation "
        "system. Its FastAPI backend extracts text from PDFs with PyMuPDF, splits it "
        "into chunks, and stores them in SQLite. Every chunk is embedded with the "
        "all-MiniLM-L6-v2 model, and cosine similarity ranks the chunks at query "
        "time. A local model served by Ollama writes the final answer with citations."
    ),
    (
        "DocMind demo - background. DocMind was created by a developer at Northfield "
        "University, during an internship at Vantage AI Labs. It extends OCR and "
        "knowledge graph research from that internship. The knowledge graph links "
        "concepts and entities across every uploaded document."
    ),
]

_ENTITIES: list[tuple[str, str]] = [
    ("DocMind", "TECHNOLOGY"),
    ("FastAPI", "TECHNOLOGY"),
    ("PyMuPDF", "TECHNOLOGY"),
    ("SQLite", "TECHNOLOGY"),
    ("all-MiniLM-L6-v2", "TECHNOLOGY"),
    ("Ollama", "TECHNOLOGY"),
    ("Cosine Similarity", "CONCEPT"),
    ("Retrieval-Augmented Generation", "CONCEPT"),
    ("Knowledge Graph", "CONCEPT"),
    ("Demo Author", "PERSON"),
    ("Northfield University", "ORGANIZATION"),
    ("Vantage AI Labs", "ORGANIZATION"),
]

# (source, label, target, chunk index)
_RELATIONSHIPS: list[tuple[str, str, str, int]] = [
    ("DocMind", "built with", "FastAPI", 0),
    ("DocMind", "extracts text with", "PyMuPDF", 0),
    ("DocMind", "stores chunks in", "SQLite", 0),
    ("DocMind", "embeds chunks with", "all-MiniLM-L6-v2", 0),
    ("DocMind", "runs models via", "Ollama", 0),
    ("DocMind", "implements", "Retrieval-Augmented Generation", 0),
    ("Retrieval-Augmented Generation", "ranks with", "Cosine Similarity", 0),
    ("all-MiniLM-L6-v2", "produces vectors for", "Cosine Similarity", 0),
    ("Demo Author", "created", "DocMind", 1),
    ("Demo Author", "studies at", "Northfield University", 1),
    ("Demo Author", "interned at", "Vantage AI Labs", 1),
    ("DocMind", "extends research from", "Vantage AI Labs", 1),
    ("DocMind", "builds", "Knowledge Graph", 1),
    ("Vantage AI Labs", "researches", "Knowledge Graph", 1),
]

# entity name -> chunk indices it is mentioned in
_MENTIONS: dict[str, list[int]] = {
    "DocMind": [0, 1],
    "FastAPI": [0],
    "PyMuPDF": [0],
    "SQLite": [0],
    "all-MiniLM-L6-v2": [0],
    "Ollama": [0],
    "Cosine Similarity": [0],
    "Retrieval-Augmented Generation": [0],
    "Demo Author": [1],
    "Northfield University": [1],
    "Vantage AI Labs": [1],
    "Knowledge Graph": [1],
}


async def seed_demo() -> None:
    async with async_session() as db:
        exists = (
            await db.execute(select(Document).where(Document.filename == DEMO_FILENAME))
        ).scalar_one_or_none()
        if exists:
            return

        # Embeddings are best-effort: with them the demo doc also works in chat;
        # without them the graph is still fully populated.
        embed_ok = False
        try:
            from app.services.embeddings import embed_texts, to_bytes

            chunk_vecs = embed_texts(_CHUNKS)
            name_vecs = embed_texts([name for name, _ in _ENTITIES])
            embed_ok = True
        except Exception as e:  # noqa: BLE001
            print(f"demo seed: embeddings unavailable ({e}); seeding graph only")

        doc = Document(filename=DEMO_FILENAME, status="ready", page_count=len(_CHUNKS))
        db.add(doc)
        await db.flush()

        chunks: list[Chunk] = []
        for i, text in enumerate(_CHUNKS):
            c = Chunk(
                document_id=doc.id,
                page_number=i + 1,
                text=text,
                chunk_index=i,
                graph_extracted=True,
                embedding=to_bytes(chunk_vecs[i]) if embed_ok else None,
            )
            db.add(c)
            chunks.append(c)
        await db.flush()

        ent_id: dict[str, str] = {}
        for idx, (name, etype) in enumerate(_ENTITIES):
            entity = Entity(
                name=name,
                norm_name=normalize_name(name),
                type=etype,
                name_embedding=to_bytes(name_vecs[idx]) if embed_ok else None,
            )
            db.add(entity)
            await db.flush()
            ent_id[name] = entity.id

        for src, label, tgt, ci in _RELATIONSHIPS:
            db.add(
                Relationship(
                    source_id=ent_id[src],
                    target_id=ent_id[tgt],
                    label=label,
                    chunk_id=chunks[ci].id,
                    document_id=doc.id,
                )
            )

        for name, chunk_indices in _MENTIONS.items():
            for ci in chunk_indices:
                db.add(
                    EntityMention(
                        entity_id=ent_id[name],
                        chunk_id=chunks[ci].id,
                        document_id=doc.id,
                        surface_form=name,
                    )
                )

        await db.commit()
        print(f"demo seed: created '{DEMO_FILENAME}' "
              f"({len(_ENTITIES)} entities, {len(_RELATIONSHIPS)} relationships)")
