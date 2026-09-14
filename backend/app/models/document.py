import uuid
from datetime import datetime

from sqlalchemy import String, DateTime, Integer, ForeignKey, Text, LargeBinary, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def gen_uuid() -> str:
    return str(uuid.uuid4())


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=gen_uuid)
    filename: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="processing")  # processing | ready | failed
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=gen_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    page_number: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    chunk_index: Mapped[int] = mapped_column(Integer)
    # Embedding vector stored as raw float32 bytes (np.ndarray.tobytes()).
    # NULL until the chunk has been embedded.
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    # Milestone 3: set once this chunk has been run through graph extraction,
    # so POST /graph/build is resumable and doesn't re-process chunks.
    graph_extracted: Mapped[bool] = mapped_column(Boolean, default=False)

    document: Mapped["Document"] = relationship(back_populates="chunks")


class Entity(Base):
    """A concept / person / org / etc. Deduplicated across all documents:
    first by exact norm_name, then by embedding similarity of the name."""
    __tablename__ = "entities"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=gen_uuid)
    name: Mapped[str] = mapped_column(String)                         # display form
    norm_name: Mapped[str] = mapped_column(String, unique=True, index=True)  # exact-match dedup key
    type: Mapped[str] = mapped_column(String)                         # PERSON | ORGANIZATION | CONCEPT | ...
    # Embedding of `name`, float32 bytes - used for fuzzy dedup ("OpenAI" ~ "OpenAI, Inc.").
    name_embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Relationship(Base):
    """A directed, labeled edge between two entities, traceable to the chunk it came from."""
    __tablename__ = "relationships"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=gen_uuid)
    source_id: Mapped[str] = mapped_column(ForeignKey("entities.id"))
    target_id: Mapped[str] = mapped_column(ForeignKey("entities.id"))
    label: Mapped[str] = mapped_column(String)                        # short verb phrase
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id"))    # provenance
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))  # for filtering / cleanup
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class EntityMention(Base):
    """Every chunk an entity was named in - even when that chunk produced no
    relationship. Backs the "where does this entity appear" panel in the UI."""
    __tablename__ = "entity_mentions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=gen_uuid)
    entity_id: Mapped[str] = mapped_column(ForeignKey("entities.id"), index=True)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id"))
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    surface_form: Mapped[str] = mapped_column(String)  # the entity name as written in this chunk


class MergeHistory(Base):
    """An undo record for one entity merge (manual or from /graph/consolidate).
    Captures the pre-merge state of everything the merge touched - the deleted
    entities, and every relationship/mention that got repointed or removed - as
    JSON blobs, so POST /graph/merges/{id}/undo can restore it verbatim. Not
    normalized into real FK'd rows on purpose: the whole point is to still have
    the data around after the live rows it describes are gone."""
    __tablename__ = "merge_history"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=gen_uuid)
    keep_id: Mapped[str] = mapped_column(String)   # not a FK - this entity may itself be merged away later
    keep_name: Mapped[str] = mapped_column(String)  # snapshot, so history reads fine even if keep_id changes
    entities_json: Mapped[str] = mapped_column(Text)        # deleted Entity rows, pre-merge
    relationships_json: Mapped[str] = mapped_column(Text)   # affected Relationship rows, pre-merge
    mentions_json: Mapped[str] = mapped_column(Text)        # affected EntityMention rows, pre-merge
    undone: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
