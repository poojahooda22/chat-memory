import uuid
from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

EMBEDDING_DIM = 1536


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Episode(SQLModel, table=True):
    """A single timestamped event with its context — the raw diary entry.

    Episodes are written once at the moment they happen (single-shot) and are
    never rewritten; semantic memories are distilled from them and link back here.
    """

    # pyrefly: ignore[bad-override]  — SQLModel types __tablename__ as a descriptor; plain-string assignment is its documented usage
    __tablename__ = "episodes"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: str = Field(default="default", index=True)
    conversation_id: str | None = Field(default=None, index=True)
    occurred_at: datetime = Field(default_factory=_utcnow, index=True)
    content: str = Field(sa_column=Column(Text, nullable=False))
    # contextual binding: who / where / why / source — the episodic paper's requirement
    context: dict = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False))
    embedding: list[float] | None = Field(
        default=None, sa_column=Column(Vector(EMBEDDING_DIM), nullable=True)
    )
    created_at: datetime = Field(default_factory=_utcnow)


class Memory(SQLModel, table=True):
    """A semantic fact distilled from one or more episodes, embedded for search."""

    # pyrefly: ignore[bad-override]
    __tablename__ = "memories"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: str = Field(default="default", index=True)
    content: str = Field(sa_column=Column(Text, nullable=False))
    embedding: list[float] | None = Field(
        default=None, sa_column=Column(Vector(EMBEDDING_DIM), nullable=True)
    )
    # provenance: which episodes this fact came from
    source_episode_ids: list[str] = Field(
        default_factory=list, sa_column=Column(JSONB, nullable=False)
    )
    is_deleted: bool = Field(default=False, index=True)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class MemoryHistory(SQLModel, table=True):
    """Audit trail: one row per ADD / UPDATE / DELETE applied to a memory."""

    # pyrefly: ignore[bad-override]
    __tablename__ = "memory_history"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    memory_id: uuid.UUID = Field(foreign_key="memories.id", index=True)
    event: str = Field(index=True)  # ADD | UPDATE | DELETE
    old_content: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    new_content: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    created_at: datetime = Field(default_factory=_utcnow)


class IngestJob(SQLModel, table=True):
    """One uploaded image working its way into memory.

    The request path writes ONLY this row (plus the file on disk) and returns 202; the
    worker reads it, runs the vision call, and inserts the Episode exactly once — so the
    Episode keeps its single-shot invariant (never a mutable placeholder). `episode_id`
    doubles as the idempotency guard: a retry of a job that already produced an episode
    can never insert a second one.
    """

    # pyrefly: ignore[bad-override]
    __tablename__ = "ingest_jobs"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: str = Field(default="default", index=True)
    kind: str = Field(default="photo")  # photo | screenshot
    status: str = Field(default="queued", index=True)  # queued | processing | done | failed
    filename: str = Field(default="")
    content_type: str = Field(default="image/jpeg")
    image_path: str = Field(sa_column=Column(Text, nullable=False))
    # parsed at upload time: captured_at, lat/lon, camera, and which source occurred_at used
    exif: dict = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False))
    episode_id: uuid.UUID | None = Field(default=None, foreign_key="episodes.id")
    error: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class Entity(SQLModel, table=True):
    """A named thing in the user's life — a person, pet, or object the user has labeled.

    Created the moment the user names a detected entity ("this is Monty"). The label is new
    knowledge arriving AFTER the photo happened, so it lives here — never written back into
    the episode, which stays single-shot. Entities are the nodes the relationship graph
    (plan §12.4) connects; co-occurrence is read off the episode_entities links.
    """

    # pyrefly: ignore[bad-override]
    __tablename__ = "entities"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    user_id: str = Field(default="default", index=True)
    name: str = Field(index=True)  # the user's label: "Monty", "Akshay", "me"
    type: str = Field(default="object")  # person | pet | object
    description: str = Field(default="", sa_column=Column(Text, nullable=False))
    embedding: list[float] | None = Field(
        default=None, sa_column=Column(Vector(EMBEDDING_DIM), nullable=True)
    )
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class EpisodeEntity(SQLModel, table=True):
    """Link: this entity appears in this episode (at detected-entity slot entity_index).

    The substrate of the future graph: two entities co-occur when they share an episode_id.
    """

    # pyrefly: ignore[bad-override]
    __tablename__ = "episode_entities"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    episode_id: uuid.UUID = Field(foreign_key="episodes.id", index=True)
    entity_id: uuid.UUID = Field(foreign_key="entities.id", index=True)
    entity_index: int = Field(default=0)  # which chip in the episode's context.entities
    created_at: datetime = Field(default_factory=_utcnow)


class ConversationSummary(SQLModel, table=True):
    """Rolling summary per conversation, consumed by the extraction step."""

    # pyrefly: ignore[bad-override]
    __tablename__ = "conversation_summaries"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    conversation_id: str = Field(unique=True, index=True)
    summary: str = Field(default="", sa_column=Column(Text, nullable=False))
    updated_at: datetime = Field(default_factory=_utcnow)