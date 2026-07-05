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

    __tablename__ = "memory_history"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    memory_id: uuid.UUID = Field(foreign_key="memories.id", index=True)
    event: str = Field(index=True)  # ADD | UPDATE | DELETE
    old_content: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    new_content: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    created_at: datetime = Field(default_factory=_utcnow)


class ConversationSummary(SQLModel, table=True):
    """Rolling summary per conversation, consumed by the extraction step."""

    __tablename__ = "conversation_summaries"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    conversation_id: str = Field(unique=True, index=True)
    summary: str = Field(default="", sa_column=Column(Text, nullable=False))
    updated_at: datetime = Field(default_factory=_utcnow)