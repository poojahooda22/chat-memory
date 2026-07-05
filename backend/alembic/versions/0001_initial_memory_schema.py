"""initial memory schema: episodes, memories, memory_history, conversation_summaries

Revision ID: 0001
Revises:
Create Date: 2026-07-05

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 1536


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "episodes",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("conversation_id", sa.String(), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("context", JSONB(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_episodes_user_id", "episodes", ["user_id"])
    op.create_index("ix_episodes_conversation_id", "episodes", ["conversation_id"])
    op.create_index("ix_episodes_occurred_at", "episodes", ["occurred_at"])

    op.create_table(
        "memories",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column("source_episode_ids", JSONB(), nullable=False),
        sa.Column("is_deleted", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_memories_user_id", "memories", ["user_id"])
    op.create_index("ix_memories_is_deleted", "memories", ["is_deleted"])

    op.create_table(
        "memory_history",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("memory_id", sa.Uuid(), sa.ForeignKey("memories.id"), nullable=False),
        sa.Column("event", sa.String(), nullable=False),
        sa.Column("old_content", sa.Text(), nullable=True),
        sa.Column("new_content", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_memory_history_memory_id", "memory_history", ["memory_id"])
    op.create_index("ix_memory_history_event", "memory_history", ["event"])

    op.create_table(
        "conversation_summaries",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("conversation_id", sa.String(), nullable=False, unique=True),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_conversation_summaries_conversation_id",
        "conversation_summaries",
        ["conversation_id"],
    )


def downgrade() -> None:
    op.drop_table("conversation_summaries")
    op.drop_table("memory_history")
    op.drop_table("memories")
    op.drop_table("episodes")
