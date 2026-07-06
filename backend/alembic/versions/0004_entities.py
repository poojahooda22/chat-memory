"""entities + episode links: the named things in the user's life

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-07

A label ("this is Monty") is knowledge arriving after the photo happened, so it lives in its
own table and links back to episodes — the episode row is never rewritten. episode_entities
is the substrate the relationship graph reads: co-occurrence = shared episode_id.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "entities",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_entities_user_id", "entities", ["user_id"])
    op.create_index("ix_entities_name", "entities", ["name"])
    op.create_index(
        "ix_entities_embedding_hnsw",
        "entities",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "episode_entities",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("episode_id", UUID(as_uuid=True), sa.ForeignKey("episodes.id"), nullable=False),
        sa.Column("entity_id", UUID(as_uuid=True), sa.ForeignKey("entities.id"), nullable=False),
        sa.Column("entity_index", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_episode_entities_episode_id", "episode_entities", ["episode_id"])
    op.create_index("ix_episode_entities_entity_id", "episode_entities", ["entity_id"])


def downgrade() -> None:
    op.drop_index("ix_episode_entities_entity_id", table_name="episode_entities")
    op.drop_index("ix_episode_entities_episode_id", table_name="episode_entities")
    op.drop_table("episode_entities")
    op.drop_index("ix_entities_embedding_hnsw", table_name="entities")
    op.drop_index("ix_entities_name", table_name="entities")
    op.drop_index("ix_entities_user_id", table_name="entities")
    op.drop_table("entities")