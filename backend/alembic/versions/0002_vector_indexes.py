"""vector indexes: HNSW cosine indexes on the memories and episodes embeddings

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-05

The pipeline now searches these embedding columns by cosine similarity on every fact, so the
columns get an index — the same rule that governs every other index in this schema. HNSW is
pgvector's approximate-nearest-neighbour index: fast recall without scanning every row.
"""
from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_memories_embedding_hnsw",
        "memories",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index(
        "ix_episodes_embedding_hnsw",
        "episodes",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_episodes_embedding_hnsw", table_name="episodes")
    op.drop_index("ix_memories_embedding_hnsw", table_name="memories")