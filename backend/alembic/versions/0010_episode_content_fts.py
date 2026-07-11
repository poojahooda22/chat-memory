"""episode content full-text index: a GIN index for keyword recall over chat episodes

Revision ID: 0010
Revises: 0009
Create Date: 2026-07-11

The chat read path now searches past chat episodes so "did we talk about X?" is answerable. Dense
(cosine) similarity alone silently misses rare proper nouns and technical identifiers ("TanStack
Query") — the token an embedding model under-weights — so retrieval is hybrid: a keyword channel
(this index) fused with the dense channel by Reciprocal Rank Fusion. The keyword channel matches
`to_tsvector('english', content) @@ to_tsquery(...)`; this GIN index makes that lookup indexed
rather than a sequential scan.

A functional expression index (not a STORED generated column) so no table rewrite is needed —
cheap at current row counts; at millions of rows build it with CREATE INDEX CONCURRENTLY instead.
"""
from collections.abc import Sequence

from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX ix_episodes_content_fts ON episodes "
        "USING gin (to_tsvector('english', content))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_episodes_content_fts")
