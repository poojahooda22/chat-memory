"""index memories.superseded_by_id

Revision ID: 0009
Revises: 0008
Create Date: 2026-07-10

The belief-chain history walk queries memories by superseded_by_id, and Postgres does not
auto-index a foreign-key referencing column — so the ON DELETE SET NULL back-reference is
scanned on every hard delete of a memories row. Index it: the walk becomes a lookup and bulk
deletes (e.g. the eval reset) stop paying an O(N) scan per deleted row.
"""
from collections.abc import Sequence

from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_memories_superseded_by_id", "memories", ["superseded_by_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_memories_superseded_by_id", table_name="memories")