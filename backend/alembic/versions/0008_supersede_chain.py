"""supersede chain: superseded_by_id on memories

Revision ID: 0008
Revises: 0007
Create Date: 2026-07-10

A corrected fact now RETIRES its predecessor instead of overwriting it in place: the old row
keeps its content (is_superseded=true) and points at its successor, so the belief chain
(old guess -> correction -> newer correction) is queryable end to end. ON DELETE SET NULL so a
hard-deleted successor never blocks or cascades into its retired predecessor.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "memories",
        sa.Column(
            "superseded_by_id",
            sa.Uuid(),
            sa.ForeignKey("memories.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("memories", "superseded_by_id")