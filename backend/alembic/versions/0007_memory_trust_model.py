"""memory trust model: source + confidence + is_superseded

Revision ID: 0007
Revises: 0006
Create Date: 2026-07-08

Every memory gets a trust model mirroring the relationship edge (mean_confidence + is_valid):
where the fact came from, how much to trust it, and whether a later observation retired it.
Existing rows backfill to source='chat', confidence=1.0, is_superseded=false.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # add with a server_default so existing rows backfill, then drop the default so future inserts
    # carry the value the app supplies (matching the model, which has no DB-side default)
    op.add_column("memories", sa.Column("source", sa.String(), nullable=False, server_default="chat"))
    op.add_column("memories", sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"))
    op.add_column(
        "memories",
        sa.Column("is_superseded", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.alter_column("memories", "source", server_default=None)
    op.alter_column("memories", "confidence", server_default=None)
    op.alter_column("memories", "is_superseded", server_default=None)
    # the hot read filter: a user's live (non-superseded) facts
    op.create_index("ix_memories_user_id_is_superseded", "memories", ["user_id", "is_superseded"])


def downgrade() -> None:
    op.drop_index("ix_memories_user_id_is_superseded", table_name="memories")
    op.drop_column("memories", "is_superseded")
    op.drop_column("memories", "confidence")
    op.drop_column("memories", "source")