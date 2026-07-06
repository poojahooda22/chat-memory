"""who labeled the link: the user by hand, or the memory by visual recognition

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-07

Auto-applied labels must be visibly auto — the glass box shows what the memory decided on
its own, and the user can undo it.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "episode_entities",
        sa.Column("labeled_by", sa.String(), nullable=False, server_default="user"),
    )


def downgrade() -> None:
    op.drop_column("episode_entities", "labeled_by")
