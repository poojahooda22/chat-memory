"""relationships: weighted co-occurrence edges between entities (the graph)

Revision ID: 0006
Revises: 0005
Create Date: 2026-07-07

Nodes are entities (existing); this is the edge table. An edge means two entities share
photos; weight = co-occurrence × recency × confidence. Invalid-not-delete for time-travel.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "relationships",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("src_entity_id", UUID(as_uuid=True), sa.ForeignKey("entities.id"), nullable=False),
        sa.Column("dst_entity_id", UUID(as_uuid=True), sa.ForeignKey("entities.id"), nullable=False),
        sa.Column("rel_type", sa.String(), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("cooccur_count", sa.Integer(), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(), nullable=True),
        sa.Column("mean_confidence", sa.Float(), nullable=False),
        sa.Column("is_valid", sa.Boolean(), nullable=False),
        sa.Column("source_episode_ids", JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_relationships_user_id", "relationships", ["user_id"])
    op.create_index("ix_relationships_src_entity_id", "relationships", ["src_entity_id"])
    op.create_index("ix_relationships_dst_entity_id", "relationships", ["dst_entity_id"])
    op.create_index("ix_relationships_is_valid", "relationships", ["is_valid"])


def downgrade() -> None:
    op.drop_index("ix_relationships_is_valid", table_name="relationships")
    op.drop_index("ix_relationships_dst_entity_id", table_name="relationships")
    op.drop_index("ix_relationships_src_entity_id", table_name="relationships")
    op.drop_index("ix_relationships_user_id", table_name="relationships")
    op.drop_table("relationships")