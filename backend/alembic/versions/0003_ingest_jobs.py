"""ingest jobs: the queue row an uploaded image rides into memory

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-06

The upload endpoint writes only this row (plus the file on disk) and returns 202. The worker
inserts the Episode exactly once and records its id here — episodes stay single-shot, and the
episode_id column is the retry-idempotency guard.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ingest_jobs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("content_type", sa.String(), nullable=False),
        sa.Column("image_path", sa.Text(), nullable=False),
        sa.Column("exif", JSONB(), nullable=False),
        sa.Column("episode_id", UUID(as_uuid=True), sa.ForeignKey("episodes.id"), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_ingest_jobs_user_id", "ingest_jobs", ["user_id"])
    op.create_index("ix_ingest_jobs_status", "ingest_jobs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_ingest_jobs_status", table_name="ingest_jobs")
    op.drop_index("ix_ingest_jobs_user_id", table_name="ingest_jobs")
    op.drop_table("ingest_jobs")