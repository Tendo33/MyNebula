"""add gin indexes for topics and ai_tags text search

Revision ID: 7e1c4b9a2f08
Revises: 3f7b9c1d2e6a
Create Date: 2026-09-10 23:15:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = "7e1c4b9a2f08"
down_revision: str | Sequence[str] | None = "3f7b9c1d2e6a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    with op.get_context().autocommit_block():
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_starred_repos_topics_gin
            ON starred_repos
            USING gin (topics)
            """
        )
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_starred_repos_ai_tags_gin
            ON starred_repos
            USING gin (ai_tags)
            """
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_starred_repos_ai_tags_gin")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_starred_repos_topics_gin")
