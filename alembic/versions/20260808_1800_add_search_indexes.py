"""add concurrent text search indexes

Revision ID: 3f7b9c1d2e6a
Revises: 8a51c7d2e4f9
Create Date: 2026-08-08 18:00:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = "3f7b9c1d2e6a"
down_revision: str | Sequence[str] | None = "8a51c7d2e4f9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SEARCH_COLUMNS = ("name", "full_name", "description", "ai_summary")


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    with op.get_context().autocommit_block():
        for column in SEARCH_COLUMNS:
            op.create_index(
                f"ix_starred_repos_{column}_trgm",
                "starred_repos",
                [column],
                unique=False,
                postgresql_using="gin",
                postgresql_ops={column: "gin_trgm_ops"},
                postgresql_concurrently=True,
            )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        for column in reversed(SEARCH_COLUMNS):
            op.drop_index(
                f"ix_starred_repos_{column}_trgm",
                table_name="starred_repos",
                postgresql_concurrently=True,
            )
