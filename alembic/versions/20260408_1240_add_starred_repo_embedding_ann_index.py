"""add starred repo embedding ann index

Revision ID: e5d4f6a7b8c9
Revises: d4c9e2f7ab10
Create Date: 2026-04-08 12:40:00.000000
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "e5d4f6a7b8c9"
down_revision = "d4c9e2f7ab10"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_starred_repos_embedding_cosine_ann
        ON starred_repos
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_starred_repos_embedding_cosine_ann")
