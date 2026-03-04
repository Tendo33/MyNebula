"""add user graph cluster defaults

Revision ID: d4c9e2f7ab10
Revises: a1f3b9c8d2e1
Create Date: 2026-03-04 19:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4c9e2f7ab10"
down_revision: str | None = "a1f3b9c8d2e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.add_column(
        "users",
        sa.Column(
            "graph_max_clusters",
            sa.Integer(),
            nullable=False,
            server_default="8",
        ),
    )
    op.add_column(
        "users",
        sa.Column(
            "graph_min_clusters",
            sa.Integer(),
            nullable=False,
            server_default="3",
        ),
    )
    op.alter_column("users", "graph_max_clusters", server_default=None)
    op.alter_column("users", "graph_min_clusters", server_default=None)


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_column("users", "graph_min_clusters")
    op.drop_column("users", "graph_max_clusters")
