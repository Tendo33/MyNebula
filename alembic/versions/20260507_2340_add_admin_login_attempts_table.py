"""add admin login attempts table

Revision ID: 1bbcbca4f4a1
Revises: 20260409_0045_add_related_feedback_unique_constraint
Create Date: 2026-05-07 23:40:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1bbcbca4f4a1"
down_revision: str | Sequence[str] | None = (
    "7c19d0a4ef21"
)
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "admin_login_attempts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("bucket_key", sa.String(length=255), nullable=False),
        sa.Column(
            "attempted_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_admin_login_attempts")),
    )
    op.create_index(
        op.f("ix_admin_login_attempts_attempted_at"),
        "admin_login_attempts",
        ["attempted_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_admin_login_attempts_bucket_key"),
        "admin_login_attempts",
        ["bucket_key"],
        unique=False,
    )
    op.create_index(
        "ix_admin_login_attempts_bucket_attempted_at",
        "admin_login_attempts",
        ["bucket_key", "attempted_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_admin_login_attempts_bucket_attempted_at",
        table_name="admin_login_attempts",
    )
    op.drop_index(
        op.f("ix_admin_login_attempts_bucket_key"),
        table_name="admin_login_attempts",
    )
    op.drop_index(
        op.f("ix_admin_login_attempts_attempted_at"),
        table_name="admin_login_attempts",
    )
    op.drop_table("admin_login_attempts")
