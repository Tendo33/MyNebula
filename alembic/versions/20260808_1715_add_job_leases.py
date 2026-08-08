"""add database-backed job leases

Revision ID: 8a51c7d2e4f9
Revises: 1bbcbca4f4a1
Create Date: 2026-08-08 17:15:00
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "8a51c7d2e4f9"
down_revision: str | Sequence[str] | None = "1bbcbca4f4a1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "admin_auth_state",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("session_version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_admin_auth_state")),
        sa.CheckConstraint("id = 1", name="singleton"),
    )
    op.execute("INSERT INTO admin_auth_state (id, session_version) VALUES (1, 0)")
    for table_name in ("pipeline_runs", "sync_tasks"):
        op.add_column(
            table_name,
            sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.add_column(
            table_name,
            sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.add_column(
            table_name,
            sa.Column("worker_id", sa.String(length=255), nullable=True),
        )

    op.create_index(
        "ix_pipeline_runs_user_active",
        "pipeline_runs",
        ["user_id", "lease_expires_at"],
        unique=False,
        postgresql_where=sa.text("status IN ('pending', 'running')"),
    )
    op.create_index(
        "ix_sync_tasks_user_type_active",
        "sync_tasks",
        ["user_id", "task_type", "lease_expires_at"],
        unique=False,
        postgresql_where=sa.text("status IN ('pending', 'running')"),
    )


def downgrade() -> None:
    op.drop_index("ix_sync_tasks_user_type_active", table_name="sync_tasks")
    op.drop_index("ix_pipeline_runs_user_active", table_name="pipeline_runs")
    for table_name in ("sync_tasks", "pipeline_runs"):
        op.drop_column(table_name, "worker_id")
        op.drop_column(table_name, "lease_expires_at")
        op.drop_column(table_name, "heartbeat_at")
    op.drop_table("admin_auth_state")
