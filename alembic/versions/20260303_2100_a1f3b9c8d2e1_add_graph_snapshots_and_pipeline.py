"""add graph snapshots and pipeline runs

Revision ID: a1f3b9c8d2e1
Revises: 3bfcdbd93f4d
Create Date: 2026-03-03 21:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1f3b9c8d2e1"
down_revision: str | None = "3bfcdbd93f4d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.create_table(
        "pipeline_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("phase", sa.String(length=50), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("details", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_pipeline_runs_user_id_users")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_pipeline_runs")),
    )
    op.create_index(
        op.f("ix_pipeline_runs_user_id"), "pipeline_runs", ["user_id"], unique=False
    )

    op.create_table(
        "graph_snapshots",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("meta", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_graph_snapshots_user_id_users")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_graph_snapshots")),
    )
    op.create_index(
        op.f("ix_graph_snapshots_user_id"), "graph_snapshots", ["user_id"], unique=False
    )
    op.create_index(
        op.f("ix_graph_snapshots_version"), "graph_snapshots", ["version"], unique=False
    )
    op.create_index(
        "ix_graph_snapshots_user_version",
        "graph_snapshots",
        ["user_id", "version"],
        unique=True,
    )

    op.create_table(
        "graph_snapshot_nodes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("snapshot_id", sa.Integer(), nullable=False),
        sa.Column("repo_id", sa.Integer(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["graph_snapshots.id"],
            name=op.f("fk_graph_snapshot_nodes_snapshot_id_graph_snapshots"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_graph_snapshot_nodes")),
    )
    op.create_index(
        op.f("ix_graph_snapshot_nodes_snapshot_id"),
        "graph_snapshot_nodes",
        ["snapshot_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_graph_snapshot_nodes_repo_id"),
        "graph_snapshot_nodes",
        ["repo_id"],
        unique=False,
    )
    op.create_index(
        "ix_graph_snapshot_nodes_snapshot_repo",
        "graph_snapshot_nodes",
        ["snapshot_id", "repo_id"],
        unique=True,
    )

    op.create_table(
        "graph_snapshot_edges",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("snapshot_id", sa.Integer(), nullable=False),
        sa.Column("edge_index", sa.Integer(), nullable=False),
        sa.Column("source", sa.Integer(), nullable=False),
        sa.Column("target", sa.Integer(), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["graph_snapshots.id"],
            name=op.f("fk_graph_snapshot_edges_snapshot_id_graph_snapshots"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_graph_snapshot_edges")),
    )
    op.create_index(
        op.f("ix_graph_snapshot_edges_snapshot_id"),
        "graph_snapshot_edges",
        ["snapshot_id"],
        unique=False,
    )
    op.create_index(
        "ix_graph_snapshot_edges_snapshot_index",
        "graph_snapshot_edges",
        ["snapshot_id", "edge_index"],
        unique=False,
    )

    op.create_table(
        "graph_snapshot_timeline",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("snapshot_id", sa.Integer(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["graph_snapshots.id"],
            name=op.f("fk_graph_snapshot_timeline_snapshot_id_graph_snapshots"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_graph_snapshot_timeline")),
        sa.UniqueConstraint(
            "snapshot_id", name=op.f("uq_graph_snapshot_timeline_snapshot_id")
        ),
    )
    op.create_index(
        op.f("ix_graph_snapshot_timeline_snapshot_id"),
        "graph_snapshot_timeline",
        ["snapshot_id"],
        unique=False,
    )

    op.add_column(
        "users", sa.Column("active_graph_snapshot_id", sa.Integer(), nullable=True)
    )
    op.create_index(
        op.f("ix_users_active_graph_snapshot_id"),
        "users",
        ["active_graph_snapshot_id"],
        unique=False,
    )
    op.create_foreign_key(
        op.f("fk_users_active_graph_snapshot_id_graph_snapshots"),
        "users",
        "graph_snapshots",
        ["active_graph_snapshot_id"],
        ["id"],
    )

    op.add_column(
        "sync_tasks", sa.Column("pipeline_run_id", sa.Integer(), nullable=True)
    )
    op.add_column(
        "sync_tasks",
        sa.Column(
            "phase", sa.String(length=50), server_default="pending", nullable=False
        ),
    )
    op.create_index(
        op.f("ix_sync_tasks_pipeline_run_id"),
        "sync_tasks",
        ["pipeline_run_id"],
        unique=False,
    )
    op.create_foreign_key(
        op.f("fk_sync_tasks_pipeline_run_id_pipeline_runs"),
        "sync_tasks",
        "pipeline_runs",
        ["pipeline_run_id"],
        ["id"],
    )
    op.alter_column("sync_tasks", "phase", server_default=None)


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_constraint(
        op.f("fk_sync_tasks_pipeline_run_id_pipeline_runs"),
        "sync_tasks",
        type_="foreignkey",
    )
    op.drop_index(op.f("ix_sync_tasks_pipeline_run_id"), table_name="sync_tasks")
    op.drop_column("sync_tasks", "phase")
    op.drop_column("sync_tasks", "pipeline_run_id")

    op.drop_constraint(
        op.f("fk_users_active_graph_snapshot_id_graph_snapshots"),
        "users",
        type_="foreignkey",
    )
    op.drop_index(op.f("ix_users_active_graph_snapshot_id"), table_name="users")
    op.drop_column("users", "active_graph_snapshot_id")

    op.drop_index(
        op.f("ix_graph_snapshot_timeline_snapshot_id"),
        table_name="graph_snapshot_timeline",
    )
    op.drop_table("graph_snapshot_timeline")

    op.drop_index(
        "ix_graph_snapshot_edges_snapshot_index", table_name="graph_snapshot_edges"
    )
    op.drop_index(
        op.f("ix_graph_snapshot_edges_snapshot_id"), table_name="graph_snapshot_edges"
    )
    op.drop_table("graph_snapshot_edges")

    op.drop_index(
        "ix_graph_snapshot_nodes_snapshot_repo", table_name="graph_snapshot_nodes"
    )
    op.drop_index(
        op.f("ix_graph_snapshot_nodes_repo_id"), table_name="graph_snapshot_nodes"
    )
    op.drop_index(
        op.f("ix_graph_snapshot_nodes_snapshot_id"), table_name="graph_snapshot_nodes"
    )
    op.drop_table("graph_snapshot_nodes")

    op.drop_index("ix_graph_snapshots_user_version", table_name="graph_snapshots")
    op.drop_index(op.f("ix_graph_snapshots_version"), table_name="graph_snapshots")
    op.drop_index(op.f("ix_graph_snapshots_user_id"), table_name="graph_snapshots")
    op.drop_table("graph_snapshots")

    op.drop_index(op.f("ix_pipeline_runs_user_id"), table_name="pipeline_runs")
    op.drop_table("pipeline_runs")
