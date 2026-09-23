"""cascade related-repo dependencies when a starred repo is removed

Revision ID: c7d2e4f91a30
Revises: 7e1c4b9a2f08
Create Date: 2026-09-22 12:00:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = "c7d2e4f91a30"
down_revision: str | Sequence[str] | None = "7e1c4b9a2f08"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for table, constraint, columns in (
        (
            "repo_related_caches",
            "fk_repo_related_caches_anchor_repo_id_starred_repos",
            "anchor_repo_id",
        ),
        (
            "repo_related_feedbacks",
            "fk_repo_related_feedbacks_anchor_repo_id_starred_repos",
            "anchor_repo_id",
        ),
        (
            "repo_related_feedbacks",
            "fk_repo_related_feedbacks_candidate_repo_id_starred_repos",
            "candidate_repo_id",
        ),
    ):
        op.drop_constraint(constraint, table_name=table, type_="foreignkey")
        op.create_foreign_key(
            constraint,
            table,
            "starred_repos",
            [columns],
            ["id"],
            ondelete="CASCADE",
        )


def downgrade() -> None:
    for table, constraint, columns in (
        (
            "repo_related_caches",
            "fk_repo_related_caches_anchor_repo_id_starred_repos",
            "anchor_repo_id",
        ),
        (
            "repo_related_feedbacks",
            "fk_repo_related_feedbacks_anchor_repo_id_starred_repos",
            "anchor_repo_id",
        ),
        (
            "repo_related_feedbacks",
            "fk_repo_related_feedbacks_candidate_repo_id_starred_repos",
            "candidate_repo_id",
        ),
    ):
        op.drop_constraint(constraint, table_name=table, type_="foreignkey")
        op.create_foreign_key(
            constraint,
            table,
            "starred_repos",
            [columns],
            ["id"],
        )
