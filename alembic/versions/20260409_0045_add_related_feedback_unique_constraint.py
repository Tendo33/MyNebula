"""add unique constraint for related feedback

Revision ID: 7c19d0a4ef21
Revises: e5d4f6a7b8c9
Create Date: 2026-04-09 00:45:00.000000
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "7c19d0a4ef21"
down_revision = "e5d4f6a7b8c9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint(
        op.f("uq_repo_related_feedbacks_user_id"),
        "repo_related_feedbacks",
        ["user_id", "anchor_repo_id", "candidate_repo_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("uq_repo_related_feedbacks_user_id"),
        "repo_related_feedbacks",
        type_="unique",
    )
