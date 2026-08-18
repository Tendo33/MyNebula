"""Smoke test proving the integration tier is wired to a real database.

If this fails, nothing else in the tier is trustworthy.
"""

import pytest
from sqlalchemy import text


@pytest.mark.asyncio
async def test_database_has_pgvector_and_alembic_head(integration_db):
    extension = await integration_db.scalar(
        text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
    )
    assert extension == "vector"

    version = await integration_db.scalar(
        text("SELECT version_num FROM alembic_version")
    )
    assert version


@pytest.mark.asyncio
async def test_schema_came_from_migrations_not_metadata_create_all(integration_db):
    """Assert on indexes only migrations produce.

    `Base.metadata.create_all()` would build the tables but none of these, so
    their presence is what proves the tier exercises the migration path — the
    drift this tier exists to catch.
    """
    for index_name in (
        "ix_starred_repos_embedding_cosine_ann",
        "ix_starred_repos_full_name_trgm",
        "ix_starred_repos_ai_summary_trgm",
    ):
        resolved = await integration_db.scalar(
            text("SELECT to_regclass(:name)"), {"name": index_name}
        )
        assert resolved is not None, f"missing migration-only index {index_name}"
