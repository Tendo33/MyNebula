from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_VERSIONS_DIR = PROJECT_ROOT / "alembic" / "versions"


@pytest.mark.asyncio
async def test_auto_upgrade_runs_when_alembic_version_missing(monkeypatch):
    from nebula.db import database as database_module

    monkeypatch.setattr(database_module, "_resolve_alembic_heads", lambda: ["head-1"])

    heads = iter([None, "head-1"])

    async def fake_read_head(_engine):
        return next(heads)

    upgraded = {"called": False}

    async def fake_upgrade():
        upgraded["called"] = True

    monkeypatch.setattr(database_module, "_read_db_migration_head", fake_read_head)
    monkeypatch.setattr(database_module, "_run_alembic_upgrade_head", fake_upgrade)

    await database_module._assert_db_schema_at_migration_head(engine=object())

    assert upgraded["called"] is True


@pytest.mark.asyncio
async def test_auto_upgrade_failure_raises_actionable_error(monkeypatch):
    from nebula.db import database as database_module

    monkeypatch.setattr(database_module, "_resolve_alembic_heads", lambda: ["head-1"])

    async def fake_read_head(_engine):
        return None

    async def fake_upgrade():
        raise RuntimeError("upgrade failed")

    monkeypatch.setattr(database_module, "_read_db_migration_head", fake_read_head)
    monkeypatch.setattr(database_module, "_run_alembic_upgrade_head", fake_upgrade)

    with pytest.raises(RuntimeError) as exc_info:
        await database_module._assert_db_schema_at_migration_head(engine=object())

    error_message = str(exc_info.value)
    assert "automatic migration failed" in error_message
    assert "uv run alembic upgrade head" in error_message


@pytest.mark.asyncio
async def test_initialized_database_does_not_run_auto_upgrade(monkeypatch):
    from nebula.db import database as database_module

    monkeypatch.setattr(database_module, "_resolve_alembic_heads", lambda: ["head-1"])

    async def fake_read_head(_engine):
        return "head-1"

    upgraded = {"called": False}

    async def fake_upgrade():
        upgraded["called"] = True

    monkeypatch.setattr(database_module, "_read_db_migration_head", fake_read_head)
    monkeypatch.setattr(database_module, "_run_alembic_upgrade_head", fake_upgrade)

    await database_module._assert_db_schema_at_migration_head(engine=object())

    assert upgraded["called"] is False


def test_taxonomy_migration_exists_for_fresh_database_bootstrap():
    taxonomy_migration = (
        ALEMBIC_VERSIONS_DIR / "20260208_1600_4b3d8e9a1c2d_add_taxonomy_tables.py"
    )

    assert taxonomy_migration.exists(), (
        "Fresh database bootstrap requires the taxonomy migration to remain in the "
        "Alembic chain."
    )


def test_repo_related_cache_migration_depends_on_taxonomy_revision():
    migration_path = (
        ALEMBIC_VERSIONS_DIR
        / "20260223_1730_3bfcdbd93f4d_add_repo_related_caches_table.py"
    )
    migration_source = migration_path.read_text(encoding="utf-8")

    assert 'down_revision: str | None = "4b3d8e9a1c2d"' in migration_source


def test_repo_related_cache_migration_creates_related_repo_tables():
    migration_path = (
        ALEMBIC_VERSIONS_DIR
        / "20260223_1730_3bfcdbd93f4d_add_repo_related_caches_table.py"
    )
    migration_source = migration_path.read_text(encoding="utf-8")

    assert '"repo_related_feedbacks"' in migration_source
    assert '"repo_related_caches"' in migration_source
