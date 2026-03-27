import pytest


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
