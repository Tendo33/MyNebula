from types import SimpleNamespace

import pytest


class _FakeDbContext:
    def __init__(self, db, events=None):
        self._db = db
        self._events = events if events is not None else []

    async def __aenter__(self):
        self._events.append("enter")
        return self._db

    async def __aexit__(self, exc_type, exc, _tb):
        self._events.append("exit")
        return False


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows

    def scalar(self):
        if isinstance(self._rows, list):
            return self._rows[0] if self._rows else None
        return self._rows


class _FakeDb:
    def __init__(self, schedules):
        self._schedules = schedules

    async def execute(self, _statement):
        return _FakeResult(self._schedules)


@pytest.mark.asyncio
async def test_scheduler_skips_tick_when_lock_not_acquired(monkeypatch):
    from nebula.core import scheduler as scheduler_module
    from nebula.db import database as db_module

    service = scheduler_module.SchedulerService()
    fake_db = _FakeDb([])
    monkeypatch.setattr(db_module, "get_db_context", lambda: _FakeDbContext(fake_db))

    async def fake_try_lock(_db):
        return False

    launched: list[int] = []

    async def fake_launch(user_id: int):
        launched.append(user_id)

    monkeypatch.setattr(service, "_try_acquire_scheduler_lock", fake_try_lock)
    monkeypatch.setattr(service, "_launch_user_pipeline", fake_launch)

    await service._check_and_trigger_syncs()

    assert launched == []


@pytest.mark.asyncio
async def test_scheduler_marks_due_schedule_then_launches_after_context_exit(
    monkeypatch,
):
    from nebula.application.services import sync_ops_service
    from nebula.core import scheduler as scheduler_module
    from nebula.db import database as db_module

    schedule = SimpleNamespace(
        user_id=7,
        timezone="UTC",
        schedule_hour=8,
        schedule_minute=30,
        last_run_at=None,
        last_run_status=None,
        last_run_error=None,
    )
    fake_db = _FakeDb([schedule])
    events: list[str] = []
    monkeypatch.setattr(
        db_module, "get_db_context", lambda: _FakeDbContext(fake_db, events)
    )

    service = scheduler_module.SchedulerService()

    async def fake_try_lock(_db):
        return True

    async def fake_has_active(_user_id: int):
        return False

    async def fake_launch(user_id: int):
        events.append(f"launch:{user_id}")

    monkeypatch.setattr(sync_ops_service, "is_schedule_due", lambda **_kwargs: True)
    monkeypatch.setattr(service, "_try_acquire_scheduler_lock", fake_try_lock)
    monkeypatch.setattr(service, "_has_active_pipeline", fake_has_active)
    monkeypatch.setattr(service, "_launch_user_pipeline", fake_launch)

    await service._check_and_trigger_syncs()

    assert schedule.last_run_status == "running"
    assert schedule.last_run_at is not None
    assert "exit" in events
    assert "launch:7" in events
    assert events.index("exit") < events.index("launch:7")


@pytest.mark.asyncio
async def test_scheduler_uses_transaction_level_advisory_lock():
    from nebula.core.scheduler import SchedulerService

    executed_sql: list[str] = []

    class _FakeLockDb:
        async def execute(self, statement, _params):
            executed_sql.append(str(statement))
            return _FakeResult(True)

    service = SchedulerService()
    acquired = await service._try_acquire_scheduler_lock(_FakeLockDb())

    assert acquired is True
    assert any("pg_try_advisory_xact_lock" in sql for sql in executed_sql)
    assert not hasattr(service, "_release_scheduler_lock")
