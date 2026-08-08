from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from nebula.application.services import job_lifecycle_service as lifecycle


class _FakeSessionContext:
    def __init__(self, record=None, *, rowcounts=(0, 0)):
        self.record = record
        self.rowcounts = iter(rowcounts)
        self.commits = 0
        self.statements = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, _model, _record_id):
        return self.record

    async def execute(self, statement, *_args, **_kwargs):
        self.statements.append(statement)
        return SimpleNamespace(rowcount=next(self.rowcounts))

    async def commit(self):
        self.commits += 1


def test_apply_and_clear_job_lease_uses_unique_execution_token():
    now = datetime(2026, 8, 8, 9, 0, tzinfo=timezone.utc)
    record = SimpleNamespace()

    token = lifecycle.apply_job_lease(record, now=now)

    assert record.heartbeat_at == now
    assert record.lease_expires_at == now + lifecycle.LEASE_DURATION
    assert record.worker_id == token
    assert token

    lifecycle.clear_job_lease(record)

    assert record.lease_expires_at is None
    assert record.worker_id is None
    assert record.heartbeat_at == now


@pytest.mark.asyncio
async def test_renew_job_lease_does_not_revive_terminal_record():
    session = _FakeSessionContext(rowcounts=(0,))

    renewed = await lifecycle.renew_job_lease(
        "pipeline", 7, "worker:test", session_factory=lambda: session
    )

    assert renewed is False
    assert session.commits == 1


@pytest.mark.asyncio
async def test_renew_job_lease_requires_current_unexpired_owner(monkeypatch):
    now = datetime(2026, 8, 8, 9, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(lifecycle, "utcnow", lambda: now)
    session = _FakeSessionContext(rowcounts=(1,))

    renewed = await lifecycle.renew_job_lease(
        "pipeline", 7, "worker:test", session_factory=lambda: session
    )

    assert renewed is True
    assert session.commits == 1
    compiled = str(
        session.statements[0].compile(compile_kwargs={"literal_binds": True})
    )
    assert "worker_id = 'worker:test'" in compiled
    assert "lease_expires_at >" in compiled
    assert "status IN ('pending', 'running')" in compiled


@pytest.mark.asyncio
async def test_heartbeat_refuses_to_start_without_owned_lease(monkeypatch):
    async def lost_renewal(*_args, **_kwargs):
        return False

    monkeypatch.setattr(lifecycle, "renew_job_lease", lost_renewal)
    heartbeat = lifecycle.JobLeaseHeartbeat(
        "pipeline",
        7,
        "worker:test",
        session_factory=lambda: _FakeSessionContext(),
    )

    with pytest.raises(lifecycle.JobLeaseLostError):
        await heartbeat.start()


@pytest.mark.asyncio
async def test_stop_does_not_replace_main_result_with_heartbeat_error():
    async def failed_heartbeat():
        raise RuntimeError("temporary database failure")

    heartbeat = lifecycle.JobLeaseHeartbeat(
        "pipeline",
        7,
        "worker:test",
        session_factory=lambda: _FakeSessionContext(),
    )
    heartbeat._task = __import__("asyncio").create_task(failed_heartbeat())
    await __import__("asyncio").sleep(0)

    await heartbeat.stop()


@pytest.mark.asyncio
async def test_online_reconciliation_interrupts_one_expired_job():
    session = _FakeSessionContext(rowcounts=(1,))

    interrupted = await lifecycle.interrupt_expired_job(
        session,
        "pipeline",
        7,
        user_id=3,
    )

    assert interrupted is True
    compiled = str(
        session.statements[0].compile(compile_kwargs={"literal_binds": True})
    )
    assert "pipeline_runs.id = 7" in compiled
    assert "pipeline_runs.user_id = 3" in compiled
    assert "lease_expires_at IS NULL" in compiled


@pytest.mark.asyncio
async def test_reconcile_orphaned_jobs_reports_each_recovered_owner():
    session = _FakeSessionContext(rowcounts=(2, 5))

    result = await lifecycle.reconcile_orphaned_jobs(session_factory=lambda: session)

    assert result == (2, 5)
    assert session.commits == 1
    assert len(session.statements) == 2
    for statement in session.statements:
        compiled = str(statement.compile(compile_kwargs={"literal_binds": True}))
        assert "lease_expires_at IS NULL" in compiled
        assert "interrupted" in compiled
