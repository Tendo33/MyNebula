"""Coverage for `SyncPipelineService` orchestration.

The pipeline orchestrator sat at 52% in the 2026-08-18 scan. The existing
`test_pipeline_state_machine.py` covers phase progression; these tests cover
the surrounding surface: run creation guards, recluster, lease loss, task
outcome inspection, and the query accessors.
"""

from types import SimpleNamespace

import pytest

from nebula.application.services import pipeline_service as module
from nebula.application.services.job_lifecycle_service import JobLeaseLostError
from nebula.application.services.pipeline_service import SyncPipelineService
from nebula.domain import PipelinePhase, PipelineStatus


def _run(run_id=1, **overrides):
    run = SimpleNamespace(
        id=run_id,
        user_id=7,
        status=PipelineStatus.pending.value,
        phase=PipelinePhase.pending.value,
        last_error=None,
        started_at=None,
        completed_at=None,
        worker_id="worker:pipeline",
        lease_expires_at=None,
        heartbeat_at=None,
    )
    for key, value in overrides.items():
        setattr(run, key, value)
    return run


def _task(task_id=1, **overrides):
    task = SimpleNamespace(
        id=task_id,
        user_id=7,
        pipeline_run_id=1,
        task_type="stars",
        status="completed",
        phase=PipelinePhase.stars.value,
        error_message=None,
        error_details=None,
        total_items=0,
        processed_items=0,
        failed_items=0,
        worker_id=None,
        lease_expires_at=None,
    )
    for key, value in overrides.items():
        setattr(task, key, value)
    return task


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None

    def scalars(self):
        return SimpleNamespace(all=lambda: list(self._rows))


class _FakeDb:
    def __init__(self, state):
        self.state = state
        self.added: list[object] = []
        self.commits = 0
        self.statements: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, _tb):
        return False

    async def get(self, model, obj_id, **_kwargs):
        return self.state.objects.get((model.__name__, obj_id))

    async def execute(self, statement, params=None):
        self.statements.append(str(statement).lower())
        return _Result(self.state.query_rows)

    def add(self, obj):
        self.added.append(obj)
        if getattr(obj, "id", None) is None:
            obj.id = self.state.next_id
            self.state.next_id += 1
        self.state.objects[(type(obj).__name__, obj.id)] = obj

    async def flush(self):
        return None

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        return None

    async def refresh(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = self.state.next_id
            self.state.next_id += 1
        self.state.objects[(type(obj).__name__, obj.id)] = obj

    def get_bind(self):
        return SimpleNamespace(dialect=SimpleNamespace(name=self.state.dialect))


class _State:
    def __init__(self, *, objects=None, query_rows=None, dialect="postgresql"):
        self.objects = objects or {}
        self.query_rows = query_rows or []
        self.dialect = dialect
        self.next_id = 100


def _install_db(monkeypatch, state):
    db = _FakeDb(state)
    monkeypatch.setattr(module, "get_db_context", lambda: db)
    return db


@pytest.fixture(autouse=True)
def _no_lease_side_effects(monkeypatch):
    async def noop(*_args, **_kwargs):
        return False

    monkeypatch.setattr(module, "interrupt_expired_job", noop)
    monkeypatch.setattr(module, "interrupt_expired_jobs_for_user", noop)
    monkeypatch.setattr(module, "apply_job_lease", lambda run: None)
    monkeypatch.setattr(module, "clear_job_lease", lambda run: None)


@pytest.mark.asyncio
async def test_create_pipeline_run_persists_a_pending_run(monkeypatch):
    state = _State()
    db = _install_db(monkeypatch, state)
    service = SyncPipelineService()

    run_id = await service.create_pipeline_run(7)

    assert run_id is not None
    created = db.added[0]
    assert created.user_id == 7
    assert created.status == PipelineStatus.pending.value
    # Serialization lock must be taken on PostgreSQL.
    assert any("pg_advisory_xact_lock" in s for s in db.statements)


@pytest.mark.asyncio
async def test_create_pipeline_run_skips_the_advisory_lock_off_postgres(monkeypatch):
    state = _State(dialect="sqlite")
    db = _install_db(monkeypatch, state)
    service = SyncPipelineService()

    await service.create_pipeline_run(7)

    assert not any("pg_advisory_xact_lock" in s for s in db.statements)


@pytest.mark.asyncio
async def test_create_pipeline_run_rejects_a_concurrent_run(monkeypatch):
    active = _run(status=PipelineStatus.running.value)
    state = _State(query_rows=[active])
    _install_db(monkeypatch, state)
    service = SyncPipelineService()

    with pytest.raises(ValueError, match="Pipeline already running"):
        await service.create_pipeline_run(7)


@pytest.mark.asyncio
async def test_run_pipeline_rejects_an_unknown_run(monkeypatch):
    _install_db(monkeypatch, _State())
    service = SyncPipelineService()

    with pytest.raises(ValueError, match="not found"):
        await service.run_pipeline(999)


@pytest.mark.asyncio
async def test_run_pipeline_requires_an_execution_token(monkeypatch):
    run = _run(worker_id=None)
    state = _State(objects={("PipelineRun", 1): run})
    _install_db(monkeypatch, state)
    service = SyncPipelineService()

    with pytest.raises(JobLeaseLostError, match="no execution token"):
        await service.run_pipeline(1)


@pytest.mark.asyncio
async def test_lock_key_is_stable_and_user_scoped():
    service = SyncPipelineService()

    assert service._pipeline_lock_key(7) == service._pipeline_lock_key(7)
    assert service._pipeline_lock_key(7) != service._pipeline_lock_key(8)


@pytest.mark.asyncio
async def test_inspect_task_outcome_raises_on_a_hard_failure(monkeypatch):
    task = _task(status="failed", error_message="stars stage exploded")
    state = _State(objects={("SyncTask", 1): task})
    _install_db(monkeypatch, state)
    service = SyncPipelineService()

    with pytest.raises(ValueError, match="stars stage exploded"):
        await service._inspect_task_outcome(1, run_id=1, phase=PipelinePhase.stars)


@pytest.mark.asyncio
async def test_inspect_task_outcome_reports_partial_failure(monkeypatch):
    task = _task(status="completed", failed_items=3)
    state = _State(objects={("SyncTask", 1): task})
    _install_db(monkeypatch, state)
    service = SyncPipelineService()

    outcome = await service._inspect_task_outcome(
        1, run_id=1, phase=PipelinePhase.stars
    )

    assert outcome is not None
    assert "failed_items=3" in outcome


@pytest.mark.asyncio
async def test_inspect_task_outcome_is_silent_on_a_clean_run(monkeypatch):
    state = _State(objects={("SyncTask", 1): _task(status="completed", failed_items=0)})
    _install_db(monkeypatch, state)
    service = SyncPipelineService()

    assert (
        await service._inspect_task_outcome(1, run_id=1, phase=PipelinePhase.stars)
        is None
    )


@pytest.mark.asyncio
async def test_inspect_task_outcome_rejects_a_missing_task(monkeypatch):
    _install_db(monkeypatch, _State())
    service = SyncPipelineService()

    with pytest.raises(ValueError, match="Pipeline task not found"):
        await service._inspect_task_outcome(404, run_id=1, phase=PipelinePhase.stars)


def test_normalize_partial_error_accepts_string_bool_and_none():
    service = SyncPipelineService()

    assert (
        service._normalize_partial_error("boom", phase=PipelinePhase.stars, task_id=1)
        == "boom"
    )
    assert (
        service._normalize_partial_error(None, phase=PipelinePhase.stars, task_id=1)
        is None
    )
    assert (
        service._normalize_partial_error(False, phase=PipelinePhase.stars, task_id=1)
        is None
    )
    generated = service._normalize_partial_error(
        True, phase=PipelinePhase.stars, task_id=42
    )
    assert "task_id=42" in generated


@pytest.mark.asyncio
async def test_should_force_full_recluster_uses_new_repo_ratio(monkeypatch):
    stars_task = _task(error_details={"new_repos": 40})
    user = SimpleNamespace(id=7, total_stars=100)
    state = _State(objects={("SyncTask", 1): stars_task, ("User", 7): user})
    _install_db(monkeypatch, state)
    service = SyncPipelineService()

    # 40 / 100 is well past the 0.2 ratio threshold.
    assert await service._should_force_full_recluster(7, 1) is True


@pytest.mark.asyncio
async def test_should_force_full_recluster_is_false_for_a_small_delta(monkeypatch):
    stars_task = _task(error_details={"new_repos": 2})
    user = SimpleNamespace(id=7, total_stars=100)
    state = _State(objects={("SyncTask", 1): stars_task, ("User", 7): user})
    _install_db(monkeypatch, state)
    service = SyncPipelineService()

    assert await service._should_force_full_recluster(7, 1) is False


@pytest.mark.asyncio
async def test_should_force_full_recluster_tolerates_missing_records(monkeypatch):
    _install_db(monkeypatch, _State())
    service = SyncPipelineService()

    # Missing task and user must not raise; 0 new repos over a 1-repo floor.
    assert await service._should_force_full_recluster(7, 1) is False


@pytest.mark.asyncio
async def test_should_force_full_recluster_tolerates_non_dict_details(monkeypatch):
    stars_task = _task(error_details="not-a-dict")
    user = SimpleNamespace(id=7, total_stars=100)
    state = _State(objects={("SyncTask", 1): stars_task, ("User", 7): user})
    _install_db(monkeypatch, state)
    service = SyncPipelineService()

    assert await service._should_force_full_recluster(7, 1) is False


@pytest.mark.asyncio
async def test_get_pipeline_returns_the_run(monkeypatch):
    run = _run()
    state = _State(objects={("PipelineRun", 1): run})
    _install_db(monkeypatch, state)
    service = SyncPipelineService()

    assert await service.get_pipeline(1) is run


@pytest.mark.asyncio
async def test_get_latest_pipeline_returns_the_newest_run(monkeypatch):
    newest = _run(run_id=9)
    state = _State(query_rows=[newest])
    db = _install_db(monkeypatch, state)
    service = SyncPipelineService()

    assert await service.get_latest_pipeline(7) is newest
    assert any("order by" in s and "desc" in s for s in db.statements)


@pytest.mark.asyncio
async def test_get_active_pipeline_returns_none_when_idle(monkeypatch):
    _install_db(monkeypatch, _State(query_rows=[]))
    service = SyncPipelineService()

    assert await service.get_active_pipeline(7) is None


@pytest.mark.asyncio
async def test_update_run_rejects_a_lost_lease(monkeypatch):
    # A run owned by a different worker must not be mutated.
    run = _run(worker_id="worker:other", status=PipelineStatus.running.value)
    state = _State(objects={("PipelineRun", 1): run})
    _install_db(monkeypatch, state)
    service = SyncPipelineService()

    with pytest.raises(JobLeaseLostError, match="lease was lost"):
        await service._update_run(
            1,
            PipelineStatus.running,
            PipelinePhase.stars,
            worker_id="worker:pipeline",
        )


@pytest.mark.asyncio
async def test_update_run_rejects_a_vanished_run(monkeypatch):
    _install_db(monkeypatch, _State())
    service = SyncPipelineService()

    with pytest.raises(JobLeaseLostError, match="no longer exists"):
        await service._update_run(
            404,
            PipelineStatus.running,
            PipelinePhase.stars,
            worker_id="worker:pipeline",
        )
