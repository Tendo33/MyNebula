from types import SimpleNamespace

import pytest

from nebula.domain import PipelinePhase, PipelineStatus


class _FakeDbContext:
    def __init__(self, run):
        self._run = run

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, model, _id):
        if model.__name__ == "PipelineRun":
            return self._run
        return None


@pytest.mark.asyncio
async def test_pipeline_marks_partial_failed_when_phase_has_failed_items(monkeypatch):
    from nebula.application.services import pipeline_service as pipeline_module

    run = SimpleNamespace(id=1, user_id=9)
    monkeypatch.setattr(pipeline_module, "get_db_context", lambda: _FakeDbContext(run))

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(pipeline_module.sync_execution_service, "sync_stars_task", noop)
    monkeypatch.setattr(
        pipeline_module.sync_execution_service, "compute_embeddings_task", noop
    )
    monkeypatch.setattr(
        pipeline_module.sync_execution_service, "run_clustering_task", noop
    )

    service = pipeline_module.SyncPipelineService(
        graph_service=SimpleNamespace(rebuild_active_snapshot=noop)
    )

    status_updates: list[tuple[PipelineStatus, PipelinePhase]] = []

    async def fake_create_task(*_args, **_kwargs):
        return len(status_updates) + 1

    outcomes = [False, True, False]

    async def fake_inspect(*_args, **_kwargs):
        return outcomes.pop(0)

    async def fake_update(_run_id, status, phase, error=None):
        status_updates.append((status, phase))

    monkeypatch.setattr(service, "_create_task", fake_create_task)
    monkeypatch.setattr(service, "_inspect_task_outcome", fake_inspect)
    monkeypatch.setattr(service, "_update_run", fake_update)
    monkeypatch.setattr(service, "_should_force_full_recluster", noop)

    await service.run_pipeline(1)

    assert status_updates[-1] == (
        PipelineStatus.partial_failed,
        PipelinePhase.completed,
    )


@pytest.mark.asyncio
async def test_pipeline_marks_failed_when_phase_raises(monkeypatch):
    from nebula.application.services import pipeline_service as pipeline_module

    run = SimpleNamespace(id=1, user_id=9)
    monkeypatch.setattr(pipeline_module, "get_db_context", lambda: _FakeDbContext(run))

    async def failing_sync(*_args, **_kwargs):
        raise RuntimeError("boom")

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        pipeline_module.sync_execution_service, "sync_stars_task", failing_sync
    )
    monkeypatch.setattr(
        pipeline_module.sync_execution_service, "compute_embeddings_task", noop
    )
    monkeypatch.setattr(
        pipeline_module.sync_execution_service, "run_clustering_task", noop
    )

    service = pipeline_module.SyncPipelineService(
        graph_service=SimpleNamespace(rebuild_active_snapshot=noop)
    )

    status_updates: list[tuple[PipelineStatus, PipelinePhase, str | None]] = []

    async def fake_create_task(*_args, **_kwargs):
        return 1

    async def fake_update(_run_id, status, phase, error=None):
        status_updates.append((status, phase, error))

    monkeypatch.setattr(service, "_create_task", fake_create_task)
    monkeypatch.setattr(service, "_update_run", fake_update)

    await service.run_pipeline(1)

    assert status_updates[-1][0] == PipelineStatus.failed
    assert status_updates[-1][1] == PipelinePhase.completed


@pytest.mark.asyncio
async def test_recluster_pipeline_runs_clustering_then_snapshot(monkeypatch):
    from nebula.application.services import pipeline_service as pipeline_module

    run = SimpleNamespace(id=7, user_id=9)
    monkeypatch.setattr(pipeline_module, "get_db_context", lambda: _FakeDbContext(run))

    clustering_calls: list[dict[str, object]] = []

    async def fake_run_clustering_task(**kwargs):
        clustering_calls.append(kwargs)

    monkeypatch.setattr(
        pipeline_module.sync_execution_service,
        "run_clustering_task",
        fake_run_clustering_task,
    )

    snapshot_calls: list[object] = []

    async def fake_rebuild_active_snapshot(db):
        snapshot_calls.append(db)
        return None

    service = pipeline_module.SyncPipelineService(
        graph_service=SimpleNamespace(
            rebuild_active_snapshot=fake_rebuild_active_snapshot
        )
    )

    status_updates: list[tuple[PipelineStatus, PipelinePhase]] = []

    async def fake_create_task(*_args, **_kwargs):
        return 42

    async def fake_update(_run_id, status, phase, error=None):
        status_updates.append((status, phase))

    async def noop(*_args, **_kwargs):
        return False

    monkeypatch.setattr(service, "_create_task", fake_create_task)
    monkeypatch.setattr(service, "_update_run", fake_update)
    monkeypatch.setattr(service, "_inspect_task_outcome", noop)

    await service.run_recluster_pipeline(run_id=7, max_clusters=11, min_clusters=4)

    assert clustering_calls == [
        {
            "user_id": 9,
            "task_id": 42,
            "use_llm": True,
            "max_clusters": 11,
            "min_clusters": 4,
            "incremental": False,
        }
    ]
    assert len(snapshot_calls) == 1
    assert status_updates[-1] == (PipelineStatus.completed, PipelinePhase.completed)
