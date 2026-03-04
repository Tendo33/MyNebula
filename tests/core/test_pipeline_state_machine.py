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

    import nebula.api.sync as sync_api

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(sync_api, "sync_stars_task", noop)
    monkeypatch.setattr(sync_api, "compute_embeddings_task", noop)
    monkeypatch.setattr(sync_api, "run_clustering_task", noop)

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

    import nebula.api.sync as sync_api

    async def failing_sync(*_args, **_kwargs):
        raise RuntimeError("boom")

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(sync_api, "sync_stars_task", failing_sync)
    monkeypatch.setattr(sync_api, "compute_embeddings_task", noop)
    monkeypatch.setattr(sync_api, "run_clustering_task", noop)

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
