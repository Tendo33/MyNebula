from datetime import datetime, timezone

import pytest
from fastapi import HTTPException
from fastapi.background import BackgroundTasks

from nebula.schemas.v2.settings import ScheduleResponse, SyncInfoResponse


@pytest.mark.asyncio
async def test_v2_settings_router_has_admin_dependency():
    from nebula.api.v2 import settings as settings_api

    assert settings_api.router.dependencies
    dependency_names = {
        getattr(dep.dependency, "__name__", "")
        for dep in settings_api.router.dependencies
    }
    assert "require_admin" in dependency_names
    assert "require_admin_csrf" in dependency_names


@pytest.mark.asyncio
async def test_get_settings_returns_typed_payload(monkeypatch):
    from nebula.api.v2 import settings as settings_api

    async def fake_resolve_single_user(*_args, **_kwargs):
        return type(
            "User", (), {"id": 1, "graph_max_clusters": 12, "graph_min_clusters": 4}
        )()

    async def fake_get_schedule(*_args, **_kwargs):
        return ScheduleResponse(
            is_enabled=True,
            schedule_hour=9,
            schedule_minute=0,
            timezone="Asia/Shanghai",
            last_run_at=None,
            last_run_status=None,
            last_run_error=None,
            next_run_at=None,
        )

    async def fake_get_sync_info(*_args, **_kwargs):
        return SyncInfoResponse(
            last_sync_at=datetime(2026, 3, 3, tzinfo=timezone.utc),
            github_token_configured=True,
            single_user_mode=True,
            read_access_mode="demo",
            total_repos=10,
            synced_repos=10,
            embedded_repos=10,
            summarized_repos=9,
            schedule=None,
        )

    monkeypatch.setattr(settings_api, "get_schedule", fake_get_schedule)
    monkeypatch.setattr(settings_api, "get_sync_info", fake_get_sync_info)
    monkeypatch.setattr(settings_api, "resolve_single_user", fake_resolve_single_user)

    payload = await settings_api.get_settings(
        user=await fake_resolve_single_user(), db=object()
    )

    assert payload.schedule.is_enabled is True
    assert payload.sync_info.total_repos == 10
    assert payload.graph_defaults.max_clusters == 12
    assert payload.graph_defaults.min_clusters == 4
    assert payload.request_id is not None


@pytest.mark.asyncio
async def test_update_graph_defaults_persists_user_values(monkeypatch):
    from nebula.api.v2 import settings as settings_api
    from nebula.schemas.v2.settings import GraphDefaultsUpdateRequest

    user = type(
        "User", (), {"id": 1, "graph_max_clusters": 8, "graph_min_clusters": 3}
    )()

    async def fake_resolve_single_user(*_args, **_kwargs):
        return user

    class _Db:
        committed = False

        async def commit(self):
            self.committed = True

    db = _Db()
    monkeypatch.setattr(settings_api, "resolve_single_user", fake_resolve_single_user)

    payload = await settings_api.update_graph_defaults(
        config=GraphDefaultsUpdateRequest(max_clusters=15, min_clusters=5),
        user=user,
        db=db,
    )

    assert db.committed is True
    assert user.graph_max_clusters == 15
    assert user.graph_min_clusters == 5
    assert payload.graph_defaults.max_clusters == 15
    assert payload.graph_defaults.min_clusters == 5


@pytest.mark.asyncio
async def test_update_graph_defaults_rejects_invalid_bounds():
    from nebula.api.v2 import settings as settings_api
    from nebula.schemas.v2.settings import GraphDefaultsUpdateRequest

    with pytest.raises(HTTPException) as exc_info:
        await settings_api.update_graph_defaults(
            config=GraphDefaultsUpdateRequest(max_clusters=4, min_clusters=5),
            db=object(),
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_trigger_full_refresh_rejects_when_pipeline_active(monkeypatch):
    from nebula.application.services import sync_ops_service
    from nebula.schemas.v2.settings import FullRefreshRequest

    user = type("User", (), {"id": 1})()

    async def noop(*_args, **_kwargs):
        return None

    class _FakeResult:
        def __init__(self, scalar_value):
            self._scalar_value = scalar_value

        def scalar_one_or_none(self):
            return self._scalar_value

        def scalar(self):
            return self._scalar_value

    class _Db:
        def __init__(self):
            self.execute_calls = 0
            self.added = []

        async def execute(self, _statement):
            self.execute_calls += 1
            if self.execute_calls == 1:
                return _FakeResult(None)
            return _FakeResult(0)

        def add(self, obj):
            self.added.append(obj)

        async def commit(self):
            return None

        async def refresh(self, obj):
            obj.id = 123

    db = _Db()
    monkeypatch.setattr(sync_ops_service, "_acquire_full_refresh_creation_lock", noop)

    async def fake_get_active_pipeline_run(*_args, **_kwargs):
        return type("Run", (), {"id": 77, "status": "running"})()

    monkeypatch.setattr(
        sync_ops_service,
        "_get_active_pipeline_run",
        fake_get_active_pipeline_run,
        raising=False,
    )

    with pytest.raises(HTTPException) as exc_info:
        await sync_ops_service.trigger_full_refresh(
            payload=FullRefreshRequest(confirm=True),
            background_tasks=BackgroundTasks(),
            user=user,
            db=db,
        )

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_trigger_full_refresh_rejects_without_github_token(monkeypatch):
    from nebula.application.services import sync_ops_service
    from nebula.schemas.v2.settings import FullRefreshRequest

    user = type("User", (), {"id": 1})()

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(sync_ops_service, "_acquire_full_refresh_creation_lock", noop)
    monkeypatch.setattr(
        sync_ops_service,
        "_get_active_pipeline_run",
        noop,
        raising=False,
    )
    monkeypatch.setattr(
        sync_ops_service,
        "get_app_settings",
        lambda: type("Settings", (), {"github_token": ""})(),
    )

    with pytest.raises(HTTPException) as exc_info:
        await sync_ops_service.trigger_full_refresh(
            payload=FullRefreshRequest(confirm=True),
            background_tasks=BackgroundTasks(),
            user=user,
            db=object(),
        )

    assert exc_info.value.status_code == 400
    assert "GitHub token" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_get_full_refresh_job_status_preserves_partial_failed_status(monkeypatch):
    from nebula.api.v2 import settings as settings_api

    async def fake_get_job_status(*_args, **_kwargs):
        return type(
            "JobStatus",
            (),
            {
                "task_id": 99,
                "task_type": "full_refresh",
                "status": "partial_failed",
                "phase": "complete",
                "progress_percent": 100.0,
                "eta_seconds": None,
                "last_error": None,
                "retryable": False,
                "started_at": None,
                "completed_at": None,
                "error_details": {
                    "partial_failures": [
                        {"phase": "stars", "task_id": 12, "failed_items": 2}
                    ]
                },
            },
        )()

    monkeypatch.setattr(settings_api, "get_job_status", fake_get_job_status)

    payload = await settings_api.get_full_refresh_job_status(
        task_id=99,
        user=type("User", (), {"id": 1})(),
        db=object(),
    )

    assert payload.job.status == "partial_failed"
    assert payload.job.error_details == {
        "partial_failures": [{"phase": "stars", "task_id": 12, "failed_items": 2}]
    }
