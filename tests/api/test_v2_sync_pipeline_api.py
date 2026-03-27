from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException


@pytest.mark.asyncio
async def test_v2_sync_router_has_admin_dependency():
    from nebula.api.v2 import sync as sync_api

    assert sync_api.router.dependencies
    dependency_names = {
        getattr(dep.dependency, "__name__", "") for dep in sync_api.router.dependencies
    }
    assert "require_admin" in dependency_names
    assert "require_admin_csrf" in dependency_names
    paths = {route.path for route in sync_api.router.routes}
    assert "/start" in paths
    assert "/recluster" in paths


@pytest.mark.asyncio
async def test_start_pipeline_sync_returns_metadata(monkeypatch):
    from nebula.api.v2 import sync as sync_api

    async def fake_get_default_user(_db):
        return SimpleNamespace(id=1, graph_max_clusters=8, graph_min_clusters=3)

    async def fake_create_pipeline_run(_user_id):
        return 99

    async def fake_get_active_pipeline(_user_id):
        return None

    monkeypatch.setattr(sync_api, "get_default_user", fake_get_default_user)
    monkeypatch.setattr(
        sync_api,
        "pipeline_service",
        SimpleNamespace(
            get_active_pipeline=fake_get_active_pipeline,
            create_pipeline_run=fake_create_pipeline_run,
            run_pipeline=lambda *_args, **_kwargs: None,
        ),
    )

    response = await sync_api.start_pipeline_sync(
        background_tasks=BackgroundTasks(),
        mode="incremental",
        use_llm=True,
        max_clusters=8,
        min_clusters=3,
        db=object(),
    )

    assert response.pipeline_run_id == 99
    assert response.request_id is not None
    assert response.version == "pipeline-99"


@pytest.mark.asyncio
async def test_start_recluster_sync_returns_metadata(monkeypatch):
    from nebula.api.v2 import sync as sync_api

    user = SimpleNamespace(id=1, graph_max_clusters=8, graph_min_clusters=3)

    async def fake_get_default_user(_db):
        return user

    async def fake_create_pipeline_run(_user_id):
        return 100

    async def fake_get_active_pipeline(_user_id):
        return None

    monkeypatch.setattr(sync_api, "get_default_user", fake_get_default_user)
    monkeypatch.setattr(
        sync_api,
        "pipeline_service",
        SimpleNamespace(
            get_active_pipeline=fake_get_active_pipeline,
            create_pipeline_run=fake_create_pipeline_run,
            run_recluster_pipeline=lambda *_args, **_kwargs: None,
        ),
    )
    db = SimpleNamespace(commit=lambda: None)

    async def _commit():
        return None

    db.commit = _commit

    response = await sync_api.start_recluster_sync(
        background_tasks=BackgroundTasks(),
        max_clusters=12,
        min_clusters=5,
        db=db,
    )

    assert response.pipeline_run_id == 100
    assert response.request_id is not None
    assert response.version == "pipeline-100"
    assert user.graph_max_clusters == 12
    assert user.graph_min_clusters == 5


@pytest.mark.asyncio
async def test_get_pipeline_status_returns_404_for_other_user(monkeypatch):
    from nebula.api.v2 import sync as sync_api

    async def fake_get_default_user(_db):
        return SimpleNamespace(id=2, graph_max_clusters=8, graph_min_clusters=3)

    async def fake_get_pipeline(_run_id):
        return SimpleNamespace(
            id=1,
            user_id=1,
            status="running",
            phase="stars",
            last_error=None,
            created_at=None,
            started_at=None,
            completed_at=None,
        )

    monkeypatch.setattr(sync_api, "get_default_user", fake_get_default_user)
    monkeypatch.setattr(
        sync_api,
        "pipeline_service",
        SimpleNamespace(get_pipeline=fake_get_pipeline),
    )

    with pytest.raises(HTTPException) as exc_info:
        await sync_api.get_pipeline_status(run_id=1, db=object())

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_start_pipeline_sync_rejects_when_active_pipeline_exists(monkeypatch):
    from nebula.api.v2 import sync as sync_api

    async def fake_get_default_user(_db):
        return SimpleNamespace(id=1, graph_max_clusters=8, graph_min_clusters=3)

    async def fake_get_active_pipeline(_user_id):
        return SimpleNamespace(id=42, status="running")

    monkeypatch.setattr(sync_api, "get_default_user", fake_get_default_user)
    monkeypatch.setattr(
        sync_api,
        "pipeline_service",
        SimpleNamespace(
            get_active_pipeline=fake_get_active_pipeline,
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        await sync_api.start_pipeline_sync(
            background_tasks=BackgroundTasks(),
            mode="incremental",
            use_llm=True,
            max_clusters=8,
            min_clusters=3,
            db=object(),
        )

    assert exc_info.value.status_code == 409
