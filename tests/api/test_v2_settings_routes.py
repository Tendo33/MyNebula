from datetime import datetime, timezone

import pytest
from fastapi import HTTPException

from nebula.schemas.v2.settings import ScheduleResponse, SyncInfoResponse


@pytest.mark.asyncio
async def test_v2_settings_router_has_admin_dependency():
    from nebula.api.v2 import settings as settings_api

    assert settings_api.router.dependencies


@pytest.mark.asyncio
async def test_get_settings_returns_typed_payload(monkeypatch):
    from nebula.api.v2 import settings as settings_api

    async def fake_get_default_user(*_args, **_kwargs):
        return type("User", (), {"id": 1, "graph_max_clusters": 12, "graph_min_clusters": 4})()

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
            total_repos=10,
            synced_repos=10,
            embedded_repos=10,
            summarized_repos=9,
            schedule=None,
        )

    monkeypatch.setattr(settings_api, "get_schedule", fake_get_schedule)
    monkeypatch.setattr(settings_api, "get_sync_info", fake_get_sync_info)
    monkeypatch.setattr(settings_api, "get_default_user", fake_get_default_user)

    payload = await settings_api.get_settings(db=object())

    assert payload.schedule.is_enabled is True
    assert payload.sync_info.total_repos == 10
    assert payload.graph_defaults.max_clusters == 12
    assert payload.graph_defaults.min_clusters == 4
    assert payload.request_id is not None


@pytest.mark.asyncio
async def test_update_graph_defaults_persists_user_values(monkeypatch):
    from nebula.api.v2 import settings as settings_api
    from nebula.schemas.v2.settings import GraphDefaultsUpdateRequest

    user = type("User", (), {"id": 1, "graph_max_clusters": 8, "graph_min_clusters": 3})()

    async def fake_get_default_user(*_args, **_kwargs):
        return user

    class _Db:
        committed = False

        async def commit(self):
            self.committed = True

    db = _Db()
    monkeypatch.setattr(settings_api, "get_default_user", fake_get_default_user)

    payload = await settings_api.update_graph_defaults(
        config=GraphDefaultsUpdateRequest(max_clusters=15, min_clusters=5),
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
