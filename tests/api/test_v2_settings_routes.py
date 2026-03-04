from datetime import datetime, timezone

import pytest

from nebula.schemas.schedule import ScheduleResponse, SyncInfoResponse


@pytest.mark.asyncio
async def test_v2_settings_router_has_admin_dependency():
    from nebula.api.v2 import settings as settings_api

    assert settings_api.router.dependencies


@pytest.mark.asyncio
async def test_get_settings_returns_typed_payload(monkeypatch):
    from nebula.api.v2 import settings as settings_api

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

    payload = await settings_api.get_settings(db=object())

    assert payload.schedule.is_enabled is True
    assert payload.sync_info.total_repos == 10
    assert payload.graph_defaults.max_clusters >= payload.graph_defaults.min_clusters
    assert payload.request_id is not None
