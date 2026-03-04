import pytest


@pytest.mark.asyncio
async def test_legacy_sync_adapter_routes_are_preserved():
    from nebula.api import sync as legacy_sync_api

    route_map = {route.path: getattr(route, "response_model", None) for route in legacy_sync_api.router.routes}

    assert "/schedule" in route_map
    assert "/info" in route_map
    assert "/full-refresh" in route_map

    assert route_map["/schedule"].__name__ == "ScheduleResponse"
    assert route_map["/info"].__name__ == "SyncInfoResponse"
    assert route_map["/full-refresh"].__name__ == "FullRefreshResponse"


@pytest.mark.asyncio
async def test_v2_settings_routes_remain_available_for_adapter_migration():
    from nebula.api.v2 import settings as v2_settings_api

    paths = {route.path for route in v2_settings_api.router.routes}

    assert "" in paths
    assert "/schedule" in paths
    assert "/full-refresh" in paths
    assert "/full-refresh/jobs/{task_id}" in paths
