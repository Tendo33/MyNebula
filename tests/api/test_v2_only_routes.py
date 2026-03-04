def test_api_router_exposes_v2_and_hides_legacy_graph_sync_routes():
    from nebula.api import api_router

    paths = {route.path for route in api_router.routes}

    assert any(path.startswith("/v2/") for path in paths)
    assert any(path.startswith("/v2/auth/") for path in paths)
    assert any(path.startswith("/v2/repos") for path in paths)
    assert not any(path.startswith("/auth/") for path in paths)
    assert not any(path.startswith("/repos") for path in paths)
    assert not any(path.startswith("/graph") for path in paths)
    assert not any(path.startswith("/sync") for path in paths)
