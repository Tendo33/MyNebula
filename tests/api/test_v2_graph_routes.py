from types import SimpleNamespace

import pytest
from starlette.requests import Request
from starlette.responses import Response

from nebula.schemas.graph import GraphData


def _build_request(if_none_match: str | None = None) -> Request:
    headers = []
    if if_none_match:
        headers.append((b"if-none-match", if_none_match.encode("utf-8")))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v2/graph",
        "headers": headers,
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_get_graph_returns_304_when_etag_matches(monkeypatch):
    from nebula.api.v2 import graph as graph_api

    called = {"count": 0}

    async def resolve_snapshot_version(_db, *, user, version: str):
        assert version == "active"
        assert user.id == 1
        return "snapshot-a"

    async def get_graph_data_with_options(*_args, **_kwargs):
        called["count"] += 1
        return GraphData(
            nodes=[],
            edges=[],
            clusters=[],
            star_lists=[],
            total_nodes=0,
            total_edges=0,
            total_clusters=0,
            total_star_lists=0,
        )

    monkeypatch.setattr(
        graph_api,
        "graph_service",
        SimpleNamespace(
            resolve_snapshot_version=resolve_snapshot_version,
            get_graph_data_with_options=get_graph_data_with_options,
        ),
    )

    request = _build_request('W/"graph:snapshot-a:edges:0"')
    response = Response()
    result = await graph_api.get_graph(
        request=request,
        response=response,
        version="active",
        include_edges=False,
        user=SimpleNamespace(id=1),
        db=object(),
    )

    assert isinstance(result, Response)
    assert result.status_code == 304
    assert called["count"] == 0


@pytest.mark.asyncio
async def test_graph_rebuild_route_requires_admin_auth_and_csrf():
    from nebula.api.v2 import graph as graph_api

    rebuild_route = next(
        (route for route in graph_api.router.routes if route.path == "/rebuild"), None
    )
    assert rebuild_route is not None
    dependency_calls = {
        getattr(getattr(dependency, "call", None), "__name__", "")
        for dependency in rebuild_route.dependant.dependencies
    }
    assert "require_admin" in dependency_calls
    assert "require_admin_csrf" in dependency_calls


@pytest.mark.asyncio
async def test_get_graph_returns_404_for_missing_snapshot_version(monkeypatch):
    from nebula.api.v2 import graph as graph_api
    from nebula.application.services.graph_query_service import (
        SnapshotVersionNotFoundError,
    )

    async def resolve_snapshot_version(_db, *, user, version: str):
        assert user.id == 1
        raise SnapshotVersionNotFoundError(f"Snapshot version not found: {version}")

    monkeypatch.setattr(
        graph_api,
        "graph_service",
        SimpleNamespace(resolve_snapshot_version=resolve_snapshot_version),
    )

    request = _build_request()
    response = Response()

    with pytest.raises(graph_api.HTTPException) as exc_info:
        await graph_api.get_graph(
            request=request,
            response=response,
            version="snapshot-missing",
            include_edges=False,
            user=SimpleNamespace(id=1),
            db=object(),
        )

    assert exc_info.value.status_code == 404
    assert "snapshot-missing" in str(exc_info.value.detail)
