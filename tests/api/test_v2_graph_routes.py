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

    async def resolve_snapshot_version(_db, *, version: str):
        assert version == "active"
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
        db=object(),
    )

    assert isinstance(result, Response)
    assert result.status_code == 304
    assert called["count"] == 0
