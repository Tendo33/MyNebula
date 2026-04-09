from types import SimpleNamespace

import pytest


class _FakeResult:
    def __init__(self, *, rows=None, scalar=None):
        self._rows = rows or []
        self._scalar = scalar

    def one(self):
        return self._scalar

    def scalar(self):
        return self._scalar

    def scalars(self):
        return self

    def all(self):
        return self._rows

    def __iter__(self):
        return iter(self._rows)


@pytest.mark.asyncio
async def test_dashboard_uses_snapshot_metadata_instead_of_hydrating_graph(monkeypatch):
    from nebula.api.v2 import dashboard as dashboard_api

    async def fake_snapshot_metadata(*_args, **_kwargs):
        return {
            "version": "snapshot-v1",
            "generated_at": "2026-04-09T00:00:00+00:00",
            "request_id": "req-1",
            "total_nodes": 10,
            "total_edges": 20,
            "total_clusters": 3,
            "total_star_lists": 1,
        }

    async def should_not_be_called(*_args, **_kwargs):
        raise AssertionError("graph hydration should not be used for dashboard metadata")

    class _FakeDb:
        def __init__(self):
            self.calls = 0

        async def execute(self, _statement, _params=None):
            self.calls += 1
            if self.calls == 1:
                return _FakeResult(
                    scalar=SimpleNamespace(total=10, embedded=8)
                )
            if self.calls == 2:
                return _FakeResult(
                    rows=[SimpleNamespace(language="TypeScript", count=6)]
                )
            if self.calls == 3:
                return _FakeResult(rows=[SimpleNamespace(topic="graph", count=4)])
            if self.calls == 4:
                return _FakeResult(scalar=17)
            return _FakeResult(
                rows=[
                    SimpleNamespace(
                        id=1,
                        name="Cluster 1",
                        repo_count=4,
                        color="#000",
                        keywords=["graph"],
                    )
                ]
            )

    monkeypatch.setattr(
        dashboard_api.graph_service,
        "get_snapshot_metadata",
        fake_snapshot_metadata,
    )
    monkeypatch.setattr(
        dashboard_api.graph_service,
        "get_graph_data_with_options",
        should_not_be_called,
    )

    payload = await dashboard_api.get_dashboard_data(
        user=SimpleNamespace(id=1),
        db=_FakeDb(),
    )

    assert payload.summary.total_edges == 20
    assert payload.summary.total_clusters == 3
    assert payload.summary.total_topics == 17
    assert payload.top_languages[0].language == "TypeScript"


@pytest.mark.asyncio
async def test_data_repos_uses_snapshot_metadata_instead_of_hydrating_graph(monkeypatch):
    from nebula.api.v2 import data as data_api

    async def fake_snapshot_metadata(*_args, **_kwargs):
        return {
            "version": "snapshot-v1",
            "generated_at": "2026-04-09T00:00:00+00:00",
            "request_id": "req-2",
            "total_nodes": 10,
            "total_edges": 20,
            "total_clusters": 3,
            "total_star_lists": 1,
        }

    async def should_not_be_called(*_args, **_kwargs):
        raise AssertionError("graph hydration should not be used for data metadata")

    repo = SimpleNamespace(
        id=1,
        full_name="octo/repo",
        name="repo",
        owner="octo",
        owner_avatar_url=None,
        description="desc",
        ai_summary="summary",
        topics=["ai"],
        language="TypeScript",
        stargazers_count=42,
        html_url="https://github.com/octo/repo",
        cluster_id=1,
        star_list_id=1,
        starred_at=None,
        repo_pushed_at=None,
    )
    cluster = SimpleNamespace(
        id=1,
        name="Cluster 1",
        color="#000",
        repo_count=1,
        keywords=[],
    )

    class _FakeDb:
        def __init__(self):
            self.calls = 0

        async def execute(self, _statement, _params=None):
            self.calls += 1
            if self.calls == 1:
                return _FakeResult(scalar=1)
            if self.calls == 2:
                return _FakeResult(scalar=3)
            if self.calls == 3:
                return _FakeResult(rows=[repo])
            return _FakeResult(rows=[cluster])

    monkeypatch.setattr(
        data_api.graph_service,
        "get_snapshot_metadata",
        fake_snapshot_metadata,
    )
    monkeypatch.setattr(
        data_api.graph_service,
        "get_graph_data_with_options",
        should_not_be_called,
    )

    payload = await data_api.get_data_repos(
        cluster_id=None,
        cluster_ids=None,
        language=None,
        min_stars=0,
        q=None,
        month=None,
        topic=None,
        sort_field="starred_at",
        sort_direction="desc",
        limit=200,
        offset=0,
        user=SimpleNamespace(id=1),
        db=_FakeDb(),
    )

    assert payload.total_repos == 3
    assert payload.items[0].full_name == "octo/repo"
    assert payload.version == "snapshot-v1"
