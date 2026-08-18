"""Coverage for `GraphSnapshotBuilderService.build_payload`.

`_build_version` was already covered; the payload builder itself — node
projection, cluster colouring, the Uncategorized star-list synthesis, edge
candidate selection, and timeline aggregation — sat at 19% in the 2026-08-18
scan.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from nebula.application.services.graph_snapshot_service import (
    UNCATEGORIZED_STAR_LIST_ID,
    UNCATEGORIZED_STAR_LIST_NAME,
    GraphSnapshotBuilderService,
)

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


def _repo(
    repo_id: int,
    *,
    cluster_id=None,
    star_list_id=None,
    embedding=None,
    starred_at=None,
    coord=(1.0, 2.0, 3.0),
    language="Python",
    topics=("topic-a",),
    stars=100,
):
    return SimpleNamespace(
        id=repo_id,
        github_repo_id=repo_id + 500,
        user_id=7,
        full_name=f"owner/repo-{repo_id}",
        name=f"repo-{repo_id}",
        owner="owner",
        owner_avatar_url="https://avatars/owner",
        description=f"description {repo_id}",
        language=language,
        html_url=f"https://github.com/owner/repo-{repo_id}",
        coord_x=coord[0],
        coord_y=coord[1],
        coord_z=coord[2],
        cluster_id=cluster_id,
        star_list_id=star_list_id,
        stargazers_count=stars,
        ai_summary=f"summary {repo_id}",
        ai_tags=["tag"],
        topics=list(topics),
        starred_at=starred_at,
        repo_pushed_at=NOW,
        embedding=embedding,
        is_embedded=True,
    )


def _cluster(cluster_id: int, color=None):
    return SimpleNamespace(
        id=cluster_id,
        user_id=7,
        name=f"Cluster {cluster_id}",
        description="desc",
        keywords=["kw"],
        color=color,
        repo_count=3,
        center_x=0.0,
        center_y=0.0,
        center_z=0.0,
    )


def _star_list(list_id: int):
    return SimpleNamespace(
        id=list_id,
        user_id=7,
        name=f"List {list_id}",
        description="list desc",
        repo_count=2,
    )


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return SimpleNamespace(all=lambda: list(self._rows))

    def scalar_one_or_none(self):
        # Used by `get_default_user`; None drives the no-user branch.
        return self._rows[0] if self._rows else None


class _FakeDb:
    """Dispatches on the queried entity, which is what build_payload varies."""

    def __init__(self, *, repos, clusters, star_lists, timeline_repos=None):
        self.repos = repos
        self.clusters = clusters
        self.star_lists = star_lists
        self.timeline_repos = timeline_repos if timeline_repos is not None else repos
        self.queries: list[str] = []

    async def execute(self, statement):
        text = str(statement).lower()
        self.queries.append(text)
        if "from clusters" in text:
            return _Result(self.clusters)
        if "from star_lists" in text:
            return _Result(self.star_lists)
        if "starred_at is not null" in text:
            return _Result(self.timeline_repos)
        if "embedding is not null" in text:
            return _Result([r for r in self.repos if r.embedding is not None])
        return _Result(self.repos)


@pytest.mark.asyncio
async def test_no_user_yields_an_empty_but_versioned_payload(monkeypatch):
    # `get_default_user` bootstraps a user rather than returning None, so the
    # empty branch is reached by stubbing the resolver the builder calls.
    import nebula.application.services.graph_snapshot_service as module

    async def no_user(_db):
        return None

    monkeypatch.setattr(module, "_get_default_user", no_user)

    service = GraphSnapshotBuilderService()
    db = _FakeDb(repos=[], clusters=[], star_lists=[])

    version, graph, timeline = await service.build_payload(db, user=None)

    assert graph.total_nodes == 0
    assert graph.total_edges == 0
    assert timeline.total_stars == 0
    assert timeline.date_range == ("", "")
    # An empty graph still carries a version, so readers never see a null one.
    assert version.startswith("snapshot-")
    assert graph.version == version
    assert timeline.version == version
    assert graph.generated_at is not None


@pytest.mark.asyncio
async def test_nodes_take_cluster_colour_and_fall_back_to_grey():
    service = GraphSnapshotBuilderService()
    user = SimpleNamespace(id=7)
    clustered = _repo(1, cluster_id=10)
    unclustered = _repo(2, cluster_id=None)
    db = _FakeDb(
        repos=[clustered, unclustered],
        clusters=[_cluster(10, color="#123456")],
        star_lists=[],
    )

    _, graph, _ = await service.build_payload(db, user=user)

    by_id = {node.id: node for node in graph.nodes}
    assert by_id[1].color == "#123456"
    assert by_id[2].color == "#808080"


@pytest.mark.asyncio
async def test_cluster_without_explicit_colour_gets_a_palette_colour():
    service = GraphSnapshotBuilderService()
    user = SimpleNamespace(id=7)
    db = _FakeDb(
        repos=[_repo(1, cluster_id=10)],
        clusters=[_cluster(10, color=None)],
        star_lists=[],
    )

    _, graph, _ = await service.build_payload(db, user=user)

    assert graph.clusters[0].color != "#808080"
    assert graph.clusters[0].color.startswith("#")


@pytest.mark.asyncio
async def test_uncategorized_star_list_is_synthesised_only_when_needed():
    service = GraphSnapshotBuilderService()
    user = SimpleNamespace(id=7)

    with_orphans = _FakeDb(
        repos=[_repo(1, star_list_id=3), _repo(2, star_list_id=None)],
        clusters=[],
        star_lists=[_star_list(3)],
    )
    _, graph, _ = await service.build_payload(with_orphans, user=user)

    synthetic = [
        info for info in graph.star_lists if info.id == UNCATEGORIZED_STAR_LIST_ID
    ]
    assert len(synthetic) == 1
    assert synthetic[0].name == UNCATEGORIZED_STAR_LIST_NAME
    assert synthetic[0].repo_count == 1
    orphan_node = next(node for node in graph.nodes if node.id == 2)
    assert orphan_node.star_list_id == UNCATEGORIZED_STAR_LIST_ID
    assert orphan_node.star_list_name == UNCATEGORIZED_STAR_LIST_NAME

    all_categorised = _FakeDb(
        repos=[_repo(1, star_list_id=3)],
        clusters=[],
        star_lists=[_star_list(3)],
    )
    _, graph2, _ = await service.build_payload(all_categorised, user=user)
    assert all(info.id != UNCATEGORIZED_STAR_LIST_ID for info in graph2.star_lists)


@pytest.mark.asyncio
async def test_node_size_grows_with_stars():
    service = GraphSnapshotBuilderService()
    user = SimpleNamespace(id=7)
    db = _FakeDb(
        repos=[_repo(1, stars=1), _repo(2, stars=10_000)],
        clusters=[],
        star_lists=[],
    )

    _, graph, _ = await service.build_payload(db, user=user)

    by_id = {node.id: node for node in graph.nodes}
    assert by_id[1].size < by_id[2].size


@pytest.mark.asyncio
async def test_repos_without_embeddings_are_excluded_from_edge_candidates():
    service = GraphSnapshotBuilderService()
    user = SimpleNamespace(id=7)
    db = _FakeDb(
        repos=[_repo(1, embedding=None), _repo(2, embedding=[])],
        clusters=[],
        star_lists=[],
    )

    _, graph, _ = await service.build_payload(db, user=user)

    # Nodes still render; only edge candidacy requires a usable embedding.
    assert graph.total_nodes == 2
    assert graph.total_edges == 0


@pytest.mark.asyncio
async def test_similar_repos_produce_edges():
    service = GraphSnapshotBuilderService()
    user = SimpleNamespace(id=7)
    repos = [
        _repo(index, embedding=[1.0, 0.02 * index], topics=("shared",))
        for index in range(1, 6)
    ]
    db = _FakeDb(repos=repos, clusters=[], star_lists=[])

    _, graph, _ = await service.build_payload(db, user=user, edge_k=3)

    assert graph.total_edges > 0
    node_ids = {node.id for node in graph.nodes}
    for edge in graph.edges:
        assert edge.source in node_ids
        assert edge.target in node_ids


@pytest.mark.asyncio
async def test_timeline_aggregates_by_month_with_top_languages_and_topics():
    service = GraphSnapshotBuilderService()
    user = SimpleNamespace(id=7)
    timeline_repos = [
        _repo(
            1,
            starred_at=datetime(2026, 1, 5, tzinfo=timezone.utc),
            language="Python",
            topics=("ai", "ml"),
        ),
        _repo(
            2,
            starred_at=datetime(2026, 1, 20, tzinfo=timezone.utc),
            language="Python",
            topics=("ai",),
        ),
        _repo(
            3,
            starred_at=datetime(2026, 3, 2, tzinfo=timezone.utc),
            language="Rust",
            topics=("systems",),
        ),
    ]
    db = _FakeDb(
        repos=timeline_repos,
        clusters=[],
        star_lists=[],
        timeline_repos=timeline_repos,
    )

    _, _, timeline = await service.build_payload(db, user=user)

    assert [point.date for point in timeline.points] == ["2026-01", "2026-03"]
    assert timeline.points[0].count == 2
    assert timeline.points[0].top_languages == ["Python"]
    assert timeline.points[0].top_topics[0] == "ai"
    assert timeline.total_stars == 3
    assert timeline.date_range == ("2026-01", "2026-03")


@pytest.mark.asyncio
async def test_timeline_is_empty_when_nothing_has_a_starred_at():
    service = GraphSnapshotBuilderService()
    user = SimpleNamespace(id=7)
    db = _FakeDb(
        repos=[_repo(1, starred_at=None)],
        clusters=[],
        star_lists=[],
        timeline_repos=[],
    )

    _, _, timeline = await service.build_payload(db, user=user)

    assert timeline.points == []
    assert timeline.total_stars == 0
    assert timeline.date_range == ("", "")


@pytest.mark.asyncio
async def test_version_reflects_the_built_counts():
    service = GraphSnapshotBuilderService()
    user = SimpleNamespace(id=7)
    db = _FakeDb(
        repos=[_repo(index) for index in range(1, 4)],
        clusters=[],
        star_lists=[],
    )

    version, graph, timeline = await service.build_payload(db, user=user)

    assert f"n{graph.total_nodes}" in version
    assert f"e{graph.total_edges}" in version
    assert graph.version == timeline.version == version
