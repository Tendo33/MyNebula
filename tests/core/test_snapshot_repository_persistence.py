"""Coverage for `SnapshotStoreRepository` persistence and validation logic.

Complements `test_snapshot_repository_lifecycle.py`. These tests exercise the
paths that were uncovered in the 2026-08-18 scan: payload persistence with its
chunked inserts, activation, and the pre-activation consistency validator.

The database is faked here. Real constraint behaviour (the
`ix_graph_snapshots_user_version` uniqueness index, foreign keys, JSONB
operators) belongs to the integration tier.
"""

from types import SimpleNamespace

import pytest

from nebula.infrastructure.repositories.snapshot_repository import (
    SNAPSHOT_INSERT_BATCH_SIZE,
    SnapshotStoreRepository,
    _batched,
)
from nebula.schemas.graph import GraphData, GraphNode, TimelineData


def _node(node_id: int) -> GraphNode:
    return GraphNode(
        id=node_id,
        github_id=node_id + 500,
        full_name=f"owner/repo-{node_id}",
        name=f"repo-{node_id}",
        html_url=f"https://github.com/owner/repo-{node_id}",
        owner="owner",
        x=1.0,
        y=2.0,
        z=3.0,
        size=1.0,
    )


def _graph(node_count: int, edge_count: int) -> GraphData:
    from nebula.schemas.graph import GraphEdge

    return GraphData(
        nodes=[_node(index) for index in range(1, node_count + 1)],
        edges=[
            GraphEdge(source=1, target=index, weight=0.5)
            for index in range(2, edge_count + 2)
        ],
        clusters=[],
        star_lists=[],
        total_nodes=node_count,
        total_edges=edge_count,
        total_clusters=0,
        total_star_lists=0,
    )


def _timeline() -> TimelineData:
    return TimelineData(points=[], total_stars=0, date_range=("", ""))


class _Result:
    def __init__(self, *, scalar_value=None, one_value=None, rows=None):
        self._scalar_value = scalar_value
        self._one_value = one_value
        self._rows = rows or []

    def scalar(self):
        return self._scalar_value

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None

    def one(self):
        return self._one_value

    def scalars(self):
        return SimpleNamespace(all=lambda: list(self._rows))


class _FakeDb:
    """Records executed statements so batching and ordering are observable."""

    def __init__(self, *, results=None, objects=None, scalar_result=None):
        self.results = results or {}
        self.objects = objects or {}
        self.scalar_result = scalar_result
        self.executed: list[tuple[str, int]] = []
        self.added: list[object] = []
        self.commits = 0
        self._next_id = 1

    async def get(self, model, obj_id, **_kwargs):
        return self.objects.get((model.__name__, obj_id))

    async def execute(self, statement, params=None):
        text = str(statement).lower()
        payload_size = len(params) if isinstance(params, list) else 0
        self.executed.append((text, payload_size))
        for key, result in self.results.items():
            if key in text:
                return result
        return _Result()

    async def scalar(self, _statement):
        return self.scalar_result

    def add(self, obj):
        self.added.append(obj)
        if getattr(obj, "id", None) is None:
            obj.id = self._next_id
            self._next_id += 1

    async def flush(self):
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = self._next_id
                self._next_id += 1

    async def commit(self):
        self.commits += 1

    async def refresh(self, _obj):
        return None


def test_batched_splits_on_the_configured_size():
    assert list(_batched(range(5), size=2)) == [[0, 1], [2, 3], [4]]
    assert list(_batched([], size=2)) == []
    assert SNAPSHOT_INSERT_BATCH_SIZE == 1000


@pytest.mark.asyncio
async def test_get_active_snapshot_returns_none_without_a_user():
    repo = SnapshotStoreRepository()
    db = _FakeDb()

    assert await repo.get_active_snapshot(db, user_id=7) is None


@pytest.mark.asyncio
async def test_get_active_snapshot_returns_none_when_no_snapshot_is_active():
    repo = SnapshotStoreRepository()
    user = SimpleNamespace(id=7, active_graph_snapshot_id=None)
    db = _FakeDb(objects={("User", 7): user})

    assert await repo.get_active_snapshot(db, user_id=7) is None


@pytest.mark.asyncio
async def test_get_active_snapshot_resolves_the_referenced_snapshot():
    repo = SnapshotStoreRepository()
    snapshot = SimpleNamespace(id=42, user_id=7, version="v1")
    user = SimpleNamespace(id=7, active_graph_snapshot_id=42)
    db = _FakeDb(objects={("User", 7): user, ("GraphSnapshot", 42): snapshot})

    assert await repo.get_active_snapshot(db, user_id=7) is snapshot


@pytest.mark.asyncio
async def test_save_snapshot_payload_rejects_a_duplicate_version(monkeypatch):
    repo = SnapshotStoreRepository()
    existing = SimpleNamespace(id=1, version="v1")

    async def existing_version(*_args, **_kwargs):
        return existing

    monkeypatch.setattr(repo, "get_snapshot_by_version", existing_version)
    db = _FakeDb()

    with pytest.raises(ValueError, match="Snapshot version already exists"):
        await repo.save_snapshot_payload(
            db,
            user_id=7,
            version="v1",
            graph_data=_graph(1, 0),
            timeline_data=_timeline(),
        )


@pytest.mark.asyncio
async def test_save_snapshot_payload_chunks_node_and_edge_inserts(monkeypatch):
    repo = SnapshotStoreRepository()

    async def no_existing(*_args, **_kwargs):
        return None

    monkeypatch.setattr(repo, "get_snapshot_by_version", no_existing)
    db = _FakeDb()

    # 2500 nodes at a batch size of 1000 -> 3 insert statements.
    await repo.save_snapshot_payload(
        db,
        user_id=7,
        version="v1",
        graph_data=_graph(2500, 1200),
        timeline_data=_timeline(),
    )

    node_inserts = [
        size for text, size in db.executed if "graph_snapshot_nodes" in text
    ]
    edge_inserts = [
        size for text, size in db.executed if "graph_snapshot_edges" in text
    ]
    assert node_inserts == [1000, 1000, 500]
    assert edge_inserts == [1000, 200]
    assert db.commits == 1


@pytest.mark.asyncio
async def test_save_snapshot_payload_stores_counts_and_timeline(monkeypatch):
    repo = SnapshotStoreRepository()

    async def no_existing(*_args, **_kwargs):
        return None

    monkeypatch.setattr(repo, "get_snapshot_by_version", no_existing)
    db = _FakeDb()

    snapshot = await repo.save_snapshot_payload(
        db,
        user_id=7,
        version="v1",
        graph_data=_graph(3, 2),
        timeline_data=_timeline(),
    )

    assert snapshot.meta["total_nodes"] == 3
    assert snapshot.meta["total_edges"] == 2
    assert snapshot.status == "ready"
    timeline_rows = [
        obj for obj in db.added if type(obj).__name__ == "GraphSnapshotTimeline"
    ]
    assert len(timeline_rows) == 1


@pytest.mark.asyncio
async def test_activate_snapshot_demotes_the_previous_active_one():
    repo = SnapshotStoreRepository()
    user = SimpleNamespace(id=7, active_graph_snapshot_id=1)
    snapshot = SimpleNamespace(id=2, user_id=7, status="ready", activated_at=None)
    db = _FakeDb(objects={("User", 7): user})

    await repo.activate_snapshot(db, user_id=7, snapshot=snapshot)

    demote = [text for text, _ in db.executed if "update graph_snapshots" in text]
    assert len(demote) == 1
    assert snapshot.status == "active"
    assert snapshot.activated_at is not None
    assert user.active_graph_snapshot_id == 2
    assert db.commits == 1


@pytest.mark.asyncio
async def test_activate_snapshot_is_a_noop_without_a_user():
    repo = SnapshotStoreRepository()
    snapshot = SimpleNamespace(id=2, user_id=7, status="ready", activated_at=None)
    db = _FakeDb()

    await repo.activate_snapshot(db, user_id=7, snapshot=snapshot)

    assert snapshot.status == "ready"
    assert db.commits == 0


def _validation_db(*, nodes: int, edges: int, timeline_id, full_name, html_url):
    return _FakeDb(
        results={
            "graph_snapshot_nodes": _Result(
                scalar_value=nodes, one_value=(full_name, html_url)
            ),
            "graph_snapshot_edges": _Result(scalar_value=edges),
        },
        scalar_result=timeline_id,
    )


@pytest.mark.asyncio
async def test_validation_rejects_a_node_count_mismatch():
    repo = SnapshotStoreRepository()
    snapshot = SimpleNamespace(id=1, meta={"total_nodes": 10, "total_edges": 5})
    db = _validation_db(nodes=9, edges=5, timeline_id=1, full_name=9, html_url=9)

    ok, reason = await repo.validate_snapshot_consistency(db, snapshot)

    assert ok is False
    assert "node count mismatch" in reason


@pytest.mark.asyncio
async def test_validation_rejects_an_edge_count_mismatch():
    repo = SnapshotStoreRepository()
    snapshot = SimpleNamespace(id=1, meta={"total_nodes": 10, "total_edges": 5})
    db = _validation_db(nodes=10, edges=4, timeline_id=1, full_name=10, html_url=10)

    ok, reason = await repo.validate_snapshot_consistency(db, snapshot)

    assert ok is False
    assert "edge count mismatch" in reason


@pytest.mark.asyncio
async def test_validation_rejects_a_missing_timeline():
    repo = SnapshotStoreRepository()
    snapshot = SimpleNamespace(id=1, meta={"total_nodes": 10, "total_edges": 5})
    db = _validation_db(nodes=10, edges=5, timeline_id=None, full_name=10, html_url=10)

    ok, reason = await repo.validate_snapshot_consistency(db, snapshot)

    assert ok is False
    assert reason == "snapshot timeline payload missing"


@pytest.mark.asyncio
async def test_validation_rejects_too_many_nodes_missing_full_name():
    repo = SnapshotStoreRepository()
    snapshot = SimpleNamespace(id=1, meta={"total_nodes": 10, "total_edges": 5})
    db = _validation_db(nodes=10, edges=5, timeline_id=1, full_name=8, html_url=10)

    ok, reason = await repo.validate_snapshot_consistency(db, snapshot)

    assert ok is False
    assert "full_name" in reason


@pytest.mark.asyncio
async def test_validation_rejects_too_many_nodes_missing_html_url():
    repo = SnapshotStoreRepository()
    snapshot = SimpleNamespace(id=1, meta={"total_nodes": 10, "total_edges": 5})
    db = _validation_db(nodes=10, edges=5, timeline_id=1, full_name=10, html_url=8)

    ok, reason = await repo.validate_snapshot_consistency(db, snapshot)

    assert ok is False
    assert "html_url" in reason


@pytest.mark.asyncio
async def test_validation_accepts_a_consistent_snapshot():
    repo = SnapshotStoreRepository()
    snapshot = SimpleNamespace(id=1, meta={"total_nodes": 10, "total_edges": 5})
    db = _validation_db(nodes=10, edges=5, timeline_id=1, full_name=10, html_url=10)

    assert await repo.validate_snapshot_consistency(db, snapshot) == (True, None)


@pytest.mark.asyncio
async def test_validation_accepts_an_empty_snapshot_without_ratio_checks():
    repo = SnapshotStoreRepository()
    snapshot = SimpleNamespace(id=1, meta={"total_nodes": 0, "total_edges": 0})
    db = _validation_db(nodes=0, edges=0, timeline_id=1, full_name=0, html_url=0)

    # A zero-node snapshot must not trip a divide-by-zero ratio check.
    assert await repo.validate_snapshot_consistency(db, snapshot) == (True, None)
