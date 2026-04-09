from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from nebula.application.services.graph_query_service import GraphQueryService
from nebula.schemas.graph import GraphData, GraphEdge


class _SnapshotRepoStub:
    def __init__(self):
        self.activated_snapshot_id: int | None = None

    async def get_active_snapshot(self, _db, _user_id):
        return SimpleNamespace(
            id=2,
            version="snapshot-active",
            created_at=datetime(2026, 3, 3, 12, 0, tzinfo=timezone.utc),
        )

    async def hydrate_graph_data(self, _db, snapshot, *, include_edges=True):
        edges = [GraphEdge(source=1, target=2, weight=0.9)] if include_edges else []
        return GraphData(
            nodes=[],
            edges=edges,
            clusters=[],
            star_lists=[],
            total_nodes=0,
            total_edges=1,
            total_clusters=0,
            total_star_lists=0,
            version=snapshot.version,
            generated_at=snapshot.created_at.isoformat(),
        )

    async def get_snapshot_by_version(self, _db, _user_id, _version):
        return None

    async def get_previous_snapshot(self, _db, *, user_id, exclude_snapshot_id):
        assert user_id == 1
        assert exclude_snapshot_id == 2
        return SimpleNamespace(
            id=1,
            version="snapshot-previous",
            created_at=datetime(2026, 3, 3, 11, 0, tzinfo=timezone.utc),
        )

    async def activate_snapshot(self, _db, _user_id, snapshot):
        self.activated_snapshot_id = snapshot.id

    async def validate_snapshot_consistency(self, _db, _snapshot):
        return True, None

    async def get_snapshot_metadata(self, _db, snapshot):
        return {
            "version": snapshot.version,
            "generated_at": snapshot.created_at.isoformat(),
            "total_nodes": 0,
            "total_edges": 1,
            "total_clusters": 0,
            "total_star_lists": 0,
        }


@pytest.mark.asyncio
async def test_get_graph_data_with_options_omits_edges():
    service = GraphQueryService(snapshot_repo=_SnapshotRepoStub())

    payload = await service.get_graph_data_with_options(
        db=object(),
        user=SimpleNamespace(id=1),
        version="active",
        include_edges=False,
    )

    assert payload.total_edges == 1
    assert payload.edges == []
    assert payload.request_id is not None


@pytest.mark.asyncio
async def test_rollback_active_snapshot_activates_previous():
    repo = _SnapshotRepoStub()
    service = GraphQueryService(snapshot_repo=repo)

    payload = await service.rollback_active_snapshot(
        db=object(),
        user=SimpleNamespace(id=1),
    )

    assert repo.activated_snapshot_id == 1
    assert payload.version == "snapshot-previous"
    assert payload.request_id is not None


@pytest.mark.asyncio
async def test_get_snapshot_metadata_returns_request_id():
    service = GraphQueryService(snapshot_repo=_SnapshotRepoStub())

    payload = await service.get_snapshot_metadata(
        db=object(),
        user=SimpleNamespace(id=1),
        version="active",
    )

    assert payload["version"] == "snapshot-active"
    assert payload["total_edges"] == 1
    assert payload["request_id"]
