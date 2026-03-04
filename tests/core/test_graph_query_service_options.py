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

    async def hydrate_graph_data(self, _db, snapshot):
        return GraphData(
            nodes=[],
            edges=[GraphEdge(source=1, target=2, weight=0.9)],
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


@pytest.mark.asyncio
async def test_get_graph_data_with_options_omits_edges(monkeypatch):
    async def _fake_user(_db):
        return SimpleNamespace(id=1)

    monkeypatch.setattr(
        "nebula.application.services.graph_query_service._get_default_user",
        _fake_user,
    )
    service = GraphQueryService(snapshot_repo=_SnapshotRepoStub())

    payload = await service.get_graph_data_with_options(
        db=object(),
        version="active",
        include_edges=False,
    )

    assert payload.total_edges == 1
    assert payload.edges == []
    assert payload.request_id is not None


@pytest.mark.asyncio
async def test_rollback_active_snapshot_activates_previous(monkeypatch):
    repo = _SnapshotRepoStub()

    async def _fake_user(_db):
        return SimpleNamespace(id=1)

    monkeypatch.setattr(
        "nebula.application.services.graph_query_service._get_default_user",
        _fake_user,
    )
    service = GraphQueryService(snapshot_repo=repo)

    payload = await service.rollback_active_snapshot(db=object())

    assert repo.activated_snapshot_id == 1
    assert payload.version == "snapshot-previous"
    assert payload.request_id is not None
