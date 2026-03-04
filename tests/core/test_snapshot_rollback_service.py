from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from nebula.application.services.graph_query_service import GraphQueryService
from nebula.schemas.graph import GraphData, TimelineData


class _BuilderStub:
    async def build_payload(self, _db):
        now = datetime(2026, 3, 4, tzinfo=timezone.utc)
        return (
            "snapshot-v1",
            GraphData(
                nodes=[],
                edges=[],
                clusters=[],
                star_lists=[],
                total_nodes=0,
                total_edges=0,
                total_clusters=0,
                total_star_lists=0,
                generated_at=now.isoformat(),
            ),
            TimelineData(points=[], total_stars=0, date_range=("", ""), generated_at=now.isoformat()),
        )


class _InvalidSnapshotRepo:
    async def save_snapshot_payload(self, *_args, **_kwargs):
        return SimpleNamespace(id=2, version="snapshot-v1", created_at=datetime.now(timezone.utc), meta={})

    async def validate_snapshot_consistency(self, *_args, **_kwargs):
        return False, "node count mismatch"


@pytest.mark.asyncio
async def test_rebuild_snapshot_requires_consistency_validation(monkeypatch):
    async def fake_user(_db):
        return SimpleNamespace(id=1)

    monkeypatch.setattr(
        "nebula.application.services.graph_query_service._get_default_user",
        fake_user,
    )
    service = GraphQueryService(
        snapshot_repo=_InvalidSnapshotRepo(),
        builder=_BuilderStub(),
    )

    with pytest.raises(ValueError, match="consistency validation failed"):
        await service.rebuild_active_snapshot(db=object())
