"""Snapshot persistence and pgvector search against a real database.

Covers the constraints and operators a fake session cannot reproduce: the
`ix_graph_snapshots_user_version` uniqueness index, chunked node/edge inserts,
retention pruning under a row lock, and the cosine-distance operator behind
`POST /api/v2/repos/search`.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

from nebula.core.config import get_embedding_settings
from nebula.db import (
    GraphSnapshot,
    GraphSnapshotEdge,
    GraphSnapshotNode,
    GraphSnapshotTimeline,
    StarredRepo,
    User,
)
from nebula.infrastructure.repositories.snapshot_repository import (
    SNAPSHOT_INSERT_BATCH_SIZE,
    SnapshotStoreRepository,
)
from nebula.schemas.graph import GraphData, GraphEdge, GraphNode, TimelineData

from .fakes import deterministic_embedding


async def _seed_user(db) -> User:
    user = User(github_id=2, username="snapshot-user")
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


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
        z=0.0,
        size=1.0,
    )


def _graph(node_count: int, edge_count: int) -> GraphData:
    return GraphData(
        nodes=[_node(i) for i in range(1, node_count + 1)],
        edges=[
            GraphEdge(source=1, target=i, weight=0.5) for i in range(2, edge_count + 2)
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


@pytest.mark.asyncio
async def test_chunked_inserts_persist_every_node_and_edge(integration_db):
    user = await _seed_user(integration_db)
    repo = SnapshotStoreRepository()

    # Deliberately over one batch so the chunking loop runs more than once.
    node_count = SNAPSHOT_INSERT_BATCH_SIZE + 250
    edge_count = 100

    snapshot = await repo.save_snapshot_payload(
        integration_db,
        user_id=user.id,
        version="v-chunked",
        graph_data=_graph(node_count, edge_count),
        timeline_data=_timeline(),
    )

    persisted_nodes = await integration_db.scalar(
        select(func.count(GraphSnapshotNode.id)).where(
            GraphSnapshotNode.snapshot_id == snapshot.id
        )
    )
    persisted_edges = await integration_db.scalar(
        select(func.count(GraphSnapshotEdge.id)).where(
            GraphSnapshotEdge.snapshot_id == snapshot.id
        )
    )
    timeline = await integration_db.scalar(
        select(GraphSnapshotTimeline.id).where(
            GraphSnapshotTimeline.snapshot_id == snapshot.id
        )
    )

    assert persisted_nodes == node_count
    assert persisted_edges == edge_count
    assert timeline is not None


@pytest.mark.asyncio
async def test_duplicate_version_is_rejected(integration_db):
    user = await _seed_user(integration_db)
    repo = SnapshotStoreRepository()

    await repo.save_snapshot_payload(
        integration_db,
        user_id=user.id,
        version="v1",
        graph_data=_graph(2, 1),
        timeline_data=_timeline(),
    )

    with pytest.raises(ValueError, match="already exists"):
        await repo.save_snapshot_payload(
            integration_db,
            user_id=user.id,
            version="v1",
            graph_data=_graph(2, 1),
            timeline_data=_timeline(),
        )


@pytest.mark.asyncio
async def test_activation_demotes_the_previous_active_snapshot(integration_db):
    user = await _seed_user(integration_db)
    repo = SnapshotStoreRepository()

    first = await repo.save_snapshot_payload(
        integration_db,
        user_id=user.id,
        version="v1",
        graph_data=_graph(2, 1),
        timeline_data=_timeline(),
    )
    second = await repo.save_snapshot_payload(
        integration_db,
        user_id=user.id,
        version="v2",
        graph_data=_graph(3, 2),
        timeline_data=_timeline(),
    )

    await repo.activate_snapshot(integration_db, user_id=user.id, snapshot=first)
    await repo.activate_snapshot(integration_db, user_id=user.id, snapshot=second)

    await integration_db.refresh(first)
    await integration_db.refresh(second)
    await integration_db.refresh(user)

    assert second.status == "active"
    assert first.status == "ready"
    assert user.active_graph_snapshot_id == second.id


@pytest.mark.asyncio
async def test_validation_matches_persisted_counts(integration_db):
    user = await _seed_user(integration_db)
    repo = SnapshotStoreRepository()

    snapshot = await repo.save_snapshot_payload(
        integration_db,
        user_id=user.id,
        version="v-valid",
        graph_data=_graph(5, 3),
        timeline_data=_timeline(),
    )

    ok, reason = await repo.validate_snapshot_consistency(integration_db, snapshot)
    assert ok is True, reason

    # Corrupt the metadata and confirm the validator notices.
    snapshot.meta = {**snapshot.meta, "total_nodes": 99}
    await integration_db.commit()
    ok, reason = await repo.validate_snapshot_consistency(integration_db, snapshot)
    assert ok is False
    assert "node count mismatch" in reason


@pytest.mark.asyncio
async def test_pruning_keeps_the_active_snapshot(integration_db):
    user = await _seed_user(integration_db)
    repo = SnapshotStoreRepository()

    snapshots = []
    for index in range(5):
        snapshots.append(
            await repo.save_snapshot_payload(
                integration_db,
                user_id=user.id,
                version=f"v{index}",
                graph_data=_graph(2, 1),
                timeline_data=_timeline(),
            )
        )
    await repo.activate_snapshot(integration_db, user_id=user.id, snapshot=snapshots[0])

    await repo.prune_snapshots(integration_db, user_id=user.id, keep_latest=2)

    remaining = set(
        (
            await integration_db.execute(
                select(GraphSnapshot.id).where(GraphSnapshot.user_id == user.id)
            )
        )
        .scalars()
        .all()
    )
    # The active snapshot is the oldest, so retention must protect it explicitly.
    assert snapshots[0].id in remaining


@pytest.mark.asyncio
async def test_vector_search_orders_by_cosine_distance(integration_db):
    """Exercises the pgvector `<=>` operator, which no fake can reproduce."""
    user = await _seed_user(integration_db)
    dimensions = get_embedding_settings().dimensions

    target = deterministic_embedding("target", dimensions, group=0)
    near = deterministic_embedding("target-neighbour", dimensions, group=0)
    far = deterministic_embedding("unrelated", dimensions, group=3)

    for index, (name, vector) in enumerate(
        [("owner/near", near), ("owner/far", far)], start=1
    ):
        integration_db.add(
            StarredRepo(
                user_id=user.id,
                github_repo_id=9000 + index,
                full_name=name,
                owner="owner",
                name=name.split("/")[1],
                html_url=f"https://github.com/{name}",
                embedding=vector,
                is_embedded=True,
            )
        )
    await integration_db.commit()

    rows = (
        await integration_db.execute(
            select(
                StarredRepo.full_name,
                (1 - StarredRepo.embedding.cosine_distance(target)).label("similarity"),
            )
            .where(StarredRepo.user_id == user.id, StarredRepo.is_embedded.is_(True))
            .order_by(StarredRepo.embedding.cosine_distance(target))
        )
    ).all()

    assert [row.full_name for row in rows] == ["owner/near", "owner/far"]
    assert rows[0].similarity > rows[1].similarity
