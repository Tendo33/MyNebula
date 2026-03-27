"""Persistence helpers for graph snapshots."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.db import (
    GraphSnapshot,
    GraphSnapshotEdge,
    GraphSnapshotNode,
    GraphSnapshotTimeline,
    User,
)
from nebula.schemas.graph import GraphData, TimelineData


class SnapshotStoreRepository:
    """Repository for creating and querying snapshot payloads."""

    async def get_active_snapshot(
        self, db: AsyncSession, user_id: int
    ) -> GraphSnapshot | None:
        user = await db.get(User, user_id)
        if not user or not user.active_graph_snapshot_id:
            return None
        return await db.get(GraphSnapshot, user.active_graph_snapshot_id)

    async def get_snapshot_by_version(
        self, db: AsyncSession, user_id: int, version: str
    ) -> GraphSnapshot | None:
        result = await db.execute(
            select(GraphSnapshot).where(
                GraphSnapshot.user_id == user_id,
                GraphSnapshot.version == version,
            )
        )
        return result.scalar_one_or_none()

    async def save_snapshot_payload(
        self,
        db: AsyncSession,
        user_id: int,
        version: str,
        graph_data: GraphData,
        timeline_data: TimelineData,
        status: str = "ready",
    ) -> GraphSnapshot:
        snapshot = await self.get_snapshot_by_version(db, user_id, version)
        if snapshot:
            await db.execute(
                delete(GraphSnapshotNode).where(
                    GraphSnapshotNode.snapshot_id == snapshot.id
                )
            )
            await db.execute(
                delete(GraphSnapshotEdge).where(
                    GraphSnapshotEdge.snapshot_id == snapshot.id
                )
            )
            await db.execute(
                delete(GraphSnapshotTimeline).where(
                    GraphSnapshotTimeline.snapshot_id == snapshot.id
                )
            )
            snapshot.status = status
            snapshot.meta = {
                "total_nodes": graph_data.total_nodes,
                "total_edges": graph_data.total_edges,
                "total_clusters": graph_data.total_clusters,
                "total_star_lists": graph_data.total_star_lists,
                "clusters": [cluster.model_dump() for cluster in graph_data.clusters],
                "star_lists": [
                    star_list.model_dump() for star_list in graph_data.star_lists
                ],
            }
        else:
            snapshot = GraphSnapshot(
                user_id=user_id,
                version=version,
                status=status,
                meta={
                    "total_nodes": graph_data.total_nodes,
                    "total_edges": graph_data.total_edges,
                    "total_clusters": graph_data.total_clusters,
                    "total_star_lists": graph_data.total_star_lists,
                    "clusters": [
                        cluster.model_dump() for cluster in graph_data.clusters
                    ],
                    "star_lists": [
                        star_list.model_dump() for star_list in graph_data.star_lists
                    ],
                },
            )
            db.add(snapshot)
            await db.flush()

        db.add_all(
            [
                GraphSnapshotNode(
                    snapshot_id=snapshot.id,
                    repo_id=node.id,
                    payload=node.model_dump(),
                )
                for node in graph_data.nodes
            ]
        )
        db.add_all(
            [
                GraphSnapshotEdge(
                    snapshot_id=snapshot.id,
                    edge_index=index,
                    source=edge.source,
                    target=edge.target,
                    weight=edge.weight,
                )
                for index, edge in enumerate(graph_data.edges)
            ]
        )
        db.add(
            GraphSnapshotTimeline(
                snapshot_id=snapshot.id,
                payload=timeline_data.model_dump(),
            )
        )
        await db.commit()
        await db.refresh(snapshot)
        return snapshot

    async def activate_snapshot(
        self, db: AsyncSession, user_id: int, snapshot: GraphSnapshot
    ) -> None:
        user = await db.get(User, user_id)
        if not user:
            return
        await db.execute(
            update(GraphSnapshot)
            .where(
                GraphSnapshot.user_id == user_id,
                GraphSnapshot.id != snapshot.id,
                GraphSnapshot.status == "active",
            )
            .values(status="ready")
        )
        snapshot.status = "active"
        snapshot.activated_at = datetime.now(timezone.utc)
        user.active_graph_snapshot_id = snapshot.id
        await db.commit()

    async def validate_snapshot_consistency(
        self,
        db: AsyncSession,
        snapshot: GraphSnapshot,
        *,
        min_required_fields_ratio: float = 0.95,
    ) -> tuple[bool, str | None]:
        """Validate persisted snapshot consistency before activation."""
        meta = snapshot.meta or {}
        expected_nodes = int(meta.get("total_nodes", 0))
        expected_edges = int(meta.get("total_edges", 0))

        node_count_result = await db.execute(
            select(func.count(GraphSnapshotNode.id)).where(
                GraphSnapshotNode.snapshot_id == snapshot.id
            )
        )
        edge_count_result = await db.execute(
            select(func.count(GraphSnapshotEdge.id)).where(
                GraphSnapshotEdge.snapshot_id == snapshot.id
            )
        )
        timeline_exists = await db.scalar(
            select(GraphSnapshotTimeline.id).where(
                GraphSnapshotTimeline.snapshot_id == snapshot.id
            )
        )

        actual_nodes = int(node_count_result.scalar() or 0)
        actual_edges = int(edge_count_result.scalar() or 0)

        if expected_nodes != actual_nodes:
            return (
                False,
                f"snapshot node count mismatch expected={expected_nodes} actual={actual_nodes}",
            )
        if expected_edges != actual_edges:
            return (
                False,
                f"snapshot edge count mismatch expected={expected_edges} actual={actual_edges}",
            )
        if timeline_exists is None:
            return (False, "snapshot timeline payload missing")

        required_fields_ok_result = await db.execute(
            select(
                func.count(GraphSnapshotNode.id).filter(
                    GraphSnapshotNode.payload["full_name"].is_not(None),
                ),
                func.count(GraphSnapshotNode.id).filter(
                    GraphSnapshotNode.payload["html_url"].is_not(None),
                ),
            ).where(GraphSnapshotNode.snapshot_id == snapshot.id)
        )
        full_name_count, html_url_count = required_fields_ok_result.one()
        if actual_nodes > 0:
            full_name_ratio = float(full_name_count or 0) / float(actual_nodes)
            html_url_ratio = float(html_url_count or 0) / float(actual_nodes)
            if full_name_ratio < min_required_fields_ratio:
                return (
                    False,
                    f"snapshot required field ratio(full_name) too low: {full_name_ratio:.3f}",
                )
            if html_url_ratio < min_required_fields_ratio:
                return (
                    False,
                    f"snapshot required field ratio(html_url) too low: {html_url_ratio:.3f}",
                )

        return (True, None)

    async def get_previous_snapshot(
        self,
        db: AsyncSession,
        *,
        user_id: int,
        exclude_snapshot_id: int | None = None,
    ) -> GraphSnapshot | None:
        query = (
            select(GraphSnapshot)
            .where(GraphSnapshot.user_id == user_id)
            .order_by(
                GraphSnapshot.activated_at.desc().nullslast(),
                GraphSnapshot.id.desc(),
            )
        )
        if exclude_snapshot_id is not None:
            query = query.where(GraphSnapshot.id != exclude_snapshot_id)
        result = await db.execute(query.limit(1))
        return result.scalar_one_or_none()

    async def hydrate_graph_data(
        self,
        db: AsyncSession,
        snapshot: GraphSnapshot,
        *,
        include_edges: bool = True,
    ) -> GraphData:
        node_rows = await db.execute(
            select(GraphSnapshotNode)
            .where(GraphSnapshotNode.snapshot_id == snapshot.id)
            .order_by(GraphSnapshotNode.id.asc())
        )

        nodes = [row.payload for row in node_rows.scalars().all()]
        edges: list[dict] = []
        if include_edges:
            edge_rows = await db.execute(
                select(GraphSnapshotEdge)
                .where(GraphSnapshotEdge.snapshot_id == snapshot.id)
                .order_by(GraphSnapshotEdge.edge_index.asc())
            )
            edges = [
                {"source": row.source, "target": row.target, "weight": row.weight}
                for row in edge_rows.scalars().all()
            ]

        meta = snapshot.meta or {}
        return GraphData(
            nodes=nodes,
            edges=edges,
            clusters=meta.get("clusters", []),
            star_lists=meta.get("star_lists", []),
            total_nodes=int(meta.get("total_nodes", len(nodes))),
            total_edges=int(meta.get("total_edges", len(edges))),
            total_clusters=int(meta.get("total_clusters", 0)),
            total_star_lists=int(meta.get("total_star_lists", 0)),
            version=snapshot.version,
            generated_at=snapshot.created_at.isoformat()
            if snapshot.created_at
            else None,
        )

    async def hydrate_timeline_data(
        self, db: AsyncSession, snapshot_id: int
    ) -> TimelineData | None:
        row = await db.scalar(
            select(GraphSnapshotTimeline).where(
                GraphSnapshotTimeline.snapshot_id == snapshot_id
            )
        )
        if row is None:
            return None
        return TimelineData(**row.payload)

    async def get_edges_page(
        self,
        db: AsyncSession,
        snapshot_id: int,
        cursor: int,
        limit: int,
    ) -> tuple[list[dict], int | None]:
        rows = await db.execute(
            select(GraphSnapshotEdge)
            .where(
                GraphSnapshotEdge.snapshot_id == snapshot_id,
                GraphSnapshotEdge.edge_index >= cursor,
            )
            .order_by(GraphSnapshotEdge.edge_index.asc())
            .limit(limit + 1)
        )
        edges = rows.scalars().all()
        has_next = len(edges) > limit
        current = edges[:limit]
        next_cursor = current[-1].edge_index + 1 if has_next and current else None
        payload = [
            {"source": edge.source, "target": edge.target, "weight": edge.weight}
            for edge in current
        ]
        return payload, next_cursor
