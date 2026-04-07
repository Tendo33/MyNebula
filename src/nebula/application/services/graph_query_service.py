"""Query services for versioned graph responses."""

from __future__ import annotations

import uuid
from time import perf_counter

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.core.config import get_app_settings
from nebula.db import GraphSnapshot, User
from nebula.infrastructure.repositories import SnapshotStoreRepository
from nebula.schemas.graph import GraphData, GraphEdge, TimelineData
from nebula.schemas.v2.graph import GraphEdgesPage
from nebula.utils import get_logger

from .graph_snapshot_service import GraphSnapshotBuilderService

logger = get_logger(__name__)
SNAPSHOT_REBUILD_LOCK_KEY_BASE = 880_000_000


class GraphQueryService:
    """Read model service for graph data endpoints."""

    def __init__(
        self,
        snapshot_repo: SnapshotStoreRepository | None = None,
        builder: GraphSnapshotBuilderService | None = None,
    ) -> None:
        self.snapshot_repo = snapshot_repo or SnapshotStoreRepository()
        self.builder = builder or GraphSnapshotBuilderService()
        self.settings = get_app_settings()

    async def ensure_active_snapshot(
        self, db: AsyncSession, *, user: User
    ) -> GraphSnapshot:
        snapshot = await self.snapshot_repo.get_active_snapshot(db, user.id)
        if snapshot:
            return snapshot

        await self._acquire_snapshot_rebuild_lock(db, user.id)
        snapshot = await self.snapshot_repo.get_active_snapshot(db, user.id)
        if snapshot:
            return snapshot

        version, graph_data, timeline_data = await self.builder.build_payload(db)
        snapshot = await self.snapshot_repo.save_snapshot_payload(
            db,
            user_id=user.id,
            version=version,
            graph_data=graph_data,
            timeline_data=timeline_data,
        )
        await self._validate_snapshot_or_raise(db, snapshot)
        await self.snapshot_repo.activate_snapshot(db, user.id, snapshot)
        logger.info(f"Built and activated initial graph snapshot: {version}")
        return snapshot

    async def get_graph_data(
        self,
        db: AsyncSession,
        *,
        user: User,
        version: str = "active",
    ) -> GraphData:
        return await self.get_graph_data_with_options(
            db,
            user=user,
            version=version,
            include_edges=True,
        )

    async def get_graph_data_with_options(
        self,
        db: AsyncSession,
        *,
        user: User,
        version: str = "active",
        include_edges: bool = True,
    ) -> GraphData:
        started = perf_counter()
        try:
            snapshot = await self._resolve_snapshot(db, user, version)
            graph_data = await self.snapshot_repo.hydrate_graph_data(
                db,
                snapshot,
                include_edges=include_edges,
            )
        except Exception:
            if not self.settings.snapshot_read_fallback_on_error:
                raise
            logger.exception("Snapshot read failed, fallback to live graph payload")
            _, graph_data, _ = await self.builder.build_payload(db)

        if not include_edges:
            graph_data.edges = []

        graph_data.request_id = str(uuid.uuid4())
        self._log_if_slow("get_graph_data_with_options", started, version=version)
        return graph_data

    async def get_timeline_data(
        self, db: AsyncSession, *, user: User, version: str = "active"
    ) -> TimelineData:
        started = perf_counter()
        try:
            snapshot = await self._resolve_snapshot(db, user, version)
            timeline = await self.snapshot_repo.hydrate_timeline_data(db, snapshot.id)
            if timeline is None:
                timeline = TimelineData(
                    points=[],
                    total_stars=0,
                    date_range=("", ""),
                    version=snapshot.version,
                    generated_at=snapshot.created_at.isoformat()
                    if snapshot.created_at
                    else None,
                )
        except Exception:
            if not self.settings.snapshot_read_fallback_on_error:
                raise
            logger.exception(
                "Snapshot timeline read failed, fallback to live timeline payload"
            )
            _, _, timeline = await self.builder.build_payload(db)

        timeline.request_id = str(uuid.uuid4())
        self._log_if_slow("get_timeline_data", started, version=version)
        return timeline

    async def get_edges_page(
        self,
        db: AsyncSession,
        *,
        user: User,
        version: str = "active",
        cursor: int = 0,
        limit: int = 500,
    ) -> GraphEdgesPage:
        started = perf_counter()
        try:
            snapshot = await self._resolve_snapshot(db, user, version)
            edge_payload, next_cursor = await self.snapshot_repo.get_edges_page(
                db,
                snapshot_id=snapshot.id,
                cursor=cursor,
                limit=limit,
            )
            page = GraphEdgesPage(
                edges=[GraphEdge(**edge) for edge in edge_payload],
                next_cursor=next_cursor,
                version=snapshot.version,
                generated_at=snapshot.created_at.isoformat()
                if snapshot.created_at
                else None,
            )
        except Exception:
            if not self.settings.snapshot_read_fallback_on_error:
                raise
            logger.exception("Snapshot edge read failed, fallback to live edge payload")
            _, graph_data, _ = await self.builder.build_payload(db)
            sliced_edges = graph_data.edges[cursor : cursor + limit]
            next_cursor = (
                cursor + limit if cursor + limit < len(graph_data.edges) else None
            )
            page = GraphEdgesPage(
                edges=sliced_edges,
                next_cursor=next_cursor,
                version=graph_data.version or "live-fallback",
                generated_at=graph_data.generated_at,
            )
        page.request_id = str(uuid.uuid4())
        self._log_if_slow(
            "get_edges_page",
            started,
            version=version,
            cursor=cursor,
            limit=limit,
        )
        return page

    async def rebuild_active_snapshot(
        self, db: AsyncSession, *, user: User
    ) -> GraphData:
        await self._acquire_snapshot_rebuild_lock(db, user.id)
        version, graph_data, timeline_data = await self.builder.build_payload(db)
        snapshot = await self.snapshot_repo.save_snapshot_payload(
            db,
            user_id=user.id,
            version=version,
            graph_data=graph_data,
            timeline_data=timeline_data,
        )
        await self._validate_snapshot_or_raise(db, snapshot)
        await self.snapshot_repo.activate_snapshot(db, user.id, snapshot)
        graph_data.version = version
        graph_data.generated_at = (
            snapshot.created_at.isoformat() if snapshot.created_at else None
        )
        graph_data.request_id = str(uuid.uuid4())
        return graph_data

    async def rollback_active_snapshot(
        self,
        db: AsyncSession,
        *,
        user: User,
        target_version: str | None = None,
    ) -> GraphData:
        current_active = await self.snapshot_repo.get_active_snapshot(db, user.id)
        if current_active is None:
            raise ValueError("No active snapshot to rollback from")

        if target_version:
            target = await self.snapshot_repo.get_snapshot_by_version(
                db,
                user_id=user.id,
                version=target_version,
            )
            if target is None:
                raise ValueError(f"Snapshot version not found: {target_version}")
        else:
            target = await self.snapshot_repo.get_previous_snapshot(
                db,
                user_id=user.id,
                exclude_snapshot_id=current_active.id,
            )
            if target is None:
                raise ValueError("No previous snapshot available for rollback")

        await self.snapshot_repo.activate_snapshot(db, user.id, target)
        await self._validate_snapshot_or_raise(db, target)
        graph_data = await self.snapshot_repo.hydrate_graph_data(db, target)
        graph_data.request_id = str(uuid.uuid4())
        return graph_data

    async def resolve_snapshot_version(
        self, db: AsyncSession, *, user: User, version: str = "active"
    ) -> str:
        snapshot = await self._resolve_snapshot(db, user, version)
        return snapshot.version

    async def _resolve_snapshot(
        self,
        db: AsyncSession,
        user: User,
        version: str,
    ) -> GraphSnapshot:
        if version == "active":
            return await self.ensure_active_snapshot(db, user=user)
        snapshot = await self.snapshot_repo.get_snapshot_by_version(
            db,
            user.id,
            version,
        )
        if snapshot is None:
            return await self.ensure_active_snapshot(db, user=user)
        return snapshot

    def _log_if_slow(
        self, operation: str, started: float, **context: int | str
    ) -> None:
        elapsed_ms = int((perf_counter() - started) * 1000)
        if elapsed_ms <= self.settings.slow_query_log_ms:
            return
        logger.warning(
            f"Slow graph query op={operation} elapsed_ms={elapsed_ms} context={context}"
        )

    async def _validate_snapshot_or_raise(
        self,
        db: AsyncSession,
        snapshot: GraphSnapshot,
    ) -> None:
        is_consistent, reason = await self.snapshot_repo.validate_snapshot_consistency(
            db,
            snapshot,
        )
        if is_consistent:
            return
        raise ValueError(
            f"Snapshot consistency validation failed for version={snapshot.version}: {reason}"
        )

    async def _acquire_snapshot_rebuild_lock(
        self,
        db: AsyncSession,
        user_id: int,
    ) -> None:
        """Serialize snapshot rebuild/creation per user when PostgreSQL is in use."""
        get_bind = getattr(db, "get_bind", None)
        if get_bind is None:
            return
        bind = get_bind()
        dialect_name = bind.dialect.name if bind is not None else ""
        if dialect_name != "postgresql":
            return
        await db.execute(
            text("SELECT pg_advisory_xact_lock(:lock_key)"),
            {"lock_key": SNAPSHOT_REBUILD_LOCK_KEY_BASE + int(user_id)},
        )
