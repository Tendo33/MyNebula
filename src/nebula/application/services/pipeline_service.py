"""Pipeline orchestration for sync, embedding, clustering, and snapshot build."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.db import PipelineRun, SyncTask, User
from nebula.db.database import get_db_context
from nebula.domain import PipelinePhase, PipelineStatus
from nebula.utils import get_logger

from . import sync_execution_service
from .graph_query_service import GraphQueryService

logger = get_logger(__name__)


class SyncPipelineService:
    """Run end-to-end sync pipeline with explicit run tracking."""

    def __init__(self, graph_service: GraphQueryService | None = None) -> None:
        self.graph_service = graph_service or GraphQueryService()

    async def create_pipeline_run(self, user_id: int) -> int:
        """Create a pipeline run record and return run id."""
        async with get_db_context() as db:
            await self._acquire_pipeline_creation_lock(db, user_id)
            active_run = await self._get_active_pipeline_from_db(db, user_id)
            if active_run is not None:
                raise ValueError(
                    "Pipeline already running "
                    f"(user_id={user_id}, run_id={active_run.id}, status={active_run.status})"
                )
            run = PipelineRun(
                user_id=user_id,
                status=PipelineStatus.pending.value,
                phase=PipelinePhase.pending.value,
                started_at=datetime.now(timezone.utc),
            )
            db.add(run)
            await db.commit()
            await db.refresh(run)
            return run.id

    async def run_pipeline(
        self,
        run_id: int,
        *,
        mode: str = "incremental",
        use_llm: bool = True,
        max_clusters: int = 8,
        min_clusters: int = 3,
    ) -> int:
        """Execute an existing pipeline run."""
        async with get_db_context() as db:
            run = await db.get(PipelineRun, run_id)
            if run is None:
                raise ValueError(f"Pipeline run {run_id} not found")
            user_id = run.user_id

        try:
            partial_failed = False

            await self._update_run(run_id, PipelineStatus.running, PipelinePhase.stars)
            stars_task_id = await self._create_task(
                user_id, run_id, "stars", PipelinePhase.stars
            )
            await sync_execution_service.sync_stars_task(user_id, stars_task_id, mode)
            partial_failed = partial_failed or await self._inspect_task_outcome(
                stars_task_id,
                run_id=run_id,
                phase=PipelinePhase.stars,
            )

            await self._update_run(
                run_id, PipelineStatus.running, PipelinePhase.embedding
            )
            embedding_task_id = await self._create_task(
                user_id, run_id, "embedding", PipelinePhase.embedding
            )
            await sync_execution_service.compute_embeddings_task(
                user_id, embedding_task_id
            )
            partial_failed = partial_failed or await self._inspect_task_outcome(
                embedding_task_id,
                run_id=run_id,
                phase=PipelinePhase.embedding,
            )

            force_full_recluster = await self._should_force_full_recluster(
                user_id, stars_task_id
            )
            await self._update_run(
                run_id, PipelineStatus.running, PipelinePhase.clustering
            )
            clustering_task_id = await self._create_task(
                user_id, run_id, "cluster", PipelinePhase.clustering
            )
            await sync_execution_service.run_clustering_task(
                user_id=user_id,
                task_id=clustering_task_id,
                use_llm=use_llm,
                max_clusters=max_clusters,
                min_clusters=min_clusters,
                incremental=not force_full_recluster,
            )
            partial_failed = partial_failed or await self._inspect_task_outcome(
                clustering_task_id,
                run_id=run_id,
                phase=PipelinePhase.clustering,
            )

            await self._update_run(
                run_id, PipelineStatus.running, PipelinePhase.snapshot
            )
            async with get_db_context() as db:
                await self.graph_service.rebuild_active_snapshot(db)

            final_status = (
                PipelineStatus.partial_failed
                if partial_failed
                else PipelineStatus.completed
            )
            await self._update_run(run_id, final_status, PipelinePhase.completed)
            return run_id
        except Exception as exc:
            logger.exception(f"Pipeline {run_id} failed: {exc}")
            await self._update_run(
                run_id,
                PipelineStatus.failed,
                PipelinePhase.completed,
                error=str(exc),
            )
            return run_id

    async def start_pipeline(
        self,
        user_id: int,
        *,
        mode: str = "incremental",
        use_llm: bool = True,
        max_clusters: int = 8,
        min_clusters: int = 3,
    ) -> int:
        run_id = await self.create_pipeline_run(user_id)
        return await self.run_pipeline(
            run_id,
            mode=mode,
            use_llm=use_llm,
            max_clusters=max_clusters,
            min_clusters=min_clusters,
        )

    async def run_recluster_pipeline(
        self,
        run_id: int,
        *,
        max_clusters: int = 8,
        min_clusters: int = 3,
    ) -> int:
        """Execute a pipeline run for full reclustering only."""
        async with get_db_context() as db:
            run = await db.get(PipelineRun, run_id)
            if run is None:
                raise ValueError(f"Pipeline run {run_id} not found")
            user_id = run.user_id

        try:
            await self._update_run(
                run_id, PipelineStatus.running, PipelinePhase.clustering
            )
            clustering_task_id = await self._create_task(
                user_id, run_id, "cluster", PipelinePhase.clustering
            )
            await sync_execution_service.run_clustering_task(
                user_id=user_id,
                task_id=clustering_task_id,
                use_llm=True,
                max_clusters=max_clusters,
                min_clusters=min_clusters,
                incremental=False,
            )
            partial_failed = await self._inspect_task_outcome(
                clustering_task_id,
                run_id=run_id,
                phase=PipelinePhase.clustering,
            )

            await self._update_run(
                run_id, PipelineStatus.running, PipelinePhase.snapshot
            )
            async with get_db_context() as db:
                await self.graph_service.rebuild_active_snapshot(db)

            final_status = (
                PipelineStatus.partial_failed
                if partial_failed
                else PipelineStatus.completed
            )
            await self._update_run(run_id, final_status, PipelinePhase.completed)
            return run_id
        except Exception as exc:
            logger.exception(f"Recluster pipeline {run_id} failed: {exc}")
            await self._update_run(
                run_id,
                PipelineStatus.failed,
                PipelinePhase.completed,
                error=str(exc),
            )
            return run_id

    async def _create_task(
        self,
        user_id: int,
        pipeline_run_id: int,
        task_type: str,
        phase: PipelinePhase,
    ) -> int:
        async with get_db_context() as db:
            task = SyncTask(
                user_id=user_id,
                pipeline_run_id=pipeline_run_id,
                task_type=task_type,
                status="pending",
                phase=phase.value,
            )
            db.add(task)
            await db.commit()
            await db.refresh(task)
            return task.id

    async def _update_run(
        self,
        run_id: int,
        status: PipelineStatus,
        phase: PipelinePhase,
        error: str | None = None,
    ) -> None:
        async with get_db_context() as db:
            run = await db.get(PipelineRun, run_id)
            if run is None:
                return
            run.status = status.value
            run.phase = phase.value
            if error:
                run.last_error = error
            if status == PipelineStatus.running and run.started_at is None:
                run.started_at = datetime.now(timezone.utc)
            if status in {
                PipelineStatus.completed,
                PipelineStatus.partial_failed,
                PipelineStatus.failed,
            }:
                run.completed_at = datetime.now(timezone.utc)
            await db.commit()

    async def _should_force_full_recluster(
        self, user_id: int, stars_task_id: int
    ) -> bool:
        async with get_db_context() as db:
            stars_task = await db.get(SyncTask, stars_task_id)
            user = await db.get(User, user_id)
            error_details = stars_task.error_details if stars_task else {}
            new_repos = (
                int(error_details.get("new_repos", 0))
                if isinstance(error_details, dict)
                else 0
            )
            total_repos = int(user.total_stars or 0) if user else 0
            return sync_execution_service.should_force_full_recluster(
                total_repos=total_repos,
                new_repos=new_repos,
                centroid_drift=None,
            )

    async def get_pipeline(self, run_id: int) -> PipelineRun | None:
        async with get_db_context() as db:
            return await db.get(PipelineRun, run_id)

    async def get_latest_pipeline(self, user_id: int) -> PipelineRun | None:
        async with get_db_context() as db:
            result = await db.execute(
                select(PipelineRun)
                .where(PipelineRun.user_id == user_id)
                .order_by(PipelineRun.id.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()

    async def get_active_pipeline(self, user_id: int) -> PipelineRun | None:
        async with get_db_context() as db:
            return await self._get_active_pipeline_from_db(db, user_id)

    async def _inspect_task_outcome(
        self,
        task_id: int,
        *,
        run_id: int,
        phase: PipelinePhase,
    ) -> bool:
        """Return whether task has partial failures, or raise on hard failure."""
        async with get_db_context() as db:
            task = await db.get(SyncTask, task_id)
            if task is None:
                raise ValueError(f"Pipeline task not found: {task_id}")

            if task.status == "failed":
                details = {
                    "run_id": run_id,
                    "task_id": task_id,
                    "phase": phase.value,
                    "status": task.status,
                    "last_error": task.error_message,
                }
                logger.error(f"Pipeline phase failed: {details}")
                raise ValueError(
                    f"Pipeline phase failed phase={phase.value} task_id={task_id}: {task.error_message}"
                )

            has_partial_failure = bool(task.failed_items and task.failed_items > 0)
            logger.info(
                f"Pipeline phase completed run_id={run_id} "
                f"task_id={task_id} "
                f"phase={phase.value} "
                f"status={task.status} "
                f"failed_items={task.failed_items}"
            )
            return has_partial_failure

    async def _acquire_pipeline_creation_lock(
        self,
        db: AsyncSession,
        user_id: int,
    ) -> None:
        """Acquire DB-level lock to serialize pipeline creation per user."""
        bind = db.get_bind()
        dialect_name = bind.dialect.name if bind is not None else ""
        if dialect_name != "postgresql":
            return
        await db.execute(
            text("SELECT pg_advisory_xact_lock(:lock_key)"),
            {"lock_key": self._pipeline_lock_key(user_id)},
        )

    def _pipeline_lock_key(self, user_id: int) -> int:
        """Build a stable advisory lock key for one user's pipeline creation."""
        return 870_000_000 + int(user_id)

    async def _get_active_pipeline_from_db(
        self,
        db: AsyncSession,
        user_id: int,
    ) -> PipelineRun | None:
        result = await db.execute(
            select(PipelineRun)
            .where(
                PipelineRun.user_id == user_id,
                PipelineRun.status.in_(
                    [PipelineStatus.pending.value, PipelineStatus.running.value]
                ),
            )
            .order_by(PipelineRun.id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
