"""Scheduler service for automatic periodic sync.

This module provides scheduling functionality for automatic
synchronization of GitHub stars at user-defined intervals.

Uses APScheduler with AsyncIOScheduler for integration with FastAPI's
async event loop.
"""

import asyncio
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.db import SyncSchedule, SyncTask, User
from nebula.utils import get_logger

logger = get_logger(__name__)

# Global scheduler instance
_scheduler_service: "SchedulerService | None" = None


class SchedulerService:
    """Service for managing scheduled sync tasks.

    Uses APScheduler's AsyncIOScheduler to run periodic checks
    and trigger sync tasks when scheduled.

    Implements singleton pattern to ensure only one scheduler
    instance exists in the application.
    """

    def __init__(self):
        """Initialize the scheduler service."""
        self._scheduler: AsyncIOScheduler | None = None
        self._running = False
        self._active_user_tasks: dict[int, asyncio.Task[None]] = {}
        self._task_lock = asyncio.Lock()

    def _get_scheduler(self) -> AsyncIOScheduler:
        """Get or create the scheduler instance."""
        if self._scheduler is None:
            self._scheduler = AsyncIOScheduler(timezone="UTC")
        return self._scheduler

    async def start(self) -> None:
        """Start the scheduler service.

        Adds a master job that runs every minute to check
        for users who need their scheduled sync executed.
        """
        if self._running:
            logger.warning("Scheduler already running")
            return

        scheduler = self._get_scheduler()

        # Add master sync checker job - runs every minute
        scheduler.add_job(
            self._check_and_trigger_syncs,
            CronTrigger(minute="*"),
            id="master_sync_checker",
            replace_existing=True,
            max_instances=1,  # Prevent overlapping executions
        )

        scheduler.start()
        self._running = True
        logger.info(
            "Scheduler service started - checking for scheduled syncs every minute"
        )

    async def stop(self) -> None:
        """Stop the scheduler service."""
        if not self._running:
            return

        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None

        self._running = False
        logger.info("Scheduler service stopped")

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    async def _check_and_trigger_syncs(self) -> None:
        """Check for users who need their scheduled sync executed.

        This job runs every minute and checks if any user's
        scheduled sync time matches the current time in their timezone.
        """
        from nebula.db.database import get_db_context

        try:
            async with get_db_context() as db:
                # Get all enabled schedules
                result = await db.execute(
                    select(SyncSchedule).where(SyncSchedule.is_enabled == True)  # noqa: E712
                )
                schedules = result.scalars().all()

                if not schedules:
                    return

                now_utc = datetime.now(timezone.utc)

                for schedule in schedules:
                    try:
                        await self._check_single_schedule(schedule, now_utc, db)
                    except Exception as e:
                        logger.error(
                            f"Error checking schedule for user {schedule.user_id}: {e}"
                        )
                        # Continue with other schedules

        except Exception as e:
            logger.error(f"Error in scheduled sync check: {e}")

    async def _check_single_schedule(
        self,
        schedule: SyncSchedule,
        now_utc: datetime,
        db: AsyncSession,
    ) -> None:
        """Check and potentially trigger a single user's schedule.

        Args:
            schedule: The sync schedule to check.
            now_utc: Current UTC time.
            db: Database session.
        """
        try:
            # Convert current UTC time to user's timezone
            user_tz = ZoneInfo(schedule.timezone)
            user_time = now_utc.astimezone(user_tz)
            last_run_local = (
                schedule.last_run_at.astimezone(user_tz) if schedule.last_run_at else None
            )

            from nebula.api.sync import _is_schedule_due

            if not _is_schedule_due(
                now_local=user_time,
                schedule_hour=schedule.schedule_hour,
                schedule_minute=schedule.schedule_minute,
                last_run_local=last_run_local,
            ):
                return

            if await self._has_active_pipeline(schedule.user_id):
                logger.info(
                    f"Skipping scheduled sync for user {schedule.user_id}: pipeline already running"
                )
                return

            logger.info(
                f"Triggering scheduled sync for user {schedule.user_id} "
                f"at {user_time.strftime('%Y-%m-%d %H:%M')} {schedule.timezone}"
            )

            schedule.last_run_status = "running"
            schedule.last_run_error = None
            schedule.last_run_at = now_utc
            await db.commit()

            await self._launch_user_pipeline(schedule.user_id)

        except Exception as e:
            logger.error(f"Error processing schedule for user {schedule.user_id}: {e}")
            schedule.last_run_status = "failed"
            schedule.last_run_error = str(e)
            await db.commit()

    async def _has_active_pipeline(self, user_id: int) -> bool:
        """Check whether a scheduler-triggered pipeline is running for the user."""
        async with self._task_lock:
            existing = self._active_user_tasks.get(user_id)
            if existing and existing.done():
                self._active_user_tasks.pop(user_id, None)
                return False
            return existing is not None

    async def _launch_user_pipeline(self, user_id: int) -> None:
        """Launch a non-blocking scheduled pipeline for the user."""
        async with self._task_lock:
            existing = self._active_user_tasks.get(user_id)
            if existing and not existing.done():
                return

            task = asyncio.create_task(
                self._run_user_sync_pipeline(user_id),
                name=f"scheduled-sync-{user_id}",
            )
            self._active_user_tasks[user_id] = task

    async def _create_task_record(
        self,
        db: AsyncSession,
        user_id: int,
        task_type: str,
    ) -> int:
        """Create and persist a sync task record."""
        task = SyncTask(
            user_id=user_id,
            task_type=task_type,
            status="pending",
        )
        db.add(task)
        await db.commit()
        await db.refresh(task)
        return task.id

    async def _run_user_sync_pipeline(
        self,
        user_id: int,
    ) -> None:
        """Run scheduled sync pipeline for a user in a detached task.

        The scheduler check loop never awaits this method directly.
        """
        from nebula.api.sync import (
            _should_force_full_recluster,
            compute_embeddings_task,
            run_clustering_task,
            sync_stars_task,
        )
        from nebula.db.database import get_db_context

        try:
            async with get_db_context() as db:
                user = await db.get(User, user_id)
                if not user:
                    raise ValueError(f"User {user_id} not found")

                active_task_result = await db.execute(
                    select(SyncTask.id).where(
                        SyncTask.user_id == user_id,
                        SyncTask.status.in_(["pending", "running"]),
                        SyncTask.task_type.in_(
                            ["stars", "embedding", "cluster", "full_refresh"]
                        ),
                    )
                )
                active_task_id = active_task_result.scalar_one_or_none()
                if active_task_id is not None:
                    logger.info(
                        f"Skipping scheduled sync for user {user_id}: existing task {active_task_id} is active"
                    )
                    schedule = await db.scalar(
                        select(SyncSchedule).where(SyncSchedule.user_id == user_id)
                    )
                    if schedule:
                        schedule.last_run_status = "success"
                        schedule.last_run_error = (
                            "Skipped because another sync task is already running"
                        )
                        await db.commit()
                    return

                stars_task_id = await self._create_task_record(db, user_id, "stars")

            # Run sync (incremental mode)
            await sync_stars_task(user_id, stars_task_id, "incremental")

            async with get_db_context() as db:
                embed_task_id = await self._create_task_record(db, user_id, "embedding")
            await compute_embeddings_task(user_id, embed_task_id)

            force_full_recluster = False
            async with get_db_context() as db:
                stars_task = await db.get(SyncTask, stars_task_id)
                user_latest = await db.get(User, user_id)
                error_details = stars_task.error_details if stars_task else {}
                new_repos = (
                    int(error_details.get("new_repos", 0))
                    if isinstance(error_details, dict)
                    else 0
                )
                total_repos = int(user_latest.total_stars or 0) if user_latest else 0
                force_full_recluster = _should_force_full_recluster(
                    total_repos=total_repos,
                    new_repos=new_repos,
                    centroid_drift=None,
                )

            async with get_db_context() as db:
                cluster_task_id = await self._create_task_record(db, user_id, "cluster")

            # Prefer incremental mode, but switch to full recluster when drift guard trips.
            await run_clustering_task(
                user_id,
                cluster_task_id,
                use_llm=True,
                incremental=not force_full_recluster,
            )

            async with get_db_context() as db:
                schedule = await db.scalar(
                    select(SyncSchedule).where(SyncSchedule.user_id == user_id)
                )
                if schedule:
                    schedule.last_run_status = "success"
                    schedule.last_run_error = None
                    await db.commit()

            logger.info(f"Scheduled sync completed for user {user_id}")

        except Exception as e:
            logger.exception(f"Scheduled sync failed for user {user_id}: {e}")
            from nebula.db.database import get_db_context

            async with get_db_context() as db:
                schedule = await db.scalar(
                    select(SyncSchedule).where(SyncSchedule.user_id == user_id)
                )
                if schedule:
                    schedule.last_run_status = "failed"
                    schedule.last_run_error = str(e)
                    await db.commit()
        finally:
            async with self._task_lock:
                existing = self._active_user_tasks.get(user_id)
                current = asyncio.current_task()
                if existing is not None and existing is current:
                    self._active_user_tasks.pop(user_id, None)


def get_scheduler_service() -> SchedulerService:
    """Get the global scheduler service instance.

    Returns:
        The singleton SchedulerService instance.
    """
    global _scheduler_service
    if _scheduler_service is None:
        _scheduler_service = SchedulerService()
    return _scheduler_service


async def close_scheduler_service() -> None:
    """Stop and cleanup the scheduler service."""
    global _scheduler_service
    if _scheduler_service is not None:
        await _scheduler_service.stop()
        _scheduler_service = None
