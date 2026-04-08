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
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.application.services.pipeline_service import SyncPipelineService
from nebula.db import SyncSchedule
from nebula.utils import get_logger

logger = get_logger(__name__)
SCHEDULER_ADVISORY_LOCK_KEY = 91234567

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

        user_ids_to_launch: list[int] = []
        try:
            async with get_db_context() as db:
                if not await self._try_acquire_scheduler_lock(db):
                    logger.debug(
                        "Skipping scheduler tick: lock held by another instance"
                    )
                    return

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
                        if await self._check_single_schedule(schedule, now_utc):
                            user_ids_to_launch.append(schedule.user_id)
                    except Exception as e:
                        logger.error(
                            f"Error checking schedule for user {schedule.user_id}: {e}"
                        )
                        schedule.last_run_status = "failed"
                        schedule.last_run_error = str(e)

        except Exception as e:
            logger.error(f"Error in scheduled sync check: {e}")
            return

        for user_id in user_ids_to_launch:
            try:
                await self._launch_user_pipeline(user_id)
            except Exception as e:
                logger.error(
                    f"Failed to launch scheduled sync pipeline for user {user_id}: {e}"
                )
                await self._mark_schedule_failed(user_id, str(e))

    async def _check_single_schedule(
        self,
        schedule: SyncSchedule,
        now_utc: datetime,
    ) -> bool:
        """Check and potentially trigger a single user's schedule.

        Args:
            schedule: The sync schedule to check.
            now_utc: Current UTC time.
        Returns:
            Whether this schedule should launch a pipeline.
        """
        # Convert current UTC time to user's timezone
        user_tz = ZoneInfo(schedule.timezone)
        user_time = now_utc.astimezone(user_tz)
        last_run_local = (
            schedule.last_run_at.astimezone(user_tz) if schedule.last_run_at else None
        )

        from nebula.application.services.sync_ops_service import is_schedule_due

        if not is_schedule_due(
            now_local=user_time,
            schedule_hour=schedule.schedule_hour,
            schedule_minute=schedule.schedule_minute,
            last_run_local=last_run_local,
        ):
            return False

        if await self._has_active_pipeline(schedule.user_id):
            logger.info(
                f"Skipping scheduled sync for user {schedule.user_id}: pipeline already running"
            )
            return False

        logger.info(
            f"Triggering scheduled sync for user {schedule.user_id} "
            f"at {user_time.strftime('%Y-%m-%d %H:%M')} {schedule.timezone}"
        )

        schedule.last_run_status = "running"
        schedule.last_run_error = None
        schedule.last_run_at = now_utc
        return True

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

    async def _run_user_sync_pipeline(
        self,
        user_id: int,
    ) -> None:
        """Run scheduled sync pipeline for a user in a detached task.

        The scheduler check loop never awaits this method directly.
        """
        from nebula.db.database import get_db_context

        try:
            pipeline_service = SyncPipelineService()
            active_run = await pipeline_service.get_active_pipeline(user_id)
            if active_run is not None:
                error = (
                    "Pipeline already running "
                    f"(user_id={user_id}, run_id={active_run.id}, "
                    f"status={active_run.status})"
                )
                logger.info(f"Skipping scheduled sync: {error}")
                await self._mark_schedule_failed(user_id, error)
                return
            run_id = await pipeline_service.create_pipeline_run(user_id)
            await pipeline_service.run_pipeline(
                run_id,
                mode="incremental",
                use_llm=True,
            )
            run = await pipeline_service.get_pipeline(run_id)
            run_failed = run is not None and run.status in {"failed", "partial_failed"}

            async with get_db_context() as db:
                schedule = await db.scalar(
                    select(SyncSchedule).where(SyncSchedule.user_id == user_id)
                )
                if schedule:
                    schedule.last_run_status = "failed" if run_failed else "success"
                    schedule.last_run_error = (
                        run.last_error if run_failed and run else None
                    )
                    await db.commit()

            logger.info(
                f"Scheduled pipeline sync completed for user {user_id} (run_id={run_id})"
            )

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

    async def _try_acquire_scheduler_lock(self, db: AsyncSession) -> bool:
        """Acquire cross-process transaction-level advisory lock for scheduler tick."""
        result = await db.execute(
            text("SELECT pg_try_advisory_xact_lock(:lock_key)"),
            {"lock_key": SCHEDULER_ADVISORY_LOCK_KEY},
        )
        return bool(result.scalar())

    async def _mark_schedule_failed(self, user_id: int, error: str) -> None:
        """Persist launch failure status for one user's schedule."""
        from nebula.db.database import get_db_context

        async with get_db_context() as db:
            schedule = await db.scalar(
                select(SyncSchedule).where(SyncSchedule.user_id == user_id)
            )
            if schedule:
                schedule.last_run_status = "failed"
                schedule.last_run_error = error
                await db.commit()


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
