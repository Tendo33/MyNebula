"""Scheduler service for automatic periodic sync.

This module provides scheduling functionality for automatic
synchronization of GitHub stars at user-defined intervals.

Uses APScheduler with AsyncIOScheduler for integration with FastAPI's
async event loop.
"""

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

            # Check if current time matches scheduled time
            if (
                user_time.hour == schedule.schedule_hour
                and user_time.minute == schedule.schedule_minute
            ):
                # Avoid running twice in the same minute
                if schedule.last_run_at:
                    last_run_local = schedule.last_run_at.astimezone(user_tz)
                    if (
                        last_run_local.date() == user_time.date()
                        and last_run_local.hour == user_time.hour
                        and last_run_local.minute == user_time.minute
                    ):
                        return  # Already ran this minute

                logger.info(
                    f"Triggering scheduled sync for user {schedule.user_id} "
                    f"at {user_time.strftime('%Y-%m-%d %H:%M')} {schedule.timezone}"
                )
                await self._trigger_user_sync(schedule, db)

        except Exception as e:
            logger.error(f"Error processing schedule for user {schedule.user_id}: {e}")
            schedule.last_run_status = "failed"
            schedule.last_run_error = str(e)
            await db.commit()

    async def _trigger_user_sync(
        self,
        schedule: SyncSchedule,
        db: AsyncSession,
    ) -> None:
        """Trigger sync tasks for a user.

        Creates a background task for incremental star sync,
        which will automatically trigger embedding and clustering
        updates as needed.

        Args:
            schedule: The sync schedule triggering this sync.
            db: Database session.
        """
        from nebula.api.sync import (
            compute_embeddings_task,
            run_clustering_task,
            sync_stars_task,
        )

        try:
            # Update schedule status
            schedule.last_run_status = "running"
            schedule.last_run_at = datetime.now(timezone.utc)
            await db.commit()

            # Get user
            user = await db.get(User, schedule.user_id)
            if not user:
                raise ValueError(f"User {schedule.user_id} not found")

            # Create sync task for stars
            stars_task = SyncTask(
                user_id=schedule.user_id,
                task_type="stars",
                status="pending",
            )
            db.add(stars_task)
            await db.commit()
            await db.refresh(stars_task)

            # Run sync (incremental mode)
            await sync_stars_task(schedule.user_id, stars_task.id, "incremental")

            # Create and run embedding task
            embed_task = SyncTask(
                user_id=schedule.user_id,
                task_type="embedding",
                status="pending",
            )
            db.add(embed_task)
            await db.commit()
            await db.refresh(embed_task)

            await compute_embeddings_task(schedule.user_id, embed_task.id)

            # Create and run clustering task
            cluster_task = SyncTask(
                user_id=schedule.user_id,
                task_type="cluster",
                status="pending",
            )
            db.add(cluster_task)
            await db.commit()
            await db.refresh(cluster_task)

            await run_clustering_task(schedule.user_id, cluster_task.id, use_llm=True)

            # Update schedule on success
            schedule.last_run_status = "success"
            schedule.last_run_error = None
            await db.commit()

            logger.info(f"Scheduled sync completed for user {schedule.user_id}")

        except Exception as e:
            logger.exception(f"Scheduled sync failed for user {schedule.user_id}: {e}")
            schedule.last_run_status = "failed"
            schedule.last_run_error = str(e)
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
