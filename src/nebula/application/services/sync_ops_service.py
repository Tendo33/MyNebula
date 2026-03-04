"""Operational sync services for status, schedule, and full-refresh."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from fastapi import BackgroundTasks, HTTPException, status
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.application.services.graph_query_service import GraphQueryService
from nebula.application.services.sync_execution_service import (
    compute_embeddings_task,
    run_clustering_task,
    sync_stars_task,
)
from nebula.application.services.user_service import get_default_user
from nebula.core.config import get_app_settings
from nebula.db import StarredRepo, SyncSchedule, SyncTask
from nebula.schemas.v2.settings import (
    FullRefreshRequest,
    FullRefreshResponse,
    JobStatusResponse,
    ScheduleConfig,
    ScheduleResponse,
    SyncInfoResponse,
)
from nebula.utils import get_logger

logger = get_logger(__name__)


def _to_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def calculate_progress_percent(total_items: int, processed_items: int) -> float:
    """Calculate bounded progress percentage from counters."""
    if total_items <= 0:
        return 0.0
    percent = (processed_items / total_items) * 100
    return max(0.0, min(100.0, float(percent)))


def estimate_eta_seconds(
    started_at: datetime | None,
    progress_percent: float,
    now: datetime | None = None,
) -> int | None:
    """Estimate remaining seconds from elapsed time and progress."""
    if started_at is None or progress_percent <= 0 or progress_percent >= 100:
        return None

    started_at_utc = _to_utc(started_at)
    now_ts = _to_utc(now) if now is not None else datetime.now(timezone.utc)
    if started_at_utc is None:
        return None

    elapsed_seconds = (now_ts - started_at_utc).total_seconds()
    if elapsed_seconds <= 0:
        return None

    total_estimated = elapsed_seconds * (100.0 / progress_percent)
    remaining = int(max(total_estimated - elapsed_seconds, 0))
    return remaining if remaining > 0 else None


def resolve_job_phase(
    task_type: str,
    status: str,
    error_details: dict | None,
) -> str:
    """Resolve user-facing job phase from task metadata."""
    if status == "completed":
        return "completed"
    if status == "failed":
        return "failed"
    if isinstance(error_details, dict):
        phase = error_details.get("phase")
        if isinstance(phase, str) and phase:
            return phase
    return task_type


def validate_full_refresh_confirmation(confirm: bool) -> None:
    """Enforce explicit full-refresh confirmation."""
    if confirm:
        return
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Full refresh requires explicit confirmation",
    )


def is_schedule_due(
    now_local: datetime,
    schedule_hour: int,
    schedule_minute: int,
    last_run_local: datetime | None,
) -> bool:
    """Check whether a schedule is due now and has not run in the same minute."""
    if now_local.hour != schedule_hour or now_local.minute != schedule_minute:
        return False
    if last_run_local is None:
        return True
    return not (
        last_run_local.date() == now_local.date()
        and last_run_local.hour == now_local.hour
        and last_run_local.minute == now_local.minute
    )


def calculate_next_run_time(schedule: SyncSchedule) -> datetime | None:
    """Calculate the next scheduled run time for a sync schedule."""
    if not schedule.is_enabled:
        return None

    try:
        tz = ZoneInfo(schedule.timezone)
        now_utc = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
        now_local = now_utc.astimezone(tz)
        scheduled_today = now_local.replace(
            hour=schedule.schedule_hour,
            minute=schedule.schedule_minute,
            second=0,
            microsecond=0,
        )

        if now_local >= scheduled_today:
            scheduled_today += timedelta(days=1)

        return scheduled_today.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    except Exception as exc:
        logger.warning("Error calculating next run time: %s", exc)
        return None


async def get_job_status(
    task_id: int,
    db: AsyncSession,
) -> JobStatusResponse:
    """Get aggregated job status for richer progress UI."""
    user = await get_default_user(db)
    result = await db.execute(
        select(SyncTask).where(
            SyncTask.id == task_id,
            SyncTask.user_id == user.id,
        )
    )
    task = result.scalar_one_or_none()

    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )

    progress_percent = calculate_progress_percent(task.total_items, task.processed_items)
    phase = resolve_job_phase(task.task_type, task.status, task.error_details)
    eta_seconds = (
        estimate_eta_seconds(task.started_at, progress_percent)
        if task.status == "running"
        else None
    )

    return JobStatusResponse(
        task_id=task.id,
        task_type=task.task_type,
        status=task.status,
        phase=phase,
        progress_percent=progress_percent,
        eta_seconds=eta_seconds,
        last_error=task.error_message,
        retryable=task.status == "failed",
        started_at=task.started_at,
        completed_at=task.completed_at,
    )


async def get_schedule(
    db: AsyncSession,
) -> ScheduleResponse:
    """Get current sync schedule configuration."""
    user = await get_default_user(db)
    result = await db.execute(
        select(SyncSchedule).where(SyncSchedule.user_id == user.id)
    )
    schedule = result.scalar_one_or_none()

    if not schedule:
        return ScheduleResponse(
            is_enabled=False,
            schedule_hour=9,
            schedule_minute=0,
            timezone="Asia/Shanghai",
            last_run_at=None,
            last_run_status=None,
            last_run_error=None,
            next_run_at=None,
        )

    next_run = calculate_next_run_time(schedule)
    return ScheduleResponse(
        is_enabled=schedule.is_enabled,
        schedule_hour=schedule.schedule_hour,
        schedule_minute=schedule.schedule_minute,
        timezone=schedule.timezone,
        last_run_at=schedule.last_run_at,
        last_run_status=schedule.last_run_status,
        last_run_error=schedule.last_run_error,
        next_run_at=next_run,
    )


async def update_schedule(
    config: ScheduleConfig,
    db: AsyncSession,
) -> ScheduleResponse:
    """Create or update sync schedule configuration."""
    user = await get_default_user(db)
    result = await db.execute(
        select(SyncSchedule).where(SyncSchedule.user_id == user.id)
    )
    schedule = result.scalar_one_or_none()

    if not schedule:
        schedule = SyncSchedule(user_id=user.id)
        db.add(schedule)

    schedule.is_enabled = config.is_enabled
    schedule.schedule_hour = config.schedule_hour
    schedule.schedule_minute = config.schedule_minute
    schedule.timezone = config.timezone

    await db.commit()
    await db.refresh(schedule)

    next_run = calculate_next_run_time(schedule)
    logger.info(
        "Schedule updated for user %s: enabled=%s, time=%s:%02d %s",
        user.id,
        config.is_enabled,
        config.schedule_hour,
        config.schedule_minute,
        config.timezone,
    )

    return ScheduleResponse(
        is_enabled=schedule.is_enabled,
        schedule_hour=schedule.schedule_hour,
        schedule_minute=schedule.schedule_minute,
        timezone=schedule.timezone,
        last_run_at=schedule.last_run_at,
        last_run_status=schedule.last_run_status,
        last_run_error=schedule.last_run_error,
        next_run_at=next_run,
    )


async def full_refresh_task(user_id: int, task_id: int):
    """Background task to perform full refresh of all repositories."""
    from nebula.db.database import get_db_context

    async def _update_main_task(**fields) -> bool:
        async with get_db_context() as db:
            task = await db.get(SyncTask, task_id)
            if not task:
                return False
            for key, value in fields.items():
                setattr(task, key, value)
            await db.commit()
            return True

    async def _create_sub_task(task_type: str) -> int:
        async with get_db_context() as db:
            sub_task = SyncTask(user_id=user_id, task_type=task_type, status="pending")
            db.add(sub_task)
            await db.commit()
            await db.refresh(sub_task)
            return sub_task.id

    try:
        task_exists = await _update_main_task(
            status="running",
            started_at=datetime.utcnow(),
            error_message=None,
            total_items=5,
            processed_items=0,
            error_details={"phase": "reset"},
        )
        if not task_exists:
            return

        logger.info("Full refresh: Resetting all repos for user %s", user_id)
        async with get_db_context() as db:
            result = await db.execute(
                update(StarredRepo)
                .where(StarredRepo.user_id == user_id)
                .values(
                    is_embedded=False,
                    is_summarized=False,
                    ai_summary=None,
                    ai_tags=None,
                    embedding=None,
                    description_hash=None,
                    topics_hash=None,
                )
            )
            reset_count = result.rowcount
            await db.commit()
        logger.info("Full refresh: Reset %s repos", reset_count)

        await _update_main_task(
            error_details={"phase": "reset", "reset_count": reset_count},
            processed_items=1,
        )

        logger.info("Full refresh: Starting full star sync for user %s", user_id)
        stars_task_id = await _create_sub_task("stars")
        await sync_stars_task(user_id, stars_task_id, "full")
        await _update_main_task(
            error_details={"phase": "stars", "reset_count": reset_count},
            processed_items=2,
        )

        logger.info("Full refresh: Computing embeddings for user %s", user_id)
        embed_task_id = await _create_sub_task("embedding")
        await compute_embeddings_task(user_id, embed_task_id)
        await _update_main_task(
            error_details={"phase": "embeddings", "reset_count": reset_count},
            processed_items=3,
        )

        logger.info("Full refresh: Running clustering for user %s", user_id)
        await _update_main_task(
            error_details={"phase": "clustering", "reset_count": reset_count},
            processed_items=3,
        )
        cluster_task_id = await _create_sub_task("cluster")
        await run_clustering_task(user_id, cluster_task_id, use_llm=True)

        await _update_main_task(
            error_details={"phase": "snapshot", "reset_count": reset_count},
            processed_items=4,
        )
        async with get_db_context() as db:
            graph_service = GraphQueryService()
            await graph_service.rebuild_active_snapshot(db)

        await _update_main_task(
            status="completed",
            completed_at=datetime.utcnow(),
            processed_items=5,
            error_details={
                "phase": "complete",
                "reset_count": reset_count,
                "steps_completed": [
                    "reset",
                    "stars",
                    "embeddings",
                    "clustering",
                    "snapshot",
                ],
            },
        )
        logger.info("Full refresh completed for user %s", user_id)
    except Exception as exc:
        logger.exception("Full refresh failed for user %s: %s", user_id, exc)
        await _update_main_task(
            status="failed",
            error_message=str(exc),
            completed_at=datetime.utcnow(),
        )


async def trigger_full_refresh(
    payload: FullRefreshRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession,
) -> FullRefreshResponse:
    """Trigger a full refresh of all repositories."""
    validate_full_refresh_confirmation(payload.confirm)
    user = await get_default_user(db)

    result = await db.execute(
        select(SyncTask).where(
            SyncTask.user_id == user.id,
            SyncTask.task_type == "full_refresh",
            SyncTask.status.in_(["pending", "running"]),
        )
    )
    existing = result.scalar_one_or_none()
    if existing:
        return FullRefreshResponse(
            task_id=existing.id,
            message="Full refresh already in progress",
            reset_count=0,
        )

    count_result = await db.execute(
        select(func.count(StarredRepo.id)).where(StarredRepo.user_id == user.id)
    )
    repo_count = count_result.scalar() or 0

    task = SyncTask(
        user_id=user.id,
        task_type="full_refresh",
        status="pending",
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)

    background_tasks.add_task(full_refresh_task, user.id, task.id)

    logger.info(
        "Full refresh started for user %s, %s repos will be reset",
        user.id,
        repo_count,
    )
    return FullRefreshResponse(
        task_id=task.id,
        message=f"Full refresh started. {repo_count} repositories will be reprocessed.",
        reset_count=repo_count,
    )


async def get_sync_info(
    db: AsyncSession,
) -> SyncInfoResponse:
    """Get comprehensive sync status information."""
    user = await get_default_user(db)
    app_settings = get_app_settings()

    stats_result = await db.execute(
        select(
            func.count(StarredRepo.id).label("total"),
            func.count(StarredRepo.id)
            .filter(StarredRepo.is_embedded == True)  # noqa: E712
            .label("embedded"),
            func.count(StarredRepo.id)
            .filter(StarredRepo.is_summarized == True)  # noqa: E712
            .label("summarized"),
        ).where(StarredRepo.user_id == user.id)
    )
    stats = stats_result.one()

    schedule_result = await db.execute(
        select(SyncSchedule).where(SyncSchedule.user_id == user.id)
    )
    schedule = schedule_result.scalar_one_or_none()

    schedule_response = None
    if schedule:
        next_run = calculate_next_run_time(schedule)
        schedule_response = ScheduleResponse(
            is_enabled=schedule.is_enabled,
            schedule_hour=schedule.schedule_hour,
            schedule_minute=schedule.schedule_minute,
            timezone=schedule.timezone,
            last_run_at=schedule.last_run_at,
            last_run_status=schedule.last_run_status,
            last_run_error=schedule.last_run_error,
            next_run_at=next_run,
        )

    return SyncInfoResponse(
        last_sync_at=user.last_sync_at,
        github_token_configured=bool(app_settings.github_token),
        single_user_mode=app_settings.single_user_mode,
        total_repos=stats.total or 0,
        synced_repos=user.synced_stars or 0,
        embedded_repos=stats.embedded or 0,
        summarized_repos=stats.summarized or 0,
        schedule=schedule_response,
    )
