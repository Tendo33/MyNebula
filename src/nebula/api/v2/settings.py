"""V2 settings routes."""

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.api.auth import require_admin
from nebula.api.sync import (
    get_job_status,
    get_schedule,
    get_sync_info,
    trigger_full_refresh,
    update_schedule,
)
from nebula.db import get_db
from nebula.schemas.schedule import FullRefreshRequest
from nebula.schemas.v2 import (
    FullRefreshJobResponse,
    FullRefreshStartResponse,
    GraphDefaults,
    ScheduleConfig,
    ScheduleUpdateResponse,
    SettingsResponse,
)
from .metadata import build_v2_metadata

router = APIRouter(dependencies=[Depends(require_admin)])


@router.get("", response_model=SettingsResponse)
async def get_settings(
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> SettingsResponse:
    """Get consolidated settings payload used by frontend."""
    schedule = await get_schedule(db=db)
    sync_info = await get_sync_info(db=db)
    version = (
        sync_info.last_sync_at.isoformat()
        if sync_info.last_sync_at is not None
        else "sync-not-run"
    )
    metadata = build_v2_metadata(version=version)
    return SettingsResponse(
        schedule=schedule,
        sync_info=sync_info,
        graph_defaults=GraphDefaults(),
        **metadata,
    )


@router.post("/schedule", response_model=ScheduleUpdateResponse)
async def update_settings_schedule(
    config: ScheduleConfig,
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> ScheduleUpdateResponse:
    """Update sync schedule configuration through v2 route."""
    schedule = await update_schedule(config=config, db=db)
    version = (
        schedule.last_run_at.isoformat()
        if schedule.last_run_at is not None
        else "schedule-not-run"
    )
    metadata = build_v2_metadata(version=version)
    return ScheduleUpdateResponse(
        schedule=schedule,
        **metadata,
    )


@router.post("/full-refresh", response_model=FullRefreshStartResponse)
async def trigger_settings_full_refresh(
    payload: FullRefreshRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> FullRefreshStartResponse:
    """Trigger full refresh via v2 settings route."""
    task = await trigger_full_refresh(payload=payload, background_tasks=background_tasks, db=db)
    return FullRefreshStartResponse(
        task=task,
        **build_v2_metadata(version=f"full-refresh-{task.task_id}"),
    )


@router.get("/full-refresh/jobs/{task_id}", response_model=FullRefreshJobResponse)
async def get_full_refresh_job_status(
    task_id: int,
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> FullRefreshJobResponse:
    """Get full refresh status via v2 settings route."""
    job = await get_job_status(task_id=task_id, db=db)
    return FullRefreshJobResponse(
        job=job,
        **build_v2_metadata(version=f"full-refresh-{task_id}"),
    )
