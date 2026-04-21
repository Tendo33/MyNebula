"""V2 sync routes backed by pipeline runs."""

from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.application.services.pipeline_service import SyncPipelineService
from nebula.db import User, get_db
from nebula.schemas.v2 import PipelineStartResponse, PipelineStatusResponse

from .access import resolve_single_user
from .auth import require_admin, require_admin_csrf
from .metadata import build_v2_metadata

router = APIRouter(dependencies=[Depends(require_admin), Depends(require_admin_csrf)])
pipeline_service = SyncPipelineService()


async def _ensure_no_active_pipeline(user_id: int) -> None:
    active_run = await pipeline_service.get_active_pipeline(user_id)
    if active_run is None:
        return
    raise HTTPException(
        status_code=409,
        detail=f"Pipeline already running (run_id={active_run.id})",
    )


@router.post("/start", response_model=PipelineStartResponse)
async def start_pipeline_sync(
    background_tasks: BackgroundTasks,
    mode: str = Query(default="incremental", pattern="^(incremental|full)$"),
    use_llm: bool = Query(default=True),
    max_clusters: int = Query(default=8, ge=2, le=30),
    min_clusters: int = Query(default=3, ge=2, le=20),
    user: User = Depends(resolve_single_user),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> PipelineStartResponse:
    """Start sync pipeline in background."""
    await _ensure_no_active_pipeline(user.id)
    try:
        run_id = await pipeline_service.create_pipeline_run(user.id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    background_tasks.add_task(
        pipeline_service.run_pipeline,
        run_id,
        mode=mode,
        use_llm=use_llm,
        max_clusters=max_clusters,
        min_clusters=min_clusters,
    )
    return PipelineStartResponse(
        pipeline_run_id=run_id,
        status="pending",
        phase="pending",
        message="Pipeline started in background",
        **build_v2_metadata(version=f"pipeline-{run_id}"),
    )


@router.post("/recluster", response_model=PipelineStartResponse)
async def start_recluster_sync(
    background_tasks: BackgroundTasks,
    max_clusters: int = Query(default=8, ge=2, le=30),
    min_clusters: int = Query(default=3, ge=2, le=30),
    user: User = Depends(resolve_single_user),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> PipelineStartResponse:
    """Start full recluster phase only, then rebuild graph snapshot."""
    if min_clusters > max_clusters:
        raise HTTPException(
            status_code=400,
            detail="min_clusters must be less than or equal to max_clusters",
        )

    await _ensure_no_active_pipeline(user.id)
    try:
        run_id = await pipeline_service.create_pipeline_run(user.id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    user.graph_max_clusters = max_clusters
    user.graph_min_clusters = min_clusters
    await db.commit()
    background_tasks.add_task(
        pipeline_service.run_recluster_pipeline,
        run_id,
        max_clusters=max_clusters,
        min_clusters=min_clusters,
    )
    return PipelineStartResponse(
        pipeline_run_id=run_id,
        status="pending",
        phase="pending",
        message="Recluster pipeline started in background",
        **build_v2_metadata(version=f"pipeline-{run_id}"),
    )


@router.get("/jobs/{run_id}", response_model=PipelineStatusResponse)
async def get_pipeline_status(
    run_id: int,
    user: User = Depends(resolve_single_user),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> PipelineStatusResponse:
    """Get pipeline run status."""
    run = await pipeline_service.get_pipeline(run_id)
    if run is None or run.user_id != user.id:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    return PipelineStatusResponse(
        pipeline_run_id=run.id,
        user_id=run.user_id,
        status=run.status,
        phase=run.phase,
        last_error=run.last_error,
        created_at=run.created_at,
        started_at=run.started_at,
        completed_at=run.completed_at,
        **build_v2_metadata(
            version=f"pipeline-{run.id}",
            generated_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
