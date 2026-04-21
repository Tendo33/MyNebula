"""V2 graph routes backed by versioned snapshots."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.application.services.graph_query_service import (
    GraphQueryService,
    SnapshotVersionNotFoundError,
)
from nebula.core.config import get_app_settings
from nebula.db import User, get_db
from nebula.schemas.graph import GraphData, TimelineData
from nebula.schemas.v2 import GraphEdgesPage

from .access import resolve_read_user
from .auth import require_admin, require_admin_csrf

router = APIRouter()
graph_service = GraphQueryService()
settings = get_app_settings()


@router.get("", response_model=GraphData)
async def get_graph(
    request: Request,
    response: Response,
    version: str = Query(default="active", description="Snapshot version or 'active'"),
    include_edges: bool = Query(
        default=False,
        description="Whether to include all edges in graph payload",
    ),
    user: User = Depends(resolve_read_user),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> GraphData:
    """Get graph payload by snapshot version."""
    try:
        resolved_version = await graph_service.resolve_snapshot_version(
            db,
            user=user,
            version=version,
        )
    except SnapshotVersionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    etag = f'W/"graph:{resolved_version}:edges:{int(include_edges)}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    try:
        payload = await asyncio.wait_for(
            graph_service.get_graph_data_with_options(
                db,
                user=user,
                version=version,
                include_edges=include_edges,
            ),
            timeout=settings.api_query_timeout_seconds,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Graph query timed out") from exc
    response.headers["ETag"] = etag
    return payload


@router.get("/edges", response_model=GraphEdgesPage)
async def get_graph_edges(
    request: Request,
    response: Response,
    version: str = Query(default="active", description="Snapshot version or 'active'"),
    cursor: int = Query(default=0, ge=0, description="Edge pagination cursor"),
    limit: int = Query(default=1000, ge=1, le=5000, description="Edge page size"),
    user: User = Depends(resolve_read_user),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> GraphEdgesPage:
    """Get paged graph edges from snapshot storage."""
    try:
        resolved_version = await graph_service.resolve_snapshot_version(
            db,
            user=user,
            version=version,
        )
    except SnapshotVersionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    etag = f'W/"graph-edges:{resolved_version}:cursor:{cursor}:limit:{limit}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})

    try:
        page = await asyncio.wait_for(
            graph_service.get_edges_page(
                db,
                user=user,
                version=version,
                cursor=cursor,
                limit=limit,
            ),
            timeout=settings.api_query_timeout_seconds,
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504, detail="Graph edges query timed out"
        ) from exc
    response.headers["ETag"] = etag
    return page


@router.get("/timeline", response_model=TimelineData)
async def get_graph_timeline(
    request: Request,
    response: Response,
    version: str = Query(default="active", description="Snapshot version or 'active'"),
    user: User = Depends(resolve_read_user),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> TimelineData:
    """Get timeline data by snapshot version."""
    try:
        resolved_version = await graph_service.resolve_snapshot_version(
            db,
            user=user,
            version=version,
        )
    except SnapshotVersionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    etag = f'W/"graph-timeline:{resolved_version}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    try:
        payload = await asyncio.wait_for(
            graph_service.get_timeline_data(db, user=user, version=version),
            timeout=settings.api_query_timeout_seconds,
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504, detail="Graph timeline query timed out"
        ) from exc
    response.headers["ETag"] = etag
    return payload


@router.post("/rebuild", response_model=GraphData)
async def rebuild_graph_snapshot(
    _: str = Depends(require_admin),  # noqa: B008
    __: None = Depends(require_admin_csrf),  # noqa: B008
    user: User = Depends(resolve_read_user),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> GraphData:
    """Rebuild and activate a new snapshot version."""
    return await graph_service.rebuild_active_snapshot(db, user=user)
