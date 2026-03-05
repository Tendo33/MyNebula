"""V2 dashboard route."""

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.application.services.graph_query_service import GraphQueryService
from nebula.application.services.user_service import get_default_user
from nebula.db import Cluster, StarredRepo, get_db
from nebula.schemas.v2 import DashboardCluster, DashboardResponse, DashboardSummary

from .metadata import build_v2_metadata

router = APIRouter()
graph_service = GraphQueryService()


@router.get("", response_model=DashboardResponse)
async def get_dashboard_data(
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> DashboardResponse:
    """Get consolidated dashboard payload."""
    user = await get_default_user(db)
    graph_data = await graph_service.get_graph_data_with_options(
        db,
        version="active",
        include_edges=False,
    )

    repo_count_result = await db.execute(
        select(
            func.count(StarredRepo.id).label("total"),
            func.count(StarredRepo.id)
            .filter(StarredRepo.is_embedded == True)  # noqa: E712
            .label("embedded"),
        ).where(StarredRepo.user_id == user.id)
    )
    counts = repo_count_result.one()

    clusters_result = await db.execute(
        select(Cluster)
        .where(Cluster.user_id == user.id)
        .order_by(Cluster.repo_count.desc())
    )
    top_clusters = clusters_result.scalars().all()[:8]
    metadata = build_v2_metadata(
        version=graph_data.version,
        generated_at=graph_data.generated_at,
        request_id=graph_data.request_id,
    )

    return DashboardResponse(
        summary=DashboardSummary(
            total_repos=int(counts.total or 0),
            embedded_repos=int(counts.embedded or 0),
            total_clusters=graph_data.total_clusters,
            total_edges=graph_data.total_edges,
        ),
        top_clusters=[
            DashboardCluster(
                id=cluster.id,
                name=cluster.name,
                repo_count=cluster.repo_count,
                color=cluster.color,
            )
            for cluster in top_clusters
        ],
        **metadata,
    )
