"""V2 dashboard route."""

from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.application.services.graph_query_service import GraphQueryService
from nebula.db import Cluster, StarredRepo, User, get_db
from nebula.schemas.v2 import (
    DashboardCluster,
    DashboardLanguageStat,
    DashboardResponse,
    DashboardSummary,
    DashboardTopicStat,
)

from .access import resolve_read_user
from .metadata import build_v2_metadata

router = APIRouter()
graph_service = GraphQueryService()


@router.get("", response_model=DashboardResponse)
async def get_dashboard_data(
    user: User = Depends(resolve_read_user),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> DashboardResponse:
    """Get consolidated dashboard payload."""
    graph_meta = await graph_service.get_snapshot_metadata(
        db,
        user=user,
        version="active",
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

    language_stats_result = await db.execute(
        select(StarredRepo.language, func.count(StarredRepo.id).label("count"))
        .where(
            StarredRepo.user_id == user.id,
            StarredRepo.language.is_not(None),
        )
        .group_by(StarredRepo.language)
        .order_by(func.count(StarredRepo.id).desc(), StarredRepo.language.asc())
        .limit(8)
    )
    topic_stats_result = await db.execute(
        text(
            """
            SELECT lower(trim(topic)) AS topic, count(*) AS count
            FROM starred_repos
            CROSS JOIN LATERAL unnest(topics) AS topic
            WHERE user_id = :user_id
              AND topic IS NOT NULL
              AND trim(topic) <> ''
            GROUP BY lower(trim(topic))
            ORDER BY count(*) DESC, lower(trim(topic)) ASC
            LIMIT 12
            """
        ),
        {"user_id": user.id},
    )
    topic_rows = topic_stats_result.all()
    total_topics_result = await db.execute(
        text(
            """
            SELECT count(*) AS total_topics
            FROM (
                SELECT DISTINCT lower(trim(topic)) AS topic
                FROM starred_repos
                CROSS JOIN LATERAL unnest(topics) AS topic
                WHERE user_id = :user_id
                  AND topic IS NOT NULL
                  AND trim(topic) <> ''
            ) AS distinct_topics
            """
        ),
        {"user_id": user.id},
    )
    total_topics = int(total_topics_result.scalar() or 0)

    clusters_result = await db.execute(
        select(Cluster)
        .where(Cluster.user_id == user.id)
        .order_by(Cluster.repo_count.desc())
        .limit(8)
    )
    top_clusters = clusters_result.scalars().all()
    metadata = build_v2_metadata(
        version=graph_meta["version"],
        generated_at=graph_meta["generated_at"],
        request_id=graph_meta["request_id"],
    )

    return DashboardResponse(
        summary=DashboardSummary(
            total_repos=int(counts.total or 0),
            embedded_repos=int(counts.embedded or 0),
            total_topics=total_topics,
            total_clusters=int(graph_meta["total_clusters"] or 0),
            total_edges=int(graph_meta["total_edges"] or 0),
        ),
        top_languages=[
            DashboardLanguageStat(language=row.language, count=row.count)
            for row in language_stats_result
        ],
        top_topics=[
            DashboardTopicStat(topic=row.topic, count=row.count) for row in topic_rows
        ],
        top_clusters=[
            DashboardCluster(
                id=cluster.id,
                name=cluster.name,
                repo_count=cluster.repo_count,
                color=cluster.color,
                keywords=cluster.keywords or [],
            )
            for cluster in top_clusters
        ],
        **metadata,
    )
