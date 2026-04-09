"""V2 data/repository routes."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Text, and_, asc, cast, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.application.services.graph_query_service import GraphQueryService
from nebula.db import Cluster, StarredRepo, User, get_db
from nebula.schemas.v2 import DataClusterInfo, DataRepoItem, DataReposResponse

from .access import resolve_read_user
from .metadata import build_v2_metadata

router = APIRouter()
graph_service = GraphQueryService()


def _normalized_query(query: str | None) -> str:
    return (query or "").strip().lower()


def _parse_stars_threshold(query: str | None) -> int | None:
    normalized = _normalized_query(query)
    if not normalized.startswith("stars:>"):
        return None
    raw_value = normalized.split("stars:>", 1)[1].strip()
    return int(raw_value) if raw_value.isdigit() else None


@router.get("/repos", response_model=DataReposResponse)
async def get_data_repos(
    cluster_id: int | None = Query(default=None),
    cluster_ids: str | None = Query(default=None, pattern=r"^\d+(,\d+)*$"),
    language: str | None = Query(default=None),
    min_stars: int = Query(default=0, ge=0),
    q: str | None = Query(default=None),
    month: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}$"),
    topic: str | None = Query(default=None),
    sort_field: str = Query(
        default="starred_at",
        pattern="^(name|language|stargazers_count|starred_at|cluster|summary|last_commit_time)$",
    ),
    sort_direction: str = Query(default="desc", pattern="^(asc|desc)$"),
    limit: int = Query(default=200, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(resolve_read_user),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> DataReposResponse:
    """List repositories for Data page use-cases."""
    conditions = [StarredRepo.user_id == user.id]

    if cluster_ids:
        parsed_cluster_ids = [int(value) for value in cluster_ids.split(",") if value]
        if parsed_cluster_ids:
            conditions.append(StarredRepo.cluster_id.in_(parsed_cluster_ids))
    elif cluster_id is not None:
        conditions.append(StarredRepo.cluster_id == cluster_id)
    if language:
        conditions.append(StarredRepo.language == language)
    if min_stars > 0:
        conditions.append(StarredRepo.stargazers_count >= min_stars)
    if q:
        stars_threshold = _parse_stars_threshold(q)
        if stars_threshold is not None:
            conditions.append(StarredRepo.stargazers_count > stars_threshold)
        else:
            like_q = f"%{q.strip()}%"
            conditions.append(
                or_(
                    StarredRepo.name.ilike(like_q),
                    StarredRepo.full_name.ilike(like_q),
                    StarredRepo.description.ilike(like_q),
                    StarredRepo.ai_summary.ilike(like_q),
                    StarredRepo.language.ilike(like_q),
                    cast(StarredRepo.ai_tags, Text).ilike(like_q),
                    cast(StarredRepo.topics, Text).ilike(like_q),
                )
            )
    if month:
        try:
            month_dt = datetime.strptime(month, "%Y-%m")
            next_month_year = month_dt.year + (1 if month_dt.month == 12 else 0)
            next_month = 1 if month_dt.month == 12 else month_dt.month + 1
            next_month_dt = month_dt.replace(year=next_month_year, month=next_month)
            conditions.append(
                StarredRepo.starred_at >= month_dt.replace(tzinfo=timezone.utc)
            )
            conditions.append(
                StarredRepo.starred_at < next_month_dt.replace(tzinfo=timezone.utc)
            )
        except ValueError:
            pass
    if topic:
        conditions.append(StarredRepo.topics.contains([topic]))

    sort_column_map = {
        "name": StarredRepo.name,
        "language": StarredRepo.language,
        "stargazers_count": StarredRepo.stargazers_count,
        "starred_at": StarredRepo.starred_at,
        "cluster": StarredRepo.cluster_id,
        "summary": StarredRepo.ai_summary,
        "last_commit_time": StarredRepo.repo_pushed_at,
    }
    sort_column = sort_column_map.get(sort_field, StarredRepo.starred_at)
    order_by = asc(sort_column) if sort_direction == "asc" else desc(sort_column)

    total_result = await db.execute(
        select(func.count(StarredRepo.id)).where(and_(*conditions))
    )
    total_count = int(total_result.scalar() or 0)
    total_repos_result = await db.execute(
        select(func.count(StarredRepo.id)).where(StarredRepo.user_id == user.id)
    )
    total_repos = int(total_repos_result.scalar() or 0)
    result = await db.execute(
        select(StarredRepo)
        .where(and_(*conditions))
        .order_by(order_by)
        .offset(offset)
        .limit(limit)
    )
    repos = result.scalars().all()
    clusters_result = await db.execute(
        select(Cluster)
        .where(Cluster.user_id == user.id)
        .order_by(Cluster.repo_count.desc())
    )
    clusters = clusters_result.scalars().all()
    graph_meta = await graph_service.get_snapshot_metadata(
        db,
        user=user,
        version="active",
    )

    return DataReposResponse(
        items=[
            DataRepoItem(
                id=repo.id,
                full_name=repo.full_name,
                name=repo.name,
                owner=repo.owner,
                owner_avatar_url=repo.owner_avatar_url,
                description=repo.description,
                ai_summary=repo.ai_summary,
                topics=repo.topics or [],
                language=repo.language,
                stargazers_count=repo.stargazers_count,
                html_url=repo.html_url,
                cluster_id=repo.cluster_id,
                star_list_id=repo.star_list_id,
                starred_at=repo.starred_at.isoformat() if repo.starred_at else None,
                last_commit_time=repo.repo_pushed_at.isoformat()
                if repo.repo_pushed_at
                else None,
            )
            for repo in repos
        ],
        clusters=[
            DataClusterInfo(
                id=cluster.id,
                name=cluster.name,
                color=cluster.color,
                repo_count=cluster.repo_count,
                keywords=cluster.keywords or [],
            )
            for cluster in clusters
        ],
        count=total_count,
        total_repos=total_repos,
        limit=limit,
        offset=offset,
        **build_v2_metadata(
            version=graph_meta["version"],
            generated_at=graph_meta["generated_at"],
            request_id=graph_meta["request_id"],
        ),
    )
