"""Build versioned graph snapshot payloads."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from time import perf_counter

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.db import Cluster, StarList, StarredRepo
from nebula.schemas.graph import (
    ClusterInfo,
    GraphData,
    GraphNode,
    StarListInfo,
    TimelineData,
    TimelinePoint,
)
from nebula.utils import get_logger

from .graph_edge_service import (
    RepoEdgeInfo,
    _build_similarity_edges_knn,
    _estimate_adaptive_relevance_threshold,
)
from .user_service import get_default_user

UNCATEGORIZED_STAR_LIST_ID = -1
UNCATEGORIZED_STAR_LIST_NAME = "Uncategorized"

CLUSTER_COLORS = [
    "#FF6B6B",
    "#4ECDC4",
    "#45B7D1",
    "#96CEB4",
    "#FFEAA7",
    "#DDA0DD",
    "#98D8C8",
    "#F7DC6F",
    "#BB8FCE",
    "#85C1E9",
    "#F8B500",
    "#00CED1",
]
logger = get_logger(__name__)


async def _get_default_user(db: AsyncSession):
    return await get_default_user(db)


class GraphSnapshotBuilderService:
    """Build snapshot payload from current repository graph state."""

    async def build_payload(
        self,
        db: AsyncSession,
        *,
        user=None,
        edge_k: int = 8,
        edge_max_nodes: int = 1000,
        adaptive_edges: bool = True,
    ) -> tuple[str, GraphData, TimelineData]:
        build_started = perf_counter()
        resolved_user = user or await _get_default_user(db)
        if not resolved_user:
            empty_graph = GraphData(
                nodes=[],
                edges=[],
                clusters=[],
                star_lists=[],
                total_nodes=0,
                total_edges=0,
                total_clusters=0,
                total_star_lists=0,
            )
            empty_timeline = TimelineData(points=[], total_stars=0, date_range=("", ""))
            generated_at = datetime.now(timezone.utc)
            version = self._build_version(empty_graph, generated_at)
            empty_graph.version = version
            empty_graph.generated_at = generated_at.isoformat()
            empty_timeline.version = version
            empty_timeline.generated_at = generated_at.isoformat()
            return version, empty_graph, empty_timeline

        nodes_started = perf_counter()
        repos_result = await db.execute(
            select(StarredRepo).where(
                StarredRepo.user_id == resolved_user.id,
                StarredRepo.is_embedded == True,  # noqa: E712
                StarredRepo.coord_x.isnot(None),
            )
        )
        repos = repos_result.scalars().all()

        clusters_result = await db.execute(
            select(Cluster).where(Cluster.user_id == resolved_user.id)
        )
        clusters = clusters_result.scalars().all()
        star_lists_result = await db.execute(
            select(StarList).where(StarList.user_id == resolved_user.id)
        )
        star_lists = star_lists_result.scalars().all()

        cluster_colors = {
            cluster.id: cluster.color or CLUSTER_COLORS[index % len(CLUSTER_COLORS)]
            for index, cluster in enumerate(clusters)
        }
        star_list_names = {star_list.id: star_list.name for star_list in star_lists}

        uncategorized_repo_count = 0
        nodes: list[GraphNode] = []
        for repo in repos:
            node_size = 1.0 + math.log10(max(repo.stargazers_count, 1)) * 0.5
            star_list_id = (
                repo.star_list_id
                if repo.star_list_id is not None
                else UNCATEGORIZED_STAR_LIST_ID
            )
            star_list_name = (
                star_list_names.get(repo.star_list_id)
                if repo.star_list_id is not None
                else UNCATEGORIZED_STAR_LIST_NAME
            )
            if repo.star_list_id is None:
                uncategorized_repo_count += 1

            nodes.append(
                GraphNode(
                    id=repo.id,
                    github_id=repo.github_repo_id,
                    full_name=repo.full_name,
                    name=repo.name,
                    description=repo.description,
                    language=repo.language,
                    html_url=repo.html_url,
                    owner=repo.owner,
                    owner_avatar_url=repo.owner_avatar_url,
                    x=repo.coord_x or 0.0,
                    y=repo.coord_y or 0.0,
                    z=repo.coord_z or 0.0,
                    cluster_id=repo.cluster_id,
                    color=cluster_colors.get(repo.cluster_id)
                    if repo.cluster_id
                    else "#808080",
                    size=node_size,
                    star_list_id=star_list_id,
                    star_list_name=star_list_name,
                    stargazers_count=repo.stargazers_count,
                    ai_summary=repo.ai_summary,
                    ai_tags=repo.ai_tags,
                    topics=repo.topics,
                    starred_at=repo.starred_at.isoformat() if repo.starred_at else None,
                    last_commit_time=repo.repo_pushed_at.isoformat()
                    if repo.repo_pushed_at
                    else None,
                )
            )

        cluster_infos = [
            ClusterInfo(
                id=cluster.id,
                name=cluster.name,
                description=cluster.description,
                keywords=cluster.keywords or [],
                color=cluster_colors.get(cluster.id, "#808080"),
                repo_count=cluster.repo_count,
                center_x=cluster.center_x,
                center_y=cluster.center_y,
                center_z=cluster.center_z,
            )
            for cluster in clusters
        ]
        star_list_infos = [
            StarListInfo(
                id=star_list.id,
                name=star_list.name,
                description=star_list.description,
                repo_count=star_list.repo_count,
            )
            for star_list in star_lists
        ]
        if uncategorized_repo_count > 0:
            star_list_infos.append(
                StarListInfo(
                    id=UNCATEGORIZED_STAR_LIST_ID,
                    name=UNCATEGORIZED_STAR_LIST_NAME,
                    description=None,
                    repo_count=uncategorized_repo_count,
                )
            )

        edge_repos_result = await db.execute(
            select(StarredRepo)
            .where(
                StarredRepo.user_id == resolved_user.id,
                StarredRepo.is_embedded == True,  # noqa: E712
                StarredRepo.embedding.isnot(None),
            )
            .order_by(StarredRepo.stargazers_count.desc())
            .limit(edge_max_nodes)
        )
        edge_repos = edge_repos_result.scalars().all()
        repo_edge_infos = [
            RepoEdgeInfo(
                repo_id=repo.id,
                embedding=repo.embedding,
                ai_tags=repo.ai_tags,
                topics=repo.topics,
                star_list_id=repo.star_list_id,
                language=repo.language,
            )
            for repo in edge_repos
            if repo.embedding is not None and len(repo.embedding) > 0
        ]
        effective_threshold = (
            _estimate_adaptive_relevance_threshold(
                repo_edge_infos,
                target_degree=edge_k,
            )
            if adaptive_edges
            else 0.5
        )
        edges_started = perf_counter()
        edges = _build_similarity_edges_knn(
            repo_edge_infos,
            min_score=effective_threshold,
            k=edge_k,
        )

        graph_data = GraphData(
            nodes=nodes,
            edges=edges,
            clusters=cluster_infos,
            star_lists=star_list_infos,
            total_nodes=len(nodes),
            total_edges=len(edges),
            total_clusters=len(cluster_infos),
            total_star_lists=len(star_list_infos),
        )
        nodes_elapsed_ms = (perf_counter() - nodes_started) * 1000
        edges_elapsed_ms = (perf_counter() - edges_started) * 1000

        timeline_started = perf_counter()
        timeline_result = await db.execute(
            select(StarredRepo)
            .where(
                StarredRepo.user_id == resolved_user.id,
                StarredRepo.starred_at.isnot(None),
            )
            .order_by(StarredRepo.starred_at)
        )
        timeline_repos = timeline_result.scalars().all()
        if not timeline_repos:
            timeline_data = TimelineData(points=[], total_stars=0, date_range=("", ""))
        else:
            monthly_data: dict[str, dict] = defaultdict(
                lambda: {
                    "count": 0,
                    "repos": [],
                    "languages": defaultdict(int),
                    "topics": defaultdict(int),
                }
            )
            for repo in timeline_repos:
                if repo.starred_at is None:
                    continue
                month_key = repo.starred_at.strftime("%Y-%m")
                monthly_data[month_key]["count"] += 1
                monthly_data[month_key]["repos"].append(repo.full_name)
                if repo.language:
                    monthly_data[month_key]["languages"][repo.language] += 1
                if repo.topics:
                    for topic in repo.topics:
                        monthly_data[month_key]["topics"][topic] += 1

            points: list[TimelinePoint] = []
            for month in sorted(monthly_data.keys()):
                data = monthly_data[month]
                top_languages = sorted(
                    data["languages"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:3]
                top_topics = sorted(
                    data["topics"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:5]
                points.append(
                    TimelinePoint(
                        date=month,
                        count=data["count"],
                        repos=data["repos"][:20],
                        top_languages=[lang for lang, _ in top_languages],
                        top_topics=[topic for topic, _ in top_topics],
                    )
                )
            timeline_data = TimelineData(
                points=points,
                total_stars=len(timeline_repos),
                date_range=(points[0].date, points[-1].date) if points else ("", ""),
            )
        timeline_elapsed_ms = (perf_counter() - timeline_started) * 1000

        generated_at = datetime.now(timezone.utc)
        version = self._build_version(graph_data, generated_at)
        graph_data.version = version
        graph_data.generated_at = generated_at.isoformat()
        timeline_data.version = version
        timeline_data.generated_at = generated_at.isoformat()
        logger.info(
            "Graph snapshot payload built "
            f"version={version} "
            f"nodes={graph_data.total_nodes} "
            f"edges={graph_data.total_edges} "
            f"timeline_points={len(timeline_data.points)} "
            f"timings_ms={{'nodes': {nodes_elapsed_ms:.1f}, "
            f"'edges': {edges_elapsed_ms:.1f}, "
            f"'timeline': {timeline_elapsed_ms:.1f}, "
            f"'total': {((perf_counter() - build_started) * 1000):.1f}}}"
        )
        return version, graph_data, timeline_data

    def _build_version(self, graph_data: GraphData, now: datetime) -> str:
        return (
            f"snapshot-{now.strftime('%Y%m%d%H%M%S')}-"
            f"n{graph_data.total_nodes}-e{graph_data.total_edges}"
        )
