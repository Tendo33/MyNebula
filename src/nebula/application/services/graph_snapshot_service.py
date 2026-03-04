"""Build versioned graph snapshot payloads."""

from __future__ import annotations

import math
from collections.abc import Sequence
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.core.relevance import cosine_similarity
from nebula.db import Cluster, StarList, StarredRepo
from nebula.schemas.graph import (
    ClusterInfo,
    GraphData,
    GraphEdge,
    GraphNode,
    StarListInfo,
    TimelineData,
    TimelinePoint,
)

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


async def _get_default_user(db: AsyncSession):
    from nebula.api.sync import get_default_user

    return await get_default_user(db)


def _build_similarity_edges_knn(
    repo_ids: list[int],
    embeddings: list[Sequence[float]],
    min_similarity: float = 0.7,
    k: int = 8,
):
    if k <= 0 or not repo_ids or not embeddings:
        return []

    n = min(len(repo_ids), len(embeddings))
    if n <= 1:
        return []

    pair_scores: dict[tuple[int, int], float] = {}

    for i in range(n):
        source_id = repo_ids[i]
        source_embedding = embeddings[i]
        if source_embedding is None or len(source_embedding) == 0:
            continue

        candidates: list[tuple[float, int]] = []
        for j in range(n):
            if i == j:
                continue
            target_embedding = embeddings[j]
            if target_embedding is None or len(target_embedding) == 0:
                continue
            similarity = cosine_similarity(source_embedding, target_embedding)
            if similarity >= min_similarity:
                candidates.append((similarity, j))

        if not candidates:
            continue

        candidates.sort(key=lambda item: item[0], reverse=True)
        for similarity, j in candidates[:k]:
            target_id = repo_ids[j]
            edge_key = (
                (source_id, target_id)
                if source_id < target_id
                else (target_id, source_id)
            )
            existing = pair_scores.get(edge_key)
            if existing is None or similarity > existing:
                pair_scores[edge_key] = similarity

    edges = [
        GraphEdge(source=source, target=target, weight=weight)
        for (source, target), weight in pair_scores.items()
    ]
    edges.sort(key=lambda edge: edge.weight, reverse=True)
    return edges


def _estimate_adaptive_similarity_threshold(
    embeddings: list[Sequence[float]],
    *,
    target_degree: int = 8,
    min_threshold: float = 0.5,
    max_threshold: float = 0.95,
    sample_limit: int = 300,
) -> float:
    if not embeddings:
        return 0.7

    n_samples = min(len(embeddings), sample_limit)
    if n_samples < 3:
        return 0.7

    arr = np.array(embeddings[:n_samples], dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    arr = arr / norms

    sim_matrix = arr @ arr.T
    tri_upper = np.triu_indices(n_samples, k=1)
    sim_values = sim_matrix[tri_upper]
    if sim_values.size == 0:
        return 0.7

    degree = max(1, min(target_degree, n_samples - 1))
    quantile = max(0.05, min(0.98, 1.0 - (degree / float(n_samples - 1))))
    threshold = float(np.quantile(sim_values, quantile))
    return max(min_threshold, min(max_threshold, threshold))


class GraphSnapshotBuilderService:
    """Build snapshot payload from current repository graph state."""

    async def build_payload(
        self,
        db: AsyncSession,
        *,
        edge_k: int = 8,
        edge_max_nodes: int = 1000,
        adaptive_edges: bool = True,
    ) -> tuple[str, GraphData, TimelineData]:
        user = await _get_default_user(db)
        if not user:
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

        repos_result = await db.execute(
            select(StarredRepo).where(
                StarredRepo.user_id == user.id,
                StarredRepo.is_embedded == True,  # noqa: E712
                StarredRepo.coord_x.isnot(None),
            )
        )
        repos = repos_result.scalars().all()

        clusters_result = await db.execute(select(Cluster).where(Cluster.user_id == user.id))
        clusters = clusters_result.scalars().all()
        star_lists_result = await db.execute(select(StarList).where(StarList.user_id == user.id))
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
                StarredRepo.user_id == user.id,
                StarredRepo.is_embedded == True,  # noqa: E712
                StarredRepo.embedding.isnot(None),
            )
            .order_by(StarredRepo.stargazers_count.desc())
            .limit(edge_max_nodes)
        )
        edge_repos = edge_repos_result.scalars().all()
        repo_ids = [
            repo.id
            for repo in edge_repos
            if repo.embedding is not None and len(repo.embedding) > 0
        ]
        embeddings = [
            repo.embedding
            for repo in edge_repos
            if repo.embedding is not None and len(repo.embedding) > 0
        ]
        effective_similarity = (
            _estimate_adaptive_similarity_threshold(
                embeddings=embeddings,
                target_degree=edge_k,
            )
            if adaptive_edges
            else 0.7
        )
        edges = _build_similarity_edges_knn(
            repo_ids=repo_ids,
            embeddings=embeddings,
            min_similarity=effective_similarity,
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

        timeline_result = await db.execute(
            select(StarredRepo)
            .where(
                StarredRepo.user_id == user.id,
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

        generated_at = datetime.now(timezone.utc)
        version = self._build_version(graph_data, generated_at)
        graph_data.version = version
        graph_data.generated_at = generated_at.isoformat()
        timeline_data.version = version
        timeline_data.generated_at = generated_at.isoformat()
        return version, graph_data, timeline_data

    def _build_version(self, graph_data: GraphData, now: datetime) -> str:
        return (
            f"snapshot-{now.strftime('%Y%m%d%H%M%S')}-"
            f"n{graph_data.total_nodes}-e{graph_data.total_edges}"
        )
