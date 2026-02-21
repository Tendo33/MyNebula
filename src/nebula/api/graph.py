"""Graph visualization API routes.

Provides data for force graph visualization and timeline analytics.
"""

import math
from collections.abc import Sequence
from typing import Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.db import Cluster, StarList, StarredRepo, User, get_db
from nebula.schemas.graph import (
    ClusterInfo,
    GraphData,
    GraphEdge,
    GraphNode,
    StarListInfo,
    TimelineData,
    TimelinePoint,
)
from nebula.utils import get_logger

logger = get_logger(__name__)
router = APIRouter()

UNCATEGORIZED_STAR_LIST_ID = -1
UNCATEGORIZED_STAR_LIST_NAME = "Uncategorized"

# Color palette for clusters
CLUSTER_COLORS = [
    "#FF6B6B",  # Red
    "#4ECDC4",  # Teal
    "#45B7D1",  # Blue
    "#96CEB4",  # Green
    "#FFEAA7",  # Yellow
    "#DDA0DD",  # Plum
    "#98D8C8",  # Mint
    "#F7DC6F",  # Gold
    "#BB8FCE",  # Purple
    "#85C1E9",  # Light Blue
    "#F8B500",  # Orange
    "#00CED1",  # Dark Cyan
]


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if vec_a is None or vec_b is None:
        return 0.0
    if len(vec_a) != len(vec_b) or len(vec_a) == 0:
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(len(vec_a)):
        a = float(vec_a[i])
        b = float(vec_b[i])
        dot += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a <= 0 or norm_b <= 0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _build_similarity_edges_knn(
    repo_ids: list[int],
    embeddings: list[Sequence[float]],
    min_similarity: float = 0.7,
    k: int = 8,
) -> list[GraphEdge]:
    """Build undirected similarity edges using per-node k-nearest neighbors."""
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
            similarity = _cosine_similarity(source_embedding, target_embedding)
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


async def get_default_user(db: AsyncSession) -> User | None:
    """Get the first user from database.

    Since authentication is disabled, we use the first available user.
    Returns None if no user exists.
    """
    result = await db.execute(select(User).limit(1))
    return result.scalar_one_or_none()


@router.get("", response_model=GraphData)
async def get_graph_data(
    include_edges: bool = Query(default=False, description="Include similarity edges"),
    min_similarity: float = Query(
        default=0.7, ge=0.5, le=0.95, description="Minimum similarity for edges"
    ),
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Get graph data for visualization.

    Args:
        include_edges: Whether to compute similarity edges
        min_similarity: Minimum similarity threshold for edges
        db: Database session

    Returns:
        Complete graph data with nodes, edges, and clusters
    """
    # Get default user
    user = await get_default_user(db)
    if not user:
        return GraphData(
            nodes=[],
            edges=[],
            clusters=[],
            total_nodes=0,
            total_edges=0,
            total_clusters=0,
        )

    # Get all embedded repos with coordinates
    repos_result = await db.execute(
        select(StarredRepo).where(
            StarredRepo.user_id == user.id,
            StarredRepo.is_embedded == True,  # noqa: E712
            StarredRepo.coord_x.isnot(None),
        )
    )
    repos = repos_result.scalars().all()

    # Get clusters
    clusters_result = await db.execute(
        select(Cluster).where(Cluster.user_id == user.id)
    )
    clusters = clusters_result.scalars().all()

    # Get user's star lists
    star_lists_result = await db.execute(
        select(StarList).where(StarList.user_id == user.id)
    )
    star_lists = star_lists_result.scalars().all()

    # Build cluster color map
    cluster_colors = {
        c.id: c.color or CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        for i, c in enumerate(clusters)
    }

    # Build star list name map
    star_list_names = {sl.id: sl.name for sl in star_lists}

    # Build nodes
    uncategorized_repo_count = 0
    nodes = []
    for repo in repos:
        # Calculate node size based on stars (log scale)
        import math

        size = 1.0 + math.log10(max(repo.stargazers_count, 1)) * 0.5

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

        node = GraphNode(
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
            color=cluster_colors.get(repo.cluster_id) if repo.cluster_id else "#808080",
            size=size,
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
        nodes.append(node)

    # Build edges (optional, bounded kNN for compatibility with older clients)
    edges = []
    if include_edges and len(repos) < 500:  # Limit for performance
        repos_with_embeddings = [
            repo for repo in repos if repo.embedding is not None and len(repo.embedding) > 0
        ]
        repo_ids = [repo.id for repo in repos_with_embeddings]
        embeddings = [repo.embedding for repo in repos_with_embeddings]
        edges = _build_similarity_edges_knn(
            repo_ids=repo_ids,
            embeddings=embeddings,
            min_similarity=min_similarity,
            k=8,
        )

    # Build cluster info
    cluster_infos = [
        ClusterInfo(
            id=c.id,
            name=c.name,
            description=c.description,
            keywords=c.keywords or [],
            color=cluster_colors.get(c.id, "#808080"),
            repo_count=c.repo_count,
            center_x=c.center_x,
            center_y=c.center_y,
            center_z=c.center_z,
        )
        for c in clusters
    ]

    # Build star list info
    star_list_infos = [
        StarListInfo(
            id=sl.id,
            name=sl.name,
            description=sl.description,
            repo_count=sl.repo_count,
        )
        for sl in star_lists
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

    return GraphData(
        nodes=nodes,
        edges=edges,
        clusters=cluster_infos,
        star_lists=star_list_infos,
        total_nodes=len(nodes),
        total_edges=len(edges),
        total_clusters=len(clusters),
        total_star_lists=len(star_list_infos),
    )


@router.get("/edges", response_model=list[GraphEdge])
async def get_graph_edges(
    strategy: Literal["knn"] = Query(default="knn"),
    k: int = Query(default=8, ge=1, le=30, description="K nearest neighbors per node"),
    min_similarity: float = Query(
        default=0.7, ge=0.5, le=0.99, description="Minimum edge similarity"
    ),
    max_nodes: int = Query(
        default=1000,
        ge=50,
        le=5000,
        description="Maximum number of nodes to include when building edges",
    ),
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Get similarity edges for the graph as a separate heavy request."""
    if strategy != "knn":
        return []

    user = await get_default_user(db)
    if not user:
        return []

    repos_result = await db.execute(
        select(StarredRepo)
        .where(
            StarredRepo.user_id == user.id,
            StarredRepo.is_embedded == True,  # noqa: E712
            StarredRepo.embedding.isnot(None),
        )
        .order_by(StarredRepo.stargazers_count.desc())
        .limit(max_nodes)
    )
    repos = repos_result.scalars().all()

    repo_ids = [repo.id for repo in repos if repo.embedding is not None and len(repo.embedding) > 0]
    embeddings = [repo.embedding for repo in repos if repo.embedding is not None and len(repo.embedding) > 0]

    return _build_similarity_edges_knn(
        repo_ids=repo_ids,
        embeddings=embeddings,
        min_similarity=min_similarity,
        k=k,
    )


@router.get("/timeline", response_model=TimelineData)
async def get_timeline_data(
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Get timeline data for visualization.

    Args:
        db: Database session

    Returns:
        Timeline data grouped by month
    """
    # Get default user
    user = await get_default_user(db)
    if not user:
        return TimelineData(
            points=[],
            total_stars=0,
            date_range=("", ""),
        )

    # Get all repos with starred_at
    repos_result = await db.execute(
        select(StarredRepo)
        .where(
            StarredRepo.user_id == user.id,
            StarredRepo.starred_at.isnot(None),
        )
        .order_by(StarredRepo.starred_at)
    )
    repos = repos_result.scalars().all()

    if not repos:
        return TimelineData(
            points=[],
            total_stars=0,
            date_range=("", ""),
        )

    # Group by month
    from collections import defaultdict

    monthly_data: dict[str, dict] = defaultdict(
        lambda: {
            "count": 0,
            "repos": [],
            "languages": defaultdict(int),
            "topics": defaultdict(int),
        }
    )

    for repo in repos:
        month_key = repo.starred_at.strftime("%Y-%m")
        monthly_data[month_key]["count"] += 1
        monthly_data[month_key]["repos"].append(repo.full_name)

        if repo.language:
            monthly_data[month_key]["languages"][repo.language] += 1

        for topic in repo.topics or []:
            monthly_data[month_key]["topics"][topic] += 1

    # Build timeline points
    points = []
    for date_key in sorted(monthly_data.keys()):
        data = monthly_data[date_key]
        top_languages = sorted(
            data["languages"].items(), key=lambda x: x[1], reverse=True
        )[:5]
        top_topics = sorted(data["topics"].items(), key=lambda x: x[1], reverse=True)[
            :5
        ]

        points.append(
            TimelinePoint(
                date=date_key,
                count=data["count"],
                repos=data["repos"][:10],  # Limit for response size
                top_languages=[lang for lang, _ in top_languages],
                top_topics=[topic for topic, _ in top_topics],
            )
        )

    # Get date range
    first_date = repos[0].starred_at.strftime("%Y-%m")
    last_date = repos[-1].starred_at.strftime("%Y-%m")

    return TimelineData(
        points=points,
        total_stars=len(repos),
        date_range=(first_date, last_date),
    )
