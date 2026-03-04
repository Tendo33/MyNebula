"""Graph visualization API routes.

Provides data for force graph visualization and timeline analytics.
"""

import math
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.core.relevance import cosine_similarity
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


@lru_cache(maxsize=1)
def _get_graph_query_service():
    from nebula.application.services import GraphQueryService

    return GraphQueryService()

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
    """Estimate adaptive similarity threshold for stable graph density."""
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
    # Approximate quantile producing ~target_degree neighbors per node.
    quantile = max(0.05, min(0.98, 1.0 - (degree / float(n_samples - 1))))
    threshold = float(np.quantile(sim_values, quantile))
    return max(min_threshold, min(max_threshold, threshold))


async def get_default_user(db: AsyncSession) -> User | None:
    """Get the first user from database.

    Since authentication is disabled, we use the first available user.
    Returns None if no user exists.
    """
    result = await db.execute(select(User).limit(1))
    return result.scalar_one_or_none()


@router.get("", response_model=GraphData)
async def get_graph_data(
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Get graph data for visualization.

    Args:
        db: Database session

    Returns:
        Complete graph data with nodes, edges, and clusters
    """
    return await _get_graph_query_service().get_graph_data_with_options(
        db,
        version="active",
        include_edges=False,
    )


@router.get("/edges", response_model=list[GraphEdge])
async def get_graph_edges(
    k: int = Query(default=8, ge=1, le=30, description="K nearest neighbors per node"),
    min_similarity: float | None = Query(
        default=None,
        ge=0.0,
        le=0.99,
        description="Manual minimum edge similarity override",
    ),
    adaptive: bool = Query(
        default=True,
        description="Use adaptive thresholding for edge density",
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
    edges_page = await _get_graph_query_service().get_edges_page(
        db,
        version="active",
        cursor=0,
        limit=max_nodes * k,
    )
    return edges_page.edges


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
    return await _get_graph_query_service().get_timeline_data(db, version="active")
