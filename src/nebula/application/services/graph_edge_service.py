"""Shared graph edge helper functions."""

from collections.abc import Sequence

import numpy as np

from nebula.core.relevance import cosine_similarity
from nebula.schemas.graph import GraphEdge


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
    quantile = max(0.05, min(0.98, 1.0 - (degree / float(n_samples - 1))))
    threshold = float(np.quantile(sim_values, quantile))
    return max(min_threshold, min(max_threshold, threshold))
