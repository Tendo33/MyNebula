"""Shared graph edge helper functions.

Uses the same unified relevance scoring as Related Repositories
(semantic + tag overlap + star list + language) so that graph edges
and recommendation ranking stay consistent.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from nebula.core.relevance import (
    LANGUAGE_WEIGHT,
    SEMANTIC_WEIGHT,
    STAR_LIST_WEIGHT,
    TAG_OVERLAP_WEIGHT,
    cosine_similarity,
    merge_repo_tags,
)
from nebula.schemas.graph import GraphEdge


@dataclass
class RepoEdgeInfo:
    """Lightweight repo metadata needed for unified edge scoring."""

    repo_id: int
    embedding: Sequence[float] | None
    ai_tags: list[str] | None = None
    topics: list[str] | None = None
    star_list_id: int | None = None
    language: str | None = None
    _merged_tags: list[str] | None = field(default=None, repr=False, init=False)

    @property
    def merged_tags(self) -> list[str]:
        if self._merged_tags is None:
            self._merged_tags = merge_repo_tags(self.ai_tags, self.topics)
        return self._merged_tags


def _weighted_tag_overlap(left: list[str], right: list[str]) -> float:
    """Weighted Jaccard overlap for pre-normalized tag lists."""
    if not left or not right:
        return 0.0
    left_set = set(left)
    right_set = set(right)
    inter = left_set & right_set
    union = left_set | right_set
    if not union:
        return 0.0
    counts = Counter(left + right)
    inter_w = sum(1.0 / counts[t] for t in inter)
    union_w = sum(1.0 / counts[t] for t in union)
    if union_w <= 0:
        return 0.0
    return max(0.0, min(1.0, inter_w / union_w))


def _relevance_score(
    semantic: float,
    tag_overlap: float,
    same_star_list: bool,
    same_language: bool,
) -> float:
    """Unified relevance score matching Related Repositories weights."""
    return max(
        0.0,
        min(
            1.0,
            SEMANTIC_WEIGHT * semantic
            + TAG_OVERLAP_WEIGHT * tag_overlap
            + STAR_LIST_WEIGHT * (1.0 if same_star_list else 0.0)
            + LANGUAGE_WEIGHT * (1.0 if same_language else 0.0),
        ),
    )


def _pair_relevance(src: RepoEdgeInfo, tgt: RepoEdgeInfo) -> float:
    """Full relevance score between two repos."""
    semantic = cosine_similarity(src.embedding, tgt.embedding)
    tag_overlap = _weighted_tag_overlap(src.merged_tags, tgt.merged_tags)
    same_star_list = (
        src.star_list_id is not None and tgt.star_list_id == src.star_list_id
    )
    same_language = (
        bool(src.language) and bool(tgt.language) and src.language == tgt.language
    )
    return _relevance_score(semantic, tag_overlap, same_star_list, same_language)


def _build_similarity_edges_knn(
    repos: list[RepoEdgeInfo],
    min_score: float = 0.5,
    k: int = 8,
) -> list[GraphEdge]:
    """Build undirected edges using per-node K-NN with unified relevance scoring."""
    if k <= 0 or not repos:
        return []

    n = len(repos)
    if n <= 1:
        return []

    pair_scores: dict[tuple[int, int], float] = {}

    for i in range(n):
        src = repos[i]
        if src.embedding is None or len(src.embedding) == 0:
            continue

        candidates: list[tuple[float, int]] = []
        for j in range(n):
            if i == j:
                continue
            tgt = repos[j]
            if tgt.embedding is None or len(tgt.embedding) == 0:
                continue
            score = _pair_relevance(src, tgt)
            if score >= min_score:
                candidates.append((score, j))

        if not candidates:
            continue

        candidates.sort(key=lambda item: item[0], reverse=True)
        for score, j in candidates[:k]:
            target_id = repos[j].repo_id
            edge_key = (
                (src.repo_id, target_id)
                if src.repo_id < target_id
                else (target_id, src.repo_id)
            )
            existing = pair_scores.get(edge_key)
            if existing is None or score > existing:
                pair_scores[edge_key] = score

    edges = [
        GraphEdge(source=s, target=t, weight=w) for (s, t), w in pair_scores.items()
    ]
    edges.sort(key=lambda e: e.weight, reverse=True)
    return edges


def _estimate_adaptive_relevance_threshold(
    repos: list[RepoEdgeInfo],
    *,
    target_degree: int = 8,
    min_threshold: float = 0.35,
    max_threshold: float = 0.85,
    sample_limit: int = 200,
) -> float:
    """Estimate adaptive relevance threshold for stable graph density.

    Thresholds are lower than pure-cosine because the relevance score
    blends semantic (65%) with sparser signals (tags 20%, star-list 10%,
    language 5%).
    """
    if not repos:
        return 0.5

    n_samples = min(len(repos), sample_limit)
    if n_samples < 3:
        return 0.5

    sample = repos[:n_samples]
    scores: list[float] = []
    for i in range(n_samples):
        if sample[i].embedding is None or len(sample[i].embedding) == 0:
            continue
        for j in range(i + 1, n_samples):
            if sample[j].embedding is None or len(sample[j].embedding) == 0:
                continue
            scores.append(_pair_relevance(sample[i], sample[j]))

    if not scores:
        return 0.5

    arr = np.array(scores, dtype=np.float32)
    degree = max(1, min(target_degree, n_samples - 1))
    quantile = max(0.05, min(0.98, 1.0 - (degree / float(n_samples - 1))))
    threshold = float(np.quantile(arr, quantile))
    return max(min_threshold, min(max_threshold, threshold))
