"""Related repository relevance scoring."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt

from nebula.core.tag_normalization import (
    merge_and_normalize_tag_sources,
    weighted_tag_overlap_score,
)

# Fixed default weights (locked by product decision).
SEMANTIC_WEIGHT = 0.65
TAG_OVERLAP_WEIGHT = 0.20
STAR_LIST_WEIGHT = 0.10
LANGUAGE_WEIGHT = 0.05


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity in [0, 1] for positive-aligned vectors."""
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

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0

    raw = dot / (sqrt(norm_a) * sqrt(norm_b))
    return max(0.0, min(1.0, float(raw)))


@dataclass
class RelevanceComponents:
    """Scoring components for explainable ranking."""

    semantic: float
    tag_overlap: float
    same_star_list: float
    same_language: float


def build_relevance_components(
    *,
    semantic_similarity: float,
    anchor_tags: list[str] | None,
    candidate_tags: list[str] | None,
    same_star_list: bool,
    same_language: bool,
) -> RelevanceComponents:
    """Build normalized scoring components."""
    tag_score = weighted_tag_overlap_score(anchor_tags, candidate_tags)
    return RelevanceComponents(
        semantic=max(0.0, min(1.0, semantic_similarity)),
        tag_overlap=tag_score,
        same_star_list=1.0 if same_star_list else 0.0,
        same_language=1.0 if same_language else 0.0,
    )


def calculate_relevance_score(components: RelevanceComponents) -> float:
    """Compute final relevance score in [0, 1]."""
    score = (
        SEMANTIC_WEIGHT * components.semantic
        + TAG_OVERLAP_WEIGHT * components.tag_overlap
        + STAR_LIST_WEIGHT * components.same_star_list
        + LANGUAGE_WEIGHT * components.same_language
    )
    return max(0.0, min(1.0, float(score)))


def collect_relevance_reasons(components: RelevanceComponents) -> list[str]:
    """Create compact reason labels for UI display."""
    reasons: list[str] = []
    if components.semantic >= 0.78:
        reasons.append("semantic:very-high")
    elif components.semantic >= 0.65:
        reasons.append("semantic:high")
    elif components.semantic >= 0.5:
        reasons.append("semantic:medium")

    if components.tag_overlap >= 0.45:
        reasons.append("tags:high-overlap")
    elif components.tag_overlap > 0:
        reasons.append("tags:overlap")

    if components.same_star_list > 0:
        reasons.append("same-star-list")
    if components.same_language > 0:
        reasons.append("same-language")
    return reasons


def merge_repo_tags(
    ai_tags: list[str] | None,
    topics: list[str] | None,
) -> list[str]:
    """Normalize and merge repo tag sources."""
    return merge_and_normalize_tag_sources(ai_tags, topics)

