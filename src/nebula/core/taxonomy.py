"""Taxonomy candidate generation and normalization utilities.

This module implements deterministic primitives used by the offline taxonomy
pipeline. It intentionally avoids provider-specific dependencies so it can run
reliably in scheduled background tasks.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Any


HIGH_CONFIDENCE_THRESHOLD = 0.78
MEDIUM_CONFIDENCE_THRESHOLD = 0.58


@dataclass(slots=True)
class TaxonomyCandidateDraft:
    """Candidate relationship between two terms before persistence."""

    left_term: str
    left_normalized: str
    right_term: str
    right_normalized: str
    score: float
    confidence_level: str
    evidence: dict[str, Any]


def normalize_taxonomy_token(token: str) -> str:
    """Normalize a taxonomy token to a stable comparable representation."""
    normalized = token.strip().lower()
    normalized = normalized.replace("_", "-")
    normalized = re.sub(r"\s+", "-", normalized)
    normalized = re.sub(r"[^a-z0-9\-+.]+", "", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.strip("-")


def map_topic_token(
    token: str,
    taxonomy_mapping: dict[str, str] | None = None,
) -> str:
    """Normalize and map a token using taxonomy mapping when available."""
    normalized = normalize_taxonomy_token(token)
    if not normalized:
        return ""
    if taxonomy_mapping and normalized in taxonomy_mapping:
        mapped = taxonomy_mapping[normalized].strip()
        return mapped or normalized
    return normalized


def normalize_topics_with_mapping(
    topics: list[str] | None,
    taxonomy_mapping: dict[str, str] | None = None,
    preserve_original: bool = False,
) -> list[str]:
    """Normalize a topic list and optionally preserve original terms."""
    if not topics:
        return []

    result: list[str] = []
    seen: set[str] = set()

    for topic in topics:
        if not topic:
            continue

        mapped = map_topic_token(topic, taxonomy_mapping)
        for candidate in (mapped, topic.strip()) if preserve_original else (mapped,):
            if not candidate:
                continue

            key = candidate.lower()
            if key in seen:
                continue

            seen.add(key)
            result.append(candidate)

    return result


def _lexical_similarity(left: str, right: str) -> float:
    """Compute lexical similarity between two normalized terms."""
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0

    left_parts = {part for part in left.split("-") if part}
    right_parts = {part for part in right.split("-") if part}
    if not left_parts or not right_parts:
        return 0.0

    overlap = len(left_parts & right_parts)
    union = len(left_parts | right_parts)
    jaccard = overlap / union if union else 0.0

    prefix_bonus = 0.15 if left.startswith(right) or right.startswith(left) else 0.0
    return min(1.0, jaccard + prefix_bonus)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right, strict=False))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return dot / (norm_left * norm_right)


def classify_confidence(score: float) -> str:
    """Map candidate score to confidence level."""
    if score >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if score >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


def _pair_key(left: str, right: str) -> tuple[str, str]:
    """Build a deterministic sorted key for a pair of tokens."""
    return (left, right) if left <= right else (right, left)


def build_taxonomy_candidates(
    raw_topic_lists: list[list[str]],
    topic_embeddings: dict[str, list[float]] | None = None,
    min_pair_support: int = 2,
    min_score: float = 0.45,
) -> list[TaxonomyCandidateDraft]:
    """Generate candidate term relations from repository topic lists.

    Scoring formula:
    - co_occurrence_ratio: 0.50 weight
    - lexical_similarity:  0.30 weight
    - embedding_similarity:0.20 weight (if vectors available)
    """
    topic_embeddings = topic_embeddings or {}

    term_frequency: dict[str, int] = {}
    pair_frequency: dict[tuple[str, str], int] = {}
    canonical_to_raw: dict[str, str] = {}

    for topics in raw_topic_lists:
        normalized_topics = {
            normalize_taxonomy_token(topic)
            for topic in topics
            if topic and normalize_taxonomy_token(topic)
        }

        for topic in topics:
            if not topic:
                continue
            normalized = normalize_taxonomy_token(topic)
            if normalized and normalized not in canonical_to_raw:
                canonical_to_raw[normalized] = topic.strip()

        for normalized in normalized_topics:
            term_frequency[normalized] = term_frequency.get(normalized, 0) + 1

        for left, right in combinations(sorted(normalized_topics), 2):
            key = _pair_key(left, right)
            pair_frequency[key] = pair_frequency.get(key, 0) + 1

    candidates: list[TaxonomyCandidateDraft] = []

    for (left, right), support in sorted(pair_frequency.items()):
        if support < min_pair_support:
            continue

        left_freq = term_frequency.get(left, 0)
        right_freq = term_frequency.get(right, 0)
        if left_freq == 0 or right_freq == 0:
            continue

        co_occurrence_ratio = support / max(left_freq, right_freq)
        lexical_similarity = _lexical_similarity(left, right)

        embedding_similarity = 0.0
        if left in topic_embeddings and right in topic_embeddings:
            embedding_similarity = _cosine_similarity(
                topic_embeddings[left],
                topic_embeddings[right],
            )

        score = (
            0.50 * co_occurrence_ratio
            + 0.30 * lexical_similarity
            + 0.20 * max(0.0, embedding_similarity)
        )

        if score < min_score:
            continue

        candidates.append(
            TaxonomyCandidateDraft(
                left_term=canonical_to_raw.get(left, left),
                left_normalized=left,
                right_term=canonical_to_raw.get(right, right),
                right_normalized=right,
                score=round(score, 6),
                confidence_level=classify_confidence(score),
                evidence={
                    "pair_support": support,
                    "left_frequency": left_freq,
                    "right_frequency": right_freq,
                    "co_occurrence_ratio": round(co_occurrence_ratio, 6),
                    "lexical_similarity": round(lexical_similarity, 6),
                    "embedding_similarity": round(embedding_similarity, 6),
                },
            )
        )

    return sorted(candidates, key=lambda item: item.score, reverse=True)


def materialize_candidate_rows(
    version_id: int,
    candidates: list[TaxonomyCandidateDraft],
) -> list[dict[str, Any]]:
    """Convert candidate drafts into rows ready for SQLAlchemy persistence."""
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        rows.append(
            {
                "version_id": version_id,
                "left_term": candidate.left_term,
                "left_normalized": candidate.left_normalized,
                "right_term": candidate.right_term,
                "right_normalized": candidate.right_normalized,
                "score": candidate.score,
                "confidence_level": candidate.confidence_level,
                "decision": "pending",
                "evidence": candidate.evidence,
            }
        )
    return rows


def summarize_candidates(candidates: list[TaxonomyCandidateDraft]) -> dict[str, int]:
    """Summarize candidate count by confidence level."""
    stats = {"high": 0, "medium": 0, "low": 0}
    for candidate in candidates:
        if candidate.confidence_level in stats:
            stats[candidate.confidence_level] += 1
    stats["total"] = len(candidates)
    return stats


async def load_active_taxonomy_mapping(db: Any, user_id: int | None) -> dict[str, str]:
    """Load active taxonomy mappings for a user with global fallback."""
    from sqlalchemy import desc, select

    from nebula.db.models import TaxonomyMapping, TaxonomyVersion

    async def _find_active_version_id(target_user_id: int | None) -> int | None:
        result = await db.execute(
            select(TaxonomyVersion.id)
            .where(
                TaxonomyVersion.user_id == target_user_id,
                TaxonomyVersion.is_active == True,  # noqa: E712
            )
            .order_by(
                desc(TaxonomyVersion.published_at),
                desc(TaxonomyVersion.updated_at),
                desc(TaxonomyVersion.id),
            )
            .limit(1)
        )
        return result.scalar_one_or_none()

    version_id = await _find_active_version_id(user_id)
    if version_id is None and user_id is not None:
        version_id = await _find_active_version_id(None)
    if version_id is None:
        return {}

    result = await db.execute(
        select(TaxonomyMapping)
        .where(TaxonomyMapping.version_id == version_id)
        .order_by(
            desc(TaxonomyMapping.confidence_score),
            desc(TaxonomyMapping.id),
        )
    )
    mappings = result.scalars().all()

    mapping_dict: dict[str, str] = {}
    for mapping in mappings:
        source = (mapping.source_normalized or "").strip().lower()
        canonical = (
            (mapping.canonical_normalized or "").strip()
            or (mapping.canonical_term or "").strip()
        )
        if source and canonical and source not in mapping_dict:
            mapping_dict[source] = canonical

    return mapping_dict
