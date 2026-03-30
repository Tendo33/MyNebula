"""Service for computing related repository recommendations.

Encapsulates ranking, caching, and serialization of related repos
so the API layer stays thin.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.core.relevance import (
    RelevanceComponents,
    build_relevance_components,
    calculate_relevance_score,
    collect_relevance_reasons,
    cosine_similarity,
    merge_repo_tags,
)
from nebula.db import RepoRelatedCache, StarredRepo, User
from nebula.schemas.repo import (
    RelatedRepoResponse,
    RelatedScoreComponents,
    RepoResponse,
)
from nebula.utils import get_logger

logger = get_logger(__name__)

RELATED_CACHE_VERSION = "related-v1"


def _build_related_cache_key(min_score: float, min_semantic: float, limit: int) -> str:
    return (
        f"{RELATED_CACHE_VERSION}"
        f"|ms={min_score:.4f}"
        f"|sem={min_semantic:.4f}"
        f"|limit={limit}"
    )


def _build_related_result_item(
    anchor_repo: StarredRepo,
    candidate_repo: StarredRepo,
) -> tuple[float, RelevanceComponents, list[str]]:
    semantic = cosine_similarity(anchor_repo.embedding, candidate_repo.embedding)
    anchor_tags = merge_repo_tags(anchor_repo.ai_tags, anchor_repo.topics)
    candidate_tags = merge_repo_tags(candidate_repo.ai_tags, candidate_repo.topics)

    components = build_relevance_components(
        semantic_similarity=semantic,
        anchor_tags=anchor_tags,
        candidate_tags=candidate_tags,
        same_star_list=(
            anchor_repo.star_list_id is not None
            and candidate_repo.star_list_id == anchor_repo.star_list_id
        ),
        same_language=(
            bool(anchor_repo.language)
            and bool(candidate_repo.language)
            and anchor_repo.language == candidate_repo.language
        ),
    )
    score = calculate_relevance_score(components)
    reasons = collect_relevance_reasons(components)
    return score, components, reasons


def rank_related_candidates(
    anchor_repo: StarredRepo,
    candidates: list[StarredRepo],
    min_score: float,
    min_semantic: float,
    limit: int,
) -> list[RelatedRepoResponse]:
    ranked: list[RelatedRepoResponse] = []
    for candidate in candidates:
        if candidate.id == anchor_repo.id:
            continue
        if candidate.embedding is None:
            continue
        score, components, reasons = _build_related_result_item(anchor_repo, candidate)
        if components.semantic < min_semantic:
            continue
        if score < min_score:
            continue

        ranked.append(
            RelatedRepoResponse(
                repo=RepoResponse.model_validate(candidate),
                score=score,
                reasons=reasons,
                components=RelatedScoreComponents(
                    semantic=components.semantic,
                    tag_overlap=components.tag_overlap,
                    same_star_list=components.same_star_list,
                    same_language=components.same_language,
                ),
            )
        )

    ranked.sort(
        key=lambda item: (
            item.score,
            item.components.semantic,
            item.repo.stargazers_count,
        ),
        reverse=True,
    )
    return ranked[:limit]


def serialize_related_results(
    ranked: list[RelatedRepoResponse],
) -> list[dict[str, Any]]:
    return [
        {
            "repo_id": item.repo.id,
            "score": item.score,
            "reasons": item.reasons,
            "components": {
                "semantic": item.components.semantic,
                "tag_overlap": item.components.tag_overlap,
                "same_star_list": item.components.same_star_list,
                "same_language": item.components.same_language,
            },
        }
        for item in ranked
    ]


def deserialize_related_results(
    items: Any,
    repo_by_id: dict[int, StarredRepo],
    limit: int,
) -> list[RelatedRepoResponse]:
    if not isinstance(items, list):
        return []

    restored: list[RelatedRepoResponse] = []
    for raw_item in items:
        if not isinstance(raw_item, dict):
            continue

        repo_id = raw_item.get("repo_id")
        if not isinstance(repo_id, int):
            continue

        repo = repo_by_id.get(repo_id)
        if repo is None:
            continue

        components = raw_item.get("components")
        if not isinstance(components, dict):
            continue

        reasons = raw_item.get("reasons")
        if not isinstance(reasons, list):
            reasons = []

        try:
            restored.append(
                RelatedRepoResponse(
                    repo=RepoResponse.model_validate(repo),
                    score=float(raw_item.get("score", 0.0)),
                    reasons=[str(reason) for reason in reasons],
                    components=RelatedScoreComponents(
                        semantic=float(components.get("semantic", 0.0)),
                        tag_overlap=float(components.get("tag_overlap", 0.0)),
                        same_star_list=float(components.get("same_star_list", 0.0)),
                        same_language=float(components.get("same_language", 0.0)),
                    ),
                )
            )
        except Exception:
            continue

        if len(restored) >= limit:
            break

    return restored


async def get_related_repos(
    *,
    db: AsyncSession,
    user: User,
    anchor_repo: StarredRepo,
    limit: int,
    min_score: float,
    min_semantic: float,
) -> list[RelatedRepoResponse]:
    """Return related repos for *anchor_repo*, using cache when valid."""

    if anchor_repo.embedding is None:
        return []

    cache_key = _build_related_cache_key(
        min_score=min_score,
        min_semantic=min_semantic,
        limit=limit,
    )

    # --- try cache ---
    cache_result = await db.execute(
        select(RepoRelatedCache).where(
            RepoRelatedCache.user_id == user.id,
            RepoRelatedCache.anchor_repo_id == anchor_repo.id,
            RepoRelatedCache.cache_key == cache_key,
        )
    )
    cache_entry = cache_result.scalar_one_or_none()

    if (
        cache_entry is not None
        and cache_entry.user_last_sync_at == user.last_sync_at
        and cache_entry.anchor_updated_at == anchor_repo.updated_at
    ):
        cached_items = cache_entry.items if isinstance(cache_entry.items, list) else []
        if not cached_items:
            return []
        cached_repo_ids = [
            item.get("repo_id")
            for item in cached_items
            if isinstance(item, dict) and isinstance(item.get("repo_id"), int)
        ]
        if cached_repo_ids:
            cached_repo_result = await db.execute(
                select(StarredRepo).where(
                    StarredRepo.user_id == user.id,
                    StarredRepo.id.in_(cached_repo_ids),
                )
            )
            repo_by_id = {repo.id: repo for repo in cached_repo_result.scalars().all()}
            restored = deserialize_related_results(
                items=cached_items,
                repo_by_id=repo_by_id,
                limit=limit,
            )
            if restored:
                return restored

    # --- ANN query ---
    ann_candidate_limit = min(limit * 5, 200)
    candidate_result = await db.execute(
        select(StarredRepo)
        .where(
            StarredRepo.user_id == user.id,
            StarredRepo.is_embedded == True,  # noqa: E712
            StarredRepo.embedding.isnot(None),
            StarredRepo.id != anchor_repo.id,
        )
        .order_by(StarredRepo.embedding.cosine_distance(anchor_repo.embedding))
        .limit(ann_candidate_limit)
    )
    candidates = list(candidate_result.scalars().all())

    ranked = rank_related_candidates(
        anchor_repo=anchor_repo,
        candidates=candidates,
        min_score=min_score,
        min_semantic=min_semantic,
        limit=limit,
    )

    # --- write-back cache ---
    serialized_items = serialize_related_results(ranked)
    await db.execute(
        pg_insert(RepoRelatedCache)
        .values(
            user_id=user.id,
            anchor_repo_id=anchor_repo.id,
            cache_key=cache_key,
            items=serialized_items,
            anchor_updated_at=anchor_repo.updated_at,
            user_last_sync_at=user.last_sync_at,
        )
        .on_conflict_do_update(
            index_elements=[
                RepoRelatedCache.user_id,
                RepoRelatedCache.anchor_repo_id,
                RepoRelatedCache.cache_key,
            ],
            set_={
                "items": serialized_items,
                "anchor_updated_at": anchor_repo.updated_at,
                "user_last_sync_at": user.last_sync_at,
                "updated_at": func.now(),
            },
        )
    )

    return ranked
