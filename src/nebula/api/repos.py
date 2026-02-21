"""Repository API routes.

Handles repository CRUD and semantic search operations.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.core.embedding import get_embedding_service
from nebula.core.relevance import (
    RelevanceComponents,
    build_relevance_components,
    calculate_relevance_score,
    collect_relevance_reasons,
    cosine_similarity,
    merge_repo_tags,
)
from nebula.db import RepoRelatedFeedback, StarredRepo, User, get_db
from nebula.schemas.repo import (
    RelatedFeedbackRequest,
    RelatedFeedbackResponse,
    RelatedRepoResponse,
    RelatedScoreComponents,
    RepoListResponse,
    RepoResponse,
    RepoSearchRequest,
    RepoSearchResponse,
)
from nebula.utils import get_logger

logger = get_logger(__name__)
router = APIRouter()


async def get_default_user(db: AsyncSession) -> User:
    """Get the first user from database.

    Since authentication is disabled, we use the first available user.
    """
    result = await db.execute(select(User).limit(1))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No user found. Please sync your GitHub stars first.",
        )

    return user


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


def _rank_related_candidates(
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


@router.get("", response_model=RepoListResponse)
async def list_repos(
    page: int = Query(default=1, ge=1, description="Page number"),
    per_page: int = Query(default=50, ge=1, le=100, description="Items per page"),
    language: str | None = Query(default=None, description="Filter by language"),
    cluster_id: int | None = Query(default=None, description="Filter by cluster"),
    sort_by: str = Query(default="starred_at", description="Sort field"),
    order: str = Query(default="desc", description="Sort order (asc/desc)"),
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """List user's starred repositories.

    Args:
        page: Page number (1-indexed)
        per_page: Items per page
        language: Filter by programming language
        cluster_id: Filter by cluster ID
        sort_by: Sort field (starred_at, stargazers_count, name)
        order: Sort order
        db: Database session

    Returns:
        Paginated list of repositories
    """
    user = await get_default_user(db)

    # Build query
    query = select(StarredRepo).where(StarredRepo.user_id == user.id)

    # Apply filters
    if language:
        query = query.where(StarredRepo.language == language)
    if cluster_id is not None:
        query = query.where(StarredRepo.cluster_id == cluster_id)

    # Apply sorting
    sort_column = getattr(StarredRepo, sort_by, StarredRepo.starred_at)
    if order == "desc":
        query = query.order_by(sort_column.desc())
    else:
        query = query.order_by(sort_column.asc())

    # Get total count
    count_query = select(func.count(StarredRepo.id)).where(
        StarredRepo.user_id == user.id
    )
    if language:
        count_query = count_query.where(StarredRepo.language == language)
    if cluster_id is not None:
        count_query = count_query.where(StarredRepo.cluster_id == cluster_id)

    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Apply pagination
    offset = (page - 1) * per_page
    query = query.offset(offset).limit(per_page)

    # Execute
    result = await db.execute(query)
    repos = result.scalars().all()

    return RepoListResponse(
        items=[RepoResponse.model_validate(r) for r in repos],
        total=total,
        page=page,
        per_page=per_page,
        has_more=(offset + len(repos)) < total,
    )


@router.get("/{repo_id}", response_model=RepoResponse)
async def get_repo(
    repo_id: int,
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Get a specific repository by ID.

    Args:
        repo_id: Repository database ID
        db: Database session

    Returns:
        Repository details
    """
    user = await get_default_user(db)

    result = await db.execute(
        select(StarredRepo).where(
            StarredRepo.id == repo_id,
            StarredRepo.user_id == user.id,
        )
    )
    repo = result.scalar_one_or_none()

    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository not found",
        )

    return RepoResponse.model_validate(repo)


@router.get("/{repo_id}/related", response_model=list[RelatedRepoResponse])
async def get_related_repositories(
    repo_id: int,
    limit: int = Query(default=20, ge=1, le=100),
    min_score: float = Query(default=0.4, ge=0.0, le=1.0),
    min_semantic: float = Query(default=0.65, ge=0.0, le=1.0),
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Get related repositories ranked by hybrid relevance score.

    Unlike graph edges, this endpoint ranks against the full embedded repository set.
    """
    user = await get_default_user(db)

    anchor_result = await db.execute(
        select(StarredRepo).where(
            StarredRepo.user_id == user.id,
            StarredRepo.id == repo_id,
        )
    )
    anchor_repo = anchor_result.scalar_one_or_none()
    if anchor_repo is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository not found",
        )

    if anchor_repo.embedding is None:
        return []

    candidate_result = await db.execute(
        select(StarredRepo).where(
            StarredRepo.user_id == user.id,
            StarredRepo.is_embedded == True,  # noqa: E712
            StarredRepo.embedding.isnot(None),
            StarredRepo.id != anchor_repo.id,
        )
    )
    candidates = candidate_result.scalars().all()

    return _rank_related_candidates(
        anchor_repo=anchor_repo,
        candidates=candidates,
        min_score=min_score,
        min_semantic=min_semantic,
        limit=limit,
    )


@router.post(
    "/{repo_id}/related-feedback",
    response_model=RelatedFeedbackResponse,
)
async def submit_related_feedback(
    repo_id: int,
    request: RelatedFeedbackRequest,
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Record user feedback for related repository recommendations."""
    user = await get_default_user(db)

    anchor_result = await db.execute(
        select(StarredRepo.id).where(
            StarredRepo.user_id == user.id,
            StarredRepo.id == repo_id,
        )
    )
    anchor_exists = anchor_result.scalar_one_or_none()
    if anchor_exists is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Anchor repository not found",
        )

    candidate_result = await db.execute(
        select(StarredRepo.id).where(
            StarredRepo.user_id == user.id,
            StarredRepo.id == request.candidate_repo_id,
        )
    )
    candidate_exists = candidate_result.scalar_one_or_none()
    if candidate_exists is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Candidate repository not found",
        )

    feedback = RepoRelatedFeedback(
        user_id=user.id,
        anchor_repo_id=repo_id,
        candidate_repo_id=request.candidate_repo_id,
        feedback=request.feedback,
        score_snapshot=request.score_snapshot,
        model_version=request.model_version,
    )
    db.add(feedback)
    await db.commit()

    return RelatedFeedbackResponse(
        status="ok",
        message="Feedback recorded",
    )


@router.post("/search", response_model=list[RepoSearchResponse])
async def search_repos(
    request: RepoSearchRequest,
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Semantic search for repositories.

    Args:
        request: Search request with query and filters
        db: Database session

    Returns:
        List of matching repositories with similarity scores
    """
    user = await get_default_user(db)

    # Get embedding for query
    embedding_service = get_embedding_service()
    query_embedding = await embedding_service.embed_text(request.query)

    # Build vector similarity query
    # Using pgvector's cosine distance operator <=>
    query = (
        select(
            StarredRepo,
            (1 - StarredRepo.embedding.cosine_distance(query_embedding)).label(
                "similarity"
            ),
        )
        .where(
            StarredRepo.user_id == user.id,
            StarredRepo.is_embedded == True,  # noqa: E712
        )
        .order_by(StarredRepo.embedding.cosine_distance(query_embedding))
        .limit(request.limit)
    )

    # Apply additional filters
    if request.language:
        query = query.where(StarredRepo.language == request.language)
    if request.cluster_id is not None:
        query = query.where(StarredRepo.cluster_id == request.cluster_id)
    if request.min_stars is not None:
        query = query.where(StarredRepo.stargazers_count >= request.min_stars)

    result = await db.execute(query)
    rows = result.all()

    return [
        RepoSearchResponse(
            repo=RepoResponse.model_validate(row.StarredRepo),
            score=float(row.similarity) if row.similarity else 0.0,
        )
        for row in rows
    ]


@router.get("/languages/stats")
async def get_language_stats(
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Get language statistics for user's starred repos.

    Args:
        db: Database session

    Returns:
        Language breakdown with counts
    """
    user = await get_default_user(db)

    query = (
        select(
            StarredRepo.language,
            func.count(StarredRepo.id).label("count"),
        )
        .where(
            StarredRepo.user_id == user.id,
            StarredRepo.language.isnot(None),
        )
        .group_by(StarredRepo.language)
        .order_by(func.count(StarredRepo.id).desc())
        .limit(20)
    )

    result = await db.execute(query)
    rows = result.all()

    return [{"language": row.language, "count": row.count} for row in rows]
