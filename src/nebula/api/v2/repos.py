"""Repository API routes.

Handles repository CRUD and semantic search operations.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.params import Depends as DependsParam
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.application.services.related_repo_service import get_related_repos
from nebula.application.services.user_service import get_default_user
from nebula.core.auth import get_admin_session_username
from nebula.core.config import AppSettings, get_app_settings
from nebula.core.embedding import get_embedding_service
from nebula.db import (
    RepoRelatedFeedback,
    StarredRepo,
    User,
    get_db,
)
from nebula.schemas.repo import (
    RelatedFeedbackRequest,
    RelatedFeedbackResponse,
    RelatedRepoResponse,
    RepoListResponse,
    RepoResponse,
    RepoSearchRequest,
    RepoSearchResponse,
)
from nebula.utils import get_logger

from .access import resolve_read_user
from .auth import ADMIN_SESSION_COOKIE, require_admin, require_admin_csrf

logger = get_logger(__name__)
router = APIRouter()


@router.get("", response_model=RepoListResponse)
async def list_repos(
    page: int = Query(default=1, ge=1, description="Page number"),
    per_page: int = Query(default=50, ge=1, le=100, description="Items per page"),
    language: str | None = Query(default=None, description="Filter by language"),
    cluster_id: int | None = Query(default=None, description="Filter by cluster"),
    sort_by: str = Query(default="starred_at", description="Sort field"),
    order: str = Query(default="desc", description="Sort order (asc/desc)"),
    user: User = Depends(resolve_read_user),  # noqa: B008
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
    # WARNING: AI 建议检查此接口是否仍被前端或外部调用方依赖；当前 frontend/src 中未发现直接调用证据。
    # Build query
    query = select(StarredRepo).where(StarredRepo.user_id == user.id)

    # Apply filters
    if language:
        query = query.where(StarredRepo.language == language)
    if cluster_id is not None:
        query = query.where(StarredRepo.cluster_id == cluster_id)

    # Apply sorting (whitelist to prevent unexpected attribute access)
    ALLOWED_SORT_FIELDS = {
        "starred_at",
        "stargazers_count",
        "full_name",
        "name",
        "language",
        "updated_at",
        "forks_count",
    }
    if sort_by not in ALLOWED_SORT_FIELDS:
        sort_by = "starred_at"
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
    user: User = Depends(resolve_read_user),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Get a specific repository by ID.

    Args:
        repo_id: Repository database ID
        db: Database session

    Returns:
        Repository details
    """
    # WARNING: AI 建议检查此接口是否仍被前端或外部调用方依赖；当前 frontend/src 中未发现直接调用证据。
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
    user: User = Depends(resolve_read_user),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Get related repositories ranked by hybrid relevance score.

    Unlike graph edges, this endpoint ranks against the full embedded repository set.
    """
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

    return await get_related_repos(
        db=db,
        user=user,
        anchor_repo=anchor_repo,
        limit=limit,
        min_score=min_score,
        min_semantic=min_semantic,
    )


@router.post(
    "/{repo_id}/related-feedback",
    response_model=RelatedFeedbackResponse,
)
async def submit_related_feedback(
    repo_id: int,
    request: RelatedFeedbackRequest,
    _: str = Depends(require_admin),  # noqa: B008
    __: None = Depends(require_admin_csrf),  # noqa: B008
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

    stmt = insert(RepoRelatedFeedback).values(
        user_id=user.id,
        anchor_repo_id=repo_id,
        candidate_repo_id=request.candidate_repo_id,
        feedback=request.feedback,
        score_snapshot=request.score_snapshot,
        model_version=request.model_version,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[
            RepoRelatedFeedback.user_id,
            RepoRelatedFeedback.anchor_repo_id,
            RepoRelatedFeedback.candidate_repo_id,
        ],
        set_={
            "feedback": stmt.excluded.feedback,
            "score_snapshot": stmt.excluded.score_snapshot,
            "model_version": stmt.excluded.model_version,
            "created_at": func.now(),
        },
    )
    await db.execute(stmt)
    await db.commit()

    return RelatedFeedbackResponse(
        status="ok",
        message="Feedback recorded",
    )


@router.post("/search", response_model=list[RepoSearchResponse])
async def search_repos(
    request: RepoSearchRequest,
    http_request: Request,
    user: User = Depends(resolve_read_user),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
    settings: AppSettings = Depends(get_app_settings),  # noqa: B008
):
    """Semantic search for repositories.

    Args:
        request: Search request with query and filters
        db: Database session

    Returns:
        List of matching repositories with similarity scores
    """
    # WARNING: AI 建议检查此接口是否仍被前端或外部调用方依赖；当前 frontend/src 中未发现直接调用证据。
    if isinstance(settings, DependsParam):
        settings = get_app_settings()

    session_username = get_admin_session_username(
        http_request,
        settings,
        cookie_name=ADMIN_SESSION_COOKIE,
    )
    is_admin_session = session_username == settings.admin_username

    if settings.effective_read_access_mode() == "demo" and not is_admin_session:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Semantic search requires authenticated read mode",
        )

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
    user: User = Depends(resolve_read_user),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Get language statistics for user's starred repos.

    Args:
        db: Database session

    Returns:
        Language breakdown with counts
    """
    # WARNING: AI 建议检查此接口是否仍被前端或外部调用方依赖；当前 frontend/src 中未发现直接调用证据。
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
