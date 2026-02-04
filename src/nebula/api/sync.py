"""Synchronization API routes.

Handles GitHub star synchronization and processing tasks.
"""

from datetime import datetime
from enum import Enum
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.core.config import get_sync_settings
from nebula.core.embedding import get_embedding_service
from nebula.core.github_client import GitHubClient
from nebula.core.llm import get_llm_service
from nebula.db import Cluster, StarList, StarredRepo, SyncTask, User, get_db
from nebula.utils import get_logger

logger = get_logger(__name__)
router = APIRouter()


class SyncMode(str, Enum):
    """Sync mode for star synchronization."""

    FULL = "full"  # Full sync: fetch all starred repos
    INCREMENTAL = "incremental"  # Incremental: only fetch new stars since last sync


class SyncStatusResponse(BaseModel):
    """Response for sync status."""

    task_id: int | None = None
    status: str
    task_type: str
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    progress_percent: float = 0.0
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class SyncStartResponse(BaseModel):
    """Response when starting a sync."""

    task_id: int
    message: str
    status: str


async def get_default_user(db: AsyncSession) -> User:
    """Get the first user from database, or create one from GitHub token.

    Since authentication is disabled, we use the first available user.
    If no user exists, we create one using the GitHub token.
    """
    result = await db.execute(select(User).limit(1))
    user = result.scalar_one_or_none()

    if user is None:
        # Try to create a user from GitHub token
        from nebula.core.config import get_app_settings

        settings = get_app_settings()

        if not settings.github_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user found and GITHUB_TOKEN not configured. Please set GITHUB_TOKEN in .env file.",
            )

        # Get user info from GitHub
        try:
            async with GitHubClient(access_token=settings.github_token) as client:
                github_user = await client.get_current_user()

            # Create new user
            user = User(
                github_id=github_user.id,
                username=github_user.login,
                email=github_user.email,
                avatar_url=github_user.avatar_url,
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
            logger.info(f"Created default user: {user.username}")
        except Exception as e:
            logger.error(f"Failed to create user from GitHub token: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create user from GitHub token: {e}",
            ) from e

    return user


async def sync_star_lists(
    user_id: int,
    github_token: str,
    db: AsyncSession,
) -> None:
    """Sync user's GitHub star lists.

    This fetches the user's custom star lists from GitHub and updates
    the database, linking repos to their respective lists.

    Args:
        user_id: User database ID
        github_token: GitHub access token
        db: Database session
    """
    try:
        async with GitHubClient(access_token=github_token) as client:
            star_lists = await client.get_star_lists()

        if not star_lists:
            logger.info(f"No star lists found for user {user_id}")
            return

        # Build map of github_repo_id -> list_id for quick lookup
        repo_to_list_map: dict[int, int] = {}

        for gh_list in star_lists:
            # Check if list exists
            result = await db.execute(
                select(StarList).where(
                    StarList.user_id == user_id,
                    StarList.github_list_id == gh_list.id,
                )
            )
            existing_list = result.scalar_one_or_none()

            if existing_list:
                # Update existing
                existing_list.name = gh_list.name
                existing_list.description = gh_list.description
                existing_list.is_public = gh_list.is_public
                existing_list.repo_count = gh_list.repos_count
                db_list = existing_list
            else:
                # Create new
                db_list = StarList(
                    user_id=user_id,
                    github_list_id=gh_list.id,
                    name=gh_list.name,
                    description=gh_list.description,
                    is_public=gh_list.is_public,
                    repo_count=gh_list.repos_count,
                )
                db.add(db_list)
                await db.flush()  # Get the ID

            # Map repo IDs to this list
            for repo_id in gh_list.repo_ids:
                repo_to_list_map[repo_id] = db_list.id

        await db.commit()

        # Update repos with their list assignments
        if repo_to_list_map:
            result = await db.execute(
                select(StarredRepo).where(
                    StarredRepo.user_id == user_id,
                    StarredRepo.github_repo_id.in_(list(repo_to_list_map.keys())),
                )
            )
            repos = result.scalars().all()

            for repo in repos:
                if repo.github_repo_id in repo_to_list_map:
                    repo.star_list_id = repo_to_list_map[repo.github_repo_id]

            await db.commit()

        logger.info(
            f"Synced {len(star_lists)} star lists for user {user_id}, "
            f"{len(repo_to_list_map)} repos assigned to lists"
        )

    except Exception as e:
        logger.warning(f"Failed to sync star lists: {e}")
        raise


async def sync_stars_task(
    user_id: int,
    task_id: int,
    sync_mode: str = "incremental",
):
    """Background task to sync GitHub stars.

    Supports two modes:
    - incremental: Only fetch stars newer than last_sync_at (fast, default)
    - full: Fetch all starred repos (complete sync)

    Args:
        user_id: User database ID
        task_id: Sync task ID
        sync_mode: Sync mode - "incremental" or "full"
    """
    from nebula.db.database import get_db_context

    async with get_db_context() as db:
        try:
            # Get user and task
            user = await db.get(User, user_id)
            task = await db.get(SyncTask, task_id)

            if not user or not task:
                logger.error(f"User or task not found: user={user_id}, task={task_id}")
                return

            # Update task status
            task.status = "running"
            task.started_at = datetime.utcnow()
            await db.commit()

            # Determine effective sync mode
            # If no previous sync, force full mode
            effective_mode = sync_mode
            stop_before = None

            if sync_mode == "incremental" and user.last_sync_at:
                stop_before = user.last_sync_at
                logger.info(
                    f"Incremental sync for {user.username}: "
                    f"fetching stars newer than {stop_before}"
                )
            elif sync_mode == "incremental" and not user.last_sync_at:
                effective_mode = "full"
                logger.info(f"First sync for {user.username}: switching to full mode")
            else:
                logger.info(f"Full sync for {user.username}")

            # Store sync mode in task metadata
            task.error_details = {"sync_mode": effective_mode}
            await db.commit()

            # Get GitHub token from settings
            from nebula.core.config import get_app_settings

            settings = get_app_settings()

            if not settings.github_token:
                error_msg = (
                    "GitHub token not configured. Please set GITHUB_TOKEN in .env file"
                )
                logger.error(error_msg)
                task.status = "failed"
                task.error_message = error_msg
                await db.commit()
                return

            # Get starred repos from GitHub
            async with GitHubClient(access_token=settings.github_token) as client:
                repos, was_truncated = await client.get_starred_repos(
                    stop_before=stop_before
                )

            if was_truncated:
                logger.info(
                    f"Incremental sync: fetched {len(repos)} new repos "
                    f"(stopped at last_sync_at)"
                )

            task.total_items = len(repos)
            await db.commit()

            # Process repos
            settings = get_sync_settings()
            processed = 0
            failed = 0
            new_count = 0
            updated_count = 0

            for repo in repos:
                try:
                    # Check if repo exists
                    result = await db.execute(
                        select(StarredRepo).where(
                            StarredRepo.user_id == user_id,
                            StarredRepo.github_repo_id == repo.id,
                        )
                    )
                    existing = result.scalar_one_or_none()

                    if existing:
                        # Update existing
                        existing.description = repo.description
                        existing.language = repo.language
                        existing.topics = repo.topics
                        existing.stargazers_count = repo.stargazers_count
                        existing.forks_count = repo.forks_count
                        existing.repo_updated_at = repo.updated_at
                        existing.repo_pushed_at = repo.pushed_at
                        existing.owner_avatar_url = repo.owner_avatar_url
                        updated_count += 1
                    else:
                        # Create new
                        new_repo = StarredRepo(
                            user_id=user_id,
                            github_repo_id=repo.id,
                            full_name=repo.full_name,
                            owner=repo.owner,
                            name=repo.name,
                            description=repo.description,
                            language=repo.language,
                            topics=repo.topics,
                            html_url=repo.html_url,
                            homepage_url=repo.homepage,
                            stargazers_count=repo.stargazers_count,
                            forks_count=repo.forks_count,
                            watchers_count=repo.watchers_count,
                            open_issues_count=repo.open_issues_count,
                            starred_at=repo.starred_at,
                            repo_created_at=repo.created_at,
                            repo_updated_at=repo.updated_at,
                            repo_pushed_at=repo.pushed_at,
                            owner_avatar_url=repo.owner_avatar_url,
                        )
                        db.add(new_repo)
                        new_count += 1

                    processed += 1
                    task.processed_items = processed

                    # Commit in batches
                    if processed % settings.batch_size == 0:
                        await db.commit()
                        logger.info(
                            f"Synced {processed}/{len(repos)} repos for user {user.username}"
                        )

                except Exception as e:
                    logger.warning(f"Failed to sync repo {repo.full_name}: {e}")
                    failed += 1
                    task.failed_items = failed

            # Final commit
            await db.commit()

            # Update user stats
            # For incremental sync, we need to count total stars from database
            if effective_mode == "incremental":
                result = await db.execute(
                    select(StarredRepo).where(StarredRepo.user_id == user_id)
                )
                total_db_repos = len(result.scalars().all())
                user.total_stars = total_db_repos
                user.synced_stars = total_db_repos
            else:
                user.total_stars = len(repos)
                user.synced_stars = processed

            user.last_sync_at = datetime.utcnow()

            # Update task with summary
            task.status = "completed"
            task.completed_at = datetime.utcnow()
            task.error_details = {
                "sync_mode": effective_mode,
                "new_repos": new_count,
                "updated_repos": updated_count,
                "was_truncated": was_truncated,
            }
            await db.commit()

            logger.info(
                f"Completed star sync for {user.username} ({effective_mode}): "
                f"{new_count} new, {updated_count} updated, {failed} failed"
            )

            # Sync star lists (user's custom categories)
            try:
                await sync_star_lists(user_id, settings.github_token, db)
            except Exception as e:
                logger.warning(f"Star lists sync failed (non-critical): {e}")

        except Exception as e:
            logger.exception(f"Star sync failed for user {user_id}: {e}")

            # Update task with error
            async with get_db_context() as db:
                task = await db.get(SyncTask, task_id)
                if task:
                    task.status = "failed"
                    task.error_message = str(e)
                    task.completed_at = datetime.utcnow()
                    await db.commit()


async def compute_embeddings_task(user_id: int, task_id: int):
    """Background task to compute embeddings for repos.

    IMPORTANT: This task now prioritizes LLM-generated content (ai_summary, ai_tags)
    over raw metadata for better semantic clustering accuracy.

    If a repo doesn't have ai_summary/ai_tags, it will be auto-generated first.

    Args:
        user_id: User database ID
        task_id: Sync task ID
    """
    from nebula.db.database import get_db_context

    async with get_db_context() as db:
        try:
            task = await db.get(SyncTask, task_id)
            if not task:
                return

            task.status = "running"
            task.started_at = datetime.utcnow()
            await db.commit()

            # Get repos without embeddings
            result = await db.execute(
                select(StarredRepo).where(
                    StarredRepo.user_id == user_id,
                    StarredRepo.is_embedded == False,  # noqa: E712
                )
            )
            repos = result.scalars().all()

            task.total_items = len(repos)
            await db.commit()

            if not repos:
                task.status = "completed"
                task.completed_at = datetime.utcnow()
                await db.commit()
                return

            # First pass: Generate summaries and tags for repos that don't have them
            # This ensures embedding is based on LLM-enhanced content
            llm_service = get_llm_service()
            repos_needing_llm = [r for r in repos if not r.ai_summary or not r.ai_tags]

            if repos_needing_llm:
                logger.info(
                    f"Generating summaries/tags for {len(repos_needing_llm)} repos before embedding"
                )
                for repo in repos_needing_llm:
                    try:
                        (
                            summary,
                            tags,
                        ) = await llm_service.generate_repo_summary_and_tags(
                            full_name=repo.full_name,
                            description=repo.description,
                            topics=repo.topics,
                            language=repo.language,
                            readme_content=repo.readme_content,
                        )
                        repo.ai_summary = summary
                        repo.ai_tags = tags
                        repo.is_summarized = True
                    except Exception as e:
                        logger.warning(
                            f"LLM generation failed for {repo.full_name}: {e}"
                        )
                        # Ensure tags are not empty even on failure
                        if not repo.ai_tags:
                            repo.ai_tags = (
                                repo.topics[:5] if repo.topics else ["开源项目"]
                            )

                await db.commit()
                logger.info(
                    f"LLM enhancement complete for {len(repos_needing_llm)} repos"
                )

            # Compute embeddings using LLM-enhanced content
            embedding_service = get_embedding_service()
            processed = 0

            # Build texts for batch processing with LLM content priority
            texts = []
            for repo in repos:
                text = embedding_service.build_repo_text(
                    full_name=repo.full_name,
                    description=repo.description,
                    topics=repo.topics,
                    language=repo.language,
                    ai_summary=repo.ai_summary,
                    ai_tags=repo.ai_tags,
                )
                texts.append(text)
                repo.embedding_text = text

            # Compute embeddings in batch
            try:
                embeddings = await embedding_service.embed_batch(texts, batch_size=32)

                for repo, embedding in zip(repos, embeddings, strict=False):
                    repo.embedding = embedding
                    repo.is_embedded = True
                    processed += 1

                task.processed_items = processed
                await db.commit()

            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                task.failed_items = len(repos)
                task.error_message = str(e)

            task.status = "completed"
            task.completed_at = datetime.utcnow()
            await db.commit()

            logger.info(f"Completed embedding for user {user_id}: {processed} embedded")

        except Exception as e:
            logger.exception(f"Embedding task failed: {e}")

            async with get_db_context() as db:
                task = await db.get(SyncTask, task_id)
                if task:
                    task.status = "failed"
                    task.error_message = str(e)
                    task.completed_at = datetime.utcnow()
                    await db.commit()


@router.post("/stars", response_model=SyncStartResponse)
async def start_star_sync(
    mode: Literal["incremental", "full"] = Query(
        default="incremental",
        description="Sync mode: 'incremental' (fast, only new stars) or 'full' (all stars)",
    ),
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Start GitHub star synchronization.

    Supports two sync modes:
    - **incremental** (default): Only fetch stars newer than last sync.
      Fast and efficient for regular updates (e.g., after starring a new repo).
    - **full**: Fetch all starred repos. Use for first sync or when you need
      to ensure complete data consistency.

    Args:
        mode: Sync mode - "incremental" or "full"
        background_tasks: FastAPI background tasks
        db: Database session

    Returns:
        Sync task information
    """
    user = await get_default_user(db)

    # Check for existing running task
    result = await db.execute(
        select(SyncTask).where(
            SyncTask.user_id == user.id,
            SyncTask.task_type == "stars",
            SyncTask.status.in_(["pending", "running"]),
        )
    )
    existing = result.scalar_one_or_none()

    if existing:
        return SyncStartResponse(
            task_id=existing.id,
            message="Sync already in progress",
            status=existing.status,
        )

    # Create new task
    task = SyncTask(
        user_id=user.id,
        task_type="stars",
        status="pending",
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)

    # Start background task with sync mode
    background_tasks.add_task(sync_stars_task, user.id, task.id, mode)

    mode_desc = "incremental (new stars only)" if mode == "incremental" else "full"
    return SyncStartResponse(
        task_id=task.id,
        message=f"Star sync started ({mode_desc})",
        status="pending",
    )


@router.post("/embeddings", response_model=SyncStartResponse)
async def start_embedding_computation(
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Start embedding computation for repos.

    Args:
        background_tasks: FastAPI background tasks
        db: Database session

    Returns:
        Task information
    """
    user = await get_default_user(db)

    # Check for existing running task
    result = await db.execute(
        select(SyncTask).where(
            SyncTask.user_id == user.id,
            SyncTask.task_type == "embedding",
            SyncTask.status.in_(["pending", "running"]),
        )
    )
    existing = result.scalar_one_or_none()

    if existing:
        return SyncStartResponse(
            task_id=existing.id,
            message="Embedding computation already in progress",
            status=existing.status,
        )

    # Create new task
    task = SyncTask(
        user_id=user.id,
        task_type="embedding",
        status="pending",
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)

    # Start background task
    background_tasks.add_task(compute_embeddings_task, user.id, task.id)

    return SyncStartResponse(
        task_id=task.id,
        message="Embedding computation started",
        status="pending",
    )


@router.get("/status/{task_id}", response_model=SyncStatusResponse)
async def get_sync_status(
    task_id: int,
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Get sync task status.

    Args:
        task_id: Task database ID
        db: Database session

    Returns:
        Task status information
    """
    user = await get_default_user(db)

    result = await db.execute(
        select(SyncTask).where(
            SyncTask.id == task_id,
            SyncTask.user_id == user.id,
        )
    )
    task = result.scalar_one_or_none()

    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )

    progress = 0.0
    if task.total_items > 0:
        progress = (task.processed_items / task.total_items) * 100

    return SyncStatusResponse(
        task_id=task.id,
        status=task.status,
        task_type=task.task_type,
        total_items=task.total_items,
        processed_items=task.processed_items,
        failed_items=task.failed_items,
        progress_percent=progress,
        error_message=task.error_message,
        started_at=task.started_at,
        completed_at=task.completed_at,
    )


@router.get("/status", response_model=list[SyncStatusResponse])
async def get_all_sync_status(
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Get all sync tasks for current user.

    Args:
        db: Database session

    Returns:
        List of task statuses
    """
    user = await get_default_user(db)

    result = await db.execute(
        select(SyncTask)
        .where(SyncTask.user_id == user.id)
        .order_by(SyncTask.created_at.desc())
        .limit(10)
    )
    tasks = result.scalars().all()

    return [
        SyncStatusResponse(
            task_id=task.id,
            status=task.status,
            task_type=task.task_type,
            total_items=task.total_items,
            processed_items=task.processed_items,
            failed_items=task.failed_items,
            progress_percent=(task.processed_items / task.total_items * 100)
            if task.total_items > 0
            else 0,
            error_message=task.error_message,
            started_at=task.started_at,
            completed_at=task.completed_at,
        )
        for task in tasks
    ]


# ==================== AI Summary Generation ====================


async def generate_summaries_task(user_id: int, task_id: int):
    """Background task to generate AI summaries for repos.

    Args:
        user_id: User database ID
        task_id: Sync task ID
    """
    from nebula.db.database import get_db_context

    async with get_db_context() as db:
        try:
            task = await db.get(SyncTask, task_id)
            if not task:
                return

            task.status = "running"
            task.started_at = datetime.utcnow()
            await db.commit()

            # Get repos without summaries
            result = await db.execute(
                select(StarredRepo).where(
                    StarredRepo.user_id == user_id,
                    StarredRepo.is_summarized == False,  # noqa: E712
                )
            )
            repos = result.scalars().all()

            task.total_items = len(repos)
            await db.commit()

            if not repos:
                task.status = "completed"
                task.completed_at = datetime.utcnow()
                await db.commit()
                return

            # Generate summaries using LLM
            llm_service = get_llm_service()
            processed = 0
            failed = 0

            # Process in small batches to avoid rate limits
            batch_size = 5
            for i in range(0, len(repos), batch_size):
                batch = repos[i : i + batch_size]

                for repo in batch:
                    try:
                        # Generate both summary and tags in one call
                        (
                            summary,
                            tags,
                        ) = await llm_service.generate_repo_summary_and_tags(
                            full_name=repo.full_name,
                            description=repo.description,
                            topics=repo.topics,
                            language=repo.language,
                            readme_content=repo.readme_content,
                        )

                        repo.ai_summary = summary
                        repo.ai_tags = tags if tags else None
                        repo.is_summarized = True
                        processed += 1

                    except Exception as e:
                        logger.warning(
                            f"Summary/tag generation failed for {repo.full_name}: {e}"
                        )
                        failed += 1

                task.processed_items = processed
                task.failed_items = failed
                await db.commit()

                logger.info(f"Generated summaries: {processed}/{len(repos)}")

            task.status = "completed"
            task.completed_at = datetime.utcnow()
            await db.commit()

            logger.info(
                f"Completed summary generation for user {user_id}: {processed} generated, {failed} failed"
            )

        except Exception as e:
            logger.exception(f"Summary generation task failed: {e}")

            async with get_db_context() as db:
                task = await db.get(SyncTask, task_id)
                if task:
                    task.status = "failed"
                    task.error_message = str(e)
                    task.completed_at = datetime.utcnow()
                    await db.commit()


@router.post("/summaries", response_model=SyncStartResponse)
async def start_summary_generation(
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Start AI summary generation for repos.

    Args:
        background_tasks: FastAPI background tasks
        db: Database session

    Returns:
        Task information
    """
    user = await get_default_user(db)

    # Check for existing running task
    result = await db.execute(
        select(SyncTask).where(
            SyncTask.user_id == user.id,
            SyncTask.task_type == "summary",
            SyncTask.status.in_(["pending", "running"]),
        )
    )
    existing = result.scalar_one_or_none()

    if existing:
        return SyncStartResponse(
            task_id=existing.id,
            message="Summary generation already in progress",
            status=existing.status,
        )

    # Create new task
    task = SyncTask(
        user_id=user.id,
        task_type="summary",
        status="pending",
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)

    # Start background task
    background_tasks.add_task(generate_summaries_task, user.id, task.id)

    return SyncStartResponse(
        task_id=task.id,
        message="Summary generation started",
        status="pending",
    )


# ==================== Clustering ====================


async def run_clustering_task(user_id: int, task_id: int, use_llm: bool = True):
    """Background task to run clustering on user's repos.

    Args:
        user_id: User database ID
        task_id: Sync task ID
        use_llm: Whether to use LLM for cluster naming
    """
    from nebula.core.clustering import (
        generate_cluster_name,
        generate_cluster_name_llm,
        get_clustering_service,
    )
    from nebula.db.database import get_db_context

    async with get_db_context() as db:
        try:
            task = await db.get(SyncTask, task_id)
            if not task:
                return

            task.status = "running"
            task.started_at = datetime.utcnow()
            await db.commit()

            # Get all embedded repos
            result = await db.execute(
                select(StarredRepo).where(
                    StarredRepo.user_id == user_id,
                    StarredRepo.is_embedded == True,  # noqa: E712
                )
            )
            repos = result.scalars().all()

            if not repos:
                task.status = "completed"
                task.completed_at = datetime.utcnow()
                task.error_message = "No embedded repos found"
                await db.commit()
                return

            task.total_items = len(repos)
            await db.commit()

            logger.info(f"Running clustering on {len(repos)} repos for user {user_id}")

            # Extract embeddings and node sizes (based on star count for collision resolution)
            embeddings = []
            node_sizes = []
            for repo in repos:
                if repo.embedding is not None:
                    embeddings.append(repo.embedding)
                    # Calculate node size based on star count (log scale)
                    # This ensures larger (more popular) repos have more space
                    import math

                    size = math.log10(max(repo.stargazers_count, 1) + 1) * 0.5 + 0.5
                    node_sizes.append(min(size, 3.0))  # Cap at 3.0

            if len(embeddings) < 5:
                task.status = "completed"
                task.completed_at = datetime.utcnow()
                task.error_message = "Not enough embedded repos for clustering (min 5)"
                await db.commit()
                return

            # Run clustering with collision resolution
            clustering_service = get_clustering_service()
            cluster_result = clustering_service.fit_transform(
                embeddings=embeddings,
                node_sizes=node_sizes,
                resolve_overlap=True,
            )

            # Delete existing clusters for this user
            await db.execute(select(Cluster).where(Cluster.user_id == user_id))
            existing_clusters = (
                (await db.execute(select(Cluster).where(Cluster.user_id == user_id)))
                .scalars()
                .all()
            )

            for cluster in existing_clusters:
                await db.delete(cluster)
            await db.commit()

            # Create new clusters
            cluster_map: dict[int, Cluster] = {}

            for cluster_id in set(cluster_result.labels):
                if cluster_id == -1:
                    continue  # Skip noise

                # Get repos in this cluster
                cluster_repo_indices = [
                    i
                    for i, label in enumerate(cluster_result.labels)
                    if label == cluster_id
                ]
                cluster_repos = [repos[i] for i in cluster_repo_indices]

                # Extract info for naming
                repo_names = [r.full_name for r in cluster_repos]
                descriptions = [r.description or "" for r in cluster_repos]
                topics = [r.topics or [] for r in cluster_repos]
                languages = [r.language or "" for r in cluster_repos]

                # Generate cluster name
                try:
                    if use_llm:
                        name, description, keywords = await generate_cluster_name_llm(
                            repo_names, descriptions, topics, languages
                        )
                    else:
                        name, description, keywords = generate_cluster_name(
                            repo_names, descriptions, topics
                        )
                except Exception as e:
                    logger.warning(f"Cluster naming failed: {e}, using heuristic")
                    name, description, keywords = generate_cluster_name(
                        repo_names, descriptions, topics
                    )

                # Get cluster center
                center = cluster_result.cluster_centers.get(cluster_id, [0, 0, 0])

                # Create cluster record
                cluster = Cluster(
                    user_id=user_id,
                    name=name,
                    description=description,
                    keywords=keywords,
                    repo_count=len(cluster_repos),
                    center_x=center[0] if len(center) > 0 else None,
                    center_y=center[1] if len(center) > 1 else None,
                    center_z=center[2] if len(center) > 2 else None,
                )
                db.add(cluster)
                await db.flush()  # Get the ID

                cluster_map[cluster_id] = cluster

            await db.commit()

            # Update repos with cluster assignments and coordinates
            for i, repo in enumerate(repos):
                if i < len(cluster_result.labels):
                    label = cluster_result.labels[i]
                    if label != -1 and label in cluster_map:
                        repo.cluster_id = cluster_map[label].id

                    if i < len(cluster_result.coords_3d):
                        coords = cluster_result.coords_3d[i]
                        repo.coord_x = coords[0] if len(coords) > 0 else None
                        repo.coord_y = coords[1] if len(coords) > 1 else None
                        repo.coord_z = coords[2] if len(coords) > 2 else None

            task.processed_items = len(repos)
            task.status = "completed"
            task.completed_at = datetime.utcnow()
            await db.commit()

            logger.info(
                f"Clustering completed for user {user_id}: "
                f"{cluster_result.n_clusters} clusters, {len(repos)} repos assigned"
            )

        except Exception as e:
            logger.exception(f"Clustering task failed: {e}")

            async with get_db_context() as db:
                task = await db.get(SyncTask, task_id)
                if task:
                    task.status = "failed"
                    task.error_message = str(e)
                    task.completed_at = datetime.utcnow()
                    await db.commit()


@router.post("/clustering", response_model=SyncStartResponse)
async def start_clustering(
    use_llm: bool = Query(default=True, description="Use LLM for cluster naming"),
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Start clustering on user's repos.

    This will:
    1. Run UMAP dimensionality reduction
    2. Apply HDBSCAN clustering
    3. Generate cluster names (using LLM if enabled)
    4. Update repo coordinates and cluster assignments

    Args:
        use_llm: Use LLM for cluster naming
        background_tasks: FastAPI background tasks
        db: Database session

    Returns:
        Task information
    """
    user = await get_default_user(db)

    # Check for existing running task
    result = await db.execute(
        select(SyncTask).where(
            SyncTask.user_id == user.id,
            SyncTask.task_type == "cluster",
            SyncTask.status.in_(["pending", "running"]),
        )
    )
    existing = result.scalar_one_or_none()

    if existing:
        return SyncStartResponse(
            task_id=existing.id,
            message="Clustering already in progress",
            status=existing.status,
        )

    # Create new task
    task = SyncTask(
        user_id=user.id,
        task_type="cluster",
        status="pending",
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)

    # Start background task
    background_tasks.add_task(run_clustering_task, user.id, task.id, use_llm)

    return SyncStartResponse(
        task_id=task.id,
        message="Clustering started",
        status="pending",
    )
