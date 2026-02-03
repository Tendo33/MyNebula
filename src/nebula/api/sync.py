"""Synchronization API routes.

Handles GitHub star synchronization and processing tasks.
"""

from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.core.config import get_sync_settings
from nebula.core.embedding import get_embedding_service
from nebula.core.github_client import GitHubClient
from nebula.db import StarredRepo, SyncTask, User, get_db
from nebula.utils import get_logger

logger = get_logger(__name__)
router = APIRouter()


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


async def get_user_by_token(token: str, db: AsyncSession) -> User:
    """Validate token and get user."""
    from nebula.api.auth import verify_jwt_token

    payload = verify_jwt_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    user_id = int(payload["sub"])
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return user


async def sync_stars_task(user_id: int, task_id: int):
    """Background task to sync GitHub stars.

    Args:
        user_id: User database ID
        task_id: Sync task ID
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

            # Get starred repos from GitHub
            async with GitHubClient(access_token=user.access_token) as client:
                repos = await client.get_starred_repos()

            task.total_items = len(repos)
            await db.commit()

            # Process repos
            settings = get_sync_settings()
            processed = 0
            failed = 0

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
                        )
                        db.add(new_repo)

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
            user.total_stars = len(repos)
            user.synced_stars = processed
            user.last_sync_at = datetime.utcnow()

            # Update task
            task.status = "completed"
            task.completed_at = datetime.utcnow()
            await db.commit()

            logger.info(
                f"Completed star sync for {user.username}: {processed} synced, {failed} failed"
            )

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

            # Compute embeddings
            embedding_service = get_embedding_service()
            processed = 0

            # Build texts for batch processing
            texts = []
            for repo in repos:
                text = embedding_service.build_repo_text(
                    full_name=repo.full_name,
                    description=repo.description,
                    topics=repo.topics,
                    language=repo.language,
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
    token: str = Query(..., description="JWT token"),
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Start GitHub star synchronization.

    Args:
        token: JWT access token
        background_tasks: FastAPI background tasks
        db: Database session

    Returns:
        Sync task information
    """
    user = await get_user_by_token(token, db)

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

    # Start background task
    background_tasks.add_task(sync_stars_task, user.id, task.id)

    return SyncStartResponse(
        task_id=task.id,
        message="Star sync started",
        status="pending",
    )


@router.post("/embeddings", response_model=SyncStartResponse)
async def start_embedding_computation(
    token: str = Query(..., description="JWT token"),
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Start embedding computation for repos.

    Args:
        token: JWT access token
        background_tasks: FastAPI background tasks
        db: Database session

    Returns:
        Task information
    """
    user = await get_user_by_token(token, db)

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
    token: str = Query(..., description="JWT token"),
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Get sync task status.

    Args:
        task_id: Task database ID
        token: JWT access token
        db: Database session

    Returns:
        Task status information
    """
    user = await get_user_by_token(token, db)

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
    token: str = Query(..., description="JWT token"),
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Get all sync tasks for current user.

    Args:
        token: JWT access token
        db: Database session

    Returns:
        List of task statuses
    """
    user = await get_user_by_token(token, db)

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
