"""Synchronization API routes.

Handles GitHub star synchronization and processing tasks.
"""

from datetime import datetime, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.core.config import get_app_settings, get_sync_settings
from nebula.core.embedding import get_embedding_service
from nebula.core.github_client import GitHubClient
from nebula.core.llm import get_llm_service
from nebula.db import (
    Cluster,
    StarList,
    StarredRepo,
    SyncSchedule,
    SyncTask,
    User,
    get_db,
)
from nebula.schemas import (
    FullRefreshRequest,
    FullRefreshResponse,
    JobStatusResponse,
    ScheduleConfig,
    ScheduleResponse,
    SyncInfoResponse,
)
from nebula.utils import compute_content_hash, compute_topics_hash, get_logger

from .auth import require_admin

logger = get_logger(__name__)
router = APIRouter(dependencies=[Depends(require_admin)])


def _to_utc(dt: datetime | None) -> datetime | None:
    """Normalize datetime to UTC for safe comparison."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _calculate_progress_percent(total_items: int, processed_items: int) -> float:
    """Calculate bounded progress percentage from counters."""
    if total_items <= 0:
        return 0.0
    percent = (processed_items / total_items) * 100
    return max(0.0, min(100.0, float(percent)))


def _estimate_eta_seconds(
    started_at: datetime | None,
    progress_percent: float,
    now: datetime | None = None,
) -> int | None:
    """Estimate remaining seconds from elapsed time and progress."""
    if started_at is None or progress_percent <= 0 or progress_percent >= 100:
        return None

    now_ts = now or datetime.utcnow()
    elapsed_seconds = (now_ts - started_at).total_seconds()
    if elapsed_seconds <= 0:
        return None

    total_estimated = elapsed_seconds * (100.0 / progress_percent)
    remaining = int(max(total_estimated - elapsed_seconds, 0))
    return remaining if remaining > 0 else None


def _resolve_job_phase(
    task_type: str,
    status: str,
    error_details: dict | None,
) -> str:
    """Resolve user-facing job phase from task metadata."""
    if status == "completed":
        return "completed"
    if status == "failed":
        return "failed"
    if isinstance(error_details, dict):
        phase = error_details.get("phase")
        if isinstance(phase, str) and phase:
            return phase
    return task_type


def _validate_full_refresh_confirmation(confirm: bool) -> None:
    """Enforce explicit full-refresh confirmation."""
    if confirm:
        return
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Full refresh requires explicit confirmation",
    )


def _is_schedule_due(
    now_local: datetime,
    schedule_hour: int,
    schedule_minute: int,
    last_run_local: datetime | None,
) -> bool:
    """Check whether a schedule is due now and has not run in the same minute."""
    if now_local.hour != schedule_hour or now_local.minute != schedule_minute:
        return False
    if last_run_local is None:
        return True
    return not (
        last_run_local.date() == now_local.date()
        and last_run_local.hour == now_local.hour
        and last_run_local.minute == now_local.minute
    )


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
    logger.info(
        f"[TASK START] sync_stars_task called: user={user_id}, task={task_id}, mode={sync_mode}"
    )
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
            settings = get_app_settings()

            if not settings.github_token:
                error_msg = (
                    "GitHub token not configured. Please set GITHUB_TOKEN in .env file"
                )
                logger.error(f"[TASK ERROR] {error_msg}")
                task.status = "failed"
                task.error_message = error_msg
                await db.commit()
                return

            logger.info("[TASK PROGRESS] GitHub token found, fetching starred repos...")
            # Get starred repos from GitHub
            try:
                async with GitHubClient(access_token=settings.github_token) as client:
                    repos, was_truncated = await client.get_starred_repos(
                        stop_before=stop_before
                    )
            except Exception as api_error:
                logger.exception(f"[TASK ERROR] GitHub API call failed: {api_error}")
                task.status = "failed"
                task.error_message = f"GitHub API error: {api_error}"
                await db.commit()
                return

            logger.info(f"[TASK PROGRESS] Fetched {len(repos)} repos from GitHub")

            if was_truncated:
                logger.info(
                    f"Incremental sync: fetched {len(repos)} new repos "
                    f"(stopped at last_sync_at)"
                )

            task.total_items = len(repos)
            await db.commit()

            # Process repos
            sync_settings = get_sync_settings()
            processed = 0
            failed = 0
            new_count = 0
            updated_count = 0

            async with GitHubClient(access_token=settings.github_token) as readme_client:
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
                            # Smart change detection: check if description or topics changed
                            new_desc_hash = compute_content_hash(repo.description)
                            new_topics_hash = compute_topics_hash(repo.topics)

                            needs_reprocess = False
                            if existing.description_hash != new_desc_hash:
                                needs_reprocess = True
                            if existing.topics_hash != new_topics_hash:
                                needs_reprocess = True

                            # If content changed, reset processing flags
                            if needs_reprocess:
                                existing.is_embedded = False
                                existing.is_summarized = False
                                existing.ai_summary = None
                                existing.ai_tags = None
                                existing.embedding = None
                                latest_readme = await readme_client.get_repo_readme(
                                    repo.full_name,
                                    max_length=sync_settings.readme_max_length,
                                )
                                if latest_readme is not None:
                                    existing.readme_content = latest_readme
                                    existing.is_readme_fetched = True
                                logger.info(
                                    f"Repo {repo.full_name} content changed, marked for reprocessing"
                                )

                            # Update metadata
                            existing.description = repo.description
                            existing.language = repo.language
                            existing.topics = repo.topics
                            existing.stargazers_count = repo.stargazers_count
                            existing.forks_count = repo.forks_count
                            existing.repo_updated_at = repo.updated_at
                            existing.repo_pushed_at = repo.pushed_at
                            existing.owner_avatar_url = repo.owner_avatar_url

                            # Update hashes for future detection
                            existing.description_hash = new_desc_hash
                            existing.topics_hash = new_topics_hash

                            updated_count += 1
                        else:
                            readme_content = await readme_client.get_repo_readme(
                                repo.full_name,
                                max_length=sync_settings.readme_max_length,
                            )
                            # Create new with content hashes for future detection
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
                                readme_content=readme_content,
                                is_readme_fetched=readme_content is not None,
                                # Content hashes for smart change detection
                                description_hash=compute_content_hash(repo.description),
                                topics_hash=compute_topics_hash(repo.topics),
                            )
                            db.add(new_repo)
                            new_count += 1

                        processed += 1
                        task.processed_items = processed

                        # Commit in batches
                        if processed % sync_settings.batch_size == 0:
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

            # Detect and remove unstarred repos
            # For incremental mode: we need to fetch full starred list just for IDs (fast)
            # For full mode: we already have the complete list
            removed_count = 0

            if effective_mode == "full" and not was_truncated:
                # Full sync already has complete data
                github_repo_ids_from_api = {repo.id for repo in repos}
            elif sync_settings.detect_unstarred_on_incremental:
                # Optional incremental deletion detection.
                # This requires fetching the full starred list and can be expensive.
                logger.info(
                    "Incremental deletion detection enabled: fetching full starred list..."
                )
                try:
                    async with GitHubClient(
                        access_token=settings.github_token
                    ) as client:
                        all_repos, _ = await client.get_starred_repos()
                    github_repo_ids_from_api = {repo.id for repo in all_repos}
                    logger.info(
                        f"Fetched {len(github_repo_ids_from_api)} starred repo IDs for deletion check"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to fetch complete starred list for deletion: {e}. "
                        "Skipping deletion detection."
                    )
                    github_repo_ids_from_api = None
            else:
                github_repo_ids_from_api = None
                logger.info(
                    "Skipping incremental unstarred deletion detection "
                    "(SYNC_DETECT_UNSTARRED_ON_INCREMENTAL=false)"
                )

            # Remove repos that are no longer starred
            if github_repo_ids_from_api is not None:
                result = await db.execute(
                    select(StarredRepo).where(
                        StarredRepo.user_id == user_id,
                        StarredRepo.github_repo_id.notin_(github_repo_ids_from_api),
                    )
                )
                unstarred_repos = result.scalars().all()

                if unstarred_repos:
                    for repo in unstarred_repos:
                        logger.info(
                            f"Removing unstarred repo: {repo.full_name} "
                            f"(github_id={repo.github_repo_id})"
                        )
                        await db.delete(repo)
                        removed_count += 1

                    await db.commit()
                    logger.info(
                        f"Removed {removed_count} unstarred repos for user {user.username}"
                    )

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
                "removed_repos": removed_count,
                "was_truncated": was_truncated,
            }
            await db.commit()

            logger.info(
                f"Completed star sync for {user.username} ({effective_mode}): "
                f"{new_count} new, {updated_count} updated, {removed_count} removed, {failed} failed"
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
                total_llm_repos = len(repos_needing_llm)
                logger.info(
                    f"Generating summaries/tags for {total_llm_repos} repos before embedding"
                )

                for i, repo in enumerate(repos_needing_llm, 1):
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

                    if i % 5 == 0:
                        logger.info(
                            f"Generating summaries progress: {i}/{total_llm_repos}"
                        )
                        await db.commit()

                await db.commit()
                logger.info(f"LLM enhancement complete for {total_llm_repos} repos")

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
                    readme_content=repo.readme_content,
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
        now_utc = datetime.now(timezone.utc)
        created_at_utc = _to_utc(existing.created_at)
        started_at_utc = _to_utc(existing.started_at)

        is_stale_pending = (
            existing.status == "pending"
            and created_at_utc is not None
            and now_utc - created_at_utc > timedelta(minutes=2)
        )
        is_stale_running = (
            existing.status == "running"
            and started_at_utc is not None
            and now_utc - started_at_utc > timedelta(hours=2)
        )

        if is_stale_pending or is_stale_running:
            logger.warning(
                f"Stale sync task detected (id={existing.id}, status={existing.status}); "
                "marking failed and starting a new task"
            )
            existing.status = "failed"
            existing.error_message = (
                "Previous sync task appears stuck. Please retry the sync."
            )
            existing.completed_at = datetime.utcnow()
            await db.commit()
        else:
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
    logger.info(f"Starting sync task {task.id} for user {user.id} in mode {mode}")
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

    progress = _calculate_progress_percent(task.total_items, task.processed_items)

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


@router.get("/jobs/{task_id}", response_model=JobStatusResponse)
async def get_job_status(
    task_id: int,
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Get aggregated job status for richer progress UI."""
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

    progress_percent = _calculate_progress_percent(task.total_items, task.processed_items)
    phase = _resolve_job_phase(task.task_type, task.status, task.error_details)
    eta_seconds = (
        _estimate_eta_seconds(task.started_at, progress_percent)
        if task.status == "running"
        else None
    )

    return JobStatusResponse(
        task_id=task.id,
        task_type=task.task_type,
        status=task.status,
        phase=phase,
        progress_percent=progress_percent,
        eta_seconds=eta_seconds,
        last_error=task.error_message,
        retryable=task.status == "failed",
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
            progress_percent=_calculate_progress_percent(
                task.total_items, task.processed_items
            ),
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


def _derive_clustering_params_for_max_clusters(
    *,
    n_samples: int,
    max_clusters: int,
    min_clusters: int | None = None,
) -> dict:
    """Derive clustering parameters from a user-friendly 'max clusters' knob.

    This is intentionally heuristic. The goal is stable UX, not perfect clustering.
    """
    safe_max_clusters = max(2, min(int(max_clusters), 20))
    safe_min_clusters = max(2, safe_max_clusters // 3)
    if min_clusters is not None:
        safe_min_clusters = max(2, min(int(min_clusters), safe_max_clusters))

    # Keep backward-compatible knobs even though clustering is now hierarchical.
    # For 180 repos: min_cluster_size = max(3, min(10, 180//20)) = max(3, 9) = 9
    approx_min_size = max(3, n_samples // 20)
    min_cluster_size = min(approx_min_size, 10)
    min_samples = max(1, min(3, min_cluster_size // 3))

    # Coarser (smaller max_clusters) => stronger consolidation pressure.
    # Finer (larger max_clusters) => preserves more local distinctions.
    n_neighbors = int(max(15, min(60, 10 + (50 - 2 * safe_max_clusters))))

    return {
        "n_neighbors": n_neighbors,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "min_clusters": safe_min_clusters,
        "target_max_clusters": safe_max_clusters,
    }


def _should_force_full_recluster(
    *,
    total_repos: int,
    new_repos: int,
    centroid_drift: float | None = None,
    new_repo_ratio_threshold: float = 0.2,
    centroid_drift_threshold: float = 0.35,
) -> bool:
    """Decide whether incremental clustering should upgrade to full recluster."""
    safe_total = max(1, total_repos)
    new_ratio = float(new_repos) / float(safe_total)
    if new_ratio > new_repo_ratio_threshold:
        return True

    return centroid_drift is not None and centroid_drift > centroid_drift_threshold


async def run_clustering_task(
    user_id: int,
    task_id: int,
    use_llm: bool = True,
    max_clusters: int = 8,
    min_clusters: int | None = None,
    incremental: bool = False,
):
    """Background task to run clustering on user's repos.

    Args:
        user_id: User database ID
        task_id: Sync task ID
        use_llm: Whether to use LLM for cluster naming
        max_clusters: User-friendly knob for controlling clustering granularity
        min_clusters: Optional lower bound hint for cluster count
        incremental: If True, only assign new repos to existing clusters
            without re-running full clustering. Existing repo positions and
            cluster assignments are preserved. Use for daily updates.
    """
    from nebula.core.clustering import (
        ClusteringService,
        assign_new_repos_incrementally,
        build_cluster_naming_inputs,
        deduplicate_cluster_entries,
        generate_cluster_name,
        generate_cluster_name_llm,
        normalize_embeddings,
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
            repos_with_embeddings: list[StarredRepo] = []
            embeddings = []
            node_sizes = []
            for repo in repos:
                if repo.embedding is not None:
                    repos_with_embeddings.append(repo)
                    embeddings.append(repo.embedding)
                    # Calculate node size based on star count (log scale)
                    # This ensures larger (more popular) repos have more space
                    import math

                    size = math.log10(max(repo.stargazers_count, 1) + 1) * 0.5 + 0.5
                    node_sizes.append(min(size, 3.0))  # Cap at 3.0

            if len(repos_with_embeddings) != len(repos):
                logger.warning(
                    "Skipping "
                    f"{len(repos) - len(repos_with_embeddings)} repos with missing embeddings "
                    "during clustering"
                )
                for repo in repos:
                    if repo.embedding is None:
                        repo.cluster_id = None
                        repo.coord_x = None
                        repo.coord_y = None
                        repo.coord_z = None

            if len(embeddings) < 5:
                task.status = "completed"
                task.completed_at = datetime.utcnow()
                task.error_message = "Not enough embedded repos for clustering (min 5)"
                await db.commit()
                return

            # ── Incremental mode: only assign NEW repos to existing clusters ──
            if incremental:
                import numpy as np

                # Separate repos into "existing" (have coords + cluster) and "new"
                existing_repos = []
                existing_embs = []
                existing_crds = []
                new_repos = []
                new_embs = []
                new_sizes = []

                # Build a map from DB cluster_id -> label index for existing repos
                existing_cluster_ids: list[int] = []

                for i, repo in enumerate(repos_with_embeddings):
                    has_position = (
                        repo.coord_x is not None
                        and repo.coord_y is not None
                        and repo.coord_z is not None
                        and repo.cluster_id is not None
                    )
                    if has_position:
                        existing_repos.append(repo)
                        existing_embs.append(embeddings[i])
                        existing_crds.append([repo.coord_x, repo.coord_y, repo.coord_z])
                        existing_cluster_ids.append(repo.cluster_id)
                    else:
                        new_repos.append(repo)
                        new_embs.append(embeddings[i])
                        new_sizes.append(node_sizes[i])

                # If no new repos, nothing to do
                if not new_repos:
                    logger.info("Incremental mode: no new repos to assign")
                    task.status = "completed"
                    task.completed_at = datetime.utcnow()
                    task.processed_items = 0
                    await db.commit()
                    return

                # If no existing repos have positions, fall through to full clustering
                if not existing_repos:
                    logger.info(
                        "Incremental mode: no existing repos with positions, "
                        "falling back to full clustering"
                    )
                    incremental = False  # Will fall through to full clustering below
                else:
                    logger.info(
                        f"Incremental mode: {len(existing_repos)} existing repos, "
                        f"{len(new_repos)} new repos to assign"
                    )

                    existing_embs_arr = normalize_embeddings(
                        np.array(existing_embs, dtype=np.float32)
                    )
                    existing_crds_arr = np.array(existing_crds, dtype=np.float64)
                    # Use DB cluster_id directly as labels for nearest-neighbor assignment
                    existing_lbls_arr = np.array(existing_cluster_ids, dtype=int)
                    new_embs_arr = normalize_embeddings(
                        np.array(new_embs, dtype=np.float32)
                    )

                    result_incr = assign_new_repos_incrementally(
                        existing_embeddings=existing_embs_arr,
                        existing_coords=existing_crds_arr,
                        existing_labels=existing_lbls_arr,
                        new_embeddings=new_embs_arr,
                        new_node_sizes=new_sizes,
                        k_neighbors=5,
                        noise_scale=0.15,
                    )

                    # Update new repos with assigned coordinates and cluster IDs
                    for i, repo in enumerate(new_repos):
                        coords = result_incr.new_coords[i]
                        repo.coord_x = coords[0]
                        repo.coord_y = coords[1]
                        repo.coord_z = coords[2]
                        # The label from incremental assign is the DB cluster_id
                        repo.cluster_id = int(result_incr.new_labels[i])

                    # Update cluster repo counts
                    cluster_ids_to_update = set(
                        int(label) for label in result_incr.new_labels
                    )
                    for cid in cluster_ids_to_update:
                        cluster_obj = await db.get(Cluster, cid)
                        if cluster_obj:
                            count_result = await db.execute(
                                select(StarredRepo).where(
                                    StarredRepo.user_id == user_id,
                                    StarredRepo.cluster_id == cid,
                                )
                            )
                            cluster_obj.repo_count = len(count_result.scalars().all())

                    task.processed_items = len(new_repos)
                    task.status = "completed"
                    task.completed_at = datetime.utcnow()
                    await db.commit()

                    logger.info(
                        f"Incremental clustering completed for user {user_id}: "
                        f"{len(new_repos)} new repos assigned to existing clusters"
                    )
                    return

            # ── Full clustering mode ──

            # Run clustering with collision resolution.
            # Use a user-friendly max-clusters control to derive stable parameters.
            derived = _derive_clustering_params_for_max_clusters(
                n_samples=len(repos_with_embeddings),
                max_clusters=max_clusters,
                min_clusters=min_clusters,
            )
            logger.info(
                "Clustering params: "
                f"max_clusters={max_clusters}, min_clusters={min_clusters}, "
                f"derived_min_clusters={derived['min_clusters']}, "
                f"derived_target_max_clusters={derived['target_max_clusters']}, "
                f"min_cluster_size={derived['min_cluster_size']}, "
                f"min_samples={derived['min_samples']}, "
                f"n_neighbors={derived['n_neighbors']}"
            )
            clustering_service = ClusteringService(
                n_neighbors=derived["n_neighbors"],
                min_dist=0.1,
                min_cluster_size=derived["min_cluster_size"],
                min_samples=derived["min_samples"],
                cluster_selection_method="eom",
                min_clusters=derived["min_clusters"],
                target_min_clusters=derived["min_clusters"],
                target_max_clusters=derived["target_max_clusters"],
            )

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
            cluster_entries: list[dict] = []
            sorted_cluster_ids = sorted(
                {cluster_id for cluster_id in cluster_result.labels if cluster_id != -1}
            )

            # Track assigned names so each subsequent LLM call avoids duplicates
            assigned_names: list[str] = []

            for cluster_id in sorted_cluster_ids:
                # Get repos in this cluster
                cluster_repo_indices = [
                    i
                    for i, label in enumerate(cluster_result.labels)
                    if label == cluster_id
                ]
                cluster_repos = [repos_with_embeddings[i] for i in cluster_repo_indices]

                if not cluster_repos:
                    continue

                # Extract info for naming
                repo_names, descriptions, topics, languages = (
                    build_cluster_naming_inputs(cluster_repos)
                )

                # Generate cluster name (pass existing names to enforce uniqueness)
                try:
                    if use_llm:
                        name, description, keywords = await generate_cluster_name_llm(
                            repo_names,
                            descriptions,
                            topics,
                            languages,
                            existing_cluster_names=assigned_names or None,
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

                assigned_names.append(name)

                # Get cluster center
                center = cluster_result.cluster_centers.get(cluster_id, [0, 0, 0])

                cluster_entries.append(
                    {
                        "cluster_id": cluster_id,
                        "name": name,
                        "description": description,
                        "keywords": keywords,
                        "repo_count": len(cluster_repos),
                        "center": center,
                    }
                )

            cluster_entries = deduplicate_cluster_entries(cluster_entries)

            for entry in cluster_entries:
                center = entry["center"]
                cluster = Cluster(
                    user_id=user_id,
                    name=entry["name"],
                    description=entry["description"],
                    keywords=entry["keywords"],
                    repo_count=entry["repo_count"],
                    center_x=center[0] if len(center) > 0 else None,
                    center_y=center[1] if len(center) > 1 else None,
                    center_z=center[2] if len(center) > 2 else None,
                )
                db.add(cluster)
                await db.flush()  # Get the ID
                cluster_map[entry["cluster_id"]] = cluster

            await db.commit()

            # Update repos with cluster assignments and coordinates
            assigned_count = 0
            unassigned_count = 0

            for i, repo in enumerate(repos_with_embeddings):
                if i < len(cluster_result.labels):
                    label = cluster_result.labels[i]

                    # Assign cluster (skip noise points, though they should be rare now)
                    if label != -1 and label in cluster_map:
                        repo.cluster_id = cluster_map[label].id
                        assigned_count += 1
                    else:
                        # Keep outliers unassigned to preserve cluster purity.
                        repo.cluster_id = None
                        unassigned_count += 1

                    # Update 3D coordinates
                    if i < len(cluster_result.coords_3d):
                        coords = cluster_result.coords_3d[i]
                        repo.coord_x = coords[0] if len(coords) > 0 else None
                        repo.coord_y = coords[1] if len(coords) > 1 else None
                        repo.coord_z = coords[2] if len(coords) > 2 else None

            task.processed_items = len(repos_with_embeddings)
            task.status = "completed"
            task.completed_at = datetime.utcnow()
            await db.commit()

            logger.info(
                f"Clustering completed for user {user_id}: "
                f"{cluster_result.n_clusters} clusters, "
                f"{assigned_count} repos assigned, {unassigned_count} unassigned"
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
    max_clusters: int = Query(
        default=8,
        ge=2,
        le=20,
        description="Maximum number of clusters (lower = coarser, higher = finer)",
    ),
    min_clusters: int = Query(
        default=3,
        ge=2,
        le=20,
        description="Minimum desired clusters (must be <= max_clusters)",
    ),
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Start clustering on user's repos.

    This will:
    1. Run PCA dimensionality reduction
    2. Apply hierarchical clustering
    3. Generate cluster names (using LLM if enabled)
    4. Update repo coordinates and cluster assignments

    Args:
        use_llm: Use LLM for cluster naming
        background_tasks: FastAPI background tasks
        db: Database session

    Returns:
        Task information
    """
    if min_clusters > max_clusters:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="min_clusters must be less than or equal to max_clusters",
        )

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
    background_tasks.add_task(
        run_clustering_task,
        user_id=user.id,
        task_id=task.id,
        use_llm=use_llm,
        max_clusters=max_clusters,
        min_clusters=min_clusters,
    )

    return SyncStartResponse(
        task_id=task.id,
        message="Clustering started",
        status="pending",
    )


# ==================== Full Refresh ====================


async def full_refresh_task(user_id: int, task_id: int):
    """Background task to perform full refresh of all repositories.

    This will:
    1. Reset all processing flags for user's repos
    2. Run full star sync from GitHub
    3. Regenerate all AI summaries
    4. Recompute all embeddings
    5. Re-run clustering

    Args:
        user_id: User database ID
        task_id: Sync task ID for tracking
    """
    from sqlalchemy import update

    from nebula.db.database import get_db_context

    async def _update_main_task(**fields) -> bool:
        async with get_db_context() as db:
            task = await db.get(SyncTask, task_id)
            if not task:
                return False
            for key, value in fields.items():
                setattr(task, key, value)
            await db.commit()
            return True

    async def _create_sub_task(task_type: str) -> int:
        async with get_db_context() as db:
            sub_task = SyncTask(user_id=user_id, task_type=task_type, status="pending")
            db.add(sub_task)
            await db.commit()
            await db.refresh(sub_task)
            return sub_task.id

    try:
        task_exists = await _update_main_task(
            status="running",
            started_at=datetime.utcnow(),
            error_message=None,
            total_items=4,
            processed_items=0,
            error_details={"phase": "reset"},
        )
        if not task_exists:
            return

        # Step 1: Reset all processing flags
        logger.info(f"Full refresh: Resetting all repos for user {user_id}")
        async with get_db_context() as db:
            result = await db.execute(
                update(StarredRepo)
                .where(StarredRepo.user_id == user_id)
                .values(
                    is_embedded=False,
                    is_summarized=False,
                    ai_summary=None,
                    ai_tags=None,
                    embedding=None,
                    description_hash=None,
                    topics_hash=None,
                )
            )
            reset_count = result.rowcount
            await db.commit()
        logger.info(f"Full refresh: Reset {reset_count} repos")

        await _update_main_task(
            error_details={"phase": "reset", "reset_count": reset_count},
            processed_items=1,
        )

        # Step 2: Create and run star sync task (full mode)
        logger.info(f"Full refresh: Starting full star sync for user {user_id}")
        stars_task_id = await _create_sub_task("stars")
        await sync_stars_task(user_id, stars_task_id, "full")

        await _update_main_task(
            error_details={"phase": "stars", "reset_count": reset_count},
            processed_items=2,
        )

        # Step 3: Create and run embedding task (includes LLM summaries)
        logger.info(f"Full refresh: Computing embeddings for user {user_id}")
        embed_task_id = await _create_sub_task("embedding")
        await compute_embeddings_task(user_id, embed_task_id)

        await _update_main_task(
            error_details={"phase": "embeddings", "reset_count": reset_count},
            processed_items=3,
        )

        # Step 4: Create and run clustering task
        logger.info(f"Full refresh: Running clustering for user {user_id}")
        await _update_main_task(
            error_details={"phase": "clustering", "reset_count": reset_count},
            processed_items=3,
        )
        cluster_task_id = await _create_sub_task("cluster")
        await run_clustering_task(user_id, cluster_task_id, use_llm=True)

        await _update_main_task(
            status="completed",
            completed_at=datetime.utcnow(),
            processed_items=4,
            error_details={
                "phase": "complete",
                "reset_count": reset_count,
                "steps_completed": ["reset", "stars", "embeddings", "clustering"],
            },
        )
        logger.info(f"Full refresh completed for user {user_id}")

    except Exception as e:
        logger.exception(f"Full refresh failed for user {user_id}: {e}")
        await _update_main_task(
            status="failed",
            error_message=str(e),
            completed_at=datetime.utcnow(),
        )


# ==================== Schedule Management ====================


def calculate_next_run_time(schedule: SyncSchedule) -> datetime | None:
    """Calculate the next scheduled run time for a sync schedule.

    Args:
        schedule: The sync schedule configuration.

    Returns:
        The next run datetime in UTC, or None if schedule is disabled.
    """
    if not schedule.is_enabled:
        return None

    try:
        tz = ZoneInfo(schedule.timezone)
        now_utc = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
        now_local = now_utc.astimezone(tz)

        # Create today's scheduled time in user's timezone
        scheduled_today = now_local.replace(
            hour=schedule.schedule_hour,
            minute=schedule.schedule_minute,
            second=0,
            microsecond=0,
        )

        # If we've passed today's time, schedule for tomorrow
        if now_local >= scheduled_today:
            from datetime import timedelta

            scheduled_today += timedelta(days=1)

        # Convert back to UTC
        return scheduled_today.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)

    except Exception as e:
        logger.warning(f"Error calculating next run time: {e}")
        return None


@router.get("/schedule", response_model=ScheduleResponse)
async def get_schedule(
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Get current sync schedule configuration.

    Returns the user's scheduled sync settings including:
    - Whether scheduled sync is enabled
    - Scheduled time (hour, minute)
    - Timezone
    - Last run status
    - Next scheduled run time

    Args:
        db: Database session

    Returns:
        Schedule configuration
    """
    user = await get_default_user(db)

    result = await db.execute(
        select(SyncSchedule).where(SyncSchedule.user_id == user.id)
    )
    schedule = result.scalar_one_or_none()

    if not schedule:
        # Return default configuration
        return ScheduleResponse(
            is_enabled=False,
            schedule_hour=9,
            schedule_minute=0,
            timezone="Asia/Shanghai",
            last_run_at=None,
            last_run_status=None,
            last_run_error=None,
            next_run_at=None,
        )

    # Calculate next run time
    next_run = calculate_next_run_time(schedule)

    return ScheduleResponse(
        is_enabled=schedule.is_enabled,
        schedule_hour=schedule.schedule_hour,
        schedule_minute=schedule.schedule_minute,
        timezone=schedule.timezone,
        last_run_at=schedule.last_run_at,
        last_run_status=schedule.last_run_status,
        last_run_error=schedule.last_run_error,
        next_run_at=next_run,
    )


@router.post("/schedule", response_model=ScheduleResponse)
async def update_schedule(
    config: ScheduleConfig,
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Update sync schedule configuration.

    Creates or updates the user's scheduled sync settings.

    Args:
        config: New schedule configuration
        db: Database session

    Returns:
        Updated schedule configuration
    """
    user = await get_default_user(db)

    result = await db.execute(
        select(SyncSchedule).where(SyncSchedule.user_id == user.id)
    )
    schedule = result.scalar_one_or_none()

    if not schedule:
        # Create new schedule
        schedule = SyncSchedule(user_id=user.id)
        db.add(schedule)

    # Update fields
    schedule.is_enabled = config.is_enabled
    schedule.schedule_hour = config.schedule_hour
    schedule.schedule_minute = config.schedule_minute
    schedule.timezone = config.timezone

    await db.commit()
    await db.refresh(schedule)

    # Calculate next run time
    next_run = calculate_next_run_time(schedule)

    logger.info(
        f"Schedule updated for user {user.id}: "
        f"enabled={config.is_enabled}, time={config.schedule_hour}:{config.schedule_minute:02d} {config.timezone}"
    )

    return ScheduleResponse(
        is_enabled=schedule.is_enabled,
        schedule_hour=schedule.schedule_hour,
        schedule_minute=schedule.schedule_minute,
        timezone=schedule.timezone,
        last_run_at=schedule.last_run_at,
        last_run_status=schedule.last_run_status,
        last_run_error=schedule.last_run_error,
        next_run_at=next_run,
    )


@router.post("/full-refresh", response_model=FullRefreshResponse)
async def trigger_full_refresh(
    payload: FullRefreshRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Trigger a full refresh of all repositories.

    This will:
    1. Reset all processing flags for user's repos
    2. Re-fetch all starred repos from GitHub
    3. Regenerate all AI summaries and tags
    4. Recompute all embeddings
    5. Re-run clustering

    WARNING: This is a resource-intensive operation that may take
    a long time and consume API quota.

    Args:
        background_tasks: FastAPI background tasks
        db: Database session

    Returns:
        Task information for tracking progress
    """
    _validate_full_refresh_confirmation(payload.confirm)

    user = await get_default_user(db)

    # Check for existing running full refresh task
    result = await db.execute(
        select(SyncTask).where(
            SyncTask.user_id == user.id,
            SyncTask.task_type == "full_refresh",
            SyncTask.status.in_(["pending", "running"]),
        )
    )
    existing = result.scalar_one_or_none()

    if existing:
        return FullRefreshResponse(
            task_id=existing.id,
            message="Full refresh already in progress",
            reset_count=0,
        )

    # Count repos that will be reset
    count_result = await db.execute(
        select(func.count(StarredRepo.id)).where(StarredRepo.user_id == user.id)
    )
    repo_count = count_result.scalar() or 0

    # Create new task
    task = SyncTask(
        user_id=user.id,
        task_type="full_refresh",
        status="pending",
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)

    # Start background task
    background_tasks.add_task(full_refresh_task, user.id, task.id)

    logger.info(
        f"Full refresh started for user {user.id}, {repo_count} repos will be reset"
    )

    return FullRefreshResponse(
        task_id=task.id,
        message=f"Full refresh started. {repo_count} repositories will be reprocessed.",
        reset_count=repo_count,
    )


@router.get("/info", response_model=SyncInfoResponse)
async def get_sync_info(
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Get comprehensive sync status information.

    Returns statistics about the user's repositories and sync status:
    - Last sync time
    - Repository counts (total, embedded, summarized)
    - Current schedule configuration

    Args:
        db: Database session

    Returns:
        Sync status information
    """
    user = await get_default_user(db)
    app_settings = get_app_settings()

    # Get repository statistics
    stats_result = await db.execute(
        select(
            func.count(StarredRepo.id).label("total"),
            func.count(StarredRepo.id)
            .filter(StarredRepo.is_embedded == True)  # noqa: E712
            .label("embedded"),
            func.count(StarredRepo.id)
            .filter(StarredRepo.is_summarized == True)  # noqa: E712
            .label("summarized"),
        ).where(StarredRepo.user_id == user.id)
    )
    stats = stats_result.one()

    # Get schedule
    schedule_result = await db.execute(
        select(SyncSchedule).where(SyncSchedule.user_id == user.id)
    )
    schedule = schedule_result.scalar_one_or_none()

    schedule_response = None
    if schedule:
        next_run = calculate_next_run_time(schedule)
        schedule_response = ScheduleResponse(
            is_enabled=schedule.is_enabled,
            schedule_hour=schedule.schedule_hour,
            schedule_minute=schedule.schedule_minute,
            timezone=schedule.timezone,
            last_run_at=schedule.last_run_at,
            last_run_status=schedule.last_run_status,
            last_run_error=schedule.last_run_error,
            next_run_at=next_run,
        )

    return SyncInfoResponse(
        last_sync_at=user.last_sync_at,
        github_token_configured=bool(app_settings.github_token),
        single_user_mode=app_settings.single_user_mode,
        total_repos=stats.total or 0,
        synced_repos=user.synced_stars or 0,
        embedded_repos=stats.embedded or 0,
        summarized_repos=stats.summarized or 0,
        schedule=schedule_response,
    )
