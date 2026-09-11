"""Star fetch/persist execution for the sync pipeline."""

from datetime import datetime, timezone
from time import perf_counter

from sqlalchemy import func, select

from nebula.db import StarredRepo, SyncTask, User
from nebula.utils import get_logger

from .sync_execution_support import collect_readme_targets, log_task_stage

logger = get_logger(__name__)


def _facade():
    from nebula.application.services import sync_execution_service as module

    return module


async def sync_stars_task(
    user_id: int,
    task_id: int,
    sync_mode: str = "incremental",
):
    """Background task to sync GitHub stars."""
    logger.info(
        f"[TASK START] sync_stars_task called: user={user_id}, "
        f"task={task_id}, mode={sync_mode}"
    )
    from nebula.db.database import get_db_context

    async with get_db_context() as db:
        try:
            user = await db.get(User, user_id)
            task = await db.get(SyncTask, task_id)

            if not user or not task:
                logger.error(f"User or task not found: user={user_id}, task={task_id}")
                return

            task.status = "running"
            task.started_at = datetime.now(timezone.utc)
            await db.commit()

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

            task.error_details = {"sync_mode": effective_mode}
            await db.commit()

            settings = _facade().get_app_settings()
            sync_settings = _facade().get_sync_settings()
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
            fetch_started = perf_counter()
            try:
                async with _facade().GitHubClient(access_token=settings.github_token) as client:
                    repos, was_truncated = await client.get_starred_repos(
                        stop_before=stop_before
                    )
            except Exception as api_error:
                logger.exception(f"[TASK ERROR] GitHub API call failed: {api_error}")
                task.status = "failed"
                task.error_message = f"GitHub API error: {api_error}"
                await db.commit()
                return
            log_task_stage(
                "sync_stars_task",
                "fetch_stars",
                fetch_started,
                repos=len(repos),
                was_truncated=was_truncated,
                mode=effective_mode,
            )

            logger.info(f"[TASK PROGRESS] Fetched {len(repos)} repos from GitHub")

            if was_truncated:
                logger.info(
                    f"Incremental sync: fetched {len(repos)} new repos "
                    "(stopped at last_sync_at)"
                )

            task.total_items = len(repos)
            await db.commit()

            processed = 0
            failed = 0
            new_count = 0
            updated_count = 0

            prefetch_started = perf_counter()
            github_ids = [repo.id for repo in repos]
            existing_map = await _facade().prefetch_existing_repos(
                db,
                user_id=user_id,
                github_ids=github_ids,
            )
            log_task_stage(
                "sync_stars_task",
                "prefetch_existing",
                prefetch_started,
                github_ids=len(github_ids),
                existing=len(existing_map),
            )

            readme_targets, repo_hashes = collect_readme_targets(repos, existing_map)

            readmes_by_repo: dict[str, str | None] = {}
            if readme_targets:
                readme_started = perf_counter()
                async with _facade().GitHubClient(
                    access_token=settings.github_token
                ) as readme_client:
                    readmes_by_repo = await _facade().fetch_readmes_in_parallel(
                        readme_client,
                        readme_targets,
                        max_length=sync_settings.readme_max_length,
                        concurrency=sync_settings.readme_fetch_concurrency,
                    )
                log_task_stage(
                    "sync_stars_task",
                    "fetch_readmes",
                    readme_started,
                    requested=len(readme_targets),
                    fetched=sum(
                        1 for content in readmes_by_repo.values() if content is not None
                    ),
                    concurrency=sync_settings.readme_fetch_concurrency,
                )

            persist_started = perf_counter()
            for repo in repos:
                try:
                    existing = existing_map.get(repo.id)
                    new_desc_hash, new_topics_hash = repo_hashes[repo.id]

                    if existing:
                        needs_reprocess = (
                            existing.description_hash != new_desc_hash
                            or existing.topics_hash != new_topics_hash
                        )

                        if needs_reprocess:
                            existing.is_embedded = False
                            existing.is_summarized = False
                            existing.ai_summary = None
                            existing.ai_tags = None
                            existing.embedding = None
                            latest_readme = readmes_by_repo.get(repo.full_name)
                            if latest_readme is not None:
                                existing.readme_content = latest_readme
                                existing.is_readme_fetched = True
                            logger.info(
                                f"Repo {repo.full_name} content changed, "
                                "marked for reprocessing"
                            )

                        existing.description = repo.description
                        existing.language = repo.language
                        existing.topics = repo.topics
                        existing.stargazers_count = repo.stargazers_count
                        existing.forks_count = repo.forks_count
                        existing.repo_updated_at = repo.updated_at
                        existing.repo_pushed_at = repo.pushed_at
                        existing.owner_avatar_url = repo.owner_avatar_url
                        existing.description_hash = new_desc_hash
                        existing.topics_hash = new_topics_hash
                        updated_count += 1
                    else:
                        readme_content = readmes_by_repo.get(repo.full_name)
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
                            description_hash=new_desc_hash,
                            topics_hash=new_topics_hash,
                        )
                        db.add(new_repo)
                        new_count += 1

                    processed += 1
                    task.processed_items = processed

                    if processed % sync_settings.progress_commit_interval == 0:
                        await db.commit()
                        logger.info(
                            f"Synced {processed}/{len(repos)} repos "
                            f"for user {user.username}"
                        )

                except Exception as exc:
                    logger.warning(f"Failed to sync repo {repo.full_name}: {exc}")
                    failed += 1
                    task.failed_items = failed

            await db.commit()
            log_task_stage(
                "sync_stars_task",
                "persist_repos",
                persist_started,
                processed=processed,
                failed=failed,
                new_repos=new_count,
                updated_repos=updated_count,
                commit_interval=sync_settings.progress_commit_interval,
            )

            removed_count = 0

            if effective_mode == "full" and not was_truncated:
                github_repo_ids_from_api = {repo.id for repo in repos}
            elif sync_settings.detect_unstarred_on_incremental:
                logger.info(
                    "Incremental deletion detection enabled: fetching full starred list..."
                )
                try:
                    async with _facade().GitHubClient(
                        access_token=settings.github_token
                    ) as client:
                        all_repos, _ = await client.get_starred_repos()
                    github_repo_ids_from_api = {repo.id for repo in all_repos}
                    logger.info(
                        f"Fetched {len(github_repo_ids_from_api)} starred repo IDs "
                        "for deletion check"
                    )
                except Exception as exc:
                    logger.warning(
                        f"Failed to fetch complete starred list for deletion: {exc}. "
                        "Skipping deletion detection."
                    )
                    github_repo_ids_from_api = None
            else:
                github_repo_ids_from_api = None
                logger.info(
                    "Skipping incremental unstarred deletion detection "
                    "(SYNC_DETECT_UNSTARRED_ON_INCREMENTAL=false)"
                )

            if github_repo_ids_from_api is not None:
                deletion_started = perf_counter()
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
                log_task_stage(
                    "sync_stars_task",
                    "remove_unstarred",
                    deletion_started,
                    removed=removed_count,
                    compared=len(github_repo_ids_from_api),
                )

            if effective_mode == "incremental":
                result = await db.execute(
                    select(func.count(StarredRepo.id)).where(
                        StarredRepo.user_id == user_id
                    )
                )
                total_db_repos = int(result.scalar() or 0)
                user.total_stars = total_db_repos
                user.synced_stars = total_db_repos
            else:
                user.total_stars = len(repos)
                user.synced_stars = processed

            user.last_sync_at = datetime.now(timezone.utc)

            task.status = "completed"
            task.completed_at = datetime.now(timezone.utc)
            task.error_details = {
                "sync_mode": effective_mode,
                "new_repos": new_count,
                "updated_repos": updated_count,
                "removed_repos": removed_count,
                "failed_items": failed,
                "was_truncated": was_truncated,
            }
            await db.commit()

            logger.info(
                f"Completed star sync for {user.username} ({effective_mode}): "
                f"{new_count} new, {updated_count} updated, "
                f"{removed_count} removed, {failed} failed"
            )

            star_lists_started = perf_counter()
            try:
                await _facade().sync_star_lists(user_id, settings.github_token, db)
                log_task_stage(
                    "sync_stars_task",
                    "sync_star_lists",
                    star_lists_started,
                    user_id=user_id,
                )
            except Exception as exc:
                logger.warning(f"Star lists sync failed (non-critical): {exc}")

        except Exception as exc:
            logger.exception(f"Star sync failed for user {user_id}: {exc}")

            async with get_db_context() as db:
                task = await db.get(SyncTask, task_id)
                if task:
                    task.status = "failed"
                    task.error_message = str(exc)
                    task.completed_at = datetime.now(timezone.utc)
                    await db.commit()


