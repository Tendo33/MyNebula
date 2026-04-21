"""Shared helpers for sync execution tasks."""

from __future__ import annotations

import asyncio
from time import perf_counter

from sqlalchemy import select

from nebula.core.github_client import GitHubClient
from nebula.db import StarList, StarredRepo
from nebula.utils import get_logger

logger = get_logger(__name__)


def log_task_stage(
    task_name: str,
    stage: str,
    started: float,
    **metrics: int | str | bool | float,
) -> None:
    """Emit a structured stage log with elapsed timing."""
    elapsed_ms = (perf_counter() - started) * 1000
    logger.info(
        f"[TASK STAGE] {task_name} "
        f"stage={stage} elapsed_ms={elapsed_ms:.1f} metrics={metrics}"
    )


async def fetch_readmes_in_parallel(
    client: GitHubClient,
    repo_names: list[str],
    *,
    max_length: int,
    concurrency: int,
) -> dict[str, str | None]:
    """Fetch repository READMEs under a bounded concurrency limit."""
    semaphore = asyncio.Semaphore(concurrency)
    readmes: dict[str, str | None] = {}

    async def fetch_one(full_name: str) -> None:
        async with semaphore:
            readmes[full_name] = await client.get_repo_readme(
                full_name,
                max_length=max_length,
            )

    await asyncio.gather(*(fetch_one(full_name) for full_name in repo_names))
    return readmes


async def generate_repo_enhancements_in_parallel(
    llm_service,
    repos: list,
    *,
    concurrency: int,
) -> dict[int, tuple[str | None, list[str] | None, Exception | None]]:
    """Generate summary/tag enhancements for repos with bounded concurrency."""
    semaphore = asyncio.Semaphore(concurrency)
    results: dict[int, tuple[str | None, list[str] | None, Exception | None]] = {}

    async def enhance(repo) -> None:
        async with semaphore:
            try:
                summary, tags = await llm_service.generate_repo_summary_and_tags(
                    full_name=repo.full_name,
                    description=repo.description,
                    topics=repo.topics,
                    language=repo.language,
                    readme_content=repo.readme_content,
                )
                results[repo.id] = (summary, tags, None)
            except Exception as exc:  # noqa: BLE001
                results[repo.id] = (None, None, exc)

    await asyncio.gather(*(enhance(repo) for repo in repos))
    return results


async def sync_star_lists(
    user_id: int,
    github_token: str,
    db,
) -> None:
    """Sync user's GitHub star lists and repo list assignment."""
    try:
        async with GitHubClient(access_token=github_token) as client:
            star_lists = await client.get_star_lists()

        if not star_lists:
            logger.info(f"No star lists found for user {user_id}")
            return

        repo_to_list_map: dict[int, int] = {}

        for gh_list in star_lists:
            result = await db.execute(
                select(StarList).where(
                    StarList.user_id == user_id,
                    StarList.github_list_id == gh_list.id,
                )
            )
            existing_list = result.scalar_one_or_none()

            if existing_list:
                existing_list.name = gh_list.name
                existing_list.description = gh_list.description
                existing_list.is_public = gh_list.is_public
                existing_list.repo_count = gh_list.repos_count
                db_list = existing_list
            else:
                db_list = StarList(
                    user_id=user_id,
                    github_list_id=gh_list.id,
                    name=gh_list.name,
                    description=gh_list.description,
                    is_public=gh_list.is_public,
                    repo_count=gh_list.repos_count,
                )
                db.add(db_list)
                await db.flush()

            for repo_id in gh_list.repo_ids:
                repo_to_list_map[repo_id] = db_list.id

        await db.commit()

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

    except Exception as exc:
        logger.warning(f"Failed to sync star lists: {exc}")
        raise
