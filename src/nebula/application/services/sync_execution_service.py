"""Execution primitives for sync, embedding, and clustering."""

from datetime import datetime, timezone

from sqlalchemy import func, select

from nebula.core.config import get_app_settings, get_sync_settings
from nebula.core.embedding import get_embedding_service
from nebula.core.github_client import GitHubClient
from nebula.core.llm import get_llm_service
from nebula.db import Cluster, StarList, StarredRepo, SyncTask, User
from nebula.utils import compute_content_hash, compute_topics_hash, get_logger

logger = get_logger(__name__)


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
                    "(stopped at last_sync_at)"
                )

            task.total_items = len(repos)
            await db.commit()

            sync_settings = get_sync_settings()
            processed = 0
            failed = 0
            new_count = 0
            updated_count = 0

            # Batch-prefetch existing repos to avoid N+1 queries
            github_ids = [repo.id for repo in repos]
            existing_map: dict[int, StarredRepo] = {}
            batch_size_lookup = 500
            for i in range(0, len(github_ids), batch_size_lookup):
                batch_ids = github_ids[i : i + batch_size_lookup]
                batch_result = await db.execute(
                    select(StarredRepo).where(
                        StarredRepo.user_id == user_id,
                        StarredRepo.github_repo_id.in_(batch_ids),
                    )
                )
                for row in batch_result.scalars():
                    existing_map[row.github_repo_id] = row

            async with GitHubClient(
                access_token=settings.github_token
            ) as readme_client:
                for repo in repos:
                    try:
                        existing = existing_map.get(repo.id)

                        if existing:
                            new_desc_hash = compute_content_hash(repo.description)
                            new_topics_hash = compute_topics_hash(repo.topics)

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
                                latest_readme = await readme_client.get_repo_readme(
                                    repo.full_name,
                                    max_length=sync_settings.readme_max_length,
                                )
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
                            readme_content = await readme_client.get_repo_readme(
                                repo.full_name,
                                max_length=sync_settings.readme_max_length,
                            )
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
                                description_hash=compute_content_hash(repo.description),
                                topics_hash=compute_topics_hash(repo.topics),
                            )
                            db.add(new_repo)
                            new_count += 1

                        processed += 1
                        task.processed_items = processed

                        if processed % sync_settings.batch_size == 0:
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

            removed_count = 0

            if effective_mode == "full" and not was_truncated:
                github_repo_ids_from_api = {repo.id for repo in repos}
            elif sync_settings.detect_unstarred_on_incremental:
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
                "was_truncated": was_truncated,
            }
            await db.commit()

            logger.info(
                f"Completed star sync for {user.username} ({effective_mode}): "
                f"{new_count} new, {updated_count} updated, "
                f"{removed_count} removed, {failed} failed"
            )

            try:
                await sync_star_lists(user_id, settings.github_token, db)
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


async def compute_embeddings_task(user_id: int, task_id: int):
    """Background task to compute embeddings for repos."""
    from nebula.db.database import get_db_context

    async with get_db_context() as db:
        try:
            task = await db.get(SyncTask, task_id)
            if not task:
                return

            task.status = "running"
            task.started_at = datetime.now(timezone.utc)
            await db.commit()

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
                task.completed_at = datetime.now(timezone.utc)
                await db.commit()
                return

            llm_service = get_llm_service()
            repos_needing_llm = [
                repo for repo in repos if not repo.ai_summary or not repo.ai_tags
            ]

            if repos_needing_llm:
                total_llm_repos = len(repos_needing_llm)
                logger.info(
                    f"Generating summaries/tags for {total_llm_repos} repos "
                    "before embedding"
                )

                for index, repo in enumerate(repos_needing_llm, 1):
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
                    except Exception as exc:
                        logger.warning(
                            f"LLM generation failed for {repo.full_name}: {exc}"
                        )
                        if not repo.ai_tags:
                            repo.ai_tags = (
                                repo.topics[:5] if repo.topics else ["开源项目"]
                            )

                    if index % 5 == 0:
                        logger.info(
                            f"Generating summaries progress: {index}/{total_llm_repos}"
                        )
                        await db.commit()

                await db.commit()
                logger.info(f"LLM enhancement complete for {total_llm_repos} repos")

            embedding_service = get_embedding_service()
            processed = 0

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

            try:
                embeddings = await embedding_service.embed_batch(texts, batch_size=32)
                if len(embeddings) != len(repos):
                    raise ValueError(
                        f"Embedding count mismatch: got {len(embeddings)} "
                        f"for {len(repos)} repos"
                    )
                for repo, embedding in zip(repos, embeddings, strict=True):
                    repo.embedding = embedding
                    repo.is_embedded = True
                    processed += 1

                task.processed_items = processed
                task.status = "completed"
                await db.commit()
            except Exception as exc:
                logger.error(f"Batch embedding failed: {exc}")
                task.failed_items = len(repos)
                task.error_message = str(exc)
                task.status = "failed"
                await db.commit()

            task.completed_at = datetime.now(timezone.utc)
            await db.commit()

            logger.info(f"Completed embedding for user {user_id}: {processed} embedded")
        except Exception as exc:
            logger.exception(f"Embedding task failed: {exc}")

            async with get_db_context() as db:
                task = await db.get(SyncTask, task_id)
                if task:
                    task.status = "failed"
                    task.error_message = str(exc)
                    task.completed_at = datetime.now(timezone.utc)
                    await db.commit()


def derive_clustering_params_for_max_clusters(
    *,
    n_samples: int,
    max_clusters: int,
    min_clusters: int | None = None,
) -> dict:
    """Derive clustering parameters from max/min cluster knobs."""
    safe_max_clusters = max(2, min(int(max_clusters), 20))
    safe_min_clusters = max(2, safe_max_clusters // 3)
    if min_clusters is not None:
        safe_min_clusters = max(2, min(int(min_clusters), safe_max_clusters))

    approx_min_size = max(3, n_samples // 20)
    min_cluster_size = min(approx_min_size, 10)
    min_samples = max(1, min(3, min_cluster_size // 3))
    n_neighbors = int(max(15, min(60, 10 + (50 - 2 * safe_max_clusters))))

    return {
        "n_neighbors": n_neighbors,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "min_clusters": safe_min_clusters,
        "target_max_clusters": safe_max_clusters,
    }


def should_force_full_recluster(
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
    """Background task to run clustering on user's repos."""
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
            task.started_at = datetime.now(timezone.utc)
            await db.commit()

            result = await db.execute(
                select(StarredRepo).where(
                    StarredRepo.user_id == user_id,
                    StarredRepo.is_embedded == True,  # noqa: E712
                )
            )
            repos = result.scalars().all()

            if not repos:
                task.status = "completed"
                task.completed_at = datetime.now(timezone.utc)
                task.error_message = "No embedded repos found"
                await db.commit()
                return

            task.total_items = len(repos)
            await db.commit()

            logger.info(f"Running clustering on {len(repos)} repos for user {user_id}")

            repos_with_embeddings: list[StarredRepo] = []
            embeddings = []
            node_sizes = []
            for repo in repos:
                if repo.embedding is not None:
                    repos_with_embeddings.append(repo)
                    embeddings.append(repo.embedding)
                    import math

                    size = math.log10(max(repo.stargazers_count, 1) + 1) * 0.5 + 0.5
                    node_sizes.append(min(size, 3.0))

            if len(repos_with_embeddings) != len(repos):
                logger.warning(
                    "Skipping "
                    f"{len(repos) - len(repos_with_embeddings)} repos with missing "
                    "embeddings during clustering"
                )
                for repo in repos:
                    if repo.embedding is None:
                        repo.cluster_id = None
                        repo.coord_x = None
                        repo.coord_y = None
                        repo.coord_z = None

            if len(embeddings) < 5:
                task.status = "completed"
                task.completed_at = datetime.now(timezone.utc)
                task.error_message = "Not enough embedded repos for clustering (min 5)"
                await db.commit()
                return

            if incremental:
                import numpy as np

                existing_repos = []
                existing_embs = []
                existing_crds = []
                new_repos = []
                new_embs = []
                new_sizes = []
                existing_cluster_ids: list[int] = []

                for index, repo in enumerate(repos_with_embeddings):
                    has_position = (
                        repo.coord_x is not None
                        and repo.coord_y is not None
                        and repo.coord_z is not None
                        and repo.cluster_id is not None
                    )
                    if has_position:
                        existing_repos.append(repo)
                        existing_embs.append(embeddings[index])
                        existing_crds.append([repo.coord_x, repo.coord_y, repo.coord_z])
                        existing_cluster_ids.append(repo.cluster_id)
                    else:
                        new_repos.append(repo)
                        new_embs.append(embeddings[index])
                        new_sizes.append(node_sizes[index])

                if not new_repos:
                    logger.info("Incremental mode: no new repos to assign")
                    task.status = "completed"
                    task.completed_at = datetime.now(timezone.utc)
                    task.processed_items = 0
                    await db.commit()
                    return

                if not existing_repos:
                    logger.info(
                        "Incremental mode: no existing repos with positions, "
                        "falling back to full clustering"
                    )
                    incremental = False
                else:
                    logger.info(
                        f"Incremental mode: {len(existing_repos)} existing repos, "
                        f"{len(new_repos)} new repos to assign"
                    )

                    existing_embs_arr = normalize_embeddings(
                        np.array(existing_embs, dtype=np.float32)
                    )
                    existing_crds_arr = np.array(existing_crds, dtype=np.float64)
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

                    for index, repo in enumerate(new_repos):
                        coords = result_incr.new_coords[index]
                        repo.coord_x = coords[0]
                        repo.coord_y = coords[1]
                        repo.coord_z = coords[2]
                        repo.cluster_id = int(result_incr.new_labels[index])

                    cluster_ids_to_update = {
                        int(label) for label in result_incr.new_labels
                    }
                    for cluster_id in cluster_ids_to_update:
                        cluster_obj = await db.get(Cluster, cluster_id)
                        if cluster_obj:
                            count_result = await db.execute(
                                select(func.count(StarredRepo.id)).where(
                                    StarredRepo.user_id == user_id,
                                    StarredRepo.cluster_id == cluster_id,
                                )
                            )
                            cluster_obj.repo_count = int(count_result.scalar() or 0)

                    task.processed_items = len(new_repos)
                    task.status = "completed"
                    task.completed_at = datetime.now(timezone.utc)
                    await db.commit()

                    logger.info(
                        f"Incremental clustering completed for user {user_id}: "
                        f"{len(new_repos)} new repos assigned"
                    )
                    return

            derived = derive_clustering_params_for_max_clusters(
                n_samples=len(repos_with_embeddings),
                max_clusters=max_clusters,
                min_clusters=min_clusters,
            )
            logger.info(
                f"Clustering params: max_clusters={max_clusters}, "
                f"min_clusters={min_clusters}, "
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

            existing_clusters = (
                (await db.execute(select(Cluster).where(Cluster.user_id == user_id)))
                .scalars()
                .all()
            )
            for cluster in existing_clusters:
                await db.delete(cluster)
            await db.commit()

            cluster_map: dict[int, Cluster] = {}
            cluster_entries: list[dict] = []
            sorted_cluster_ids = sorted(
                {cluster_id for cluster_id in cluster_result.labels if cluster_id != -1}
            )

            assigned_names: list[str] = []
            for cluster_id in sorted_cluster_ids:
                cluster_repo_indices = [
                    idx
                    for idx, label in enumerate(cluster_result.labels)
                    if label == cluster_id
                ]
                cluster_repos = [
                    repos_with_embeddings[idx] for idx in cluster_repo_indices
                ]
                if not cluster_repos:
                    continue

                repo_names, descriptions, topics, languages = (
                    build_cluster_naming_inputs(cluster_repos)
                )

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
                except Exception as exc:
                    logger.warning(f"Cluster naming failed: {exc}, using heuristic")
                    name, description, keywords = generate_cluster_name(
                        repo_names, descriptions, topics
                    )

                assigned_names.append(name)
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
                await db.flush()
                cluster_map[entry["cluster_id"]] = cluster

            await db.commit()

            assigned_count = 0
            unassigned_count = 0
            for index, repo in enumerate(repos_with_embeddings):
                if index < len(cluster_result.labels):
                    label = cluster_result.labels[index]
                    if label != -1 and label in cluster_map:
                        repo.cluster_id = cluster_map[label].id
                        assigned_count += 1
                    else:
                        repo.cluster_id = None
                        unassigned_count += 1

                    if index < len(cluster_result.coords_3d):
                        coords = cluster_result.coords_3d[index]
                        repo.coord_x = coords[0] if len(coords) > 0 else None
                        repo.coord_y = coords[1] if len(coords) > 1 else None
                        repo.coord_z = coords[2] if len(coords) > 2 else None

            task.processed_items = len(repos_with_embeddings)
            task.status = "completed"
            task.completed_at = datetime.now(timezone.utc)
            await db.commit()

            logger.info(
                f"Clustering completed for user {user_id}: {cluster_result.n_clusters} "
                f"clusters, {assigned_count} assigned, {unassigned_count} unassigned"
            )

        except Exception as exc:
            logger.exception(f"Clustering task failed: {exc}")

            async with get_db_context() as db:
                task = await db.get(SyncTask, task_id)
                if task:
                    task.status = "failed"
                    task.error_message = str(exc)
                    task.completed_at = datetime.now(timezone.utc)
                    await db.commit()
