"""Clustering execution for the sync pipeline."""

import math
from datetime import datetime, timezone

from sqlalchemy import delete, func, select, update

from nebula.db import Cluster, StarredRepo, SyncTask
from nebula.utils import get_logger

logger = get_logger(__name__)


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

            # ---- Phase 1: compute everything, write nothing --------------
            # Cluster naming calls an LLM, which can take minutes. Doing that
            # after the destructive delete (as this task used to) meant a crash
            # or lease loss in that window left the user with zero clusters and
            # every repo unassigned, with no automatic recovery.
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

            # ---- Phase 2: swap in one transaction ------------------------
            # Detach every repo from its cluster first so the
            # starred_repos.cluster_id -> clusters.id foreign key stays
            # satisfied through the delete. Both statements are set-based; the
            # previous per-row ORM delete issued one child-nullification query
            # per cluster.
            await db.execute(
                update(StarredRepo)
                .where(StarredRepo.user_id == user_id)
                .values(cluster_id=None)
                .execution_options(synchronize_session=False)
            )
            # The bulk UPDATE bypasses the identity map, so realign the loaded
            # objects before assigning new ids. Without this, an assignment
            # that happens to match the stale in-session value would emit no
            # UPDATE and silently leave the row NULL.
            for repo in repos_with_embeddings:
                repo.cluster_id = None

            await db.execute(delete(Cluster).where(Cluster.user_id == user_id))

            cluster_map: dict[int, Cluster] = {}
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
                cluster_map[entry["cluster_id"]] = cluster
            await db.flush()

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

