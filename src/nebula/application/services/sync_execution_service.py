"""Public sync execution API.

Task implementations live in `sync_stars_execution`, `sync_embeddings_execution`,
and `sync_clustering_execution`. This module stays the import and monkeypatch
surface: tests and pipeline code patch names here, and the split modules look
them up through `_facade()`.
"""

from nebula.core.config import get_app_settings, get_sync_settings
from nebula.core.embedding import get_embedding_service
from nebula.core.github_client import GitHubClient
from nebula.core.llm import get_llm_service

from .sync_clustering_execution import (
    derive_clustering_params_for_max_clusters,
    run_clustering_task,
    should_force_full_recluster,
)
from .sync_embeddings_execution import _embed_one_chunk, compute_embeddings_task
from .sync_execution_support import (
    collect_readme_targets,
    fetch_readmes_in_parallel,
    generate_repo_enhancements_in_parallel,
    log_task_stage,
    prefetch_existing_repos,
    sync_star_lists,
)
from .sync_stars_execution import sync_stars_task

__all__ = [
    "GitHubClient",
    "_embed_one_chunk",
    "collect_readme_targets",
    "compute_embeddings_task",
    "derive_clustering_params_for_max_clusters",
    "fetch_readmes_in_parallel",
    "generate_repo_enhancements_in_parallel",
    "get_app_settings",
    "get_embedding_service",
    "get_llm_service",
    "get_sync_settings",
    "log_task_stage",
    "prefetch_existing_repos",
    "run_clustering_task",
    "should_force_full_recluster",
    "sync_star_lists",
    "sync_stars_task",
]
