"""End-to-end pipeline against a real PostgreSQL + pgvector database.

GitHub, embedding, and LLM providers are faked; every database interaction is
real. This is the only place where transaction boundaries, foreign keys, and
the advisory lock actually execute.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import func, select

from nebula.application.services import sync_execution_service
from nebula.application.services.pipeline_service import SyncPipelineService
from nebula.core.config import get_embedding_settings
from nebula.db import Cluster, GraphSnapshot, PipelineRun, StarredRepo, SyncTask, User

from .fakes import (
    FakeEmbeddingService,
    FakeGitHubClient,
    FakeLLMService,
    make_github_repo,
)


@pytest.fixture
def wired_providers(monkeypatch, bound_session_factory):
    """Install fakes for the three network boundaries."""
    dimensions = get_embedding_settings().dimensions
    repos = [make_github_repo(index) for index in range(1, 13)]
    FakeGitHubClient.repos = repos
    FakeGitHubClient.truncated = False
    FakeGitHubClient.star_lists = []
    FakeGitHubClient.readmes = {}

    groups = {
        f"text::owner/repo-{index}": 0 if index <= 6 else 1 for index in range(1, 13)
    }
    embedding = FakeEmbeddingService(dimensions, groups)
    llm = FakeLLMService()

    monkeypatch.setattr(sync_execution_service, "GitHubClient", FakeGitHubClient)
    monkeypatch.setattr(
        sync_execution_service, "get_embedding_service", lambda: embedding
    )
    monkeypatch.setattr(sync_execution_service, "get_llm_service", lambda: llm)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_integration")

    from nebula.core.config import get_app_settings, get_sync_settings

    get_app_settings.cache_clear()
    get_sync_settings.cache_clear()
    yield {"embedding": embedding, "llm": llm, "repos": repos}
    get_app_settings.cache_clear()
    get_sync_settings.cache_clear()


async def _seed_user(db) -> User:
    user = User(github_id=1, username="integration-user", total_stars=0, synced_stars=0)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@pytest.mark.asyncio
async def test_full_pipeline_produces_an_activated_snapshot(
    integration_db, wired_providers
):
    user = await _seed_user(integration_db)
    service = SyncPipelineService()

    run_id = await service.start_pipeline(user.id, mode="full", use_llm=False)

    run = await integration_db.get(PipelineRun, run_id)
    await integration_db.refresh(run)
    assert run.status in {"completed", "partial_failed"}, run.last_error
    assert run.phase == "completed"

    # One task per phase, all terminal.
    tasks = (
        (
            await integration_db.execute(
                select(SyncTask).where(SyncTask.pipeline_run_id == run_id)
            )
        )
        .scalars()
        .all()
    )
    assert {task.task_type for task in tasks} == {"stars", "embedding", "cluster"}
    assert all(task.status in {"completed", "failed"} for task in tasks)

    # Repos landed and were embedded.
    embedded = await integration_db.scalar(
        select(func.count(StarredRepo.id)).where(
            StarredRepo.user_id == user.id,
            StarredRepo.is_embedded.is_(True),
        )
    )
    assert embedded == 12

    # A snapshot was built and activated.
    await integration_db.refresh(user)
    assert user.active_graph_snapshot_id is not None
    snapshot = await integration_db.get(GraphSnapshot, user.active_graph_snapshot_id)
    assert snapshot.status == "active"
    assert snapshot.meta["total_nodes"] == 12


@pytest.mark.asyncio
async def test_embedding_resumes_after_a_failed_chunk(
    integration_db, wired_providers, monkeypatch
):
    """A failed chunk must not discard the chunks around it."""
    monkeypatch.setenv("SYNC_BATCH_SIZE", "10")
    from nebula.core.config import get_sync_settings

    get_sync_settings.cache_clear()

    user = await _seed_user(integration_db)
    task = SyncTask(user_id=user.id, task_type="stars", status="pending", phase="stars")
    integration_db.add(task)
    await integration_db.commit()
    await integration_db.refresh(task)

    await sync_execution_service.sync_stars_task(user.id, task.id, "full")

    embed_task = SyncTask(
        user_id=user.id, task_type="embedding", status="pending", phase="embedding"
    )
    integration_db.add(embed_task)
    await integration_db.commit()
    await integration_db.refresh(embed_task)

    # 12 repos at chunk size 10 -> two chunks; fail the first.
    wired_providers["embedding"].fail_on_calls = {1}
    await sync_execution_service.compute_embeddings_task(user.id, embed_task.id)

    await integration_db.refresh(embed_task)
    embedded = await integration_db.scalar(
        select(func.count(StarredRepo.id)).where(
            StarredRepo.user_id == user.id, StarredRepo.is_embedded.is_(True)
        )
    )
    # Second chunk committed even though the first failed.
    assert embedded == 2
    assert embed_task.failed_items == 10
    assert embed_task.status == "completed"

    # Re-running embeds only the remainder.
    wired_providers["embedding"].fail_on_calls = set()
    rerun = SyncTask(
        user_id=user.id, task_type="embedding", status="pending", phase="embedding"
    )
    integration_db.add(rerun)
    await integration_db.commit()
    await integration_db.refresh(rerun)

    await sync_execution_service.compute_embeddings_task(user.id, rerun.id)
    await integration_db.refresh(rerun)

    assert rerun.total_items == 10
    embedded_after = await integration_db.scalar(
        select(func.count(StarredRepo.id)).where(
            StarredRepo.user_id == user.id, StarredRepo.is_embedded.is_(True)
        )
    )
    assert embedded_after == 12

    get_sync_settings.cache_clear()


@pytest.mark.asyncio
async def test_pipeline_creation_is_serialized_by_the_advisory_lock(
    integration_db, wired_providers
):
    """`pg_advisory_xact_lock` never executes off PostgreSQL, so this is the
    only place its serialization guarantee is exercised."""
    user = await _seed_user(integration_db)
    service = SyncPipelineService()

    results = await asyncio.gather(
        service.create_pipeline_run(user.id),
        service.create_pipeline_run(user.id),
        return_exceptions=True,
    )

    succeeded = [r for r in results if isinstance(r, int)]
    rejected = [r for r in results if isinstance(r, ValueError)]

    assert len(succeeded) == 1
    assert len(rejected) == 1
    assert "already running" in str(rejected[0])


@pytest.mark.asyncio
async def test_clusters_and_repos_satisfy_the_foreign_key_after_a_swap(
    integration_db, wired_providers
):
    user = await _seed_user(integration_db)
    service = SyncPipelineService()
    await service.start_pipeline(user.id, mode="full", use_llm=False)

    cluster_ids = set(
        (
            await integration_db.execute(
                select(Cluster.id).where(Cluster.user_id == user.id)
            )
        )
        .scalars()
        .all()
    )
    repo_cluster_ids = (
        (
            await integration_db.execute(
                select(StarredRepo.cluster_id).where(StarredRepo.user_id == user.id)
            )
        )
        .scalars()
        .all()
    )

    dangling = [
        cid for cid in repo_cluster_ids if cid is not None and cid not in cluster_ids
    ]
    assert dangling == []
    assert cluster_ids, "clustering produced no clusters"
