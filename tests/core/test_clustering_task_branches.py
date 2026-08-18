"""Branch coverage for `run_clustering_task` early returns and incremental mode.

Complements `test_cluster_swap_atomicity.py`, which covers the full-recluster
swap. These tests cover the guards that decide whether a swap happens at all.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from nebula.application.services import sync_execution_service


def _repo(repo_id: int, *, embedding, cluster_id=None, coords=(None, None, None)):
    return SimpleNamespace(
        id=repo_id,
        user_id=7,
        full_name=f"owner/repo-{repo_id}",
        description=f"description {repo_id}",
        topics=["topic"],
        language="Python",
        stargazers_count=10,
        embedding=embedding,
        cluster_id=cluster_id,
        coord_x=coords[0],
        coord_y=coords[1],
        coord_z=coords[2],
    )


def _task():
    return SimpleNamespace(
        id=1,
        user_id=7,
        task_type="cluster",
        status="pending",
        started_at=None,
        completed_at=None,
        error_message=None,
        error_details=None,
        total_items=0,
        processed_items=0,
        failed_items=0,
    )


class _Result:
    def __init__(self, rows, scalar_value=None):
        self._rows = rows
        self._scalar_value = scalar_value

    def scalars(self):
        return SimpleNamespace(all=lambda: list(self._rows))

    def scalar(self):
        return self._scalar_value if self._scalar_value is not None else len(self._rows)


class _FakeDb:
    def __init__(self, state):
        self.state = state
        self.commits = 0
        self.statements: list[str] = []
        self.added: list[object] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, _tb):
        return False

    async def get(self, model, obj_id, **_kwargs):
        if model.__name__ == "SyncTask":
            return self.state.task if self.state.task.id == obj_id else None
        if model.__name__ == "Cluster":
            return self.state.clusters.get(obj_id)
        return None

    async def execute(self, statement):
        text = str(statement).lower()
        self.statements.append(text)
        if text.startswith("delete") or text.startswith("update"):
            return _Result([])
        if "count(" in text:
            return _Result([], scalar_value=self.state.cluster_repo_count)
        return _Result(self.state.repos)

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        return None

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        return None


class _State:
    def __init__(self, repos, task, clusters=None, cluster_repo_count=3):
        self.repos = repos
        self.task = task
        self.clusters = clusters or {}
        self.cluster_repo_count = cluster_repo_count


def _install(monkeypatch, state):
    db = _FakeDb(state)
    monkeypatch.setattr("nebula.db.database.get_db_context", lambda: db, raising=False)
    return db


@pytest.mark.asyncio
async def test_no_embedded_repos_completes_with_message(monkeypatch):
    task = _task()
    state = _State([], task)
    _install(monkeypatch, state)

    await sync_execution_service.run_clustering_task(user_id=7, task_id=1)

    assert task.status == "completed"
    assert task.error_message == "No embedded repos found"


@pytest.mark.asyncio
async def test_fewer_than_five_embeddings_completes_without_clustering(monkeypatch):
    repos = [_repo(index, embedding=[0.1, 0.2]) for index in range(4)]
    task = _task()
    state = _State(repos, task)
    db = _install(monkeypatch, state)

    await sync_execution_service.run_clustering_task(user_id=7, task_id=1)

    assert task.status == "completed"
    assert task.error_message == "Not enough embedded repos for clustering (min 5)"
    # No destructive statement may run when clustering is skipped.
    assert not any(s.startswith("delete") for s in db.statements)


@pytest.mark.asyncio
async def test_repos_missing_embeddings_are_reset(monkeypatch):
    repos = [_repo(index, embedding=[0.1, 0.2]) for index in range(3)]
    orphan = _repo(99, embedding=None, cluster_id=5, coords=(1.0, 2.0, 3.0))
    repos.append(orphan)
    task = _task()
    state = _State(repos, task)
    _install(monkeypatch, state)

    await sync_execution_service.run_clustering_task(user_id=7, task_id=1)

    assert orphan.cluster_id is None
    assert orphan.coord_x is None
    assert orphan.coord_y is None
    assert orphan.coord_z is None


@pytest.mark.asyncio
async def test_incremental_with_no_new_repos_is_a_noop(monkeypatch):
    repos = [
        _repo(index, embedding=[0.1, 0.2], cluster_id=1, coords=(1.0, 2.0, 3.0))
        for index in range(6)
    ]
    task = _task()
    state = _State(repos, task)
    db = _install(monkeypatch, state)

    await sync_execution_service.run_clustering_task(
        user_id=7, task_id=1, incremental=True
    )

    assert task.status == "completed"
    assert task.processed_items == 0
    assert not any(s.startswith("delete") for s in db.statements)


@pytest.mark.asyncio
async def test_incremental_assigns_new_repos_to_existing_clusters(monkeypatch):
    positioned = [
        _repo(
            index,
            embedding=[1.0, 0.0] if index < 3 else [0.0, 1.0],
            cluster_id=1 if index < 3 else 2,
            coords=(float(index), float(index), float(index)),
        )
        for index in range(6)
    ]
    fresh = _repo(100, embedding=[1.0, 0.05])
    task = _task()
    state = _State(
        positioned + [fresh],
        task,
        clusters={
            1: SimpleNamespace(id=1, user_id=7, repo_count=0),
            2: SimpleNamespace(id=2, user_id=7, repo_count=0),
        },
        cluster_repo_count=4,
    )
    db = _install(monkeypatch, state)

    await sync_execution_service.run_clustering_task(
        user_id=7, task_id=1, incremental=True
    )

    assert task.status == "completed"
    assert task.processed_items == 1
    # The new repo lands in the cluster of its nearest neighbours, and gets
    # real coordinates rather than staying unpositioned.
    assert fresh.cluster_id == 1
    assert fresh.coord_x is not None
    # Incremental mode must never delete clusters.
    assert not any(s.startswith("delete") for s in db.statements)
    # Affected cluster repo_count is recomputed.
    assert state.clusters[1].repo_count == 4


@pytest.mark.asyncio
async def test_incremental_falls_back_to_full_when_nothing_is_positioned(monkeypatch):
    repos = [_repo(index, embedding=[float(index), 0.0]) for index in range(6)]
    task = _task()
    state = _State(repos, task)
    db = _install(monkeypatch, state)

    class _FakeClusterResult:
        labels = [0, 0, 0, 1, 1, 1]
        n_clusters = 2
        cluster_centers = {0: [0.0, 0.0, 0.0], 1: [1.0, 1.0, 1.0]}
        coords_3d = [[float(i)] * 3 for i in range(6)]

    class _FakeClusteringService:
        def __init__(self, **_kwargs):
            pass

        def fit_transform(self, **_kwargs):
            return _FakeClusterResult()

    import nebula.core.clustering as clustering

    monkeypatch.setattr(clustering, "ClusteringService", _FakeClusteringService)
    monkeypatch.setattr(
        clustering,
        "build_cluster_naming_inputs",
        lambda repos: (["a"], ["b"], ["c"], ["Python"]),
    )
    monkeypatch.setattr(
        clustering,
        "generate_cluster_name",
        lambda *_a, **_k: ("Name", "Description", ["kw"]),
    )
    monkeypatch.setattr(
        clustering, "deduplicate_cluster_entries", lambda entries: entries
    )

    await sync_execution_service.run_clustering_task(
        user_id=7, task_id=1, use_llm=False, incremental=True
    )

    # Fell through to the full path, which does run the swap.
    assert any(s.startswith("delete from clusters") for s in db.statements)
    assert task.status == "completed"


def test_normalize_embeddings_produces_unit_vectors():
    from nebula.core.clustering import normalize_embeddings

    normalized = normalize_embeddings(
        np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    )

    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)
