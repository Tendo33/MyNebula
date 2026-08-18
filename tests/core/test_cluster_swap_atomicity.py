"""Tests for the atomic cluster swap in `run_clustering_task`.

Regression guard for the non-atomic swap found in the 2026-08-18 scan: the task
used to delete every cluster and commit, then perform LLM naming (minutes of
network latency), then insert and reassign. An interruption in that window left
the user with zero clusters and every repo unassigned, with no recovery path.
"""

from types import SimpleNamespace

import pytest

from nebula.application.services import sync_execution_service


def _make_repo(repo_id: int, cluster_id: int | None, dimensions: int = 4):
    return SimpleNamespace(
        id=repo_id,
        user_id=7,
        full_name=f"owner/repo-{repo_id}",
        description=f"description {repo_id}",
        topics=["topic"],
        language="Python",
        stargazers_count=10 * repo_id,
        embedding=[float(repo_id)] * dimensions,
        cluster_id=cluster_id,
        coord_x=1.0,
        coord_y=2.0,
        coord_z=3.0,
        ai_summary=None,
        ai_tags=None,
    )


def _make_task():
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


class _ScalarsResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return SimpleNamespace(all=lambda: list(self._rows))

    def scalar(self):
        return len(self._rows)


class _FakeDb:
    """Fake session that models transaction boundaries.

    Statements take effect immediately, as they would inside a real
    transaction, so later ORM assignments override them. A checkpoint is taken
    at each commit and restored on rollback, which is what lets a test assert
    what survives an exception.
    """

    def __init__(self, state):
        self.state = state
        self.commits = 0
        self.statements: list[str] = []
        self.added: list[object] = []
        self._next_cluster_id = 100
        self._checkpoint = self._snapshot()

    def _snapshot(self):
        return (
            dict(self.state.clusters),
            {repo.id: repo.cluster_id for repo in self.state.repos},
        )

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
        if text.startswith("delete from clusters"):
            self.state.clusters = {}
            return _ScalarsResult([])
        if text.startswith("update starred_repos"):
            for repo in self.state.repos:
                repo.cluster_id = None
            return _ScalarsResult([])
        if "count(" in text:
            return _ScalarsResult([])
        return _ScalarsResult(self.state.repos)

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = self._next_cluster_id
                self._next_cluster_id += 1
            if type(obj).__name__ == "Cluster":
                self.state.clusters[obj.id] = obj
        self.added = []

    async def commit(self):
        self.commits += 1
        await self.flush()
        self._checkpoint = self._snapshot()

    async def rollback(self):
        self.added = []
        clusters, repo_cluster_ids = self._checkpoint
        self.state.clusters = dict(clusters)
        for repo in self.state.repos:
            repo.cluster_id = repo_cluster_ids[repo.id]


class _State:
    def __init__(self, repos, clusters, task):
        self.repos = repos
        self.clusters = clusters
        self.task = task


class _ClusterResult:
    def __init__(self, labels, n_clusters):
        self.labels = labels
        self.n_clusters = n_clusters
        self.cluster_centers = {
            label: [float(label), float(label), float(label)]
            for label in set(labels)
            if label != -1
        }
        self.coords_3d = [[float(i), float(i) + 1, float(i) + 2] for i in range(8)]


def _install(monkeypatch, state, *, naming_raises=False, labels=None):
    db = _FakeDb(state)
    monkeypatch.setattr("nebula.db.database.get_db_context", lambda: db, raising=False)

    resolved_labels = labels if labels is not None else [0, 0, 0, 1, 1, 1, -1, -1]

    class _FakeClusteringService:
        def __init__(self, **_kwargs):
            pass

        def fit_transform(self, **_kwargs):
            return _ClusterResult(resolved_labels, n_clusters=2)

    async def fake_name_llm(*_args, **_kwargs):
        if naming_raises:
            raise RuntimeError("llm naming down")
        return "Name", "Description", ["kw"]

    def fake_name_heuristic(*_args, **_kwargs):
        if naming_raises:
            raise RuntimeError("heuristic naming down")
        return "Heuristic", "Description", ["kw"]

    import nebula.core.clustering as clustering

    monkeypatch.setattr(clustering, "ClusteringService", _FakeClusteringService)
    monkeypatch.setattr(clustering, "generate_cluster_name_llm", fake_name_llm)
    monkeypatch.setattr(clustering, "generate_cluster_name", fake_name_heuristic)
    monkeypatch.setattr(
        clustering,
        "build_cluster_naming_inputs",
        lambda repos: (["a"], ["b"], ["c"], ["Python"]),
    )
    monkeypatch.setattr(
        clustering, "deduplicate_cluster_entries", lambda entries: entries
    )
    return db


@pytest.mark.asyncio
async def test_naming_failure_leaves_previous_clusters_intact(monkeypatch):
    repos = [_make_repo(index, cluster_id=1 if index < 5 else 2) for index in range(8)]
    clusters = {
        1: SimpleNamespace(id=1, user_id=7, name="Old A", repo_count=5),
        2: SimpleNamespace(id=2, user_id=7, name="Old B", repo_count=3),
    }
    task = _make_task()
    state = _State(repos, clusters, task)
    _install(monkeypatch, state, naming_raises=True)

    await sync_execution_service.run_clustering_task(
        user_id=7, task_id=1, use_llm=True, incremental=False
    )

    # Naming raised before any destructive statement ran, so the old graph is
    # untouched. Under the pre-fix ordering the clusters were already gone.
    assert set(state.clusters) == {1, 2}
    assert [repo.cluster_id for repo in repos] == [1, 1, 1, 1, 1, 2, 2, 2]
    assert task.status == "failed"


@pytest.mark.asyncio
async def test_swap_detaches_repos_before_deleting_clusters(monkeypatch):
    repos = [_make_repo(index, cluster_id=1) for index in range(8)]
    clusters = {1: SimpleNamespace(id=1, user_id=7, name="Old", repo_count=8)}
    task = _make_task()
    state = _State(repos, clusters, task)
    db = _install(monkeypatch, state)

    await sync_execution_service.run_clustering_task(
        user_id=7, task_id=1, use_llm=False, incremental=False
    )

    detach_index = next(
        i for i, s in enumerate(db.statements) if s.startswith("update starred_repos")
    )
    delete_index = next(
        i for i, s in enumerate(db.statements) if s.startswith("delete from clusters")
    )
    # The foreign key starred_repos.cluster_id -> clusters.id is only safe if
    # the detach happens first.
    assert detach_index < delete_index


@pytest.mark.asyncio
async def test_swap_uses_set_based_statements_not_per_row_deletes(monkeypatch):
    repos = [_make_repo(index, cluster_id=index % 3 + 1) for index in range(8)]
    clusters = {
        cluster_id: SimpleNamespace(
            id=cluster_id, user_id=7, name=f"Old {cluster_id}", repo_count=3
        )
        for cluster_id in (1, 2, 3)
    }
    task = _make_task()
    state = _State(repos, clusters, task)
    db = _install(monkeypatch, state)

    await sync_execution_service.run_clustering_task(
        user_id=7, task_id=1, use_llm=False, incremental=False
    )

    deletes = [s for s in db.statements if s.startswith("delete from clusters")]
    updates = [s for s in db.statements if s.startswith("update starred_repos")]
    # One statement each, regardless of how many clusters or repos exist.
    assert len(deletes) == 1
    assert len(updates) == 1


@pytest.mark.asyncio
async def test_successful_swap_replaces_clusters_and_assigns_coordinates(monkeypatch):
    repos = [_make_repo(index, cluster_id=1) for index in range(8)]
    clusters = {1: SimpleNamespace(id=1, user_id=7, name="Old", repo_count=8)}
    task = _make_task()
    state = _State(repos, clusters, task)
    _install(monkeypatch, state)

    await sync_execution_service.run_clustering_task(
        user_id=7, task_id=1, use_llm=False, incremental=False
    )

    assert 1 not in state.clusters
    assert len(state.clusters) == 2
    new_ids = set(state.clusters)
    # Labels [0,0,0,1,1,1,-1,-1]: six assigned, two noise.
    assigned = [repo.cluster_id for repo in repos if repo.cluster_id is not None]
    assert len(assigned) == 6
    assert set(assigned) <= new_ids
    assert [repo.cluster_id for repo in repos[6:]] == [None, None]
    # Coordinates come from the fit result, keyed by position.
    assert repos[0].coord_x == 0.0
    assert repos[3].coord_y == 4.0
    assert task.status == "completed"
    assert task.processed_items == 8


@pytest.mark.asyncio
async def test_no_repo_points_at_a_deleted_cluster_after_swap(monkeypatch):
    repos = [_make_repo(index, cluster_id=1) for index in range(8)]
    clusters = {1: SimpleNamespace(id=1, user_id=7, name="Old", repo_count=8)}
    task = _make_task()
    state = _State(repos, clusters, task)
    _install(monkeypatch, state)

    await sync_execution_service.run_clustering_task(
        user_id=7, task_id=1, use_llm=False, incremental=False
    )

    live_cluster_ids = set(state.clusters)
    dangling = [
        repo.id
        for repo in repos
        if repo.cluster_id is not None and repo.cluster_id not in live_cluster_ids
    ]
    assert dangling == []
