from datetime import datetime, timezone
from types import SimpleNamespace

import pytest


class _FakeExecuteResult:
    def __init__(self, *, rowcount: int = 0):
        self.rowcount = rowcount


class _FakeDb:
    def __init__(self, state):
        self.state = state
        self._added: list[object] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, _tb):
        return False

    async def get(self, model, obj_id):
        if model.__name__ == "SyncTask":
            return self.state.tasks.get(obj_id)
        if model.__name__ == "User":
            return self.state.user if self.state.user.id == obj_id else None
        return None

    async def execute(self, _statement):
        return _FakeExecuteResult(rowcount=self.state.reset_count)

    def add(self, obj):
        self._added.append(obj)

    async def commit(self):
        return None

    async def refresh(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = self.state.next_task_id
            self.state.next_task_id += 1
        self.state.tasks[obj.id] = obj


class _DbContextFactory:
    def __init__(self, state):
        self.state = state

    def __call__(self):
        return _FakeDb(self.state)


class _FakeState:
    def __init__(self):
        self.user = SimpleNamespace(id=7)
        self.tasks = {
            1: SimpleNamespace(
                id=1,
                user_id=7,
                task_type="full_refresh",
                status="pending",
                started_at=None,
                completed_at=None,
                error_message=None,
                error_details=None,
                total_items=0,
                processed_items=0,
                failed_items=0,
            )
        }
        self.next_task_id = 2
        self.reset_count = 5
        self.snapshot_calls: list[tuple[int, int]] = []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failed_phase", "subtask_type", "expected_message"),
    [
        ("stars", "stars", "stars exploded"),
        ("embeddings", "embedding", "embedding exploded"),
        ("clustering", "cluster", "clustering exploded"),
    ],
)
async def test_full_refresh_marks_parent_failed_when_subtask_sets_failed_status(
    monkeypatch,
    failed_phase,
    subtask_type,
    expected_message,
):
    from nebula.application.services import sync_ops_service

    state = _FakeState()

    def _build_subtask_failure(message: str):
        async def _fail(_user_id, task_id, *_args, **_kwargs):
            task = state.tasks[task_id]
            task.status = "failed"
            task.error_message = message
            task.completed_at = datetime.now(timezone.utc)

        return _fail

    async def complete_subtask(_user_id, task_id, *_args, **_kwargs):
        task = state.tasks[task_id]
        task.status = "completed"
        task.completed_at = datetime.now(timezone.utc)

    async def fake_snapshot(db, *, user):
        state.snapshot_calls.append((db.state.user.id, user.id))

    monkeypatch.setattr(
        sync_ops_service,
        "get_db_context",
        _DbContextFactory(state),
        raising=False,
    )
    monkeypatch.setattr(
        sync_ops_service,
        "sync_stars_task",
        _build_subtask_failure("stars exploded")
        if subtask_type == "stars"
        else complete_subtask,
    )
    monkeypatch.setattr(
        sync_ops_service,
        "compute_embeddings_task",
        _build_subtask_failure("embedding exploded")
        if subtask_type == "embedding"
        else complete_subtask,
    )
    monkeypatch.setattr(
        sync_ops_service,
        "run_clustering_task",
        _build_subtask_failure("clustering exploded")
        if subtask_type == "cluster"
        else complete_subtask,
    )
    monkeypatch.setattr(
        sync_ops_service,
        "GraphQueryService",
        lambda: SimpleNamespace(rebuild_active_snapshot=fake_snapshot),
    )

    await sync_ops_service.full_refresh_task(user_id=7, task_id=1)

    main_task = state.tasks[1]
    assert main_task.status == "failed"
    assert expected_message in main_task.error_message
    assert main_task.completed_at is not None
    assert main_task.error_details["phase"] == failed_phase
    assert state.snapshot_calls == []
