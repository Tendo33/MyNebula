from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from nebula.infrastructure.repositories.snapshot_repository import (
    SnapshotStoreRepository,
    _batched,
)


class _ScalarResult:
    def __init__(self, values):
        self.values = values

    def scalars(self):
        return self

    def all(self):
        return self.values


class _RetentionSession:
    def __init__(self):
        self.results = iter(
            [
                _ScalarResult(list(range(1, 31))),
                _ScalarResult([40, 41]),
                SimpleNamespace(),
                SimpleNamespace(),
                SimpleNamespace(),
                SimpleNamespace(),
            ]
        )
        self.statements = []
        self.commits = 0

    async def get(self, _model, _record_id, *, with_for_update=False):
        assert with_for_update is True
        return SimpleNamespace(active_graph_snapshot_id=99)

    async def execute(self, statement, *_args):
        self.statements.append(statement)
        return next(self.results)

    async def commit(self):
        self.commits += 1


class _ActivationSession:
    def __init__(self):
        self.user = SimpleNamespace(active_graph_snapshot_id=7)
        self.locked = False
        self.commits = 0

    async def get(self, _model, _record_id, *, with_for_update=False):
        self.locked = with_for_update
        return self.user

    async def execute(self, _statement):
        return SimpleNamespace()

    async def commit(self):
        self.commits += 1


class _PreviousSnapshotSession:
    def __init__(self):
        self.statement = None

    async def execute(self, statement):
        self.statement = statement
        return SimpleNamespace(scalar_one_or_none=lambda: None)


def test_snapshot_insert_batches_are_bounded():
    batches = list(_batched(range(2501), size=1000))

    assert [len(batch) for batch in batches] == [1000, 1000, 501]


@pytest.mark.asyncio
async def test_activation_locks_user_row_shared_with_retention():
    repo = SnapshotStoreRepository()
    session = _ActivationSession()
    snapshot = SimpleNamespace(id=11, status="ready", activated_at=None)

    await repo.activate_snapshot(session, user_id=5, snapshot=snapshot)

    assert session.locked is True
    assert session.user.active_graph_snapshot_id == 11
    assert snapshot.status == "active"
    assert session.commits == 1


@pytest.mark.asyncio
async def test_previous_snapshot_excludes_failed_or_building_snapshots():
    repo = SnapshotStoreRepository()
    session = _PreviousSnapshotSession()

    await repo.get_previous_snapshot(session, user_id=5, exclude_snapshot_id=11)

    sql = str(session.statement.compile(compile_kwargs={"literal_binds": True}))
    assert "graph_snapshots.status IN ('active', 'ready')" in sql


@pytest.mark.asyncio
async def test_save_snapshot_refuses_to_overwrite_existing_version(monkeypatch):
    repo = SnapshotStoreRepository()

    async def existing(*_args, **_kwargs):
        return SimpleNamespace(id=7)

    monkeypatch.setattr(repo, "get_snapshot_by_version", existing)

    with pytest.raises(ValueError, match="already exists"):
        await repo.save_snapshot_payload(
            db=SimpleNamespace(),
            user_id=1,
            version="snapshot-existing",
            graph_data=SimpleNamespace(),
            timeline_data=SimpleNamespace(),
        )


@pytest.mark.asyncio
async def test_retention_protects_active_and_latest_snapshots():
    repo = SnapshotStoreRepository()
    session = _RetentionSession()

    deleted = await repo.prune_snapshots(
        session,
        user_id=5,
        now=datetime(2026, 8, 8, tzinfo=timezone.utc),
    )

    assert deleted == 2
    assert session.commits == 1
    candidate_sql = str(
        session.statements[1].compile(compile_kwargs={"literal_binds": True})
    )
    assert "graph_snapshots.created_at <" in candidate_sql
    assert "99" in candidate_sql
    assert "graph_snapshots.user_id = 5" in candidate_sql
    delete_sql = "\n".join(
        str(statement.compile(compile_kwargs={"literal_binds": True}))
        for statement in session.statements[2:]
    )
    assert "40" in delete_sql and "41" in delete_sql
    assert "99" not in delete_sql
    assert "graph_snapshots.user_id = 5" in delete_sql
