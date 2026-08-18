"""Unit coverage for `sync_stars_task`.

The star-sync stage owns new/existing repo handling, content-hash reprocessing,
incremental truncation, unstarred detection, and the `total_stars` /
`synced_stars` accounting difference between modes. It sat at 4% coverage in
the 2026-08-18 scan.

Database interaction is faked here; the integration tier covers the persistence
semantics these tests cannot reach (large `NOT IN`, cascade behaviour).
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from nebula.application.services import sync_execution_service
from nebula.core.config import get_app_settings, get_sync_settings
from nebula.core.github_client import GitHubRepo

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


def _gh_repo(repo_id: int, **overrides) -> GitHubRepo:
    payload = {
        "id": repo_id,
        "full_name": f"owner/repo-{repo_id}",
        "owner": "owner",
        "name": f"repo-{repo_id}",
        "description": f"description {repo_id}",
        "language": "Python",
        "topics": ["topic-a"],
        "html_url": f"https://github.com/owner/repo-{repo_id}",
        "stargazers_count": repo_id * 10,
        "forks_count": repo_id,
        "watchers_count": repo_id,
        "open_issues_count": 0,
        "starred_at": NOW - timedelta(days=repo_id),
        "created_at": NOW - timedelta(days=365),
        "updated_at": NOW,
        "pushed_at": NOW,
        "owner_avatar_url": "https://avatars/owner",
    }
    payload.update(overrides)
    return GitHubRepo(**payload)


def _existing_repo(github_id: int, **overrides):
    repo = SimpleNamespace(
        id=github_id + 1000,
        user_id=7,
        github_repo_id=github_id,
        full_name=f"owner/repo-{github_id}",
        description=f"description {github_id}",
        language="Python",
        topics=["topic-a"],
        stargazers_count=0,
        forks_count=0,
        repo_updated_at=None,
        repo_pushed_at=None,
        owner_avatar_url=None,
        readme_content=None,
        is_readme_fetched=False,
        is_embedded=True,
        is_summarized=True,
        ai_summary="summary",
        ai_tags=["tag"],
        embedding=[0.1, 0.2],
        description_hash=None,
        topics_hash=None,
    )
    for key, value in overrides.items():
        setattr(repo, key, value)
    return repo


def _make_task():
    return SimpleNamespace(
        id=1,
        user_id=7,
        task_type="stars",
        status="pending",
        started_at=None,
        completed_at=None,
        error_message=None,
        error_details=None,
        total_items=0,
        processed_items=0,
        failed_items=0,
    )


def _make_user(last_sync_at=None):
    return SimpleNamespace(
        id=7,
        username="tester",
        total_stars=0,
        synced_stars=0,
        last_sync_at=last_sync_at,
    )


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return SimpleNamespace(all=lambda: list(self._rows))

    def scalar(self):
        return len(self._rows)


class _FakeDb:
    def __init__(self, state):
        self.state = state
        self.added: list[object] = []
        self.deleted: list[object] = []
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, _tb):
        return False

    async def get(self, model, obj_id, **_kwargs):
        if model.__name__ == "SyncTask":
            return self.state.task if self.state.task.id == obj_id else None
        if model.__name__ == "User":
            return self.state.user if self.state.user.id == obj_id else None
        return None

    async def execute(self, statement):
        text = str(statement).lower()
        if "count(" in text:
            return _Result(self.state.stored)
        if "not in" in text or "not_in" in text:
            return _Result(self.state.unstarred)
        return _Result(self.state.stored)

    def add(self, obj):
        self.added.append(obj)

    async def delete(self, obj):
        self.deleted.append(obj)

    async def flush(self):
        return None

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        return None


class _State:
    def __init__(self, user, task, stored=None, unstarred=None):
        self.user = user
        self.task = task
        self.stored = stored or []
        self.unstarred = unstarred or []


class _FakeGitHubClient:
    """Records `stop_before` so incremental behaviour is observable."""

    instances: list["_FakeGitHubClient"] = []

    def __init__(self, access_token: str):
        self.access_token = access_token
        self.stop_before_calls: list[object] = []
        type(self).instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, _tb):
        return False

    async def get_starred_repos(self, stop_before=None):
        self.stop_before_calls.append(stop_before)
        return type(self).repos, type(self).truncated


def _install(
    monkeypatch,
    state,
    *,
    repos,
    truncated=False,
    existing_map=None,
    github_token="ghp_test",
):
    db = _FakeDb(state)
    monkeypatch.setattr("nebula.db.database.get_db_context", lambda: db, raising=False)

    _FakeGitHubClient.instances = []
    _FakeGitHubClient.repos = repos
    _FakeGitHubClient.truncated = truncated
    monkeypatch.setattr(sync_execution_service, "GitHubClient", _FakeGitHubClient)

    async def fake_prefetch(_db, *, user_id, github_ids):
        return dict(existing_map or {})

    async def fake_fetch_readmes(_client, targets, *, max_length, concurrency):
        return {name: f"readme for {name}" for name in targets}

    async def fake_sync_star_lists(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        sync_execution_service, "prefetch_existing_repos", fake_prefetch
    )
    monkeypatch.setattr(
        sync_execution_service, "fetch_readmes_in_parallel", fake_fetch_readmes
    )
    monkeypatch.setattr(sync_execution_service, "sync_star_lists", fake_sync_star_lists)
    monkeypatch.setenv("GITHUB_TOKEN", github_token)
    get_app_settings.cache_clear()
    get_sync_settings.cache_clear()
    return db


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    yield
    get_app_settings.cache_clear()
    get_sync_settings.cache_clear()


@pytest.mark.asyncio
async def test_new_repos_are_created_with_readme(monkeypatch):
    user, task = _make_user(), _make_task()
    state = _State(user, task)
    db = _install(monkeypatch, state, repos=[_gh_repo(1), _gh_repo(2)])

    await sync_execution_service.sync_stars_task(7, 1, "full")

    created = [obj for obj in db.added if type(obj).__name__ == "StarredRepo"]
    assert len(created) == 2
    assert all(repo.is_readme_fetched for repo in created)
    assert task.status == "completed"
    assert task.error_details["new_repos"] == 2
    assert task.error_details["updated_repos"] == 0


@pytest.mark.asyncio
async def test_existing_repo_is_updated_not_recreated(monkeypatch):
    user, task = _make_user(), _make_task()
    existing = _existing_repo(1)
    from nebula.utils import compute_content_hash, compute_topics_hash

    existing.description_hash = compute_content_hash("description 1")
    existing.topics_hash = compute_topics_hash(["topic-a"])
    state = _State(user, task)
    db = _install(monkeypatch, state, repos=[_gh_repo(1)], existing_map={1: existing})

    await sync_execution_service.sync_stars_task(7, 1, "full")

    assert [obj for obj in db.added if type(obj).__name__ == "StarredRepo"] == []
    assert task.error_details["updated_repos"] == 1
    assert task.error_details["new_repos"] == 0
    # Unchanged content must not trigger reprocessing.
    assert existing.is_embedded is True
    assert existing.ai_summary == "summary"


@pytest.mark.asyncio
async def test_changed_content_marks_repo_for_reprocessing(monkeypatch):
    user, task = _make_user(), _make_task()
    existing = _existing_repo(1, description_hash="stale", topics_hash="stale")
    state = _State(user, task)
    _install(monkeypatch, state, repos=[_gh_repo(1)], existing_map={1: existing})

    await sync_execution_service.sync_stars_task(7, 1, "full")

    assert existing.is_embedded is False
    assert existing.is_summarized is False
    assert existing.ai_summary is None
    assert existing.ai_tags is None
    assert existing.embedding is None
    assert existing.readme_content == "readme for owner/repo-1"


@pytest.mark.asyncio
async def test_incremental_first_sync_upgrades_to_full(monkeypatch):
    user, task = _make_user(last_sync_at=None), _make_task()
    state = _State(user, task)
    _install(monkeypatch, state, repos=[_gh_repo(1)])

    await sync_execution_service.sync_stars_task(7, 1, "incremental")

    assert task.error_details["sync_mode"] == "full"
    assert _FakeGitHubClient.instances[0].stop_before_calls == [None]


@pytest.mark.asyncio
async def test_incremental_sync_passes_last_sync_as_cutoff(monkeypatch):
    cutoff = NOW - timedelta(days=3)
    user, task = _make_user(last_sync_at=cutoff), _make_task()
    state = _State(user, task, stored=[_existing_repo(1)])
    _install(monkeypatch, state, repos=[_gh_repo(1)], truncated=True)

    await sync_execution_service.sync_stars_task(7, 1, "incremental")

    assert _FakeGitHubClient.instances[0].stop_before_calls == [cutoff]
    assert task.error_details["sync_mode"] == "incremental"
    assert task.error_details["was_truncated"] is True


@pytest.mark.asyncio
async def test_full_sync_removes_unstarred_repos(monkeypatch):
    user, task = _make_user(), _make_task()
    stale = _existing_repo(99)
    state = _State(user, task, unstarred=[stale])
    db = _install(monkeypatch, state, repos=[_gh_repo(1)])

    await sync_execution_service.sync_stars_task(7, 1, "full")

    assert stale in db.deleted
    assert task.error_details["removed_repos"] == 1


@pytest.mark.asyncio
async def test_incremental_skips_unstarred_detection_by_default(monkeypatch):
    monkeypatch.setenv("SYNC_DETECT_UNSTARRED_ON_INCREMENTAL", "false")
    user, task = _make_user(last_sync_at=NOW - timedelta(days=1)), _make_task()
    stale = _existing_repo(99)
    state = _State(user, task, unstarred=[stale])
    db = _install(monkeypatch, state, repos=[_gh_repo(1)], truncated=True)

    await sync_execution_service.sync_stars_task(7, 1, "incremental")

    assert db.deleted == []
    assert task.error_details["removed_repos"] == 0
    # Only the initial truncated fetch; no second full-list fetch.
    assert len(_FakeGitHubClient.instances[0].stop_before_calls) == 1


@pytest.mark.asyncio
async def test_incremental_detects_unstarred_when_enabled(monkeypatch):
    monkeypatch.setenv("SYNC_DETECT_UNSTARRED_ON_INCREMENTAL", "true")
    user, task = _make_user(last_sync_at=NOW - timedelta(days=1)), _make_task()
    stale = _existing_repo(99)
    state = _State(user, task, unstarred=[stale])
    db = _install(monkeypatch, state, repos=[_gh_repo(1)], truncated=True)

    await sync_execution_service.sync_stars_task(7, 1, "incremental")

    assert stale in db.deleted
    assert task.error_details["removed_repos"] == 1


@pytest.mark.asyncio
async def test_incremental_counts_stars_from_database_not_page(monkeypatch):
    user, task = _make_user(last_sync_at=NOW - timedelta(days=1)), _make_task()
    # Three repos stored, but only one arrived in this incremental page.
    state = _State(
        user,
        task,
        stored=[_existing_repo(1), _existing_repo(2), _existing_repo(3)],
    )
    _install(monkeypatch, state, repos=[_gh_repo(1)], truncated=True)

    await sync_execution_service.sync_stars_task(7, 1, "incremental")

    assert user.total_stars == 3
    assert user.synced_stars == 3


@pytest.mark.asyncio
async def test_full_sync_counts_stars_from_the_fetched_page(monkeypatch):
    user, task = _make_user(), _make_task()
    state = _State(user, task)
    _install(monkeypatch, state, repos=[_gh_repo(1), _gh_repo(2), _gh_repo(3)])

    await sync_execution_service.sync_stars_task(7, 1, "full")

    assert user.total_stars == 3
    assert user.synced_stars == 3
    assert user.last_sync_at is not None


@pytest.mark.asyncio
async def test_missing_github_token_fails_the_task(monkeypatch):
    user, task = _make_user(), _make_task()
    state = _State(user, task)
    _install(monkeypatch, state, repos=[_gh_repo(1)], github_token="")

    await sync_execution_service.sync_stars_task(7, 1, "full")

    assert task.status == "failed"
    assert "GitHub token not configured" in task.error_message


@pytest.mark.asyncio
async def test_github_api_error_fails_the_task(monkeypatch):
    user, task = _make_user(), _make_task()
    state = _State(user, task)
    _install(monkeypatch, state, repos=[_gh_repo(1)])

    async def boom(self, stop_before=None):
        raise RuntimeError("github down")

    monkeypatch.setattr(_FakeGitHubClient, "get_starred_repos", boom)

    await sync_execution_service.sync_stars_task(7, 1, "full")

    assert task.status == "failed"
    assert "GitHub API error" in task.error_message


@pytest.mark.asyncio
async def test_star_list_sync_failure_is_non_critical(monkeypatch):
    user, task = _make_user(), _make_task()
    state = _State(user, task)
    _install(monkeypatch, state, repos=[_gh_repo(1)])

    async def failing_star_lists(*_args, **_kwargs):
        raise RuntimeError("star lists down")

    monkeypatch.setattr(sync_execution_service, "sync_star_lists", failing_star_lists)

    await sync_execution_service.sync_stars_task(7, 1, "full")

    # The star sync itself already completed before star lists were attempted.
    assert task.status == "completed"


@pytest.mark.asyncio
async def test_missing_user_or_task_returns_without_side_effects(monkeypatch):
    user, task = _make_user(), _make_task()
    state = _State(user, task)
    db = _install(monkeypatch, state, repos=[_gh_repo(1)])

    await sync_execution_service.sync_stars_task(999, 1, "full")

    assert db.added == []
    assert task.status == "pending"
