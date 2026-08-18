"""Tests for chunked, resumable embedding in `compute_embeddings_task`.

Regression guard for the all-or-nothing embedding stage found in the
2026-08-18 scan: the task used to load every un-embedded repo at once, issue a
single `embed_batch` call, and on any failure persist zero embeddings while
marking every repo failed.
"""

from types import SimpleNamespace

import pytest

from nebula.application.services import sync_execution_service
from nebula.core.config import get_sync_settings

CHUNK_SIZE = 10


def _make_repo(repo_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=repo_id,
        user_id=7,
        full_name=f"owner/repo-{repo_id}",
        description=f"description {repo_id}",
        topics=["topic-a", "topic-b"],
        language="Python",
        readme_content=f"readme {repo_id}",
        ai_summary=None,
        ai_tags=None,
        is_summarized=False,
        embedding_text=None,
        embedding=None,
        is_embedded=False,
    )


def _make_task(task_id: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        id=task_id,
        user_id=7,
        task_type="embedding",
        status="pending",
        started_at=None,
        completed_at=None,
        error_message=None,
        error_details=None,
        total_items=0,
        processed_items=0,
        failed_items=0,
    )


class _CountResult:
    def __init__(self, value: int):
        self._value = value

    def scalar(self):
        return self._value


class _ScalarsResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return SimpleNamespace(all=lambda: list(self._rows))


class _FakeDb:
    """Minimal async session over an in-memory repo list.

    Interprets the two statement shapes the task issues: a pending-count query
    and a keyset-paged chunk query. Records commits and rollbacks so tests can
    assert per-chunk persistence.
    """

    def __init__(self, state):
        self.state = state
        self.commits = 0
        self.rollbacks = 0
        # Snapshot of is_embedded flags at each commit, so a rollback can undo
        # in-memory mutations the way a real transaction would.
        self._checkpoint = self._snapshot()

    def _snapshot(self):
        return {
            repo.id: (repo.is_embedded, repo.embedding, repo.embedding_text)
            for repo in self.state.repos
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, _tb):
        return False

    async def get(self, model, obj_id, **_kwargs):
        if model.__name__ == "SyncTask":
            return self.state.task if self.state.task.id == obj_id else None
        return None

    async def execute(self, statement):
        text = str(statement).lower()
        pending = [repo for repo in self.state.repos if not repo.is_embedded]
        # Match the aggregate call, not the substring: `select(StarredRepo)`
        # renders columns such as `open_issues_count`.
        if "count(" in text:
            return _CountResult(len(pending))

        cursor = self.state.cursor_hint
        rows = [repo for repo in pending if repo.id > cursor]
        rows.sort(key=lambda repo: repo.id)
        return _ScalarsResult(rows[:CHUNK_SIZE])

    async def commit(self):
        self.commits += 1
        self._checkpoint = self._snapshot()

    async def rollback(self):
        self.rollbacks += 1
        for repo in self.state.repos:
            embedded, embedding, embedding_text = self._checkpoint[repo.id]
            repo.is_embedded = embedded
            repo.embedding = embedding
            repo.embedding_text = embedding_text


class _State:
    def __init__(self, repos, task):
        self.repos = repos
        self.task = task
        self.cursor_hint = 0


class _FakeEmbeddingService:
    """Embeds by chunk; can be told which chunk ordinals must fail."""

    def __init__(self, dimensions: int = 4, fail_on_chunks: set[int] | None = None):
        self.dimensions = dimensions
        self.fail_on_chunks = fail_on_chunks or set()
        self.chunk_calls = 0
        self.embedded_texts: list[list[str]] = []
        self.batch_sizes: list[int | None] = []

    def build_repo_text(self, *, full_name: str, **_kwargs) -> str:
        return f"text::{full_name}"

    async def embed_batch(self, texts, batch_size=None):
        self.chunk_calls += 1
        self.batch_sizes.append(batch_size)
        if self.chunk_calls in self.fail_on_chunks:
            raise RuntimeError(f"embedding failure on chunk {self.chunk_calls}")
        self.embedded_texts.append(list(texts))
        return [[float(index)] * self.dimensions for index in range(len(texts))]


class _FakeLLMService:
    def __init__(self, fail: bool = False):
        self.fail = fail

    async def generate_repo_summary_and_tags(self, *, full_name: str, **_kwargs):
        if self.fail:
            raise RuntimeError("llm unavailable")
        return f"summary for {full_name}", ["tag-a", "tag-b"]


@pytest.fixture
def chunked_settings(monkeypatch):
    monkeypatch.setenv("SYNC_BATCH_SIZE", str(CHUNK_SIZE))
    get_sync_settings.cache_clear()
    yield
    get_sync_settings.cache_clear()


def _install(monkeypatch, state, embedding_service, llm_service):
    """Wire fakes into the module under test and track cursor advancement."""
    db = _FakeDb(state)

    def db_context():
        return db

    monkeypatch.setattr("nebula.db.database.get_db_context", db_context, raising=False)
    monkeypatch.setattr(
        sync_execution_service, "get_embedding_service", lambda: embedding_service
    )
    monkeypatch.setattr(sync_execution_service, "get_llm_service", lambda: llm_service)

    original_chunk = sync_execution_service._embed_one_chunk

    async def tracking_chunk(**kwargs):
        repos = kwargs["repos"]
        # Mirror the production cursor: the task advances past the chunk before
        # processing it, so the fake session must do the same.
        state.cursor_hint = repos[-1].id
        return await original_chunk(**kwargs)

    monkeypatch.setattr(sync_execution_service, "_embed_one_chunk", tracking_chunk)
    return db


@pytest.mark.asyncio
async def test_each_chunk_is_committed_before_the_next(chunked_settings, monkeypatch):
    repos = [_make_repo(index) for index in range(1, 26)]
    task = _make_task()
    state = _State(repos, task)
    embedding = _FakeEmbeddingService()
    db = _install(monkeypatch, state, embedding, _FakeLLMService())

    await sync_execution_service.compute_embeddings_task(7, 1)

    assert all(repo.is_embedded for repo in repos)
    assert task.status == "completed"
    assert task.total_items == 25
    assert task.processed_items == 25
    assert task.failed_items == 0
    # 25 repos at chunk size 10 -> 3 chunks.
    assert embedding.chunk_calls == 3
    # start + total + three chunk commits + final commit.
    assert db.commits >= 5


@pytest.mark.asyncio
async def test_failed_chunk_does_not_discard_earlier_or_later_chunks(
    chunked_settings, monkeypatch
):
    repos = [_make_repo(index) for index in range(1, 26)]
    task = _make_task()
    state = _State(repos, task)
    embedding = _FakeEmbeddingService(fail_on_chunks={2})
    db = _install(monkeypatch, state, embedding, _FakeLLMService())

    await sync_execution_service.compute_embeddings_task(7, 1)

    embedded_ids = {repo.id for repo in repos if repo.is_embedded}
    # Chunk 1 (ids 1-10) and chunk 3 (ids 21-25) survived; chunk 2 did not.
    assert embedded_ids == set(range(1, 11)) | set(range(21, 26))
    assert task.status == "completed"
    assert task.processed_items == 15
    assert task.failed_items == 10
    assert db.rollbacks == 1
    assert task.error_details["chunks_succeeded"] == 2
    assert task.error_details["chunks_failed"] == 1


@pytest.mark.asyncio
async def test_rerun_after_partial_failure_only_embeds_the_remainder(
    chunked_settings, monkeypatch
):
    repos = [_make_repo(index) for index in range(1, 26)]
    task = _make_task()
    state = _State(repos, task)
    failing = _FakeEmbeddingService(fail_on_chunks={2})
    _install(monkeypatch, state, failing, _FakeLLMService())
    await sync_execution_service.compute_embeddings_task(7, 1)

    # Second run with a healthy provider and a fresh task.
    state.task = _make_task(task_id=1)
    state.cursor_hint = 0
    healthy = _FakeEmbeddingService()
    _install(monkeypatch, state, healthy, _FakeLLMService())
    await sync_execution_service.compute_embeddings_task(7, 1)

    assert all(repo.is_embedded for repo in repos)
    assert state.task.total_items == 10
    assert state.task.processed_items == 10
    assert healthy.chunk_calls == 1


@pytest.mark.asyncio
async def test_persistently_failing_chunk_terminates(chunked_settings, monkeypatch):
    repos = [_make_repo(index) for index in range(1, 26)]
    task = _make_task()
    state = _State(repos, task)
    embedding = _FakeEmbeddingService(fail_on_chunks={1, 2, 3, 4, 5, 6})
    _install(monkeypatch, state, embedding, _FakeLLMService())

    await sync_execution_service.compute_embeddings_task(7, 1)

    # Terminates after visiting each chunk once, rather than re-selecting the
    # still-unembedded rows forever.
    assert embedding.chunk_calls == 3
    assert task.failed_items == 25
    assert task.processed_items == 0


@pytest.mark.asyncio
async def test_all_chunks_failing_is_a_hard_failure(chunked_settings, monkeypatch):
    repos = [_make_repo(index) for index in range(1, 26)]
    task = _make_task()
    state = _State(repos, task)
    _install(
        monkeypatch,
        state,
        _FakeEmbeddingService(fail_on_chunks={1, 2, 3}),
        _FakeLLMService(),
    )

    await sync_execution_service.compute_embeddings_task(7, 1)

    # `_inspect_task_outcome` raises on "failed", which is what stops the
    # pipeline from building a snapshot over no embeddings.
    assert task.status == "failed"
    assert task.error_message is not None


@pytest.mark.asyncio
async def test_partial_failure_reports_completed_so_pipeline_reports_partial(
    chunked_settings, monkeypatch
):
    repos = [_make_repo(index) for index in range(1, 26)]
    task = _make_task()
    state = _State(repos, task)
    _install(
        monkeypatch, state, _FakeEmbeddingService(fail_on_chunks={2}), _FakeLLMService()
    )

    await sync_execution_service.compute_embeddings_task(7, 1)

    # status "completed" plus failed_items > 0 is what `_inspect_task_outcome`
    # turns into PipelineStatus.partial_failed.
    assert task.status == "completed"
    assert task.failed_items > 0


@pytest.mark.asyncio
async def test_nothing_to_embed_completes_immediately(chunked_settings, monkeypatch):
    repos = [_make_repo(index) for index in range(1, 4)]
    for repo in repos:
        repo.is_embedded = True
    task = _make_task()
    state = _State(repos, task)
    embedding = _FakeEmbeddingService()
    _install(monkeypatch, state, embedding, _FakeLLMService())

    await sync_execution_service.compute_embeddings_task(7, 1)

    assert task.status == "completed"
    assert task.total_items == 0
    assert embedding.chunk_calls == 0


@pytest.mark.asyncio
async def test_llm_failure_falls_back_to_topics_and_still_embeds(
    chunked_settings, monkeypatch
):
    repos = [_make_repo(index) for index in range(1, 6)]
    task = _make_task()
    state = _State(repos, task)
    embedding = _FakeEmbeddingService()
    _install(monkeypatch, state, embedding, _FakeLLMService(fail=True))

    await sync_execution_service.compute_embeddings_task(7, 1)

    assert all(repo.is_embedded for repo in repos)
    assert all(repo.ai_tags == ["topic-a", "topic-b"] for repo in repos)
    assert task.status == "completed"
    assert task.error_details["llm_failed_items"] == 5


@pytest.mark.asyncio
async def test_chunk_embedding_uses_configured_batch_size(
    chunked_settings, monkeypatch
):
    repos = [_make_repo(index) for index in range(1, 12)]
    task = _make_task()
    state = _State(repos, task)
    embedding = _FakeEmbeddingService()
    _install(monkeypatch, state, embedding, _FakeLLMService())

    await sync_execution_service.compute_embeddings_task(7, 1)

    assert embedding.batch_sizes == [CHUNK_SIZE, CHUNK_SIZE]
