"""Tests for embedding batch retry granularity and batch-size configuration.

Regression guard for two defects found in the 2026-08-18 scan:

- `embed_batch` used to carry the retry decorator on the whole loop, so a
  failure in slice N replayed slices 0..N-1 against a metered provider.
- `SYNC_BATCH_SIZE` was declared, validated, and documented, but no code read
  it; the embedding stage hardcoded 32.
"""

from types import SimpleNamespace

import pytest

from nebula.core.config import get_sync_settings
from nebula.core.embedding import EmbeddingService


class _FakeEmbeddings:
    """Records every provider request and can fail on chosen call numbers."""

    def __init__(self, dimensions: int, fail_on_calls: set[int] | None = None):
        self.dimensions = dimensions
        self.fail_on_calls = fail_on_calls or set()
        self.requests: list[list[str]] = []
        self.call_count = 0

    async def create(self, *, model: str, input: list[str]):
        self.call_count += 1
        self.requests.append(list(input))
        if self.call_count in self.fail_on_calls:
            raise RuntimeError(f"provider failure on call {self.call_count}")
        data = [
            SimpleNamespace(index=index, embedding=[float(index)] * self.dimensions)
            for index in range(len(input))
        ]
        return SimpleNamespace(data=data)


class _FakeClient:
    def __init__(self, embeddings: _FakeEmbeddings):
        self.embeddings = embeddings


def _service(fake: _FakeEmbeddings) -> EmbeddingService:
    service = EmbeddingService()
    service._client = _FakeClient(fake)  # type: ignore[assignment]
    return service


@pytest.fixture
def fast_retry(monkeypatch):
    """Remove retry sleeps so the tests stay fast."""
    import nebula.utils.decorator_utils as decorator_utils

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(decorator_utils.asyncio, "sleep", no_sleep)


@pytest.mark.asyncio
async def test_successful_slices_are_not_resent_when_a_later_slice_fails(fast_retry):
    """The whole point of A1: retry must not replay the successful prefix."""
    service = _service(_FakeEmbeddings(dimensions=4, fail_on_calls={2}))
    fake = service._client.embeddings  # type: ignore[union-attr]

    texts = [f"text-{index}" for index in range(6)]
    embeddings = await service.embed_batch(texts, batch_size=2)

    assert len(embeddings) == 6
    # 3 slices, one of which failed once and was retried: 4 provider calls.
    # The pre-fix behaviour replayed the whole loop and would issue 6.
    assert fake.call_count == 4
    # Slice 1 must have been requested exactly once.
    assert fake.requests.count(["text-0", "text-1"]) == 1
    # Slice 2 is the retried one.
    assert fake.requests.count(["text-2", "text-3"]) == 2


@pytest.mark.asyncio
async def test_slice_retry_exhaustion_propagates(fast_retry):
    service = _service(_FakeEmbeddings(dimensions=4, fail_on_calls={1, 2, 3, 4}))
    fake = service._client.embeddings  # type: ignore[union-attr]

    with pytest.raises(RuntimeError):
        await service.embed_batch(["a", "b"], batch_size=1)

    # One initial attempt plus three retries on the first slice, then give up.
    assert fake.call_count == 4


@pytest.mark.asyncio
async def test_batch_size_defaults_to_sync_settings(monkeypatch):
    # SyncSettings validates batch_size >= 10, so 10 is the smallest usable value.
    monkeypatch.setenv("SYNC_BATCH_SIZE", "10")
    get_sync_settings.cache_clear()
    try:
        service = _service(_FakeEmbeddings(dimensions=4))
        fake = service._client.embeddings  # type: ignore[union-attr]

        await service.embed_batch([f"text-{index}" for index in range(25)])

        assert [len(request) for request in fake.requests] == [10, 10, 5]
    finally:
        get_sync_settings.cache_clear()


@pytest.mark.asyncio
async def test_explicit_batch_size_overrides_settings(monkeypatch):
    monkeypatch.setenv("SYNC_BATCH_SIZE", "100")
    get_sync_settings.cache_clear()
    try:
        service = _service(_FakeEmbeddings(dimensions=4))
        fake = service._client.embeddings  # type: ignore[union-attr]

        await service.embed_batch(["a", "b", "c"], batch_size=2)

        assert [len(request) for request in fake.requests] == [2, 1]
    finally:
        get_sync_settings.cache_clear()


@pytest.mark.asyncio
async def test_empty_text_becomes_zero_vector_and_order_is_preserved():
    service = EmbeddingService()
    # Placeholder substitution uses the service's configured dimensions, so the
    # fake provider must agree with them for the assertion to be meaningful.
    dimensions = service.settings.dimensions
    fake = _FakeEmbeddings(dimensions=dimensions)
    service._client = _FakeClient(fake)  # type: ignore[assignment]

    embeddings = await service.embed_batch(["alpha", "   ", "gamma"], batch_size=3)

    assert embeddings[1] == [0.0] * dimensions
    # The fake returns [index] * dims, so ordering is observable.
    assert embeddings[0] == [0.0] * dimensions
    assert embeddings[2] == [2.0] * dimensions
    # The blank text is sent as a single space, never as an empty string.
    assert fake.requests == [["alpha", " ", "gamma"]]


@pytest.mark.asyncio
async def test_no_texts_makes_no_provider_call():
    service = _service(_FakeEmbeddings(dimensions=3))
    fake = service._client.embeddings  # type: ignore[union-attr]

    assert await service.embed_batch([]) == []
    assert fake.call_count == 0
