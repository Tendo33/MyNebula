"""Tests for `async_retry_decorator`.

这是 `nebula.utils.decorator_utils` 中唯一保留的装饰器,
被 embedding / LLM 供应商调用依赖,必须保持重试语义稳定。
"""

import asyncio

import pytest

from nebula.utils import async_retry_decorator


@pytest.mark.asyncio
async def test_returns_result_without_retry_on_success() -> None:
    calls = 0

    @async_retry_decorator(max_retries=3, delay=0)
    async def succeed() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    assert await succeed() == "ok"
    assert calls == 1


@pytest.mark.asyncio
async def test_retries_until_success() -> None:
    calls = 0

    @async_retry_decorator(max_retries=3, delay=0)
    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError("transient")
        return "ok"

    assert await flaky() == "ok"
    assert calls == 3


@pytest.mark.asyncio
async def test_raises_last_exception_after_exhausting_retries() -> None:
    calls = 0

    @async_retry_decorator(max_retries=2, delay=0)
    async def always_fails() -> None:
        nonlocal calls
        calls += 1
        raise ValueError(f"attempt {calls}")

    with pytest.raises(ValueError, match="attempt 3"):
        await always_fails()

    # max_retries=2 means one initial attempt plus two retries.
    assert calls == 3


@pytest.mark.asyncio
async def test_does_not_retry_unlisted_exception_types() -> None:
    calls = 0

    @async_retry_decorator(max_retries=3, delay=0, exceptions=(ValueError,))
    async def raises_type_error() -> None:
        nonlocal calls
        calls += 1
        raise TypeError("not retried")

    with pytest.raises(TypeError):
        await raises_type_error()

    assert calls == 1


@pytest.mark.asyncio
async def test_applies_exponential_backoff_between_attempts() -> None:
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        await real_sleep(0)

    @async_retry_decorator(max_retries=3, delay=0.5, backoff=2.0)
    async def always_fails() -> None:
        raise RuntimeError("boom")

    original = asyncio.sleep
    asyncio.sleep = record_sleep  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError):
            await always_fails()
    finally:
        asyncio.sleep = original  # type: ignore[assignment]

    assert sleeps == [0.5, 1.0, 2.0]


@pytest.mark.asyncio
async def test_preserves_function_metadata() -> None:
    @async_retry_decorator(max_retries=1, delay=0)
    async def documented() -> None:
        """Original docstring."""

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "Original docstring."


@pytest.mark.asyncio
async def test_forwards_arguments() -> None:
    @async_retry_decorator(max_retries=1, delay=0)
    async def add(a: int, b: int = 0) -> int:
        return a + b

    assert await add(2, b=3) == 5
