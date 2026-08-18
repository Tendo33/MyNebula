"""Decorator utilities module.

提供 MyNebula 运行时实际使用的装饰器。

当前只保留 `async_retry_decorator`：外部 embedding / LLM 供应商调用需要带
指数退避的重试。重试粒度必须落在单次 API 调用上，调用方不要再叠加一层重试。
"""

import asyncio
import traceback
from collections.abc import Callable, Coroutine
from functools import wraps

from .logger_util import get_logger

logger = get_logger(__name__)


def async_retry_decorator(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., Coroutine]], Callable[..., Coroutine]]:
    """异步失败重试装饰器。

    Args:
        max_retries: 最大重试次数
        delay: 初始延迟时间(秒)
        backoff: 延迟时间的倍增系数
        exceptions: 要捕获的异常类型元组

    Returns:
        装饰器函数

    Example:
        @async_retry_decorator(max_retries=3, delay=1.0)
        async def unstable_api_call():
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.json()
    """

    def decorator(func: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception: BaseException | None = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"🔄 Async function '{func.__name__}' failed "
                            f"(attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {current_delay:.2f}s"
                        )
                        logger.debug(f"Traceback:\n{traceback.format_exc()}")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"❌ Async function '{func.__name__}' failed after "
                            f"{max_retries + 1} attempts: {e}"
                        )
                        logger.debug(f"Traceback:\n{traceback.format_exc()}")

            if last_exception is not None:
                raise last_exception
            raise RuntimeError("Unexpected state: no exception captured")

        return wrapper

    return decorator
