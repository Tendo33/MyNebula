"""Decorator utilities module.

提供常用的装饰器函数。
"""

import asyncio
import time
import traceback
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any

from .logger_util import get_logger

logger = get_logger(__name__)


def timing_decorator(func: Callable) -> Callable:
    """计算函数执行时间的装饰器。

    Args:
        func: 要装饰的函数

    Returns:
        装饰后的函数
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {e}")
            logger.debug(f"Traceback:\n{traceback.format_exc()}")
            raise
        finally:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            logger.info(
                f"⏱️  Function '{func.__name__}' executed in {execution_time:.4f} seconds"
            )

    return wrapper


def retry_decorator(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable:
    """失败重试装饰器。

    Args:
        max_retries: 最大重试次数
        delay: 初始延迟时间(秒)
        backoff: 延迟时间的倍增系数
        exceptions: 要捕获的异常类型元组

    Returns:
        装饰器函数
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception: BaseException | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"🔄 Function '{func.__name__}' failed (attempt {attempt + 1}/"
                            f"{max_retries + 1}): {e}. Retrying in {current_delay:.2f}s"
                        )
                        logger.debug(f"Traceback:\n{traceback.format_exc()}")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"❌ Function '{func.__name__}' failed after "
                            f"{max_retries + 1} attempts: {e}"
                        )
                        logger.debug(f"Traceback:\n{traceback.format_exc()}")

            if last_exception is not None:
                raise last_exception
            raise RuntimeError("Unexpected state: no exception captured")

        return wrapper

    return decorator


def catch_exceptions(
    default_return: Any = None,
    log_traceback: bool = True,
    reraise: bool = False,
) -> Callable:
    """捕获异常的装饰器。

    Args:
        default_return: 发生异常时的默认返回值
        log_traceback: 是否记录完整的 traceback
        reraise: 是否重新抛出异常

    Returns:
        装饰器函数
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Exception in {func.__name__}: {e}")
                if log_traceback:
                    logger.debug(f"Traceback:\n{traceback.format_exc()}")

                if reraise:
                    raise

                return default_return

        return wrapper

    return decorator


def log_calls(log_args: bool = True, log_result: bool = True) -> Callable:
    """记录函数调用的装饰器。

    Args:
        log_args: 是否记录参数
        log_result: 是否记录返回值

    Returns:
        装饰器函数
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__

            if log_args:
                logger.debug(
                    f"📞 Calling {func_name} with args={args}, kwargs={kwargs}"
                )
            else:
                logger.debug(f"📞 Calling {func_name}")

            try:
                result = func(*args, **kwargs)

                if log_result:
                    logger.debug(f"✅ {func_name} returned: {result}")
                else:
                    logger.debug(f"✅ {func_name} completed")

                return result

            except Exception as e:
                logger.error(f"❌ Exception in {func_name}: {e}")
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
                raise

        return wrapper

    return decorator


def deprecated(reason: str = "", alternative: str | None = None) -> Callable:
    """标记函数为已弃用的装饰器。

    Args:
        reason: 弃用原因
        alternative: 推荐的替代方案

    Returns:
        装饰器函数
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            message = f"⚠️  Function '{func.__name__}' is deprecated"
            if reason:
                message += f": {reason}"
            if alternative:
                message += f". Use '{alternative}' instead"

            logger.warning(message)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def singleton(cls):
    """单例模式装饰器。

    Args:
        cls: 要装饰的类

    Returns:
        装饰后的类
    """
    instances = {}

    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
            logger.debug(f"Created singleton instance of {cls.__name__}")
        return instances[cls]

    return get_instance


class ContextTimer:
    """上下文管理器,用于计时代码块执行时间。

    Example:
        with ContextTimer("数据处理"):
            # 执行耗时操作
            process_data()
    """

    def __init__(self, name: str = "operation", log_level: str = "INFO"):
        """初始化计时器。

        Args:
            name: 操作名称
            log_level: 日志级别
        """
        self.name = name
        self.log_level = log_level.upper()
        self.start_time: float | None = None
        self.end_time: float | None = None

    def __enter__(self):
        """开始计时。"""
        self.start_time = time.perf_counter()
        logger.debug(f"⏱️  Starting timer for: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """停止计时并记录结果。"""
        self.end_time = time.perf_counter()
        if self.start_time is None:
            return
        execution_time = self.end_time - self.start_time

        if exc_type is not None:
            logger.error(
                f"❌ Operation '{self.name}' failed after {execution_time:.4f} seconds"
            )
            logger.debug(f"Traceback:\n{traceback.format_exc()}")
        else:
            log_func = getattr(logger, self.log_level.lower(), logger.info)
            log_func(
                f"✅ Operation '{self.name}' completed in {execution_time:.4f} seconds"
            )

    @property
    def elapsed_time(self) -> float | None:
        """获取已经过的时间。

        Returns:
            已经过的时间(秒),如果未开始则返回 None
        """
        if self.start_time is None:
            return None

        end_time = self.end_time or time.perf_counter()
        return end_time - self.start_time


# =============================================================================
# Async Decorators (异步装饰器)
# =============================================================================


def async_timing_decorator(func: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
    """计算异步函数执行时间的装饰器。

    Args:
        func: 要装饰的异步函数

    Returns:
        装饰后的异步函数

    Example:
        @async_timing_decorator
        async def fetch_data():
            await asyncio.sleep(1)
            return "data"
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {e}")
            logger.debug(f"Traceback:\n{traceback.format_exc()}")
            raise
        finally:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            logger.info(
                f"⏱️  Async function '{func.__name__}' executed in {execution_time:.4f} seconds"
            )

    return wrapper


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


def async_catch_exceptions(
    default_return: Any = None,
    log_traceback: bool = True,
    reraise: bool = False,
) -> Callable[[Callable[..., Coroutine]], Callable[..., Coroutine]]:
    """捕获异步函数异常的装饰器。

    Args:
        default_return: 发生异常时的默认返回值
        log_traceback: 是否记录完整的 traceback
        reraise: 是否重新抛出异常

    Returns:
        装饰器函数

    Example:
        @async_catch_exceptions(default_return=None)
        async def safe_fetch():
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.json()
    """

    def decorator(func: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Exception in {func.__name__}: {e}")
                if log_traceback:
                    logger.debug(f"Traceback:\n{traceback.format_exc()}")

                if reraise:
                    raise

                return default_return

        return wrapper

    return decorator


def async_log_calls(
    log_args: bool = True, log_result: bool = True
) -> Callable[[Callable[..., Coroutine]], Callable[..., Coroutine]]:
    """记录异步函数调用的装饰器。

    Args:
        log_args: 是否记录参数
        log_result: 是否记录返回值

    Returns:
        装饰器函数

    Example:
        @async_log_calls(log_args=True, log_result=True)
        async def process_data(data):
            await asyncio.sleep(0.1)
            return {"processed": data}
    """

    def decorator(func: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            func_name = func.__name__

            if log_args:
                logger.debug(
                    f"📞 Calling async {func_name} with args={args}, kwargs={kwargs}"
                )
            else:
                logger.debug(f"📞 Calling async {func_name}")

            try:
                result = await func(*args, **kwargs)

                if log_result:
                    logger.debug(f"✅ Async {func_name} returned: {result}")
                else:
                    logger.debug(f"✅ Async {func_name} completed")

                return result

            except Exception as e:
                logger.error(f"❌ Exception in async {func_name}: {e}")
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
                raise

        return wrapper

    return decorator


class AsyncContextTimer:
    """异步上下文管理器,用于计时异步代码块执行时间。

    Example:
        async with AsyncContextTimer("异步数据处理"):
            await process_data()
    """

    def __init__(self, name: str = "operation", log_level: str = "INFO"):
        """初始化计时器。

        Args:
            name: 操作名称
            log_level: 日志级别
        """
        self.name = name
        self.log_level = log_level.upper()
        self.start_time: float | None = None
        self.end_time: float | None = None

    async def __aenter__(self) -> "AsyncContextTimer":
        """开始计时。"""
        self.start_time = time.perf_counter()
        logger.debug(f"⏱️  Starting async timer for: {self.name}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """停止计时并记录结果。"""
        self.end_time = time.perf_counter()
        if self.start_time is None:
            return

        execution_time = self.end_time - self.start_time

        if exc_type is not None:
            logger.error(
                f"❌ Async operation '{self.name}' failed after {execution_time:.4f} seconds"
            )
            logger.debug(f"Traceback:\n{traceback.format_exc()}")
        else:
            log_func = getattr(logger, self.log_level.lower(), logger.info)
            log_func(
                f"✅ Async operation '{self.name}' completed in {execution_time:.4f} seconds"
            )

    @property
    def elapsed_time(self) -> float | None:
        """获取已经过的时间。

        Returns:
            已经过的时间(秒),如果未开始则返回 None
        """
        if self.start_time is None:
            return None

        end_time = self.end_time or time.perf_counter()
        return end_time - self.start_time
