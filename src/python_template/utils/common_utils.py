"""Common utility functions module.

提供各种通用的工具函数。
"""

import traceback
import uuid
from typing import Any

from .logger_util import get_logger

logger = get_logger(__name__)


def chunk_list(data: list[Any], chunk_size: int) -> list[list[Any]]:
    """将列表分割成指定大小的块。

    Args:
        data: 要分割的列表
        chunk_size: 每个块的大小

    Returns:
        分割后的列表

    Raises:
        ValueError: 如果 chunk_size 不是正数
    """
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i : i + chunk_size])

    logger.debug(f"Split list of {len(data)} items into {len(chunks)} chunks")
    return chunks


def flatten_dict(
    nested_dict: dict[str, Any], separator: str = ".", prefix: str = ""
) -> dict[str, Any]:
    """展平嵌套字典。

    Args:
        nested_dict: 嵌套字典
        separator: 键的分隔符
        prefix: 键的前缀

    Returns:
        展平后的字典
    """
    flattened = {}

    for key, value in nested_dict.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key

        if isinstance(value, dict):
            flattened.update(flatten_dict(value, separator, new_key))
        else:
            flattened[new_key] = value

    return flattened


def unflatten_dict(
    flattened_dict: dict[str, Any], separator: str = "."
) -> dict[str, Any]:
    """将展平的字典还原为嵌套结构。

    Args:
        flattened_dict: 展平的字典
        separator: 键的分隔符

    Returns:
        嵌套字典
    """
    unflattened = {}

    for key, value in flattened_dict.items():
        keys = key.split(separator)
        current = unflattened

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    return unflattened


def merge_dicts(*dicts: dict[str, Any]) -> dict[str, Any]:
    """合并多个字典。

    Args:
        *dicts: 要合并的字典

    Returns:
        合并后的字典
    """
    merged = {}

    for d in dicts:
        if isinstance(d, dict):
            merged.update(d)

    return merged


def filter_dict(
    data: dict[str, Any],
    allowed_keys: list[str] | None = None,
    excluded_keys: list[str] | None = None,
) -> dict[str, Any]:
    """根据允许或排除的键过滤字典。

    Args:
        data: 要过滤的字典
        allowed_keys: 允许的键列表(白名单)
        excluded_keys: 排除的键列表(黑名单)

    Returns:
        过滤后的字典
    """
    if allowed_keys is not None:
        return {k: v for k, v in data.items() if k in allowed_keys}

    if excluded_keys is not None:
        return {k: v for k, v in data.items() if k not in excluded_keys}

    return data.copy()


def generate_uuid() -> str:
    """生成 UUID4 字符串。

    Returns:
        UUID4 字符串
    """
    return str(uuid.uuid4())


def deep_merge_dict(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """深度合并两个字典。

    Args:
        base: 基础字典
        update: 更新字典

    Returns:
        合并后的字典
    """
    result = base.copy()

    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value

    return result


def safe_get(
    data: dict[str, Any],
    key_path: str,
    default: Any = None,
    separator: str = ".",
) -> Any:
    """安全地从嵌套字典中获取值。

    Args:
        data: 字典数据
        key_path: 键路径,使用分隔符分隔
        default: 默认值
        separator: 分隔符

    Returns:
        获取的值,失败时返回 default

    Example:
        >>> data = {"a": {"b": {"c": 123}}}
        >>> safe_get(data, "a.b.c")
        123
        >>> safe_get(data, "a.b.d", default=0)
        0
    """
    try:
        keys = key_path.split(separator)
        current = data

        for key in keys:
            current = current[key]

        return current
    except (KeyError, TypeError, AttributeError):
        return default


def safe_set(
    data: dict[str, Any],
    key_path: str,
    value: Any,
    separator: str = ".",
    create_missing: bool = True,
) -> bool:
    """安全地设置嵌套字典中的值。

    Args:
        data: 字典数据
        key_path: 键路径,使用分隔符分隔
        value: 要设置的值
        separator: 分隔符
        create_missing: 是否创建缺失的中间字典

    Returns:
        成功返回 True,失败返回 False

    Example:
        >>> data = {}
        >>> safe_set(data, "a.b.c", 123)
        True
        >>> data
        {'a': {'b': {'c': 123}}}
    """
    try:
        keys = key_path.split(separator)
        current = data

        for key in keys[:-1]:
            if key not in current:
                if create_missing:
                    current[key] = {}
                else:
                    return False
            current = current[key]

        current[keys[-1]] = value
        return True
    except (TypeError, AttributeError) as e:
        logger.error(f"Failed to set value at '{key_path}': {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return False


def remove_none_values(data: dict[str, Any], recursive: bool = False) -> dict[str, Any]:
    """移除字典中值为 None 的项。

    Args:
        data: 字典数据
        recursive: 是否递归处理嵌套字典

    Returns:
        处理后的字典
    """
    result = {}

    for key, value in data.items():
        if value is None:
            continue

        if recursive and isinstance(value, dict):
            result[key] = remove_none_values(value, recursive=True)
        else:
            result[key] = value

    return result


def remove_empty_values(
    data: dict[str, Any],
    recursive: bool = False,
) -> dict[str, Any]:
    """移除字典中的空值(None, "", [], {})。

    Args:
        data: 字典数据
        recursive: 是否递归处理嵌套字典

    Returns:
        处理后的字典
    """
    result = {}

    for key, value in data.items():
        # 检查是否为空值
        if value is None or value == "" or value == [] or value == {}:
            continue

        if recursive and isinstance(value, dict):
            result[key] = remove_empty_values(value, recursive=True)
        else:
            result[key] = value

    return result


def batch_process(
    items: list[Any],
    batch_size: int,
    process_func,
    *args,
    **kwargs,
) -> list[Any]:
    """批量处理列表项。

    Args:
        items: 要处理的项列表
        batch_size: 批次大小
        process_func: 处理函数
        *args: 传递给处理函数的额外参数
        **kwargs: 传递给处理函数的额外关键字参数

    Returns:
        处理结果列表
    """
    results = []
    batches = chunk_list(items, batch_size)

    for i, batch in enumerate(batches):
        logger.debug(f"Processing batch {i + 1}/{len(batches)}")
        try:
            batch_result = process_func(batch, *args, **kwargs)
            results.extend(
                batch_result if isinstance(batch_result, list) else [batch_result]
            )
        except Exception as e:
            logger.error(f"Error processing batch {i + 1}: {e}")
            logger.debug(f"Traceback:\n{traceback.format_exc()}")

    return results


def retry_on_exception(
    func,
    max_retries: int = 3,
    delay: float = 1.0,
    *args,
    **kwargs,
) -> Any:
    """重试执行函数直到成功或达到最大重试次数。

    Args:
        func: 要执行的函数
        max_retries: 最大重试次数
        delay: 重试间隔(秒)
        *args: 传递给函数的参数
        **kwargs: 传递给函数的关键字参数

    Returns:
        函数执行结果

    Raises:
        最后一次执行的异常
    """
    import time

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries + 1} attempts failed")
                logger.debug(f"Traceback:\n{traceback.format_exc()}")

    raise last_exception
