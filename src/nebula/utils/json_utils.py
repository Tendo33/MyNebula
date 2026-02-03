"""JSON utilities module.

提供 JSON 读写、序列化、反序列化等常用功能。
"""

import json
import traceback
from pathlib import Path
from typing import Any

import aiofiles

from .logger_util import get_logger

logger = get_logger(__name__)


def read_json(
    file_path: str | Path,
    encoding: str = "utf-8",
    default: Any = None,
) -> Any:
    """读取 JSON 文件。

    Args:
        file_path: JSON 文件路径
        encoding: 文件编码
        default: 读取失败时返回的默认值

    Returns:
        解析后的 JSON 数据,失败时返回 default
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"JSON file not found: {file_path}")
            return default

        with open(file_path, encoding=encoding) as f:
            data = json.load(f)
            logger.debug(f"Successfully read JSON from: {file_path}")
            return data

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {file_path}: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return default
    except Exception as e:
        logger.error(f"Failed to read JSON file {file_path}: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return default


def write_json(
    data: Any,
    file_path: str | Path,
    encoding: str = "utf-8",
    indent: int = 2,
    ensure_ascii: bool = False,
    create_dirs: bool = True,
) -> bool:
    """写入 JSON 文件。

    Args:
        data: 要写入的数据
        file_path: JSON 文件路径
        encoding: 文件编码
        indent: 缩进空格数
        ensure_ascii: 是否确保 ASCII 编码
        create_dirs: 是否自动创建父目录

    Returns:
        写入成功返回 True,失败返回 False
    """
    try:
        file_path = Path(file_path)

        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding=encoding) as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            logger.debug(f"Successfully wrote JSON to: {file_path}")
            return True

    except TypeError as e:
        logger.error(f"Data is not JSON serializable: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return False
    except Exception as e:
        logger.error(f"Failed to write JSON file {file_path}: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return False


def safe_json_loads(
    json_str: str,
    default: Any = None,
) -> Any:
    """安全地解析 JSON 字符串。

    Args:
        json_str: JSON 字符串
        default: 解析失败时返回的默认值

    Returns:
        解析后的数据,失败时返回 default
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        logger.debug(f"Failed to parse JSON string: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return default


def safe_json_dumps(
    obj: Any,
    default: Any = None,
    indent: int | None = None,
    ensure_ascii: bool = False,
    **kwargs,
) -> str | None:
    """安全地序列化对象为 JSON 字符串。

    Args:
        obj: 要序列化的对象
        default: 不可序列化对象的默认处理函数
        indent: 缩进空格数
        ensure_ascii: 是否确保 ASCII 编码
        **kwargs: 其他 json.dumps 参数

    Returns:
        JSON 字符串,失败时返回 None
    """
    try:
        return json.dumps(
            obj,
            default=default,
            indent=indent,
            ensure_ascii=ensure_ascii,
            **kwargs,
        )
    except (TypeError, ValueError) as e:
        logger.debug(f"Failed to serialize to JSON: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return None


def merge_json_files(
    file_paths: list[str | Path],
    output_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """合并多个 JSON 文件。

    Args:
        file_paths: JSON 文件路径列表
        output_path: 输出文件路径(可选)

    Returns:
        合并后的字典,失败时返回 None
    """
    try:
        merged = {}

        for file_path in file_paths:
            data = read_json(file_path)
            if isinstance(data, dict):
                merged.update(data)
            else:
                logger.warning(f"Skipping non-dict JSON file: {file_path}")

        if output_path:
            write_json(merged, output_path)

        logger.info(f"Merged {len(file_paths)} JSON files")
        return merged

    except Exception as e:
        logger.error(f"Failed to merge JSON files: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return None


def pretty_print_json(data: Any, indent: int = 2) -> None:
    """美化打印 JSON 数据。

    Args:
        data: 要打印的数据
        indent: 缩进空格数
    """
    try:
        print(json.dumps(data, indent=indent, ensure_ascii=False))
    except Exception as e:
        logger.error(f"Failed to pretty print JSON: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        print(data)


def validate_json_schema(data: dict[str, Any], required_keys: list[str]) -> bool:
    """验证 JSON 数据是否包含必需的键。

    Args:
        data: 要验证的数据
        required_keys: 必需的键列表

    Returns:
        验证通过返回 True,否则返回 False
    """
    try:
        missing_keys = [key for key in required_keys if key not in data]

        if missing_keys:
            logger.warning(f"Missing required keys: {missing_keys}")
            return False

        logger.debug("JSON schema validation passed")
        return True

    except Exception as e:
        logger.error(f"Failed to validate JSON schema: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return False


# Async functions for I/O-bound operations


async def async_read_json(
    file_path: str | Path,
    encoding: str = "utf-8",
    default: Any = None,
) -> Any:
    """异步读取 JSON 文件。

    Args:
        file_path: JSON 文件路径
        encoding: 文件编码
        default: 读取失败时返回的默认值

    Returns:
        解析后的 JSON 数据,失败时返回 default
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"JSON file not found: {file_path}")
            return default

        async with aiofiles.open(file_path, encoding=encoding) as f:
            content = await f.read()
            data = json.loads(content)
            logger.debug(f"Successfully read JSON from: {file_path}")
            return data

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {file_path}: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return default
    except Exception as e:
        logger.error(f"Failed to read JSON file {file_path}: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return default


async def async_write_json(
    data: Any,
    file_path: str | Path,
    encoding: str = "utf-8",
    indent: int = 2,
    ensure_ascii: bool = False,
    create_dirs: bool = True,
) -> bool:
    """异步写入 JSON 文件。

    Args:
        data: 要写入的数据
        file_path: JSON 文件路径
        encoding: 文件编码
        indent: 缩进空格数
        ensure_ascii: 是否确保 ASCII 编码
        create_dirs: 是否自动创建父目录

    Returns:
        写入成功返回 True,失败返回 False
    """
    try:
        file_path = Path(file_path)

        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        content = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)
        async with aiofiles.open(file_path, "w", encoding=encoding) as f:
            await f.write(content)
            logger.debug(f"Successfully wrote JSON to: {file_path}")
            return True

    except TypeError as e:
        logger.error(f"Data is not JSON serializable: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return False
    except Exception as e:
        logger.error(f"Failed to write JSON file {file_path}: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return False
