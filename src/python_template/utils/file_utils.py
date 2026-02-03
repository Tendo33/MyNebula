"""File utilities module.

提供文件操作相关的常用功能。
"""

import hashlib
import re
import shutil
import traceback
from functools import lru_cache
from pathlib import Path

import aiofiles

from .logger_util import get_logger

logger = get_logger(__name__)

# Pre-compiled regex patterns for filename sanitization
FILENAME_ILLEGAL_PATTERN = re.compile(r'[<>:"/\\|?*]')
FILENAME_CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def ensure_directory(directory_path: str | Path) -> Path | None:
    """确保目录存在,不存在则创建。

    Args:
        directory_path: 目录路径

    Returns:
        Path 对象,失败时返回 None
    """
    try:
        dir_path = Path(directory_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {dir_path}")
        return dir_path
    except Exception as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return None


def get_file_size(file_path: str | Path) -> int | None:
    """获取文件大小(字节)。

    Args:
        file_path: 文件路径

    Returns:
        文件大小(字节),失败时返回 None
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None

        size = file_path.stat().st_size
        logger.debug(f"File size: {file_path} = {size} bytes")
        return size
    except Exception as e:
        logger.error(f"Failed to get file size for {file_path}: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return None


@lru_cache(maxsize=128)
def format_file_size(size_bytes: int) -> str:
    """格式化文件大小为人类可读格式。

    Args:
        size_bytes: 字节数

    Returns:
        格式化后的文件大小字符串
    """
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB", "PB"]
    import math

    i = int(math.floor(math.log(size_bytes, 1024)))
    i = min(i, len(size_names) - 1)  # 防止索引越界
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)

    return f"{s} {size_names[i]}"


def calculate_file_hash(
    file_path: str | Path,
    algorithm: str = "sha256",
) -> str | None:
    """计算文件哈希值。

    Args:
        file_path: 文件路径
        algorithm: 哈希算法 (md5, sha1, sha256, sha512)

    Returns:
        哈希值的十六进制字符串,失败时返回 None
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None

        hasher = hashlib.new(algorithm)
        logger.debug(f"Calculating {algorithm} hash for: {file_path}")

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)

        hash_value = hasher.hexdigest()
        logger.debug(f"File hash ({algorithm}): {hash_value}")
        return hash_value

    except ValueError:
        logger.error(f"Unsupported hash algorithm: {algorithm}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return None
    except Exception as e:
        logger.error(f"Failed to calculate hash for {file_path}: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return None


def copy_file(
    src: str | Path,
    dst: str | Path,
    create_dirs: bool = True,
) -> bool:
    """复制文件。

    Args:
        src: 源文件路径
        dst: 目标文件路径
        create_dirs: 是否自动创建目标目录

    Returns:
        成功返回 True,失败返回 False
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            logger.error(f"Source file not found: {src_path}")
            return False

        if create_dirs:
            dst_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src_path, dst_path)
        logger.info(f"Copied file: {src_path} -> {dst_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to copy file {src} to {dst}: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return False


def move_file(
    src: str | Path,
    dst: str | Path,
    create_dirs: bool = True,
) -> bool:
    """移动文件。

    Args:
        src: 源文件路径
        dst: 目标文件路径
        create_dirs: 是否自动创建目标目录

    Returns:
        成功返回 True,失败返回 False
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            logger.error(f"Source file not found: {src_path}")
            return False

        if create_dirs:
            dst_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(str(src_path), str(dst_path))
        logger.info(f"Moved file: {src_path} -> {dst_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to move file {src} to {dst}: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return False


def delete_file(file_path: str | Path, missing_ok: bool = True) -> bool:
    """删除文件。

    Args:
        file_path: 文件路径
        missing_ok: 文件不存在时是否报错

    Returns:
        成功返回 True,失败返回 False
    """
    try:
        file_path = Path(file_path)

        if not file_path.exists():
            if missing_ok:
                logger.debug(f"File not found (ignored): {file_path}")
                return True
            else:
                logger.error(f"File not found: {file_path}")
                return False

        file_path.unlink()
        logger.info(f"Deleted file: {file_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to delete file {file_path}: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return False


def list_files(
    directory: str | Path,
    pattern: str = "*",
    recursive: bool = False,
) -> list[Path]:
    """列出目录中的文件。

    Args:
        directory: 目录路径
        pattern: 文件名模式 (支持通配符)
        recursive: 是否递归搜索子目录

    Returns:
        文件路径列表
    """
    try:
        dir_path = Path(directory)

        if not dir_path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            return []

        if recursive:
            files = list(dir_path.rglob(pattern))
        else:
            files = list(dir_path.glob(pattern))

        # 只返回文件,不包括目录
        files = [f for f in files if f.is_file()]

        logger.debug(f"Found {len(files)} files in {dir_path}")
        return files

    except Exception as e:
        logger.error(f"Failed to list files in {directory}: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return []


def read_text_file(
    file_path: str | Path,
    encoding: str = "utf-8",
    default: str = "",
) -> str:
    """读取文本文件。

    Args:
        file_path: 文件路径
        encoding: 文件编码
        default: 读取失败时返回的默认值

    Returns:
        文件内容,失败时返回 default
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return default

        with open(file_path, encoding=encoding) as f:
            content = f.read()
            logger.debug(f"Read {len(content)} characters from: {file_path}")
            return content

    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return default


def write_text_file(
    content: str,
    file_path: str | Path,
    encoding: str = "utf-8",
    create_dirs: bool = True,
) -> bool:
    """写入文本文件。

    Args:
        content: 要写入的内容
        file_path: 文件路径
        encoding: 文件编码
        create_dirs: 是否自动创建父目录

    Returns:
        成功返回 True,失败返回 False
    """
    try:
        file_path = Path(file_path)

        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding=encoding) as f:
            f.write(content)
            logger.debug(f"Wrote {len(content)} characters to: {file_path}")
            return True

    except Exception as e:
        logger.error(f"Failed to write file {file_path}: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return False


@lru_cache(maxsize=128)
def sanitize_filename(filename: str, replacement: str = "_") -> str:
    """清理文件名,移除或替换非法字符。

    Args:
        filename: 原始文件名
        replacement: 替换字符

    Returns:
        清理后的文件名
    """
    # 移除或替换非法字符
    sanitized = FILENAME_ILLEGAL_PATTERN.sub(replacement, filename)

    # 移除控制字符
    sanitized = FILENAME_CONTROL_PATTERN.sub("", sanitized)

    # 限制长度
    if len(sanitized) > 255:
        name, ext = Path(sanitized).stem, Path(sanitized).suffix
        max_name_len = 255 - len(ext)
        sanitized = name[:max_name_len] + ext

    return sanitized.strip()


# Async functions for I/O-bound operations


async def async_read_text_file(
    file_path: str | Path,
    encoding: str = "utf-8",
    default: str = "",
) -> str:
    """异步读取文本文件。

    Args:
        file_path: 文件路径
        encoding: 文件编码
        default: 读取失败时返回的默认值

    Returns:
        文件内容,失败时返回 default
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return default

        async with aiofiles.open(file_path, encoding=encoding) as f:
            content = await f.read()
            logger.debug(f"Read {len(content)} characters from: {file_path}")
            return content

    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return default


async def async_write_text_file(
    content: str,
    file_path: str | Path,
    encoding: str = "utf-8",
    create_dirs: bool = True,
) -> bool:
    """异步写入文本文件。

    Args:
        content: 要写入的内容
        file_path: 文件路径
        encoding: 文件编码
        create_dirs: 是否自动创建父目录

    Returns:
        成功返回 True,失败返回 False
    """
    try:
        file_path = Path(file_path)

        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(file_path, "w", encoding=encoding) as f:
            await f.write(content)
            logger.debug(f"Wrote {len(content)} characters to: {file_path}")
            return True

    except Exception as e:
        logger.error(f"Failed to write file {file_path}: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        return False
