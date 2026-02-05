"""Hash utilities for content change detection.

This module provides hash computation functions for detecting
changes in repository metadata (description, topics) to enable
smart incremental updates.
"""

import hashlib


def compute_content_hash(content: str | None) -> str:
    """Compute MD5 hash of text content.

    Args:
        content: Text content to hash. None or empty string returns empty string.

    Returns:
        MD5 hash as 32-character hexadecimal string, or empty string for empty content.
    """
    if not content:
        return ""
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def compute_topics_hash(topics: list[str] | None) -> str:
    """Compute MD5 hash of topics list.

    Topics are sorted before hashing to ensure order-independence.

    Args:
        topics: List of topic strings. None or empty list returns empty string.

    Returns:
        MD5 hash as 32-character hexadecimal string, or empty string for empty topics.
    """
    if not topics:
        return ""
    # Sort topics to ensure consistent hash regardless of order
    sorted_topics = ",".join(sorted(topics))
    return hashlib.md5(sorted_topics.encode("utf-8")).hexdigest()


def content_has_changed(
    old_hash: str | None,
    new_content: str | None,
) -> bool:
    """Check if content has changed by comparing hashes.

    Args:
        old_hash: Previously stored hash (can be None for new records).
        new_content: New content to check.

    Returns:
        True if content has changed, False otherwise.
    """
    new_hash = compute_content_hash(new_content)
    # Treat None and empty string as equivalent
    old = old_hash or ""
    return old != new_hash


def topics_have_changed(
    old_hash: str | None,
    new_topics: list[str] | None,
) -> bool:
    """Check if topics have changed by comparing hashes.

    Args:
        old_hash: Previously stored hash (can be None for new records).
        new_topics: New topics list to check.

    Returns:
        True if topics have changed, False otherwise.
    """
    new_hash = compute_topics_hash(new_topics)
    # Treat None and empty string as equivalent
    old = old_hash or ""
    return old != new_hash
