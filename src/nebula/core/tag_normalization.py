"""Tag and topic normalization utilities.

This module centralizes normalization logic used by:
- related repository ranking
- clustering naming/topic aggregation
- frontend/backed tag matching consistency
"""

from __future__ import annotations

import re
from collections import Counter

_TOKEN_SPLIT_RE = re.compile(r"[,\s/|;，、]+")
_NON_TOKEN_RE = re.compile(r"[^a-z0-9\-\u4e00-\u9fff]+")
_MULTI_DASH_RE = re.compile(r"-+")

TAG_SYNONYMS = {
    # Memory / agent memory
    "agent-memory": "agent-memory",
    "agent_memory": "agent-memory",
    "long-term-memory": "agent-memory",
    "longterm-memory": "agent-memory",
    "rag-memory": "agent-memory",
    "memory-augmented": "agent-memory",
    "mem0": "agent-memory",
    "记忆": "agent-memory",
    "长期记忆": "agent-memory",
    # RAG
    "retrieval-augmented-generation": "rag",
    "retrieval-augmented": "rag",
    "检索增强": "rag",
    "检索增强生成": "rag",
    # Agents
    "ai-agent": "agent",
    "ai-agents": "agent",
    "agents": "agent",
    "智能体": "agent",
    "agentic": "agent",
    # LLM
    "large-language-model": "llm",
    "大语言模型": "llm",
    # Training
    "distributed-training-framework": "distributed-training",
    "distributed-systems": "distributed-training",
    "deepspeed": "distributed-training",
    "fsdp": "distributed-training",
    "megatron": "distributed-training",
}


def normalize_tag_token(token: str) -> str:
    """Normalize one tag token to a stable canonical form."""
    cleaned = token.strip().lower()
    cleaned = cleaned.replace("_", "-")
    cleaned = _NON_TOKEN_RE.sub("-", cleaned)
    cleaned = _MULTI_DASH_RE.sub("-", cleaned)
    cleaned = cleaned.strip("-")
    if not cleaned:
        return ""
    return TAG_SYNONYMS.get(cleaned, cleaned)


def normalize_tag_list(tags: list[str] | None) -> list[str]:
    """Normalize a list of tags with uniqueness and stable order."""
    if not tags:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in tags:
        if not raw:
            continue

        parts = [raw]
        if any(ch in raw for ch in ",/|;，、 "):
            parts = [p for p in _TOKEN_SPLIT_RE.split(raw) if p]

        for part in parts:
            token = normalize_tag_token(part)
            if not token or token in seen:
                continue
            seen.add(token)
            normalized.append(token)

    return normalized


def weighted_tag_overlap_score(
    left: list[str] | None,
    right: list[str] | None,
) -> float:
    """Calculate weighted overlap score in [0, 1].

    Uses weighted Jaccard-like ratio where rarer tokens carry larger weight.
    """
    left_norm = normalize_tag_list(left)
    right_norm = normalize_tag_list(right)
    if not left_norm or not right_norm:
        return 0.0

    left_set = set(left_norm)
    right_set = set(right_norm)
    union = left_set | right_set
    inter = left_set & right_set
    if not union:
        return 0.0

    token_counts = Counter(left_norm + right_norm)

    def _weight(token: str) -> float:
        # lower frequency => higher discriminative weight.
        return 1.0 / float(token_counts[token])

    inter_weight = sum(_weight(t) for t in inter)
    union_weight = sum(_weight(t) for t in union)
    if union_weight <= 0:
        return 0.0

    return max(0.0, min(1.0, inter_weight / union_weight))


def merge_and_normalize_tag_sources(*tag_sources: list[str] | None) -> list[str]:
    """Merge multiple tag sources and normalize once."""
    merged: list[str] = []
    for source in tag_sources:
        if source:
            merged.extend(source)
    return normalize_tag_list(merged)

