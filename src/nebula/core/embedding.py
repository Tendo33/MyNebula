"""Unified embedding service supporting multiple providers.

This module provides a unified interface for text embeddings using
OpenAI-compatible APIs, supporting providers like:
- OpenAI
- SiliconFlow
- Jina AI
- Ollama (local)
- 智谱 AI (Zhipu)
- Any OpenAI-compatible endpoint
"""

import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel

from nebula.core.config import EmbeddingSettings, get_embedding_settings
from nebula.utils import get_logger
from nebula.utils.decorator_utils import async_retry_decorator

logger = get_logger(__name__)


def _normalize_semantic_tags(tags: list[str] | None) -> list[str]:
    """Normalize tags for more consistent semantic embeddings."""
    if not tags:
        return []

    from nebula.core.clustering import normalize_topic_token

    normalized_tags: list[str] = []
    seen: set[str] = set()
    for raw_tag in tags:
        if not raw_tag:
            continue

        canonical_tag = normalize_topic_token(raw_tag)
        for candidate in (canonical_tag, raw_tag.strip()):
            if not candidate:
                continue
            candidate_key = candidate.lower()
            if candidate_key in seen:
                continue
            seen.add(candidate_key)
            normalized_tags.append(candidate)

    return normalized_tags


class EmbeddingResult(BaseModel):
    """Result of an embedding operation."""

    text: str
    embedding: list[float]
    model: str
    dimensions: int


class EmbeddingService:
    """Unified embedding service using OpenAI-compatible API.

    This service supports multiple embedding providers through a single interface.
    All providers use the OpenAI SDK since they expose OpenAI-compatible endpoints.

    Usage:
        service = EmbeddingService()
        embedding = await service.embed_text("Hello world")
        embeddings = await service.embed_batch(["Hello", "World"])
    """

    def __init__(self, settings: EmbeddingSettings | None = None):
        """Initialize embedding service.

        Args:
            settings: Optional settings override. If not provided, uses global settings.
        """
        self.settings = settings or get_embedding_settings()
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        """Get or create async OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
            )
        return self._client

    @async_retry_decorator(max_retries=3, delay=1.0, backoff=2.0)
    async def embed_text(self, text: str) -> list[float]:
        """Compute embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        if not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.settings.dimensions

        response = await self.client.embeddings.create(
            model=self.settings.model,
            input=text,
        )
        return response.data[0].embedding

    @async_retry_decorator(max_retries=3, delay=1.0, backoff=2.0)
    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Compute embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Replace empty strings with placeholder
            processed_batch = [t if t.strip() else " " for t in batch]

            response = await self.client.embeddings.create(
                model=self.settings.model,
                input=processed_batch,
            )

            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [d.embedding for d in sorted_data]

            # Replace placeholder embeddings with zero vectors
            for j, text in enumerate(batch):
                if not text.strip():
                    batch_embeddings[j] = [0.0] * self.settings.dimensions

            all_embeddings.extend(batch_embeddings)

            logger.debug(
                f"Embedded batch {i // batch_size + 1}, total: {len(all_embeddings)}/{len(texts)}"
            )

        return all_embeddings

    def build_repo_text(
        self,
        full_name: str,
        description: str | None = None,
        topics: list[str] | None = None,
        readme_summary: str | None = None,
        language: str | None = None,
        ai_summary: str | None = None,
        ai_tags: list[str] | None = None,
    ) -> str:
        """Build text for repository embedding.

        Combines repository metadata into a single text optimized for embedding.
        Prioritizes LLM-generated content (ai_summary, ai_tags) over raw metadata
        for better semantic clustering accuracy.

        Args:
            full_name: Repository full name (owner/repo)
            description: Repository description
            topics: Repository topics/tags
            readme_summary: Summarized README content
            language: Primary programming language
            ai_summary: LLM-generated summary (preferred over description)
            ai_tags: LLM-generated semantic tags (preferred over topics)

        Returns:
            Combined text for embedding
        """
        parts = [full_name]

        if language:
            parts.append(f"Language: {language}")

        # Prefer LLM-generated summary over raw description
        # LLM summary is more concise and semantically rich
        if ai_summary:
            parts.append(ai_summary)
        elif description:
            parts.append(description)

        # Prefer LLM-generated tags over raw topics
        # LLM tags are semantically normalized and more meaningful for clustering
        normalized_ai_tags = _normalize_semantic_tags(ai_tags)
        normalized_topics = _normalize_semantic_tags(topics)

        if normalized_ai_tags:
            parts.append(f"Tags: {', '.join(normalized_ai_tags)}")
        elif normalized_topics:
            parts.append(f"Topics: {', '.join(normalized_topics)}")

        # Include readme summary if available (additional context)
        if readme_summary:
            parts.append(readme_summary)

        return "\n".join(parts)

    async def compute_similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0 to 1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    async def close(self) -> None:
        """Close the client connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None


# Global service instance
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get global embedding service instance.

    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


async def close_embedding_service() -> None:
    """Close global embedding service."""
    global _embedding_service
    if _embedding_service is not None:
        await _embedding_service.close()
        _embedding_service = None
