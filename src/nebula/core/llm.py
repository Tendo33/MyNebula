"""LLM service for text generation tasks.

This module provides a unified interface for LLM completions using
OpenAI-compatible APIs, supporting providers like:
- OpenAI
- SiliconFlow
- Zhipu AI
- Ollama (local)
- Any OpenAI-compatible endpoint

Used for:
- Generating cluster names and descriptions
- Creating repository summaries
- Natural language processing tasks
"""

from openai import AsyncOpenAI
from pydantic import BaseModel

from nebula.core.config import LLMSettings, get_llm_settings
from nebula.utils import get_logger
from nebula.utils.decorator_utils import async_retry_decorator

logger = get_logger(__name__)


class LLMResponse(BaseModel):
    """Response from LLM completion."""

    content: str
    model: str
    usage: dict | None = None


class LLMService:
    """Unified LLM service using OpenAI-compatible API.

    This service supports multiple LLM providers through a single interface.
    All providers use the OpenAI SDK since they expose OpenAI-compatible endpoints.

    Usage:
        service = LLMService()
        response = await service.complete("Summarize this text...")
        response = await service.complete_json(prompt, schema)
    """

    def __init__(self, settings: LLMSettings | None = None):
        """Initialize LLM service.

        Args:
            settings: Optional settings override. If not provided, uses global settings.
        """
        self.settings = settings or get_llm_settings()
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
    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """Generate completion for a prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-2)

        Returns:
            Generated text content
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.settings.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response.choices[0].message.content or ""

    @async_retry_decorator(max_retries=3, delay=1.0, backoff=2.0)
    async def complete_with_metadata(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Generate completion with full response metadata.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-2)

        Returns:
            LLMResponse with content and metadata
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.settings.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            usage=usage,
        )

    async def generate_cluster_info(
        self,
        repo_names: list[str],
        descriptions: list[str],
        topics: list[list[str]],
        languages: list[str],
    ) -> tuple[str, str, list[str]]:
        """Generate cluster name, description, and keywords using LLM.

        Args:
            repo_names: Names of repos in cluster
            descriptions: Descriptions of repos
            topics: Topics/tags of repos
            languages: Primary languages of repos

        Returns:
            Tuple of (name, description, keywords)
        """
        # Build context
        repos_info = []
        for i, name in enumerate(repo_names[:10]):  # Limit to 10 repos
            desc = descriptions[i] if i < len(descriptions) else ""
            topic = ", ".join(topics[i][:5]) if i < len(topics) and topics[i] else ""
            lang = languages[i] if i < len(languages) else ""
            repos_info.append(f"- {name}: {desc} [{lang}] ({topic})")

        repos_text = "\n".join(repos_info)
        all_topics = []
        for t in topics:
            if t:
                all_topics.extend(t)
        unique_topics = list(set(all_topics))[:20]

        all_languages = [lang for lang in languages if lang]
        unique_languages = list(set(all_languages))

        prompt = f"""分析以下 GitHub 仓库集合，生成一个简洁的集群描述。

仓库列表 ({len(repo_names)} 个):
{repos_text}

主要编程语言: {", ".join(unique_languages[:5])}
常见标签: {", ".join(unique_topics[:10])}

请按以下格式输出（使用 | 分隔）:
名称|描述|关键词1,关键词2,关键词3

要求:
1. 名称: 2-4 个词，概括这组仓库的主题
2. 描述: 一句话说明这组仓库的共同特点
3. 关键词: 3-5 个关键词

示例输出格式:
机器学习工具|用于数据科学和机器学习的 Python 工具库|机器学习,数据科学,Python,深度学习"""

        try:
            response = await self.complete(
                prompt=prompt,
                system_prompt="你是一个技术分析专家，擅长分析和总结 GitHub 项目。请用中文回答。",
                max_tokens=200,
                temperature=0.5,
            )

            # Parse response
            parts = response.strip().split("|")
            if len(parts) >= 3:
                name = parts[0].strip()
                description = parts[1].strip()
                keywords = [k.strip() for k in parts[2].split(",")]
                return name, description, keywords[:5]
            else:
                # Fallback parsing
                lines = response.strip().split("\n")
                name = lines[0] if lines else "技术项目集"
                description = (
                    lines[1] if len(lines) > 1 else f"包含 {len(repo_names)} 个相关仓库"
                )
                keywords = unique_topics[:5] if unique_topics else ["github", "开源"]
                return name, description, keywords

        except Exception as e:
            logger.warning(f"LLM cluster naming failed: {e}, using fallback")
            # Fallback to heuristic method
            from nebula.core.clustering import generate_cluster_name

            return generate_cluster_name(repo_names, descriptions, topics)

    async def generate_repo_summary(
        self,
        full_name: str,
        description: str | None,
        topics: list[str] | None,
        language: str | None,
        readme_content: str | None,
    ) -> str:
        """Generate a one-line summary for a repository.

        Args:
            full_name: Repository full name (owner/repo)
            description: Repository description
            topics: Repository topics/tags
            language: Primary programming language
            readme_content: README content (truncated)

        Returns:
            One-line summary in Chinese
        """
        # Build context
        context_parts = [f"仓库: {full_name}"]

        if language:
            context_parts.append(f"语言: {language}")

        if description:
            context_parts.append(f"描述: {description}")

        if topics:
            context_parts.append(f"标签: {', '.join(topics[:10])}")

        if readme_content:
            # Truncate README to avoid token limits
            readme_truncated = readme_content[:2000]
            context_parts.append(f"README 摘要:\n{readme_truncated}")

        context = "\n".join(context_parts)

        prompt = f"""请为以下 GitHub 项目生成一句话中文摘要（不超过 50 字）。

{context}

要求:
1. 简洁明了，突出项目核心功能
2. 使用中文
3. 不要使用 "这是一个" 开头
4. 不超过 50 个字

直接输出摘要内容，不要有其他文字。"""

        try:
            response = await self.complete(
                prompt=prompt,
                system_prompt="你是一个技术文档专家，擅长用简洁的语言描述开源项目。",
                max_tokens=100,
                temperature=0.3,
            )

            # Clean up response
            summary = response.strip()
            # Remove common prefixes
            for prefix in ["摘要:", "总结:", "简介:"]:
                if summary.startswith(prefix):
                    summary = summary[len(prefix) :].strip()

            return summary[:100]  # Ensure max length

        except Exception as e:
            logger.warning(f"LLM summary generation failed for {full_name}: {e}")
            # Fallback to description or default
            if description:
                return description[:100]
            return f"{full_name.split('/')[-1]} - 开源项目"

    async def generate_repo_tags(
        self,
        full_name: str,
        description: str | None,
        topics: list[str] | None,
        language: str | None,
        readme_content: str | None,
    ) -> list[str]:
        """Generate AI tags for a repository based on README and metadata.

        Args:
            full_name: Repository full name (owner/repo)
            description: Repository description
            topics: Repository topics/tags from GitHub
            language: Primary programming language
            readme_content: README content (truncated)

        Returns:
            List of 3-7 AI-generated tags in Chinese
        """
        # Build context
        context_parts = [f"仓库: {full_name}"]

        if language:
            context_parts.append(f"语言: {language}")

        if description:
            context_parts.append(f"描述: {description}")

        if topics:
            context_parts.append(f"GitHub 标签: {', '.join(topics[:10])}")

        if readme_content:
            # Truncate README to avoid token limits
            readme_truncated = readme_content[:3000]
            context_parts.append(f"README 内容:\n{readme_truncated}")

        context = "\n".join(context_parts)

        prompt = f"""请为以下 GitHub 项目生成 3-7 个精准的中文标签。

{context}

要求:
1. 标签应该描述项目的核心功能、技术栈、应用场景
2. 使用中文
3. 每个标签 2-4 个字
4. 标签之间用逗号分隔
5. 不要重复 GitHub 已有的标签
6. 关注项目的实际用途和技术特点

直接输出标签列表，格式：标签1,标签2,标签3"""

        try:
            response = await self.complete(
                prompt=prompt,
                system_prompt="你是一个技术标签专家，擅长为开源项目生成精准的分类标签。",
                max_tokens=100,
                temperature=0.3,
            )

            # Parse response
            tags = [tag.strip() for tag in response.strip().split(",")]
            # Clean up tags
            tags = [tag for tag in tags if tag and len(tag) <= 20]

            return tags[:7]  # Limit to 7 tags

        except Exception as e:
            logger.warning(f"AI tag generation failed for {full_name}: {e}")
            # Fallback to empty list
            return []

    async def generate_repo_summary_and_tags(
        self,
        full_name: str,
        description: str | None,
        topics: list[str] | None,
        language: str | None,
        readme_content: str | None,
    ) -> tuple[str, list[str]]:
        """Generate both summary and tags for a repository in one call.

        More efficient than calling generate_repo_summary and generate_repo_tags separately.

        Args:
            full_name: Repository full name (owner/repo)
            description: Repository description
            topics: Repository topics/tags from GitHub
            language: Primary programming language
            readme_content: README content (truncated)

        Returns:
            Tuple of (summary, tags)
        """
        # Build context
        context_parts = [f"仓库: {full_name}"]

        if language:
            context_parts.append(f"语言: {language}")

        if description:
            context_parts.append(f"描述: {description}")

        if topics:
            context_parts.append(f"GitHub 标签: {', '.join(topics[:10])}")

        if readme_content:
            # Truncate README to avoid token limits
            readme_truncated = readme_content[:2500]
            context_parts.append(f"README 内容:\n{readme_truncated}")

        context = "\n".join(context_parts)

        prompt = f"""请为以下 GitHub 项目生成中文摘要和标签。

{context}

请按以下格式输出（使用 | 分隔）:
摘要|标签1,标签2,标签3,标签4,标签5

要求:
1. 摘要: 一句话概括项目核心功能，不超过 50 字，不要用"这是一个"开头
2. 标签: 3-7 个精准的中文标签，每个 2-4 字，用逗号分隔

直接输出，不要有其他文字。"""

        try:
            response = await self.complete(
                prompt=prompt,
                system_prompt="你是一个技术文档专家，擅长用简洁的语言描述和分类开源项目。",
                max_tokens=150,
                temperature=0.3,
            )

            # Parse response
            parts = response.strip().split("|")
            if len(parts) >= 2:
                summary = parts[0].strip()
                tags = [tag.strip() for tag in parts[1].split(",")]
                tags = [tag for tag in tags if tag and len(tag) <= 20][:7]
                return summary[:100], tags
            else:
                # Fallback: treat entire response as summary
                return response.strip()[:100], []

        except Exception as e:
            logger.warning(f"Summary/tag generation failed for {full_name}: {e}")
            # Fallback
            if description:
                return description[:100], []
            return f"{full_name.split('/')[-1]} - 开源项目", []

    async def generate_batch_summaries(
        self,
        repos: list[dict],
        batch_size: int = 5,
    ) -> list[str]:
        """Generate summaries for multiple repositories.

        Args:
            repos: List of repo dicts with keys: full_name, description, topics, language, readme_content
            batch_size: Number of repos to process in parallel

        Returns:
            List of summaries in same order as input
        """
        import asyncio

        summaries = []

        for i in range(0, len(repos), batch_size):
            batch = repos[i : i + batch_size]
            tasks = [
                self.generate_repo_summary(
                    full_name=r.get("full_name", ""),
                    description=r.get("description"),
                    topics=r.get("topics"),
                    language=r.get("language"),
                    readme_content=r.get("readme_content"),
                )
                for r in batch
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.warning(f"Summary generation failed: {result}")
                    repo = batch[j]
                    summaries.append(repo.get("description", "")[:100] or "开源项目")
                else:
                    summaries.append(result)

            logger.debug(f"Generated summaries: {len(summaries)}/{len(repos)}")

        return summaries

    async def close(self) -> None:
        """Close the client connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None


# Global service instance
_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    """Get global LLM service instance.

    Returns:
        LLMService instance
    """
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


async def close_llm_service() -> None:
    """Close global LLM service."""
    global _llm_service
    if _llm_service is not None:
        await _llm_service.close()
        _llm_service = None
