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

import json_repair
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
        max_tokens: int = 2048,
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
        max_tokens: int = 2048,
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
                max_tokens=4096,
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
        """Generate a summary for a repository.

        Args:
            full_name: Repository full name (owner/repo)
            description: Repository description
            topics: Repository topics/tags
            language: Primary programming language
            readme_content: README content (truncated)

        Returns:
            Summary in Chinese
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
            readme_truncated = readme_content[:4000]
            context_parts.append(f"README 摘要:\n{readme_truncated}")

        context = "\n".join(context_parts)

        prompt = f"""请为以下 GitHub 项目生成一段中文摘要（100-150 字）。

{context}

要求:
1. 涵盖项目的核心功能、技术特点和适用场景
2. 使用专业、流畅的中文
3. 不要使用 "这是一个" 开头
4. 字数控制在 100-150 字之间

直接输出摘要内容，不要有其他文字。"""

        try:
            response = await self.complete(
                prompt=prompt,
                system_prompt="你是一个技术文档专家，擅长用清晰的语言描述开源项目。",
                max_tokens=10000,
                temperature=0.3,
            )

            # Clean up response
            summary = response.strip()
            # Remove common prefixes
            for prefix in ["摘要:", "总结:", "简介:"]:
                if summary.startswith(prefix):
                    summary = summary[len(prefix) :].strip()

            return summary

        except Exception as e:
            logger.warning(f"LLM summary generation failed for {full_name}: {e}")
            # Fallback to description or default
            if description:
                return description
            return f"{full_name.split('/')[-1]} - 开源项目"

    async def generate_repo_summary_and_tags(
        self,
        full_name: str,
        description: str | None,
        topics: list[str] | None,
        language: str | None,
        readme_content: str | None,
    ) -> tuple[str, list[str]]:
        """Generate both summary and tags for a repository in one LLM call.

        Uses JSON output format for reliable parsing.
        IMPORTANT: This method guarantees non-empty tags through fallback mechanisms.

        Args:
            full_name: Repository full name (owner/repo)
            description: Repository description
            topics: Repository topics/tags from GitHub
            language: Primary programming language
            readme_content: README content (truncated)

        Returns:
            Tuple of (summary, tags) - tags is guaranteed to be non-empty
        """
        import json

        # Build structured context
        context_parts = [f"仓库名称: {full_name}"]

        if language:
            context_parts.append(f"主要语言: {language}")

        if description:
            context_parts.append(f"官方描述: {description}")

        if topics:
            context_parts.append(f"GitHub 标签: {', '.join(topics[:10])}")

        if readme_content:
            # Truncate README to avoid token limits
            readme_truncated = readme_content[:5000]
            context_parts.append(f"README 内容:\n{readme_truncated}")

        context = "\n".join(context_parts)

        system_prompt = """你是一名资深的开源项目分析师，专注于技术项目的分类和文档撰写。

你的职责：
1. 深入理解项目的技术架构和核心价值
2. 用专业、简洁的中文描述项目
3. 提取精准的分类标签

输出要求：
- 必须以纯 JSON 格式输出
- 不要添加 markdown 代码块标记
- 不要添加任何额外说明文字"""

        user_prompt = f"""分析以下 GitHub 项目，生成结构化的摘要和标签。

## 项目信息
{context}

## 输出要求

### summary (摘要)
- 字数：100-150 字
- 内容：核心功能、技术特点、适用场景
- 风格：专业流畅，避免"这是一个"等冗余开头
- 语言：中文

### tags (标签)
- 数量：3-7 个
- 长度：每个 2-6 字
- 覆盖：功能类型、技术领域、应用场景
- 语言：中文优先，技术专有名词可用英文

## 输出格式 (严格按此 JSON 格式)
{{"summary": "项目摘要内容", "tags": ["标签1", "标签2", "标签3"]}}"""

        tags: list[str] = []
        summary: str = ""

        try:
            response = await self.complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=2048,
                temperature=0.3,
            )

            # Parse JSON response
            response_text = response.strip()
            # Remove potential markdown code block markers
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(
                    lines[1:-1] if lines[-1] == "```" else lines[1:]
                )
            response_text = response_text.strip()

            try:
                data = json_repair.loads(response_text)
                summary = data.get("summary", "").strip()
                raw_tags = data.get("tags", [])
                if isinstance(raw_tags, list):
                    tags = [str(tag).strip() for tag in raw_tags if tag]
                    tags = [tag for tag in tags if tag and len(tag) <= 20][:7]
            except json.JSONDecodeError:
                # Fallback: try to extract from malformed response
                logger.warning(f"JSON parse failed for {full_name}, trying fallback")
                if "summary" in response_text and "tags" in response_text:
                    # Attempt regex extraction
                    import re

                    summary_match = re.search(
                        r'"summary"\s*:\s*"([^"]+)"', response_text
                    )
                    if summary_match:
                        summary = summary_match.group(1)
                    tags_match = re.search(r'"tags"\s*:\s*\[([^\]]+)\]', response_text)
                    if tags_match:
                        tags_str = tags_match.group(1)
                        tags = [t.strip().strip("\"'") for t in tags_str.split(",")]
                        tags = [tag for tag in tags if tag and len(tag) <= 20][:7]
                else:
                    summary = response_text[:200]

        except Exception as e:
            logger.warning(f"Summary/tag generation failed for {full_name}: {e}")
            summary = (
                description if description else f"{full_name.split('/')[-1]} - 开源项目"
            )

        # CRITICAL: Ensure tags are never empty
        # This is essential for accurate clustering
        tags = self._ensure_tags_not_empty(
            tags=tags,
            topics=topics,
            language=language,
            full_name=full_name,
        )

        return summary, tags

    def _ensure_tags_not_empty(
        self,
        tags: list[str],
        topics: list[str] | None,
        language: str | None,
        full_name: str,
    ) -> list[str]:
        """Ensure tags list is never empty through fallback mechanisms.

        Fallback priority:
        1. Use existing tags if available
        2. Use GitHub topics as fallback
        3. Use language as fallback
        4. Generate default tag from repo name

        Args:
            tags: LLM-generated tags (may be empty)
            topics: GitHub topics/tags
            language: Programming language
            full_name: Repository full name

        Returns:
            Non-empty list of tags
        """
        if tags and len(tags) >= 1:
            return tags

        result_tags: list[str] = []

        # Fallback 1: Use GitHub topics
        if topics and len(topics) > 0:
            # Convert English topics to simple Chinese tags or keep as-is
            for topic in topics[:5]:
                # Keep short topics as-is, they work as tags
                if len(topic) <= 15:
                    result_tags.append(topic)
            if result_tags:
                return result_tags

        # Fallback 2: Use language
        if language:
            result_tags.append(language)
            result_tags.append("开源项目")
            return result_tags

        # Fallback 3: Generate from repo name
        repo_name = full_name.split("/")[-1] if "/" in full_name else full_name
        # Extract meaningful parts from repo name
        parts = repo_name.replace("-", " ").replace("_", " ").split()
        if parts:
            result_tags.append(parts[0])
        result_tags.append("开源项目")

        return result_tags

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
