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
        existing_cluster_names: list[str] | None = None,
    ) -> tuple[str, str, list[str]]:
        """Generate cluster name, description, and keywords using LLM.

        Args:
            repo_names: Names of repos in cluster
            descriptions: Descriptions of repos
            topics: Topics/tags of repos
            languages: Primary languages of repos
            existing_cluster_names: Names already assigned to other clusters,
                used to ensure this cluster gets a unique and distinctive name.

        Returns:
            Tuple of (name, description, keywords)
        """
        from nebula.core.clustering import normalize_topic_lists, sanitize_cluster_name

        normalized_topics = normalize_topic_lists(topics)

        # Build context
        repos_info = []
        for i, name in enumerate(repo_names[:10]):  # Limit to 10 repos
            desc = descriptions[i] if i < len(descriptions) else ""
            topic = (
                ", ".join(normalized_topics[i][:5])
                if i < len(normalized_topics) and normalized_topics[i]
                else ""
            )
            lang = languages[i] if i < len(languages) else ""
            repos_info.append(f"- {name}: {desc} [{lang}] ({topic})")

        repos_text = "\n".join(repos_info)
        topic_counts: dict[str, int] = {}
        for t in normalized_topics:
            if t:
                for topic in t:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1

        unique_topics = [
            topic
            for topic, _ in sorted(
                topic_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:20]
        ]

        language_counts: dict[str, int] = {}
        for language in languages:
            if language:
                language_counts[language] = language_counts.get(language, 0) + 1

        unique_languages = [
            language
            for language, _ in sorted(
                language_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]

        is_chinese = self.settings.output_language == "zh"

        # Build the exclusion constraint for uniqueness
        exclusion_block = ""
        if existing_cluster_names:
            names_list = ", ".join(f'"{n}"' for n in existing_cluster_names)
            if is_chinese:
                exclusion_block = (
                    f"\n\n已有的集群名称（你必须避免使用相同或相似的名称）:\n{names_list}"
                )
            else:
                exclusion_block = (
                    "\n\nExisting cluster names (you must avoid reusing or paraphrasing them):\n"
                    f"{names_list}"
                )

        if is_chinese:
            prompt = f"""分析以下 GitHub 仓库集合，生成一个简洁且有区分度的集群描述。

仓库列表 ({len(repo_names)} 个):
{repos_text}

主要编程语言: {", ".join(unique_languages[:5])}
常见标签: {", ".join(unique_topics[:10])}{exclusion_block}

请按以下格式输出（使用 | 分隔，只输出一行）:
名称|描述|关键词1,关键词2,关键词3

要求:
1. 名称: 2-5 个中文词，准确概括这组仓库的**核心差异化主题**（不要使用宽泛的"LLM"、"AI"等词，要具体到应用场景或技术领域）
2. 描述: 一句话说明这组仓库区别于其他集群的独特共同特点
3. 关键词: 3-5 个关键词，尽量具体

示例输出格式:
RAG检索增强|基于向量检索的大模型知识增强框架|RAG,向量检索,知识库,LangChain
代码生成助手|AI驱动的代码自动生成与补全工具|代码生成,Copilot,IDE插件"""
            system_prompt = (
                "你是一个技术分析专家，擅长分析和分类 GitHub 项目。"
                "你的任务是为每个集群生成独特且有区分度的名称。"
                "请用中文回答，只输出要求的格式，不要输出其他内容。"
            )
        else:
            prompt = f"""Analyze the following GitHub repositories and generate a concise, distinctive cluster description.

Repositories ({len(repo_names)}):
{repos_text}

Primary languages: {", ".join(unique_languages[:5])}
Common topics: {", ".join(unique_topics[:10])}{exclusion_block}

Return one line only with this format (use | as separator):
Name|Description|keyword1,keyword2,keyword3

Requirements:
1. Name: 2-5 words, specific and differentiating (avoid broad labels like "AI" or "LLM")
2. Description: one sentence describing what makes this cluster unique
3. Keywords: 3-5 specific keywords
"""
            system_prompt = (
                "You are a technical analyst specializing in GitHub project taxonomy. "
                "Return only the requested single-line format."
            )

        try:
            response = await self.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=2560,
                temperature=0.3,
            )

            # Parse response - take only the first non-empty line
            response_line = ""
            for line in response.strip().split("\n"):
                line = line.strip()
                if line and "|" in line:
                    response_line = line
                    break

            if not response_line:
                response_line = response.strip().split("\n")[0]

            parts = response_line.split("|")
            if len(parts) >= 3:
                name = sanitize_cluster_name(parts[0].strip())
                description = parts[1].strip()
                keywords = [k.strip() for k in parts[2].split(",") if k.strip()]
                return name, description, keywords[:5]
            else:
                # Fallback parsing
                name = sanitize_cluster_name(
                    response_line if response_line else ("技术项目集" if is_chinese else "Tech Cluster")
                )
                description = (
                    f"包含 {len(repo_names)} 个相关仓库"
                    if is_chinese
                    else f"Contains {len(repo_names)} related repositories"
                )
                keywords = (
                    unique_topics[:5]
                    if unique_topics
                    else (["github", "开源"] if is_chinese else ["github", "open-source"])
                )
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
            Summary text in configured output language
        """
        is_chinese = self.settings.output_language == "zh"

        # Build context
        context_parts = [f"仓库: {full_name}" if is_chinese else f"Repository: {full_name}"]

        if language:
            context_parts.append(f"语言: {language}" if is_chinese else f"Language: {language}")

        if description:
            context_parts.append(f"描述: {description}" if is_chinese else f"Description: {description}")

        if topics:
            context_parts.append(f"标签: {', '.join(topics[:10])}" if is_chinese else f"Topics: {', '.join(topics[:10])}")

        if readme_content:
            # Truncate README to avoid token limits
            readme_truncated = readme_content[:4000]
            context_parts.append(
                f"README 摘要:\n{readme_truncated}"
                if is_chinese
                else f"README excerpt:\n{readme_truncated}"
            )

        context = "\n".join(context_parts)

        if is_chinese:
            prompt = f"""请为以下 GitHub 项目生成一段中文摘要（100-150 字）。

{context}

要求:
1. 涵盖项目的核心功能、技术特点和适用场景
2. 使用专业、流畅的中文
3. 不要使用 "这是一个" 开头
4. 字数控制在 100-150 字之间

直接输出摘要内容，不要有其他文字。"""
            system_prompt = "你是一个技术文档专家，擅长用清晰的语言描述开源项目。"
        else:
            prompt = f"""Write a concise English summary (70-120 words) for the GitHub project below.

{context}

Requirements:
1. Cover core functionality, technical highlights, and use cases
2. Use clear and professional language
3. Avoid starting with "This is a..."

Return only the summary text."""
            system_prompt = (
                "You are a technical writer who explains open-source projects clearly and accurately."
            )

        try:
            response = await self.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=10000,
                temperature=0.3,
            )

            # Clean up response
            summary = response.strip()
            # Remove common prefixes
            for prefix in (
                ["摘要:", "总结:", "简介:"]
                if is_chinese
                else ["Summary:", "Overview:", "Description:"]
            ):
                if summary.startswith(prefix):
                    summary = summary[len(prefix) :].strip()

            return summary

        except Exception as e:
            logger.warning(f"LLM summary generation failed for {full_name}: {e}")
            # Fallback to description or default
            if description:
                return description
            return (
                f"{full_name.split('/')[-1]} - 开源项目"
                if is_chinese
                else f"{full_name.split('/')[-1]} - open source project"
            )

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

        is_chinese = self.settings.output_language == "zh"

        # Build structured context
        context_parts = (
            [f"仓库名称: {full_name}"]
            if is_chinese
            else [f"Repository: {full_name}"]
        )

        if language:
            context_parts.append(
                f"主要语言: {language}"
                if is_chinese
                else f"Primary language: {language}"
            )

        if description:
            context_parts.append(
                f"官方描述: {description}"
                if is_chinese
                else f"Description: {description}"
            )

        if topics:
            context_parts.append(
                f"GitHub 标签: {', '.join(topics[:10])}"
                if is_chinese
                else f"GitHub topics: {', '.join(topics[:10])}"
            )

        if readme_content:
            # Truncate README to avoid token limits
            readme_truncated = readme_content[:5000]
            context_parts.append(
                f"README 内容:\n{readme_truncated}"
                if is_chinese
                else f"README excerpt:\n{readme_truncated}"
            )

        context = "\n".join(context_parts)

        if is_chinese:
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
        else:
            system_prompt = """You are a senior technical open-source project analyst focused on software classification and documentation.

Your responsibilities:
1. Understand project architecture and core value quickly
2. Write concise, professional English summaries
3. Extract precise classification tags

Output requirements:
- Return pure JSON only
- Do not wrap output in markdown code fences
- Do not include any extra commentary"""
            user_prompt = f"""Analyze the GitHub project below and generate a structured summary and tags.

## Repository Info
{context}

## Output Requirements

### summary
- length: 70-120 words
- content: core functionality, technical highlights, and practical use cases
- style: professional and concise; avoid boilerplate openings
- language: English

### tags
- count: 3-7
- length: 1-4 words each
- coverage: function category, technical domain, and use case
- language: English preferred

## Output Format (strict JSON)
{{"summary": "Project summary content", "tags": ["tag1", "tag2", "tag3"]}}"""

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
                description
                if description
                else (
                    f"{full_name.split('/')[-1]} - 开源项目"
                    if is_chinese
                    else f"{full_name.split('/')[-1]} - open source project"
                )
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
            # Keep short topics as-is, they generally work as tags.
            for topic in topics[:5]:
                if len(topic) <= 15:
                    result_tags.append(topic)
            if result_tags:
                return result_tags

        # Fallback 2: Use language
        if language:
            result_tags.append(language)
            result_tags.append(
                "开源项目" if self.settings.output_language == "zh" else "open-source"
            )
            return result_tags

        # Fallback 3: Generate from repo name
        repo_name = full_name.split("/")[-1] if "/" in full_name else full_name
        # Extract meaningful parts from repo name
        parts = repo_name.replace("-", " ").replace("_", " ").split()
        if parts:
            result_tags.append(parts[0])
        result_tags.append(
            "开源项目" if self.settings.output_language == "zh" else "open-source"
        )

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
                    summaries.append(
                        repo.get("description", "")[:100]
                        or (
                            "开源项目"
                            if self.settings.output_language == "zh"
                            else "open source project"
                        )
                    )
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
