import asyncio

from nebula.core.config import LLMSettings
from nebula.core.llm import LLMService


def test_generate_repo_summary_and_tags_uses_english_prompt_when_configured():
    async def run_test() -> None:
        service = LLMService(
            settings=LLMSettings(
                api_key="test-key",
                base_url="https://example.com/v1",
                model="test-model",
                output_language="en",
            )
        )
        captured: dict[str, str] = {}

        async def fake_complete(
            prompt: str,
            system_prompt: str | None = None,
            max_tokens: int = 2048,
            temperature: float = 0.3,
        ) -> str:
            _ = max_tokens, temperature
            captured["prompt"] = prompt
            captured["system_prompt"] = system_prompt or ""
            return '{"summary": "English summary", "tags": ["agent", "workflow", "cli"]}'

        service.complete = fake_complete  # type: ignore[method-assign]

        summary, tags = await service.generate_repo_summary_and_tags(
            full_name="owner/repo",
            description="A CLI agent framework",
            topics=["agent", "automation"],
            language="Python",
            readme_content="README content",
        )

        assert summary == "English summary"
        assert tags == ["agent", "workflow", "cli"]
        assert "## Repository Info" in captured["prompt"]
        assert "language: English" in captured["prompt"]
        assert "technical open-source project analyst" in captured["system_prompt"]

    asyncio.run(run_test())


def test_generate_repo_summary_and_tags_uses_chinese_prompt_when_configured():
    async def run_test() -> None:
        service = LLMService(
            settings=LLMSettings(
                api_key="test-key",
                base_url="https://example.com/v1",
                model="test-model",
                output_language="zh",
            )
        )
        captured: dict[str, str] = {}

        async def fake_complete(
            prompt: str,
            system_prompt: str | None = None,
            max_tokens: int = 2048,
            temperature: float = 0.3,
        ) -> str:
            _ = max_tokens, temperature
            captured["prompt"] = prompt
            captured["system_prompt"] = system_prompt or ""
            return '{"summary": "中文摘要", "tags": ["智能体", "工作流", "命令行"]}'

        service.complete = fake_complete  # type: ignore[method-assign]

        summary, tags = await service.generate_repo_summary_and_tags(
            full_name="owner/repo",
            description="一个 CLI 智能体框架",
            topics=["agent", "automation"],
            language="Python",
            readme_content="README 内容",
        )

        assert summary == "中文摘要"
        assert tags == ["智能体", "工作流", "命令行"]
        assert "## 项目信息" in captured["prompt"]
        assert "语言：中文" in captured["prompt"]
        assert "资深的开源项目分析师" in captured["system_prompt"]

    asyncio.run(run_test())


def test_generate_cluster_info_uses_english_prompt_when_configured():
    async def run_test() -> None:
        service = LLMService(
            settings=LLMSettings(
                api_key="test-key",
                base_url="https://example.com/v1",
                model="test-model",
                output_language="en",
            )
        )
        captured: dict[str, str] = {}

        async def fake_complete(
            prompt: str,
            system_prompt: str | None = None,
            max_tokens: int = 2048,
            temperature: float = 0.3,
        ) -> str:
            _ = max_tokens, temperature
            captured["prompt"] = prompt
            captured["system_prompt"] = system_prompt or ""
            return "Agent Ops|Tooling for autonomous agent workflows|agent,automation,tooling"

        service.complete = fake_complete  # type: ignore[method-assign]

        name, description, keywords = await service.generate_cluster_info(
            repo_names=["owner/repo-a", "owner/repo-b"],
            descriptions=["desc a", "desc b"],
            topics=[["agent"], ["automation"]],
            languages=["Python", "TypeScript"],
        )

        assert name == "Agent Ops"
        assert description == "Tooling for autonomous agent workflows"
        assert keywords == ["agent", "automation", "tooling"]
        assert "Analyze the following GitHub repositories" in captured["prompt"]
        assert "technical analyst specializing in GitHub project taxonomy" in captured[
            "system_prompt"
        ]

    asyncio.run(run_test())


def test_generate_repo_summary_and_tags_fallback_uses_language_specific_defaults():
    async def run_test() -> None:
        service = LLMService(
            settings=LLMSettings(
                api_key="test-key",
                base_url="https://example.com/v1",
                model="test-model",
                output_language="en",
            )
        )

        async def fake_complete(*args, **kwargs) -> str:
            _ = args, kwargs
            raise RuntimeError("llm failed")

        service.complete = fake_complete  # type: ignore[method-assign]

        summary, tags = await service.generate_repo_summary_and_tags(
            full_name="owner/repo",
            description=None,
            topics=None,
            language=None,
            readme_content=None,
        )

        assert summary == "repo - open source project"
        assert tags == ["repo", "open-source"]

    asyncio.run(run_test())
