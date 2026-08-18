"""Deterministic provider fakes for the integration tier.

The database is real here; GitHub, the embedding provider, and the LLM are not.
This tier adds persistence realism, not network realism.

Embeddings are derived from a hash of the repository name so clustering results
are reproducible across runs — a random vector would make cluster-membership
assertions flaky.
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta, timezone

from nebula.core.github_client import GitHubRepo

EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def deterministic_embedding(
    seed: str, dimensions: int, *, group: int = 0
) -> list[float]:
    """Unit vector keyed by `seed`, offset into a cone per `group`.

    Same group -> vectors close together; different group -> well separated.
    That gives clustering something real to find without randomness.
    """
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    values = [0.0] * dimensions
    # Dominant component per group keeps groups apart.
    values[group % dimensions] = 10.0
    for index in range(min(dimensions, len(digest))):
        values[index] += digest[index] / 255.0
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


class FakeGitHubClient:
    """Async context manager matching the surface `sync_stars_task` uses."""

    repos: list[GitHubRepo] = []
    truncated: bool = False
    star_lists: list = []
    readmes: dict[str, str | None] = {}

    def __init__(self, access_token: str):
        self.access_token = access_token

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, _tb):
        return False

    async def get_starred_repos(self, stop_before=None):
        if stop_before is None:
            return type(self).repos, type(self).truncated
        fresh = [
            repo
            for repo in type(self).repos
            if repo.starred_at is None or repo.starred_at > stop_before
        ]
        return fresh, True

    async def get_repo_readme(self, full_name: str, max_length: int = 10000):
        return type(self).readmes.get(full_name, f"readme for {full_name}")

    async def get_star_lists(self):
        return type(self).star_lists


class FakeEmbeddingService:
    """Embeds by name hash; `fail_on_calls` injects provider failures."""

    def __init__(self, dimensions: int, groups: dict[str, int] | None = None):
        self.dimensions = dimensions
        self.groups = groups or {}
        self.calls = 0
        self.fail_on_calls: set[int] = set()

    def build_repo_text(self, *, full_name: str, **_kwargs) -> str:
        return f"text::{full_name}"

    async def embed_text(self, text: str) -> list[float]:
        return deterministic_embedding(text, self.dimensions)

    async def embed_batch(self, texts, batch_size=None):
        self.calls += 1
        if self.calls in self.fail_on_calls:
            raise RuntimeError(f"embedding provider failure on call {self.calls}")
        return [
            deterministic_embedding(
                text, self.dimensions, group=self.groups.get(text, 0)
            )
            for text in texts
        ]


class FakeLLMService:
    def __init__(self, *, fail: bool = False):
        self.fail = fail

    async def generate_repo_summary_and_tags(self, *, full_name: str, **_kwargs):
        if self.fail:
            raise RuntimeError("llm unavailable")
        return f"summary for {full_name}", ["tag-a", "tag-b"]

    async def complete(self, *_args, **_kwargs):
        if self.fail:
            raise RuntimeError("llm unavailable")
        return "Fake Cluster"


def make_github_repo(index: int, *, starred_days_ago: int | None = None) -> GitHubRepo:
    starred_at = (
        EPOCH - timedelta(days=starred_days_ago)
        if starred_days_ago is not None
        else EPOCH - timedelta(days=index)
    )
    return GitHubRepo(
        id=1000 + index,
        full_name=f"owner/repo-{index}",
        owner="owner",
        name=f"repo-{index}",
        description=f"description {index}",
        language="Python" if index % 2 == 0 else "Rust",
        topics=["topic-a"] if index % 2 == 0 else ["topic-b"],
        html_url=f"https://github.com/owner/repo-{index}",
        stargazers_count=index * 10,
        forks_count=index,
        watchers_count=index,
        open_issues_count=0,
        starred_at=starred_at,
        created_at=EPOCH - timedelta(days=365),
        updated_at=EPOCH,
        pushed_at=EPOCH,
        owner_avatar_url="https://avatars/owner",
    )
