"""GitHub API client wrapper.

This module provides async GitHub API operations including:
- Starred repositories retrieval
- README content fetching
- Repository metadata
"""

import base64
from datetime import datetime

import httpx
from pydantic import BaseModel

from nebula.utils import get_logger
from nebula.utils.decorator_utils import async_retry_decorator

logger = get_logger(__name__)

# GitHub API endpoints
GITHUB_API_BASE = "https://api.github.com"


class GitHubUser(BaseModel):
    """GitHub user information."""

    id: int
    login: str
    email: str | None = None
    avatar_url: str | None = None
    name: str | None = None


class GitHubRepo(BaseModel):
    """GitHub repository information."""

    id: int
    full_name: str
    owner: str
    name: str
    description: str | None = None
    language: str | None = None
    topics: list[str] = []
    html_url: str
    homepage: str | None = None
    stargazers_count: int = 0
    forks_count: int = 0
    watchers_count: int = 0
    open_issues_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    pushed_at: datetime | None = None
    starred_at: datetime | None = None


class GitHubClient:
    """Async GitHub API client.

    Supports Personal Access Token authentication.

    Usage:
        # With PAT
        client = GitHubClient(access_token="ghp_xxx")
        user = await client.get_current_user()
        stars = await client.get_starred_repos()
    """

    def __init__(self, access_token: str):
        """Initialize GitHub client.

        Args:
            access_token: Personal Access Token for GitHub API
        """
        self._access_token = access_token
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            headers = {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            if self._access_token:
                headers["Authorization"] = f"Bearer {self._access_token}"

            self._client = httpx.AsyncClient(
                base_url=GITHUB_API_BASE,
                headers=headers,
                timeout=30.0,
            )
        return self._client

    async def get_current_user(self) -> GitHubUser:
        """Get current authenticated user.

        Returns:
            GitHubUser object

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        response = await self.client.get("/user")
        response.raise_for_status()

        data = response.json()
        return GitHubUser(
            id=data["id"],
            login=data["login"],
            email=data.get("email"),
            avatar_url=data.get("avatar_url"),
            name=data.get("name"),
        )

    async def get_starred_repos(
        self,
        per_page: int = 100,
        max_pages: int | None = None,
        stop_before: datetime | None = None,
    ) -> tuple[list[GitHubRepo], bool]:
        """Get starred repositories for current user.

        GitHub API returns repos sorted by starred_at DESC (newest first).
        This allows efficient incremental sync by stopping when we reach
        repos starred before a given timestamp.

        Args:
            per_page: Items per page (max 100)
            max_pages: Maximum pages to fetch (None for all)
            stop_before: Stop fetching when starred_at <= this timestamp.
                         Used for incremental sync to skip already-synced repos.

        Returns:
            Tuple of (repos list, was_truncated).
            was_truncated is True if we stopped early due to stop_before.

        Yields progress through logger.
        """
        repos: list[GitHubRepo] = []
        page = 1
        was_truncated = False

        while True:
            if max_pages and page > max_pages:
                break

            response = await self.client.get(
                "/user/starred",
                params={"per_page": per_page, "page": page},
                headers={
                    "Accept": "application/vnd.github.star+json"
                },  # Include starred_at
            )
            response.raise_for_status()

            data = response.json()
            if not data:
                break

            should_stop = False
            for item in data:
                repo_data = item.get("repo", item)  # Handle both formats
                starred_at = item.get("starred_at")

                # Parse starred_at for comparison
                starred_at_dt = (
                    datetime.fromisoformat(starred_at.replace("Z", "+00:00"))
                    if starred_at
                    else None
                )

                # Check if we should stop (incremental sync optimization)
                if stop_before and starred_at_dt and starred_at_dt <= stop_before:
                    logger.info(
                        f"Incremental sync: stopping at {starred_at_dt} "
                        f"(<= last_sync {stop_before})"
                    )
                    should_stop = True
                    was_truncated = True
                    break

                repo = GitHubRepo(
                    id=repo_data["id"],
                    full_name=repo_data["full_name"],
                    owner=repo_data["owner"]["login"],
                    name=repo_data["name"],
                    description=repo_data.get("description"),
                    language=repo_data.get("language"),
                    topics=repo_data.get("topics", []),
                    html_url=repo_data["html_url"],
                    homepage=repo_data.get("homepage"),
                    stargazers_count=repo_data.get("stargazers_count", 0),
                    forks_count=repo_data.get("forks_count", 0),
                    watchers_count=repo_data.get("watchers_count", 0),
                    open_issues_count=repo_data.get("open_issues_count", 0),
                    created_at=datetime.fromisoformat(
                        repo_data["created_at"].replace("Z", "+00:00")
                    )
                    if repo_data.get("created_at")
                    else None,
                    updated_at=datetime.fromisoformat(
                        repo_data["updated_at"].replace("Z", "+00:00")
                    )
                    if repo_data.get("updated_at")
                    else None,
                    pushed_at=datetime.fromisoformat(
                        repo_data["pushed_at"].replace("Z", "+00:00")
                    )
                    if repo_data.get("pushed_at")
                    else None,
                    starred_at=starred_at_dt,
                )
                repos.append(repo)

            if should_stop:
                break

            logger.info(f"Fetched page {page}, total repos: {len(repos)}")
            page += 1

        return repos, was_truncated

    @async_retry_decorator(max_retries=2, delay=0.5)
    async def get_readme(
        self,
        owner: str,
        repo: str,
        max_length: int = 10000,
    ) -> str | None:
        """Get README content for a repository.

        Args:
            owner: Repository owner
            repo: Repository name
            max_length: Maximum content length to return

        Returns:
            README content as string, or None if not found
        """
        try:
            response = await self.client.get(f"/repos/{owner}/{repo}/readme")

            if response.status_code == 404:
                return None

            response.raise_for_status()
            data = response.json()

            # Decode base64 content
            content = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")

            # Truncate if too long
            if len(content) > max_length:
                content = content[:max_length] + "\n... [truncated]"

            return content

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logger.warning(f"Failed to get README for {owner}/{repo}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error getting README for {owner}/{repo}: {e}")
            return None

    async def get_rate_limit(self) -> dict:
        """Get current rate limit status.

        Returns:
            Rate limit information
        """
        response = await self.client.get("/rate_limit")
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "GitHubClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
