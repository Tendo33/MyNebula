import pytest

from nebula.core.github_client import GitHubClient


class _FakeResponse:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400 and self.status_code != 404:
            raise RuntimeError("http error")


class _FakeClient:
    def __init__(self, response: _FakeResponse):
        self._response = response

    async def get(self, *_args, **_kwargs):
        return self._response


@pytest.mark.asyncio
async def test_get_repo_readme_truncates_by_max_length():
    client = GitHubClient(access_token="token")
    client._client = _FakeClient(_FakeResponse(200, "a" * 200))  # type: ignore[attr-defined]

    readme = await client.get_repo_readme("owner/repo", max_length=50)
    assert readme == "a" * 50


@pytest.mark.asyncio
async def test_get_repo_readme_returns_none_on_404():
    client = GitHubClient(access_token="token")
    client._client = _FakeClient(_FakeResponse(404, ""))  # type: ignore[attr-defined]

    readme = await client.get_repo_readme("owner/missing")
    assert readme is None
