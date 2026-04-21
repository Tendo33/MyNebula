import pytest

from nebula.core.github_client import GitHubClient


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeAsyncClient:
    def __init__(self, responses: list[dict]):
        self._responses = [_FakeResponse(payload) for payload in responses]
        self.calls: list[dict] = []

    async def post(self, _url: str, *, json: dict, headers: dict):
        self.calls.append({"json": json, "headers": headers})
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_get_star_lists_paginates_items_for_user_list():
    client = GitHubClient(access_token="token")
    fake_client = _FakeAsyncClient(
        [
            {
                "data": {
                    "viewer": {
                        "lists": {
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                            "nodes": [
                                {
                                    "id": "list-1",
                                    "name": "Infra",
                                    "description": "Infrastructure repos",
                                    "isPrivate": False,
                                    "items": {"totalCount": 3},
                                }
                            ],
                        }
                    }
                }
            },
            {
                "data": {
                    "node": {
                        "__typename": "UserList",
                        "items": {
                            "pageInfo": {"hasNextPage": True, "endCursor": "cursor-2"},
                            "totalCount": 3,
                            "nodes": [{"databaseId": 11}, {"databaseId": 22}],
                        },
                    }
                }
            },
            {
                "data": {
                    "node": {
                        "__typename": "UserList",
                        "items": {
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                            "totalCount": 3,
                            "nodes": [{"databaseId": 33}],
                        },
                    }
                }
            },
        ]
    )
    client._client = fake_client

    star_lists = await client.get_star_lists()

    assert len(star_lists) == 1
    assert star_lists[0].repo_ids == [11, 22, 33]
    assert fake_client.calls[1]["json"]["variables"]["listId"] == "list-1"


@pytest.mark.asyncio
async def test_get_star_lists_handles_missing_items_payload_gracefully():
    client = GitHubClient(access_token="token")
    fake_client = _FakeAsyncClient(
        [
            {
                "data": {
                    "viewer": {
                        "lists": {
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                            "nodes": [
                                {
                                    "id": "list-1",
                                    "name": "Infra",
                                    "description": None,
                                    "isPrivate": False,
                                    "items": {"totalCount": 1},
                                }
                            ],
                        }
                    }
                }
            },
            {"data": {"node": {"__typename": "UnexpectedListType"}}},
        ]
    )
    client._client = fake_client

    star_lists = await client.get_star_lists()

    assert len(star_lists) == 1
    assert star_lists[0].repo_ids == []
