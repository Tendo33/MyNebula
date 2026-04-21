from types import SimpleNamespace

import pytest
from sqlalchemy.exc import IntegrityError


class _FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _FakeDb:
    def __init__(self, existing_user):
        self._existing_user = existing_user
        self.added = []
        self.rollback_called = False
        self.commit_calls = 0

    async def execute(self, _query):
        return _FakeResult(self._existing_user)

    def add(self, value):
        self.added.append(value)

    async def commit(self):
        self.commit_calls += 1
        raise IntegrityError("insert into users", {}, Exception("duplicate key"))

    async def rollback(self):
        self.rollback_called = True

    async def refresh(self, _value):
        return None


@pytest.mark.asyncio
async def test_get_default_user_recovers_from_concurrent_insert(monkeypatch):
    from nebula.application.services import user_service

    existing_user = SimpleNamespace(id=7, username="existing-user")
    db = _FakeDb(existing_user)

    sequence = iter([None, None])

    async def fake_get_first_user(_db):
        return next(sequence)

    async def fake_acquire_lock(_db):
        return None

    class _FakeGitHubClient:
        def __init__(self, access_token: str):
            self.access_token = access_token

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get_current_user(self):
            return SimpleNamespace(
                id=42,
                login="bootstrap-user",
                email="bootstrap@example.com",
                avatar_url="https://example.com/avatar.png",
            )

    monkeypatch.setattr(
        user_service,
        "get_app_settings",
        lambda: SimpleNamespace(github_token="token"),
    )
    monkeypatch.setattr(user_service, "_get_first_user", fake_get_first_user)
    monkeypatch.setattr(
        user_service,
        "_acquire_default_user_bootstrap_lock",
        fake_acquire_lock,
    )
    monkeypatch.setattr(user_service, "GitHubClient", _FakeGitHubClient)

    resolved = await user_service.get_default_user(db)

    assert resolved is existing_user
    assert db.rollback_called is True
    assert db.added
