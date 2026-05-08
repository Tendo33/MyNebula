from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from nebula.core.auth import create_signed_session_token
from nebula.core.config import AppSettings


class _FakeRateLimitDb:
    def __init__(self):
        self.attempts: list[tuple[str, datetime]] = []


def _install_fake_rate_limit_helpers(monkeypatch, auth_api):
    fake_db = _FakeRateLimitDb()

    async def fake_delete_stale(db, *, cutoff):
        db.attempts = [
            (bucket_key, attempted_at)
            for bucket_key, attempted_at in db.attempts
            if attempted_at >= cutoff
        ]

    async def fake_count_recent(db, *, bucket_key, cutoff):
        return sum(
            1
            for current_key, attempted_at in db.attempts
            if current_key == bucket_key and attempted_at >= cutoff
        )

    async def fake_store(db, *, keys, attempted_at):
        for key in keys:
            db.attempts.append((key, attempted_at))

    async def fake_clear(db, *, keys):
        db.attempts = [
            (bucket_key, attempted_at)
            for bucket_key, attempted_at in db.attempts
            if bucket_key not in keys
        ]

    monkeypatch.setattr(auth_api, "_delete_stale_login_attempts", fake_delete_stale)
    monkeypatch.setattr(auth_api, "_count_recent_login_attempts", fake_count_recent)
    monkeypatch.setattr(auth_api, "_store_login_attempts", fake_store)
    monkeypatch.setattr(auth_api, "_clear_login_attempts", fake_clear)
    return fake_db


def _build_request(
    *,
    scheme: str = "http",
    forwarded_proto: str | None = None,
    session_cookie: str | None = None,
) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if forwarded_proto:
        headers.append((b"x-forwarded-proto", forwarded_proto.encode("utf-8")))
    if session_cookie:
        headers.append((b"cookie", f"nebula_admin_session={session_cookie}".encode()))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v2/graph",
        "headers": headers,
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "scheme": scheme,
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_resolve_read_user_allows_demo_mode(monkeypatch):
    from nebula.api.v2 import access as access_api

    async def fake_get_default_user(_db):
        return SimpleNamespace(id=1)

    monkeypatch.setattr(access_api, "get_default_user", fake_get_default_user)

    user = await access_api.resolve_read_user(
        request=_build_request(),
        db=object(),
        settings=AppSettings(read_access_mode="demo"),
    )

    assert user.id == 1


@pytest.mark.asyncio
async def test_resolve_read_user_rejects_authenticated_mode_without_session(
    monkeypatch,
):
    from nebula.api.v2 import access as access_api

    async def fake_get_default_user(_db):
        return SimpleNamespace(id=1)

    monkeypatch.setattr(access_api, "get_default_user", fake_get_default_user)

    with pytest.raises(HTTPException) as exc_info:
        await access_api.resolve_read_user(
            request=_build_request(),
            db=object(),
            settings=AppSettings(
                read_access_mode="authenticated",
                admin_username="owner",
                admin_password="topsecret",
                admin_session_secret="session-secret",
            ),
        )

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_resolve_read_user_accepts_authenticated_mode_with_valid_session(
    monkeypatch,
):
    from nebula.api.v2 import access as access_api

    async def fake_get_default_user(_db):
        return SimpleNamespace(id=1)

    monkeypatch.setattr(access_api, "get_default_user", fake_get_default_user)
    settings = AppSettings(
        read_access_mode="authenticated",
        admin_username="owner",
        admin_password="topsecret",
        admin_session_secret="session-secret",
    )
    session_cookie = create_signed_session_token(
        username="owner",
        secret=settings.admin_session_secret,
        expires_in_seconds=60,
    )

    user = await access_api.resolve_read_user(
        request=_build_request(session_cookie=session_cookie),
        db=object(),
        settings=settings,
    )

    assert user.id == 1


@pytest.mark.asyncio
async def test_login_admin_uses_secure_cookie_behind_trusted_proxy(monkeypatch):
    from nebula.api.v2 import auth as auth_api

    fake_db = _install_fake_rate_limit_helpers(monkeypatch, auth_api)
    settings = AppSettings(
        admin_username="owner",
        admin_password="topsecret",
        admin_session_secret="session-secret",
        trust_proxy_headers=True,
        trusted_proxy_ips="127.0.0.1",
    )
    response = Response()

    payload = auth_api.LoginRequest(username="owner", password="topsecret")
    await auth_api.login_admin(
        payload=payload,
        request=_build_request(forwarded_proto="https"),
        response=response,
        settings=settings,
        db=fake_db,
    )

    set_cookie_headers = response.headers.getlist("set-cookie")
    assert len(set_cookie_headers) == 2
    assert all("Secure" in header for header in set_cookie_headers)


@pytest.mark.asyncio
async def test_login_admin_ignores_forwarded_proto_when_proxy_trust_disabled(
    monkeypatch,
):
    from nebula.api.v2 import auth as auth_api

    fake_db = _install_fake_rate_limit_helpers(monkeypatch, auth_api)
    settings = AppSettings(
        admin_username="owner",
        admin_password="topsecret",
        admin_session_secret="session-secret",
        trust_proxy_headers=False,
        trusted_proxy_ips="127.0.0.1",
    )
    response = Response()

    payload = auth_api.LoginRequest(username="owner", password="topsecret")
    await auth_api.login_admin(
        payload=payload,
        request=_build_request(forwarded_proto="https"),
        response=response,
        settings=settings,
        db=fake_db,
    )

    set_cookie_headers = response.headers.getlist("set-cookie")
    assert len(set_cookie_headers) == 2
    assert all("Secure" not in header for header in set_cookie_headers)


@pytest.mark.asyncio
async def test_login_admin_ignores_forwarded_proto_from_untrusted_proxy(monkeypatch):
    from nebula.api.v2 import auth as auth_api

    fake_db = _install_fake_rate_limit_helpers(monkeypatch, auth_api)
    settings = AppSettings(
        admin_username="owner",
        admin_password="topsecret",
        admin_session_secret="session-secret",
        trust_proxy_headers=True,
        trusted_proxy_ips="10.0.0.1",
    )
    response = Response()

    payload = auth_api.LoginRequest(username="owner", password="topsecret")
    await auth_api.login_admin(
        payload=payload,
        request=_build_request(forwarded_proto="https"),
        response=response,
        settings=settings,
        db=fake_db,
    )

    set_cookie_headers = response.headers.getlist("set-cookie")
    assert len(set_cookie_headers) == 2
    assert all("Secure" not in header for header in set_cookie_headers)


@pytest.mark.asyncio
async def test_login_admin_rate_limits_repeated_failures(monkeypatch):
    from nebula.api.v2 import auth as auth_api

    fake_db = _install_fake_rate_limit_helpers(monkeypatch, auth_api)
    settings = AppSettings(
        admin_username="owner",
        admin_password="topsecret",
        admin_session_secret="session-secret",
        admin_login_rate_limit_max_attempts=1,
    )
    request = _build_request()
    response = Response()

    with pytest.raises(HTTPException) as first_exc:
        await auth_api.login_admin(
            payload=auth_api.LoginRequest(username="owner", password="wrong"),
            request=request,
            response=response,
            settings=settings,
            db=fake_db,
        )
    assert first_exc.value.status_code == 401

    with pytest.raises(HTTPException) as second_exc:
        await auth_api.login_admin(
            payload=auth_api.LoginRequest(username="owner", password="wrong"),
            request=request,
            response=Response(),
            settings=settings,
            db=fake_db,
        )
    assert second_exc.value.status_code == 429


@pytest.mark.asyncio
async def test_login_rate_limit_prunes_stale_buckets(monkeypatch):
    from nebula.api.v2 import auth as auth_api

    fake_db = _install_fake_rate_limit_helpers(monkeypatch, auth_api)
    fake_db.attempts = [
        ("ip:stale", datetime(2026, 1, 1, tzinfo=timezone.utc)),
        ("user:stale", datetime(2026, 1, 1, tzinfo=timezone.utc)),
    ]

    settings = AppSettings(
        admin_username="owner",
        admin_password="topsecret",
        admin_session_secret="session-secret",
        admin_login_rate_limit_window_seconds=60,
    )

    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 1, 0, 2, 0, tzinfo=tz or timezone.utc)

    monkeypatch.setattr(auth_api, "datetime", _FrozenDateTime)

    await auth_api.login_admin(
        payload=auth_api.LoginRequest(username="owner", password="topsecret"),
        request=_build_request(),
        response=Response(),
        settings=settings,
        db=fake_db,
    )

    assert (
        "ip:stale",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    ) not in fake_db.attempts
    assert (
        "user:stale",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
    ) not in fake_db.attempts


@pytest.mark.asyncio
async def test_logout_admin_requires_csrf_dependency():
    from nebula.api.v2 import auth as auth_api

    route = next(
        (route for route in auth_api.router.routes if route.path == "/logout"),
        None,
    )

    assert route is not None
    dependency_calls = {
        getattr(getattr(dependency, "call", None), "__name__", "")
        for dependency in route.dependant.dependencies
    }
    assert "require_admin" in dependency_calls
    assert "require_admin_csrf" in dependency_calls


@pytest.mark.asyncio
async def test_repo_search_allows_demo_mode_without_admin_session(monkeypatch):
    from nebula.api.v2 import repos as repos_api
    from nebula.schemas.repo import RepoSearchRequest

    class _FakeEmbeddingService:
        def __init__(self):
            self.calls = 0

        async def embed_text(self, _query: str):
            self.calls += 1
            return [0.1, 0.2, 0.3]

    class _FakeResult:
        def all(self):
            return []

    class _FakeDb:
        async def execute(self, _statement):
            return _FakeResult()

    embedding_service = _FakeEmbeddingService()
    monkeypatch.setattr(
        repos_api,
        "get_app_settings",
        lambda: AppSettings(read_access_mode="demo"),
        raising=False,
    )
    monkeypatch.setattr(repos_api, "get_embedding_service", lambda: embedding_service)
    repos_api._SEMANTIC_SEARCH_CACHE.clear()

    results = await repos_api.search_repos(
        request=RepoSearchRequest(query="vector search"),
        http_request=_build_request(),
        user=SimpleNamespace(id=1, active_graph_snapshot_id=None),
        db=_FakeDb(),
    )

    assert results == []
    assert embedding_service.calls == 1


@pytest.mark.asyncio
async def test_repo_search_allows_demo_mode_with_valid_admin_session(monkeypatch):
    from nebula.api.v2 import repos as repos_api
    from nebula.schemas.repo import RepoSearchRequest

    class _FakeEmbeddingService:
        async def embed_text(self, _query: str):
            return [0.1, 0.2, 0.3]

    class _FakeResult:
        def all(self):
            return []

    class _FakeDb:
        async def execute(self, _statement):
            return _FakeResult()

    settings = AppSettings(
        read_access_mode="demo",
        admin_username="owner",
        admin_password="topsecret",
        admin_session_secret="session-secret",
    )
    session_cookie = create_signed_session_token(
        username="owner",
        secret=settings.admin_session_secret,
        expires_in_seconds=60,
    )

    monkeypatch.setattr(
        repos_api, "get_embedding_service", lambda: _FakeEmbeddingService()
    )
    repos_api._SEMANTIC_SEARCH_CACHE.clear()

    results = await repos_api.search_repos(
        request=RepoSearchRequest(query="vector search"),
        user=SimpleNamespace(id=1, active_graph_snapshot_id=None),
        db=_FakeDb(),
        settings=settings,
        http_request=_build_request(session_cookie=session_cookie),
    )

    assert results == []


@pytest.mark.asyncio
async def test_repo_search_rejects_blank_query_after_trimming(monkeypatch):
    from nebula.api.v2 import repos as repos_api
    from nebula.schemas.repo import RepoSearchRequest

    class _FakeEmbeddingService:
        async def embed_text(self, _query: str):
            raise AssertionError("blank query should fail before embedding")

    class _FakeDb:
        async def execute(self, _statement):
            raise AssertionError("blank query should fail before DB access")

    monkeypatch.setattr(
        repos_api,
        "get_app_settings",
        lambda: AppSettings(read_access_mode="demo"),
        raising=False,
    )
    monkeypatch.setattr(
        repos_api, "get_embedding_service", lambda: _FakeEmbeddingService()
    )
    repos_api._SEMANTIC_SEARCH_CACHE.clear()

    with pytest.raises(HTTPException) as exc_info:
            await repos_api.search_repos(
                request=RepoSearchRequest(query="   "),
                http_request=_build_request(),
                user=SimpleNamespace(id=1, active_graph_snapshot_id=None),
                db=_FakeDb(),
            )

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_repo_search_wraps_embedding_failures(monkeypatch):
    from nebula.api.v2 import repos as repos_api
    from nebula.schemas.repo import RepoSearchRequest

    class _FakeEmbeddingService:
        async def embed_text(self, _query: str):
            raise RuntimeError("provider boom")

    class _FakeDb:
        async def execute(self, _statement):
            raise AssertionError("provider failures should fail before DB access")

    monkeypatch.setattr(
        repos_api,
        "get_app_settings",
        lambda: AppSettings(read_access_mode="demo"),
        raising=False,
    )
    monkeypatch.setattr(
        repos_api, "get_embedding_service", lambda: _FakeEmbeddingService()
    )
    repos_api._SEMANTIC_SEARCH_CACHE.clear()

    with pytest.raises(HTTPException) as exc_info:
            await repos_api.search_repos(
                request=RepoSearchRequest(query="vector search"),
                http_request=_build_request(),
                user=SimpleNamespace(id=1, active_graph_snapshot_id=None),
                db=_FakeDb(),
            )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Semantic search is temporarily unavailable"


@pytest.mark.asyncio
async def test_repo_search_caches_identical_requests(monkeypatch):
    from nebula.api.v2 import repos as repos_api
    from nebula.schemas.repo import RepoSearchRequest

    class _FakeEmbeddingService:
        def __init__(self):
            self.calls = 0

        async def embed_text(self, _query: str):
            self.calls += 1
            return [0.1, 0.2, 0.3]

    class _FakeResult:
        def all(self):
            return []

    class _FakeDb:
        def __init__(self):
            self.calls = 0

        async def execute(self, _statement):
            self.calls += 1
            return _FakeResult()

    embedding_service = _FakeEmbeddingService()
    fake_db = _FakeDb()
    monkeypatch.setattr(
        repos_api,
        "get_app_settings",
        lambda: AppSettings(read_access_mode="demo"),
        raising=False,
    )
    monkeypatch.setattr(repos_api, "get_embedding_service", lambda: embedding_service)
    repos_api._SEMANTIC_SEARCH_CACHE.clear()

    request = RepoSearchRequest(query="vector search", limit=10, min_stars=5)
    kwargs = {
        "request": request,
        "http_request": _build_request(),
        "user": SimpleNamespace(id=1, active_graph_snapshot_id=None),
        "db": fake_db,
    }
    first = await repos_api.search_repos(**kwargs)
    second = await repos_api.search_repos(**kwargs)

    assert first == []
    assert second == []
    assert embedding_service.calls == 1
    assert fake_db.calls == 1


@pytest.mark.asyncio
async def test_repo_search_cache_is_scoped_to_active_snapshot(monkeypatch):
    from nebula.api.v2 import repos as repos_api
    from nebula.schemas.repo import RepoSearchRequest

    class _FakeEmbeddingService:
        def __init__(self):
            self.calls = 0

        async def embed_text(self, _query: str):
            self.calls += 1
            return [0.1, 0.2, 0.3]

    class _FakeResult:
        def all(self):
            return []

    class _FakeDb:
        def __init__(self):
            self.calls = 0

        async def execute(self, _statement):
            self.calls += 1
            return _FakeResult()

    embedding_service = _FakeEmbeddingService()
    fake_db = _FakeDb()
    monkeypatch.setattr(
        repos_api,
        "get_app_settings",
        lambda: AppSettings(read_access_mode="demo"),
        raising=False,
    )
    monkeypatch.setattr(repos_api, "get_embedding_service", lambda: embedding_service)
    repos_api._SEMANTIC_SEARCH_CACHE.clear()

    request = RepoSearchRequest(query="vector search", limit=10, min_stars=5)
    first = await repos_api.search_repos(
        request=request,
        http_request=_build_request(),
        user=SimpleNamespace(id=1, active_graph_snapshot_id=10),
        db=fake_db,
    )
    second = await repos_api.search_repos(
        request=request,
        http_request=_build_request(),
        user=SimpleNamespace(id=1, active_graph_snapshot_id=11),
        db=fake_db,
    )

    assert first == []
    assert second == []
    assert embedding_service.calls == 2
    assert fake_db.calls == 2
