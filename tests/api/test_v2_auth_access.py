from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from nebula.core.auth import create_signed_session_token
from nebula.core.config import AppSettings


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
async def test_resolve_read_user_rejects_authenticated_mode_without_session(monkeypatch):
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
async def test_resolve_read_user_accepts_authenticated_mode_with_valid_session(monkeypatch):
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
async def test_login_admin_uses_secure_cookie_behind_trusted_proxy():
    from nebula.api.v2 import auth as auth_api

    auth_api._LOGIN_ATTEMPTS.clear()
    settings = AppSettings(
        admin_username="owner",
        admin_password="topsecret",
        admin_session_secret="session-secret",
        trust_proxy_headers=True,
    )
    response = Response()

    payload = auth_api.LoginRequest(username="owner", password="topsecret")
    await auth_api.login_admin(
        payload=payload,
        request=_build_request(forwarded_proto="https"),
        response=response,
        settings=settings,
    )

    set_cookie_headers = response.headers.getlist("set-cookie")
    assert len(set_cookie_headers) == 2
    assert all("Secure" in header for header in set_cookie_headers)


@pytest.mark.asyncio
async def test_login_admin_rate_limits_repeated_failures():
    from nebula.api.v2 import auth as auth_api

    auth_api._LOGIN_ATTEMPTS.clear()
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
        )
    assert first_exc.value.status_code == 401

    with pytest.raises(HTTPException) as second_exc:
        await auth_api.login_admin(
            payload=auth_api.LoginRequest(username="owner", password="wrong"),
            request=request,
            response=Response(),
            settings=settings,
        )
    assert second_exc.value.status_code == 429

    auth_api._LOGIN_ATTEMPTS.clear()
