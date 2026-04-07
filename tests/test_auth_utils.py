"""Tests for admin auth utilities."""

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from nebula.api.v2.auth import (
    ADMIN_CSRF_COOKIE,
    ADMIN_CSRF_HEADER,
    _get_csrf_secret,
    require_admin_csrf,
)
from nebula.core.auth import (
    create_signed_session_token,
    get_client_ip,
    is_admin_auth_enabled,
    verify_admin_credentials,
    verify_signed_session_token,
)
from nebula.core.config import AppSettings


def test_verify_admin_credentials_plaintext_password():
    settings = AppSettings(
        admin_username="owner",
        admin_password="topsecret",
        admin_session_secret="session-secret",
    )

    assert verify_admin_credentials("owner", "topsecret", settings) is True
    assert verify_admin_credentials("owner", "wrong", settings) is False
    assert verify_admin_credentials("guest", "topsecret", settings) is False


def test_admin_auth_disabled_when_password_not_set():
    settings = AppSettings(
        admin_username="owner",
        admin_password="",
        admin_session_secret="session-secret",
    )

    assert is_admin_auth_enabled(settings) is False
    assert verify_admin_credentials("owner", "anything", settings) is False


def test_admin_auth_disabled_when_session_secret_not_set():
    settings = AppSettings(
        admin_username="owner",
        admin_password="topsecret",
        admin_session_secret="",
    )

    assert is_admin_auth_enabled(settings) is False
    assert verify_admin_credentials("owner", "topsecret", settings) is False


def test_signed_session_token_roundtrip():
    token = create_signed_session_token(
        username="owner",
        secret="session-secret",
        expires_in_seconds=60,
    )

    payload = verify_signed_session_token(token, "session-secret")

    assert payload is not None
    assert payload["u"] == "owner"


def test_signed_session_token_rejects_invalid_signature():
    token = create_signed_session_token(
        username="owner",
        secret="session-secret",
        expires_in_seconds=60,
    )

    assert verify_signed_session_token(token, "wrong-secret") is None


def test_signed_session_token_rejects_expired_token():
    token = create_signed_session_token(
        username="owner",
        secret="session-secret",
        expires_in_seconds=-1,
    )

    assert verify_signed_session_token(token, "session-secret") is None


def _build_post_request(csrf_cookie: str | None, csrf_header: str | None) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if csrf_cookie:
        headers.append((b"cookie", f"{ADMIN_CSRF_COOKIE}={csrf_cookie}".encode()))
    if csrf_header:
        headers.append((ADMIN_CSRF_HEADER.encode(), csrf_header.encode()))
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v2/sync/start",
        "headers": headers,
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope)


def test_require_admin_csrf_rejects_missing_token():
    settings = AppSettings(
        admin_username="owner",
        admin_password="topsecret",
        admin_session_secret="session-secret",
    )
    request = _build_post_request(csrf_cookie=None, csrf_header=None)

    with pytest.raises(HTTPException) as exc_info:
        require_admin_csrf(request=request, settings=settings, _="owner")

    assert exc_info.value.status_code == 403


def test_require_admin_csrf_accepts_valid_token():
    settings = AppSettings(
        admin_username="owner",
        admin_password="topsecret",
        admin_session_secret="session-secret",
    )
    csrf_token = create_signed_session_token(
        username="owner",
        secret=_get_csrf_secret(settings),
        expires_in_seconds=60,
    )
    request = _build_post_request(csrf_cookie=csrf_token, csrf_header=csrf_token)

    require_admin_csrf(request=request, settings=settings, _="owner")


def test_get_client_ip_ignores_forwarded_for_by_default():
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v2/auth/me",
        "headers": [(b"x-forwarded-for", b"203.0.113.9")],
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    request = Request(scope)

    assert get_client_ip(request) == "127.0.0.1"
