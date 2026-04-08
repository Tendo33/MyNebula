"""Admin authentication utilities."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

from fastapi import Request

from nebula.core.config import AppSettings


def is_admin_auth_enabled(settings: AppSettings) -> bool:
    """Check whether admin auth is configured."""
    return bool(settings.admin_password and settings.admin_session_secret)


def verify_admin_credentials(
    username: str,
    password: str,
    settings: AppSettings,
) -> bool:
    """Verify admin username/password against configured settings."""
    if not is_admin_auth_enabled(settings):
        return False

    if not hmac.compare_digest(username, settings.admin_username):
        return False

    return hmac.compare_digest(password, settings.admin_password)


def get_admin_session_username(
    request: Request,
    settings: AppSettings,
    *,
    cookie_name: str,
) -> str | None:
    """Read the signed admin session cookie and return the username when valid."""
    token = request.cookies.get(cookie_name)
    if not token:
        return None

    payload = verify_signed_session_token(token, settings.admin_session_secret)
    if not payload:
        return None

    username = payload.get("u")
    if not isinstance(username, str):
        return None
    return username


def _trusted_proxy_ip_list(
    *,
    settings: AppSettings | None,
    trusted_proxy_ips: list[str] | None,
) -> list[str]:
    if trusted_proxy_ips is not None:
        return trusted_proxy_ips
    if settings is not None:
        return settings.trusted_proxy_ips_list()
    return []


def _trust_proxy_headers_enabled(
    *,
    settings: AppSettings | None,
    trust_proxy_headers: bool | None,
) -> bool:
    if trust_proxy_headers is not None:
        return trust_proxy_headers
    if settings is not None:
        return settings.trust_proxy_headers
    return False


def request_uses_trusted_proxy(
    request: Request,
    *,
    settings: AppSettings | None = None,
    trust_proxy_headers: bool | None = None,
    trusted_proxy_ips: list[str] | None = None,
) -> bool:
    """Return whether forwarded headers should be trusted for this request."""
    if not _trust_proxy_headers_enabled(
        settings=settings,
        trust_proxy_headers=trust_proxy_headers,
    ):
        return False

    proxy_ips = _trusted_proxy_ip_list(
        settings=settings,
        trusted_proxy_ips=trusted_proxy_ips,
    )
    if not proxy_ips:
        return False

    client_host = request.client.host if request.client and request.client.host else None
    return bool(client_host and client_host in proxy_ips)


def get_client_ip(
    request: Request,
    *,
    settings: AppSettings | None = None,
    trust_proxy_headers: bool | None = None,
    trusted_proxy_ips: list[str] | None = None,
) -> str:
    """Best-effort client IP for audit logging and coarse rate limiting."""
    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if request_uses_trusted_proxy(
        request,
        settings=settings,
        trust_proxy_headers=trust_proxy_headers,
        trusted_proxy_ips=trusted_proxy_ips,
    ) and forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _urlsafe_b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _urlsafe_b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_signed_session_token(
    username: str,
    secret: str,
    expires_in_seconds: int,
) -> str:
    """Create signed session token for admin user."""
    exp = int(time.time()) + expires_in_seconds
    payload = {"u": username, "exp": exp}
    payload_b64 = _urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    )
    signature = hmac.new(
        secret.encode("utf-8"),
        payload_b64.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    return f"{payload_b64}.{signature}"


def verify_signed_session_token(
    token: str,
    secret: str,
) -> dict[str, Any] | None:
    """Verify signed session token and return payload when valid."""
    if not token or "." not in token:
        return None

    try:
        payload_b64, signature = token.split(".", 1)
        expected_signature = hmac.new(
            secret.encode("utf-8"),
            payload_b64.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(signature, expected_signature):
            return None

        payload = json.loads(_urlsafe_b64decode(payload_b64).decode("utf-8"))
        exp = int(payload.get("exp", 0))
        if exp <= int(time.time()):
            return None

        username = payload.get("u")
        if not isinstance(username, str) or not username:
            return None

        return payload
    except Exception:
        return None
