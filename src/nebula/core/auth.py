"""Admin authentication utilities."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

from nebula.core.config import AppSettings


def is_admin_auth_enabled(settings: AppSettings) -> bool:
    """Check whether admin auth is configured."""
    return bool(settings.admin_password)


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
