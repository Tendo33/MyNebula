"""Tests for admin auth utilities."""

from nebula.core.auth import (
    create_signed_session_token,
    is_admin_auth_enabled,
    verify_admin_credentials,
    verify_signed_session_token,
)
from nebula.core.config import AppSettings


def test_verify_admin_credentials_plaintext_password():
    settings = AppSettings(admin_username="owner", admin_password="topsecret")

    assert verify_admin_credentials("owner", "topsecret", settings) is True
    assert verify_admin_credentials("owner", "wrong", settings) is False
    assert verify_admin_credentials("guest", "topsecret", settings) is False


def test_admin_auth_disabled_when_password_not_set():
    settings = AppSettings(admin_username="owner")

    assert is_admin_auth_enabled(settings) is False
    assert verify_admin_credentials("owner", "anything", settings) is False


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
