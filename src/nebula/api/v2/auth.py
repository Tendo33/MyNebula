"""Admin authentication API routes."""

import hmac
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from nebula.core.auth import (
    create_signed_session_token,
    is_admin_auth_enabled,
    verify_admin_credentials,
    verify_signed_session_token,
)
from nebula.core.config import AppSettings, get_app_settings

router = APIRouter()

ADMIN_SESSION_COOKIE = "nebula_admin_session"
ADMIN_CSRF_COOKIE = "nebula_admin_csrf"
ADMIN_CSRF_HEADER = "x-csrf-token"


class LoginRequest(BaseModel):
    """Admin login request payload."""

    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=512)


class AdminSessionResponse(BaseModel):
    """Admin session response."""

    authenticated: bool = True
    username: str


class AdminAuthConfigResponse(BaseModel):
    """Admin auth runtime configuration."""

    enabled: bool


def _get_session_secret(settings: AppSettings) -> str:
    return settings.admin_session_secret


def _get_csrf_secret(settings: AppSettings) -> str:
    return f"{settings.admin_session_secret}:csrf"


def _read_session_username(request: Request, settings: AppSettings) -> str | None:
    token = request.cookies.get(ADMIN_SESSION_COOKIE)
    if not token:
        return None

    payload = verify_signed_session_token(token, _get_session_secret(settings))
    if not payload:
        return None

    username = payload.get("u")
    if not isinstance(username, str):
        return None
    return username


def _read_csrf_token(request: Request, settings: AppSettings) -> str | None:
    csrf_token = request.cookies.get(ADMIN_CSRF_COOKIE)
    if not csrf_token:
        return None
    payload = verify_signed_session_token(csrf_token, _get_csrf_secret(settings))
    if not payload:
        return None
    username = payload.get("u")
    if not isinstance(username, str):
        return None
    if username != settings.admin_username:
        return None
    return csrf_token


def require_admin(
    request: Request,
    settings: AppSettings = Depends(get_app_settings),  # noqa: B008
) -> str:
    """Dependency that enforces admin authentication."""
    if not is_admin_auth_enabled(settings):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin auth is not configured",
        )

    username = _read_session_username(request, settings)
    if username != settings.admin_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin login required",
        )

    return username


def require_admin_csrf(
    request: Request,
    settings: AppSettings = Depends(get_app_settings),  # noqa: B008
    _: str = Depends(require_admin),  # noqa: B008
) -> None:
    """Dependency that enforces CSRF checks on mutating admin endpoints."""
    if request.method.upper() in {"GET", "HEAD", "OPTIONS", "TRACE"}:
        return

    cookie_token = _read_csrf_token(request, settings)
    header_token = request.headers.get(ADMIN_CSRF_HEADER)

    if not cookie_token or not header_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF token required",
        )
    if not hmac.compare_digest(cookie_token, header_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF token mismatch",
        )


@router.post("/login", response_model=AdminSessionResponse)
async def login_admin(
    payload: LoginRequest,
    request: Request,
    response: Response,
    settings: AppSettings = Depends(get_app_settings),  # noqa: B008
):
    """Login as admin and set signed session cookie."""
    if not is_admin_auth_enabled(settings):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin auth is not configured",
        )

    if not verify_admin_credentials(payload.username, payload.password, settings):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    expires_delta = timedelta(hours=settings.admin_session_ttl_hours)
    token = create_signed_session_token(
        username=settings.admin_username,
        secret=_get_session_secret(settings),
        expires_in_seconds=int(expires_delta.total_seconds()),
    )
    csrf_token = create_signed_session_token(
        username=settings.admin_username,
        secret=_get_csrf_secret(settings),
        expires_in_seconds=int(expires_delta.total_seconds()),
    )

    response.set_cookie(
        key=ADMIN_SESSION_COOKIE,
        value=token,
        max_age=int(expires_delta.total_seconds()),
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
        path="/",
    )
    response.set_cookie(
        key=ADMIN_CSRF_COOKIE,
        value=csrf_token,
        max_age=int(expires_delta.total_seconds()),
        httponly=False,
        secure=request.url.scheme == "https",
        samesite="lax",
        path="/",
    )

    return AdminSessionResponse(username=settings.admin_username)


@router.post("/logout", response_model=AdminSessionResponse)
async def logout_admin(response: Response):
    """Logout admin and clear session cookie."""
    response.delete_cookie(
        key=ADMIN_SESSION_COOKIE,
        path="/",
        samesite="lax",
    )
    response.delete_cookie(
        key=ADMIN_CSRF_COOKIE,
        path="/",
        samesite="lax",
    )
    return AdminSessionResponse(authenticated=False, username="")


@router.get("/me", response_model=AdminSessionResponse)
async def get_admin_session(username: str = Depends(require_admin)):  # noqa: B008
    """Get current admin session info."""
    return AdminSessionResponse(authenticated=True, username=username)


@router.get("/config", response_model=AdminAuthConfigResponse)
async def get_admin_auth_config(
    settings: AppSettings = Depends(get_app_settings),  # noqa: B008
):
    """Get admin auth availability for UI guidance."""
    return AdminAuthConfigResponse(enabled=is_admin_auth_enabled(settings))
