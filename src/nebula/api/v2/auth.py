"""Admin authentication API routes."""

import hmac
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.core.auth import (
    create_signed_session_token,
    get_admin_session_username,
    get_client_ip,
    is_admin_auth_enabled,
    request_uses_trusted_proxy,
    verify_admin_credentials,
    verify_signed_session_token,
)
from nebula.core.config import AppSettings, get_app_settings
from nebula.db import AdminLoginAttempt, get_db
from nebula.utils import get_logger

router = APIRouter()
logger = get_logger(__name__)

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
    return get_admin_session_username(
        request,
        settings,
        cookie_name=ADMIN_SESSION_COOKIE,
    )


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


def _mask_username(username: str) -> str:
    if len(username) <= 2:
        return "*" * len(username)
    return f"{username[:2]}***"


def _login_rate_limit_keys(
    request: Request,
    username: str,
    settings: AppSettings,
) -> tuple[str, str]:
    client_ip = get_client_ip(request, settings=settings)
    return (
        f"ip:{client_ip}",
        f"user:{username.strip().lower()}",
    )


async def _delete_stale_login_attempts(
    db: AsyncSession,
    *,
    cutoff: datetime,
) -> None:
    await db.execute(
        delete(AdminLoginAttempt).where(AdminLoginAttempt.attempted_at < cutoff)
    )
    await db.commit()


async def _count_recent_login_attempts(
    db: AsyncSession,
    *,
    bucket_key: str,
    cutoff: datetime,
) -> int:
    result = await db.execute(
        select(func.count(AdminLoginAttempt.id)).where(
            AdminLoginAttempt.bucket_key == bucket_key,
            AdminLoginAttempt.attempted_at >= cutoff,
        )
    )
    return int(result.scalar() or 0)


async def _store_login_attempts(
    db: AsyncSession,
    *,
    keys: tuple[str, str],
    attempted_at: datetime,
) -> None:
    db.add_all(
        [
            AdminLoginAttempt(bucket_key=keys[0], attempted_at=attempted_at),
            AdminLoginAttempt(bucket_key=keys[1], attempted_at=attempted_at),
        ]
    )
    await db.commit()


async def _clear_login_attempts(
    db: AsyncSession,
    *,
    keys: tuple[str, str],
) -> None:
    await db.execute(
        delete(AdminLoginAttempt).where(AdminLoginAttempt.bucket_key.in_(keys))
    )
    await db.commit()


async def _enforce_login_rate_limit(
    db: AsyncSession,
    request: Request,
    username: str,
    settings: AppSettings,
) -> None:
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(seconds=settings.admin_login_rate_limit_window_seconds)
    await _delete_stale_login_attempts(db, cutoff=cutoff)
    keys = _login_rate_limit_keys(request, username, settings)
    for key in keys:
        attempt_count = await _count_recent_login_attempts(
            db,
            bucket_key=key,
            cutoff=cutoff,
        )
        if attempt_count >= settings.admin_login_rate_limit_max_attempts:
            logger.warning(
                "Admin login rate limit exceeded "
                f"bucket={key.split(':', 1)[0]} "
                f"username={_mask_username(username)} "
                f"client_ip={get_client_ip(request, settings=settings)}"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many login attempts. Please try again later.",
            )


async def _record_failed_login(
    db: AsyncSession,
    request: Request,
    username: str,
    settings: AppSettings,
) -> None:
    attempted_at = datetime.now(timezone.utc)
    keys = _login_rate_limit_keys(request, username, settings)
    cutoff = attempted_at - timedelta(
        seconds=settings.admin_login_rate_limit_window_seconds
    )
    await _delete_stale_login_attempts(db, cutoff=cutoff)
    await _store_login_attempts(db, keys=keys, attempted_at=attempted_at)
    logger.warning(
        "Admin login failed "
        f"username={_mask_username(username)} "
        f"client_ip={get_client_ip(request, settings=settings)}"
    )


async def _clear_login_failures(
    db: AsyncSession,
    request: Request,
    username: str,
    settings: AppSettings,
) -> None:
    await _clear_login_attempts(
        db,
        keys=_login_rate_limit_keys(request, username, settings),
    )


def _request_is_secure(request: Request, settings: AppSettings) -> bool:
    if settings.force_secure_cookies:
        return True
    if request.url.scheme == "https":
        return True
    if request_uses_trusted_proxy(request, settings=settings):
        forwarded_proto = request.headers.get("x-forwarded-proto", "")
        if forwarded_proto.split(",")[0].strip().lower() == "https":
            return True
    return False


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
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Login as admin and set signed session cookie."""
    if not is_admin_auth_enabled(settings):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin auth is not configured",
        )

    await _enforce_login_rate_limit(db, request, payload.username, settings)
    if not verify_admin_credentials(payload.username, payload.password, settings):
        await _record_failed_login(db, request, payload.username, settings)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    await _clear_login_failures(db, request, payload.username, settings)

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
    secure_cookie = _request_is_secure(request, settings)
    logger.info(
        "Admin login succeeded "
        f"username={_mask_username(settings.admin_username)} "
        f"client_ip={get_client_ip(request, settings=settings)} "
        f"secure_cookie={secure_cookie}"
    )

    response.set_cookie(
        key=ADMIN_SESSION_COOKIE,
        value=token,
        max_age=int(expires_delta.total_seconds()),
        httponly=True,
        secure=secure_cookie,
        samesite="lax",
        path="/",
    )
    response.set_cookie(
        key=ADMIN_CSRF_COOKIE,
        value=csrf_token,
        max_age=int(expires_delta.total_seconds()),
        httponly=False,
        secure=secure_cookie,
        samesite="lax",
        path="/",
    )

    return AdminSessionResponse(username=settings.admin_username)


@router.post("/logout", response_model=AdminSessionResponse)
async def logout_admin(
    response: Response,
    _: str = Depends(require_admin),  # noqa: B008
    __: None = Depends(require_admin_csrf),  # noqa: B008
):
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
