"""Admin authentication API routes."""

import hashlib
import hmac
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select, text
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
from nebula.db import AdminAuthState, AdminLoginAttempt, get_db
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


def _read_session_username(
    request: Request, settings: AppSettings, *, session_version: int = 0
) -> str | None:
    return get_admin_session_username(
        request,
        settings,
        cookie_name=ADMIN_SESSION_COOKIE,
        expected_session_version=session_version,
    )


def _read_csrf_token(
    request: Request, settings: AppSettings, *, session_version: int = 0
) -> str | None:
    csrf_token = request.cookies.get(ADMIN_CSRF_COOKIE)
    if not csrf_token:
        return None
    payload = verify_signed_session_token(csrf_token, _get_csrf_secret(settings))
    if not payload:
        return None
    if payload.get("sv") != session_version:
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


async def get_admin_session_version(db: AsyncSession) -> int:
    """Return the singleton revocation version, creating it when necessary."""
    get = getattr(db, "get", None)
    if get is None:
        return 0
    state = await get(AdminAuthState, 1)
    if state is not None:
        return int(state.session_version)
    state = AdminAuthState(id=1, session_version=0)
    db.add(state)
    await db.commit()
    return 0


async def increment_admin_session_version(db: AsyncSession) -> int:
    """Revoke every previously issued admin session token."""
    get = getattr(db, "get", None)
    if get is None:
        return 1
    state = await get(AdminAuthState, 1)
    if state is None:
        state = AdminAuthState(id=1, session_version=1)
        db.add(state)
    else:
        state.session_version += 1
    await db.commit()
    return int(state.session_version)


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
    keys: tuple[str, str],
    cutoff: datetime,
) -> None:
    await db.execute(
        delete(AdminLoginAttempt).where(
            AdminLoginAttempt.bucket_key.in_(keys),
            AdminLoginAttempt.attempted_at < cutoff,
        )
    )


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


async def _clear_login_attempts(
    db: AsyncSession,
    *,
    keys: tuple[str, str],
) -> None:
    await db.execute(
        delete(AdminLoginAttempt).where(AdminLoginAttempt.bucket_key.in_(keys))
    )
    await db.commit()


async def _commit_if_supported(db: AsyncSession) -> None:
    commit = getattr(db, "commit", None)
    if commit is not None:
        await commit()


async def _acquire_login_bucket_locks(
    db: AsyncSession,
    *,
    keys: tuple[str, str],
) -> None:
    """Serialize rate-limit reservations across application processes."""
    get_bind = getattr(db, "get_bind", None)
    if get_bind is None:
        return
    bind = get_bind()
    if bind is None or bind.dialect.name != "postgresql":
        return
    for key in sorted(keys):
        digest = hashlib.sha256(key.encode("utf-8")).digest()[:8]
        lock_key = int.from_bytes(digest, byteorder="big", signed=True)
        await db.execute(
            text("SELECT pg_advisory_xact_lock(:lock_key)"),
            {"lock_key": lock_key},
        )


async def _reserve_login_attempt(
    db: AsyncSession,
    request: Request,
    username: str,
    settings: AppSettings,
) -> None:
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(seconds=settings.admin_login_rate_limit_window_seconds)
    keys = _login_rate_limit_keys(request, username, settings)
    await _acquire_login_bucket_locks(db, keys=keys)
    await _delete_stale_login_attempts(db, keys=keys, cutoff=cutoff)
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
                headers={
                    "Retry-After": str(settings.admin_login_rate_limit_window_seconds)
                },
            )
    await _store_login_attempts(db, keys=keys, attempted_at=now_utc)
    await _commit_if_supported(db)


def _log_failed_login(request: Request, username: str, settings: AppSettings) -> None:
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


async def require_admin(
    request: Request,
    settings: AppSettings = Depends(get_app_settings),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> str:
    """Dependency that enforces admin authentication."""
    if not is_admin_auth_enabled(settings):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin auth is not configured",
        )

    session_version = await get_admin_session_version(db)
    username = _read_session_username(
        request, settings, session_version=session_version
    )
    if username != settings.admin_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin login required",
        )

    return username


async def require_admin_csrf(
    request: Request,
    settings: AppSettings = Depends(get_app_settings),  # noqa: B008
    _: str = Depends(require_admin),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> None:
    """Dependency that enforces CSRF checks on mutating admin endpoints."""
    if request.method.upper() in {"GET", "HEAD", "OPTIONS", "TRACE"}:
        return

    session_version = await get_admin_session_version(db)
    cookie_token = _read_csrf_token(request, settings, session_version=session_version)
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

    await _reserve_login_attempt(db, request, payload.username, settings)
    if not verify_admin_credentials(payload.username, payload.password, settings):
        _log_failed_login(request, payload.username, settings)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    await _clear_login_failures(db, request, payload.username, settings)

    expires_delta = timedelta(hours=settings.admin_session_ttl_hours)
    session_version = await get_admin_session_version(db)
    token = create_signed_session_token(
        username=settings.admin_username,
        secret=_get_session_secret(settings),
        expires_in_seconds=int(expires_delta.total_seconds()),
        session_version=session_version,
    )
    csrf_token = create_signed_session_token(
        username=settings.admin_username,
        secret=_get_csrf_secret(settings),
        expires_in_seconds=int(expires_delta.total_seconds()),
        session_version=session_version,
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
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Logout admin and clear session cookie."""
    await increment_admin_session_version(db)
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
