"""Shared access-policy dependencies for v2 APIs."""

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.application.services.user_service import get_default_user
from nebula.core.auth import get_admin_session_username, is_admin_auth_enabled
from nebula.core.config import AppSettings, get_app_settings
from nebula.db import User, get_db

from .auth import ADMIN_SESSION_COOKIE


async def resolve_single_user(
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> User:
    """Resolve the single-user deployment scope explicitly at the API boundary."""
    return await get_default_user(db)


async def resolve_read_user(
    request: Request,
    db: AsyncSession = Depends(get_db),  # noqa: B008
    settings: AppSettings = Depends(get_app_settings),  # noqa: B008
) -> User:
    """Resolve the current read user under the configured read-access policy."""
    mode = settings.effective_read_access_mode()

    if mode == "authenticated":
        if not is_admin_auth_enabled(settings):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authenticated read mode requires admin auth configuration",
            )

        username = get_admin_session_username(
            request,
            settings,
            cookie_name=ADMIN_SESSION_COOKIE,
        )
        if username != settings.admin_username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authenticated read access requires admin login",
            )

    return await resolve_single_user(db)
