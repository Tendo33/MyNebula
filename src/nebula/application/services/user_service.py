"""Shared user resolution helpers."""

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.core.config import get_app_settings
from nebula.core.github_client import GitHubClient
from nebula.db import User
from nebula.utils import get_logger

logger = get_logger(__name__)


async def get_default_user(db: AsyncSession) -> User:
    """Get first user from DB or bootstrap one via GitHub token."""
    result = await db.execute(select(User).limit(1))
    user = result.scalar_one_or_none()

    if user is None:
        settings = get_app_settings()

        if not settings.github_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user found and GITHUB_TOKEN not configured. Please set GITHUB_TOKEN in .env file.",
            )

        try:
            async with GitHubClient(access_token=settings.github_token) as client:
                github_user = await client.get_current_user()

            user = User(
                github_id=github_user.id,
                username=github_user.login,
                email=github_user.email,
                avatar_url=github_user.avatar_url,
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
            logger.info(f"Created default user: {user.username}")
        except Exception as exc:
            logger.error(f"Failed to create user from GitHub token: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create user from GitHub token: {exc}",
            ) from exc

    return user
