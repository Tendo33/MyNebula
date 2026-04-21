"""Shared user resolution helpers."""

from fastapi import HTTPException, status
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.core.config import get_app_settings
from nebula.core.github_client import GitHubClient
from nebula.db import User
from nebula.utils import get_logger

logger = get_logger(__name__)
DEFAULT_USER_BOOTSTRAP_LOCK_KEY = 860_000_001


async def _acquire_default_user_bootstrap_lock(db: AsyncSession) -> None:
    get_bind = getattr(db, "get_bind", None)
    if get_bind is None:
        return

    bind = get_bind()
    dialect_name = bind.dialect.name if bind is not None else ""
    if dialect_name != "postgresql":
        return

    await db.execute(
        text("SELECT pg_advisory_xact_lock(:lock_key)"),
        {"lock_key": DEFAULT_USER_BOOTSTRAP_LOCK_KEY},
    )


async def _get_first_user(db: AsyncSession) -> User | None:
    result = await db.execute(select(User).limit(1))
    return result.scalar_one_or_none()


async def get_default_user(db: AsyncSession) -> User:
    """Get first user from DB or bootstrap one via GitHub token."""
    user = await _get_first_user(db)

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

            await _acquire_default_user_bootstrap_lock(db)
            user = await _get_first_user(db)
            if user is not None:
                return user

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
        except IntegrityError as exc:
            await db.rollback()
            existing_result = await db.execute(
                select(User).where(
                    (User.github_id == github_user.id)
                    | (User.username == github_user.login)
                )
            )
            existing_user = existing_result.scalar_one_or_none()
            if existing_user is not None:
                logger.info(
                    "Recovered default user bootstrap after concurrent insert: "
                    f"{existing_user.username}"
                )
                return existing_user
            logger.exception(
                "Default user bootstrap hit integrity error and recovery failed"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "Failed to create default user from GitHub token. "
                    "Please verify GITHUB_TOKEN and try again."
                ),
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            await db.rollback()
            logger.error(f"Failed to create user from GitHub token: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "Failed to create default user from GitHub token. "
                    "Please verify GITHUB_TOKEN and try again."
                ),
            ) from exc

    return user
