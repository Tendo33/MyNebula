"""Authentication API routes.

Handles GitHub OAuth flow and JWT token management.
"""

import secrets
from datetime import datetime, timedelta

import jwt
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nebula.core.config import get_app_settings, get_github_settings
from nebula.core.github_client import GitHubClient
from nebula.db import User, get_db
from nebula.schemas.user import AuthResponse, UserResponse
from nebula.utils import get_logger

logger = get_logger(__name__)
router = APIRouter()

# In-memory state storage (use Redis in production)
_oauth_states: dict[str, datetime] = {}


def create_jwt_token(user_id: int, username: str) -> str:
    """Create JWT token for user.

    Args:
        user_id: User database ID
        username: GitHub username

    Returns:
        JWT token string
    """
    settings = get_app_settings()
    payload = {
        "sub": str(user_id),
        "username": username,
        "exp": datetime.utcnow() + timedelta(days=7),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, settings.secret_key, algorithm="HS256")


def verify_jwt_token(token: str) -> dict | None:
    """Verify and decode JWT token.

    Args:
        token: JWT token string

    Returns:
        Decoded payload or None if invalid
    """
    settings = get_app_settings()
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        return None


@router.get("/login")
async def login():
    """Initiate GitHub OAuth flow.

    Returns:
        Redirect to GitHub authorization page
    """
    settings = get_github_settings()

    # Generate random state
    state = secrets.token_urlsafe(32)
    _oauth_states[state] = datetime.utcnow()

    # Clean old states (older than 10 minutes)
    _oauth_states.clear()  # Simple cleanup
    _oauth_states[state] = datetime.utcnow()

    # Build OAuth URL
    client = GitHubClient(settings=settings)
    oauth_url = client.get_oauth_url(state)

    return RedirectResponse(url=oauth_url)


@router.get("/callback")
async def oauth_callback(
    code: str = Query(..., description="OAuth authorization code"),
    state: str = Query(..., description="OAuth state parameter"),
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Handle GitHub OAuth callback.

    Args:
        code: Authorization code from GitHub
        state: State parameter for CSRF verification
        db: Database session

    Returns:
        AuthResponse with JWT token and user info
    """
    # Verify state
    if state not in _oauth_states:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OAuth state",
        )
    del _oauth_states[state]

    settings = get_github_settings()

    try:
        # Exchange code for token
        client = GitHubClient(settings=settings)
        access_token = await client.exchange_code_for_token(code)

        # Get user info
        client = GitHubClient(access_token=access_token, settings=settings)
        github_user = await client.get_current_user()
        await client.close()

        # Find or create user
        result = await db.execute(select(User).where(User.github_id == github_user.id))
        user = result.scalar_one_or_none()

        if user is None:
            # Create new user
            user = User(
                github_id=github_user.id,
                username=github_user.login,
                email=github_user.email,
                avatar_url=github_user.avatar_url,
                access_token=access_token,  # Store encrypted in production
            )
            db.add(user)
            await db.flush()
            logger.info(f"Created new user: {github_user.login}")
        else:
            # Update existing user
            user.access_token = access_token
            user.avatar_url = github_user.avatar_url
            user.email = github_user.email
            logger.info(f"Updated user: {github_user.login}")

        await db.commit()
        await db.refresh(user)

        # Create JWT token
        jwt_token = create_jwt_token(user.id, user.username)

        return AuthResponse(
            access_token=jwt_token,
            token_type="bearer",
            user=UserResponse.model_validate(user),
        )

    except ValueError as e:
        logger.error(f"OAuth error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.exception(f"OAuth callback failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed",
        ) from e


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    token: str = Query(..., description="JWT token"),
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    """Get current authenticated user.

    Args:
        token: JWT access token
        db: Database session

    Returns:
        Current user information
    """
    payload = verify_jwt_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    user_id = int(payload["sub"])
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return UserResponse.model_validate(user)


@router.post("/logout")
async def logout():
    """Logout user (client should discard token).

    Returns:
        Success message
    """
    # JWT tokens are stateless, so logout is handled client-side
    # In production, you might maintain a token blacklist
    return {"message": "Logged out successfully"}
