"""Database connection and session management.

This module provides async PostgreSQL connection using SQLAlchemy 2.0
with pgvector extension support.
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import make_url, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from nebula.core.config import get_database_settings
from nebula.utils import get_logger

logger = get_logger(__name__)

# Global engine instance
_engine: AsyncEngine | None = None
AsyncSessionLocal: async_sessionmaker[AsyncSession] | None = None


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _build_alembic_config():
    from alembic.config import Config

    project_root = _get_project_root()
    cfg = Config(str(project_root / "alembic.ini"))
    cfg.set_main_option("script_location", str(project_root / "alembic"))
    return cfg


def _resolve_alembic_heads() -> list[str]:
    """Resolve Alembic migration heads from local migration scripts."""
    from alembic.script import ScriptDirectory

    project_root = _get_project_root()
    alembic_ini = project_root / "alembic.ini"
    if not alembic_ini.exists():
        return []

    cfg = _build_alembic_config()
    script = ScriptDirectory.from_config(cfg)
    return script.get_heads()


async def _read_db_migration_head(engine: AsyncEngine) -> str | None:
    """Read current DB migration head from alembic_version."""
    async with engine.connect() as conn:
        version_table = await conn.scalar(text("SELECT to_regclass('alembic_version')"))
        if version_table is None:
            return None
        current_head = await conn.scalar(
            text("SELECT version_num FROM alembic_version LIMIT 1")
        )
    return str(current_head) if current_head else None


async def _run_alembic_upgrade_head() -> None:
    """Run Alembic upgrade head in a worker thread."""
    from alembic import command

    cfg = _build_alembic_config()
    await asyncio.to_thread(command.upgrade, cfg, "head")


async def _assert_db_schema_at_migration_head(engine: AsyncEngine) -> None:
    """Fail fast when DB schema is not managed by Alembic head."""
    heads = _resolve_alembic_heads()
    if not heads:
        logger.warning("alembic.ini not found, skipping migration head validation")
        return
    if len(heads) > 1:
        raise RuntimeError(
            f"Multiple Alembic heads detected ({heads}). Merge heads before startup."
        )
    expected_head = heads[0]
    current_head = await _read_db_migration_head(engine)
    if current_head is None:
        logger.warning(
            "alembic_version table not found; attempting automatic migration to head"
        )
        try:
            await _run_alembic_upgrade_head()
        except Exception as exc:
            raise RuntimeError(
                "Database schema is not initialized via Alembic and automatic "
                "migration failed. Run `uv run alembic upgrade head` and retry. "
                f"Original error: {exc}"
            ) from exc
        current_head = await _read_db_migration_head(engine)
        if current_head is None:
            raise RuntimeError(
                "Automatic Alembic migration did not initialize alembic_version. "
                "Run `uv run alembic upgrade head` and retry."
            )

    if current_head != expected_head:
        raise RuntimeError(
            "Database migration version mismatch. "
            f"expected={expected_head}, actual={current_head}. "
            "Run `uv run alembic upgrade head`."
        )


async def init_db() -> None:
    """Initialize database connection and create pgvector extension.

    This function should be called once at application startup.
    """
    global _engine, AsyncSessionLocal

    settings = get_database_settings()
    # Log actual connection URL (masked password)
    safe_url = make_url(settings.async_url).render_as_string(hide_password=True)
    logger.info(f"Connecting to database at {safe_url}")

    _engine = create_async_engine(
        settings.async_url,
        echo=False,  # Set to True for SQL debugging
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,  # Verify connections before use
    )

    AsyncSessionLocal = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    # Create pgvector extension if not exists
    async with _engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("pgvector extension ensured")

    await _assert_db_schema_at_migration_head(_engine)


async def close_db() -> None:
    """Close database connection.

    This function should be called at application shutdown.
    """
    global _engine, AsyncSessionLocal

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        AsyncSessionLocal = None
        logger.info("Database connection closed")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for dependency injection.

    Usage with FastAPI:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...

    Yields:
        AsyncSession: Database session
    """
    if AsyncSessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """Get database session as async context manager.

    Usage:
        async with get_db_context() as db:
            result = await db.execute(...)

    Yields:
        AsyncSession: Database session
    """
    if AsyncSessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def check_db_connection() -> bool:
    """Check if database connection is healthy.

    Returns:
        bool: True if connection is healthy, False otherwise
    """
    if _engine is None:
        return False

    try:
        async with _engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
