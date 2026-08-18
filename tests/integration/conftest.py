"""Integration tier: real PostgreSQL + pgvector, faked providers.

Selection is by the `integration` marker, applied automatically to everything
in this package. The tier skips — loudly — when no test database is reachable,
so `uv run pytest -q` stays usable without a container running.

## Why `TEST_DATABASE_URL` is required

An earlier design resolved `TEST_DATABASE_URL` -> `DATABASE_URL` -> split
`DATABASE_*` settings. That is unsafe: a developer `.env` may point
`DATABASE_URL` at a real remote database, and this tier runs
`alembic upgrade head` and mutates rows. Falling back to the application's own
connection string would run destructive migrations against production.

So: the URL must be given explicitly, and a non-local host is refused unless
`MYNEBULA_ALLOW_REMOTE_TEST_DB=true` is also set.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", ""}
SKIP_HINT = (
    "integration tier skipped: set TEST_DATABASE_URL to a disposable "
    "PostgreSQL+pgvector database, e.g.\n"
    "  docker run -d --name mynebula-test-db "
    "-e POSTGRES_USER=mynebula_test -e POSTGRES_PASSWORD=mynebula_test "
    "-e POSTGRES_DB=mynebula_test -p 127.0.0.1:5433:5432 pgvector/pgvector:pg16\n"
    "  export TEST_DATABASE_URL="
    "postgresql://mynebula_test:mynebula_test@127.0.0.1:5433/mynebula_test"
)


def pytest_collection_modifyitems(items):
    """Mark everything in this package, so a test cannot forget the marker."""
    for item in items:
        if "tests/integration" in str(item.fspath).replace("\\", "/"):
            item.add_marker(pytest.mark.integration)


def _to_async_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return f"postgresql+asyncpg://{url}"


def _to_sync_url(url: str) -> str:
    return url.replace("+asyncpg", "")


@pytest.fixture(scope="session")
def integration_database_url() -> str:
    raw = os.environ.get("TEST_DATABASE_URL", "").strip()
    if not raw:
        pytest.skip(SKIP_HINT, allow_module_level=True)

    host = urlparse(raw).hostname or ""
    if host not in LOCAL_HOSTS and os.environ.get(
        "MYNEBULA_ALLOW_REMOTE_TEST_DB", ""
    ).lower() not in {"1", "true", "yes"}:
        pytest.skip(
            f"integration tier refused: TEST_DATABASE_URL host {host!r} is not local. "
            "This tier runs migrations and mutates rows. Set "
            "MYNEBULA_ALLOW_REMOTE_TEST_DB=true only for a database you are "
            "willing to have wiped.",
            allow_module_level=True,
        )
    return raw


@pytest.fixture(scope="session")
def migrated_database(integration_database_url: str) -> str:
    """Apply the real Alembic head, not `metadata.create_all`.

    Two migrations create indexes that `create_all` would never produce — the
    ivfflat cosine index the vector search path depends on, and the pg_trgm GIN
    indexes behind Data-page search. Building the schema from the models would
    silently hide migration drift, which is exactly what this tier exists to
    catch.
    """
    import psycopg2

    from alembic import command
    from alembic.config import Config

    sync_url = _to_sync_url(integration_database_url)

    try:
        connection = psycopg2.connect(sync_url)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"integration tier skipped: cannot reach {sync_url}: {exc}")

    with connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute("DROP SCHEMA public CASCADE")
            cursor.execute("CREATE SCHEMA public")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    connection.close()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cfg = Config(os.path.join(project_root, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(project_root, "alembic"))
    cfg.set_main_option("sqlalchemy.url", sync_url)
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = sync_url
    try:
        command.upgrade(cfg, "head")
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous

    return integration_database_url


@pytest_asyncio.fixture
async def integration_engine(migrated_database: str):
    engine = create_async_engine(_to_async_url(migrated_database), poolclass=None)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def integration_db(integration_engine):
    """Session with truncate-based cleanup.

    Transaction-rollback isolation is not usable here: the code under test
    commits, and several of these tests assert on what survives a commit or a
    rollback. Truncating is slower but is the only isolation that does not
    change the behaviour being measured.
    """
    session_factory = async_sessionmaker(
        bind=integration_engine, expire_on_commit=False, autoflush=False
    )
    async with session_factory() as session:
        yield session

    async with integration_engine.begin() as conn:
        await conn.execute(
            text(
                "TRUNCATE graph_snapshot_nodes, graph_snapshot_edges, "
                "graph_snapshot_timeline, graph_snapshots, repo_related_caches, "
                "repo_related_feedbacks, starred_repos, clusters, star_lists, "
                "sync_tasks, pipeline_runs, sync_schedules, users, "
                "admin_login_attempts RESTART IDENTITY CASCADE"
            )
        )


@pytest_asyncio.fixture
async def bound_session_factory(integration_engine, monkeypatch):
    """Point `nebula.db.database` at the test engine.

    Production code reaches the database through `get_db_context()`, so the
    module-level session factory has to be swapped rather than passed in.
    """
    import nebula.db.database as database

    factory = async_sessionmaker(
        bind=integration_engine, expire_on_commit=False, autoflush=False
    )
    monkeypatch.setattr(database, "_engine", integration_engine, raising=False)
    monkeypatch.setattr(database, "AsyncSessionLocal", factory, raising=False)
    return factory
