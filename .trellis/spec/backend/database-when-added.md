# Database and Operations

MyNebula already uses PostgreSQL + pgvector through SQLAlchemy async and
Alembic migrations.

## Current Runtime

- Docker Compose runs `db` and `api`; frontend build assets are served by the
  API image.
- Local development usually runs backend and frontend separately while the
  database comes from Docker.
- `frontend/dist` exists in deploy images and FastAPI serves static resources.
- Current version example: `APP_VERSION=1.2.10`.

## Key Environment Groups

- GitHub: `GITHUB_TOKEN`
- Embedding: `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`
- LLM: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_OUTPUT_LANGUAGE`
- Admin auth: `ADMIN_USERNAME`, `ADMIN_PASSWORD`, `ADMIN_SESSION_SECRET`
- Read boundary: `READ_ACCESS_MODE`
- Proxy trust: `TRUST_PROXY_HEADERS`, `TRUSTED_PROXY_IPS`, `TRUSTED_HOSTS`
- Runtime: `DEBUG`, `API_PORT`, `SLOW_QUERY_LOG_MS`,
  `API_QUERY_TIMEOUT_SECONDS`
- Sync tuning: `SYNC_*`

## Rules

- `/docs` and `/redoc` are available only when `DEBUG=true`.
- Admin login/write ability requires both `ADMIN_PASSWORD` and
  `ADMIN_SESSION_SECRET`.
- Prefer Settings full refresh for rerunning business data without schema
  changes.
- Local reset path is `uv run python scripts/reset_db.py` followed by
  `uv run alembic upgrade head`.
